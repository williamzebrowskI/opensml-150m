"""Complete conversational targets and own-history evaluation from preserved 2048."""
import argparse
import gc
import importlib.metadata
import json
import shutil
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sml_v1.common import atomic_json, file_sha256, fingerprint, lock, read_json
from sft.transfer_control.launch import Tee, preflight

DIR = Path(__file__).resolve().parent
OUTPUT = ROOT/'runs/sft_conversation_completion_2048_v1'


def code_files():
    paths = list(DIR.glob('*.py'))
    for name in ('constraint_completion_1920/engine.py','skill_balance/engine.py',
                 'skill_balance/data.py','skill_balance/evaluate.py',
                 'conversation_generalization/data.py','natural_control/data.py',
                 'conversation_ab/data.py','transfer_control/engine.py',
                 'reading_repair/generation.py','reading_repair/evaluate.py','transfer_control/data.py'):
        paths.append(ROOT/'sft'/name)
    paths += list((ROOT/'sml_v1').glob('*.py'))
    paths += [ROOT/'sft/conversation_ab/data.py',ROOT/'sft/reading_repair/source.py',ROOT/'sft/transfer_control/launch.py']
    paths += [ROOT/'evaluation/generation_comparison/run.py',ROOT/'scripts/completion_playground.py']
    return sorted(set(paths))


def input_files():
    return code_files()+[DIR/n for n in ('config.json','selection.json','review.json',
                                       'review_samples.json','source.json','prepared.json')]


def snapshot_inputs(directory):
    # Helpers may live in the sibling v1 checkout. Preserve repository-relative
    # paths for both versions, and validate the entire mapping before copying.
    pairs=[(src,Path(directory)/src.resolve().relative_to(ROOT.parent)) for src in input_files()]
    if len({dest for _,dest in pairs}) != len(pairs):
        raise ValueError('Input snapshot path collision')
    for src,dest in pairs:
        if dest.exists() and file_sha256(dest) != file_sha256(src):
            raise ValueError('Existing input snapshot changed: '+str(dest))
    for src,dest in pairs:
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(src,dest)


def validate(cfg):
    if cfg['source_step'] != 2048 or cfg['updates'] != 256:
        raise ValueError('Expected preserved 2048 plus 256 updates')
    if cfg['per_update'] != dict(complete=8,commonsense=2,instruction=3,reading=3):
        raise ValueError('Changed balanced batch')
    if sum(cfg['new_counts'].values()) != 8*cfg['updates']:
        raise ValueError('New data exposure count changed')
    if cfg['evaluation_updates'] != list(range(0,257,32)):
        raise ValueError('Changed evaluation schedule')
    if not 0 < cfg['final_lr'] <= cfg['peak_lr'] or not 0 < cfg['warmup_updates'] < cfg['updates']:
        raise ValueError('Invalid LR schedule')
    source = ROOT/cfg['source_bundle']
    if read_json(source/'model.safetensors.json')['step'] != 2048:
        raise ValueError('Wrong source bundle')
    if file_sha256(source/'model.safetensors') != cfg['source_model_sha256']:
        raise ValueError('Preserved source weights changed')


def contract(cfg):
    validate(cfg)
    selection = read_json(DIR/'selection.json')
    review = read_json(DIR/'review.json')
    if selection['config'] != fingerprint(cfg) or review['status'] != 'sample-reviewed':
        raise ValueError('Prepare and review the selected data first')
    if review['sample_sha256'] != file_sha256(DIR/'review_samples.json'):
        raise ValueError('Data review sample changed')
    paths = code_files()+[DIR/n for n in ('config.json','selection.json','review.json','review_samples.json','source.json','prepared.json')]
    paths += [ROOT/cfg['source_bundle']/n for n in ('model.safetensors','model.safetensors.json')]
    paths += [p for p in (ROOT/'tokenizer/bytebpe32k_v1').iterdir() if p.is_file()]
    paths += [ROOT/'runs/stage_b_prose_v1/step_0073243_7f3070237eea/model.safetensors.json']
    paths += [ROOT/'sft'/n for n in ('skill_balance/config.json','skill_balance/selection.json','skill_balance/probes.json')]
    paths += [ROOT/'diagnostics/base_capability_20260924/items.json']
    return dict(source=str(ROOT/cfg['source_bundle']),config=fingerprint(cfg),selection=fingerprint(selection),
                protected={str(p):file_sha256(p) for p in paths},
                runtime={p:importlib.metadata.version(p) for p in ('mlx','numpy','tokenizers','pyarrow','requests','datasets')})


def verify(frozen):
    for name,sha in frozen['protected'].items():
        if file_sha256(name) != sha:
            raise ValueError('Protected input changed: '+name)
    if frozen['runtime'] != {p:importlib.metadata.version(p) for p in frozen['runtime']}:
        raise ValueError('Runtime changed')


def rebuild(cfg,cancelled=lambda: False):
    if cancelled():raise InterruptedError('Stopped')
    cached=read_json(DIR/'prepared.json')
    if cached['receipt'] != read_json(DIR/'selection.json') or cached['receipt']['config'] != fingerprint(cfg):
        raise ValueError('Prepared selection changed')
    if fingerprint(cached['data']) != cached['receipt']['data_hash']:
        raise ValueError('Prepared rows changed')
    return cached['data']


def metadata(cfg,frozen,u):
    return dict(step=2048+u,source_step=2048,additional_updates=u,optimizer_local_step=u,
                next_examples={f:u*n for f,n in cfg['per_update'].items()},contract=fingerprint(frozen),
                experiment='conversation-completion-2048-v1',training_format='plain-user-assistant-eos-v1',
                experimental=True,automatic_promotion=False)


def restore(b,opt,path,frozen,cfg):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from sml_v1.checkpoint_bundle import resolve_bundle
    weights = resolve_bundle(path)
    meta = read_json(weights+'.json')
    u = meta['additional_updates']
    if meta['contract'] != fingerprint(frozen) or not 0 <= u <= cfg['updates']:
        raise ValueError('Resume identity mismatch')
    if meta['step'] != 2048+u or meta['next_examples'] != {f:u*n for f,n in cfg['per_update'].items()}:
        raise ValueError('Resume cursor mismatch')
    b.model.load_weights(weights,strict=True)
    opt.state = tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()))
    mx.eval(b.model.parameters(),opt.state)
    if int(opt.state['step'].item()) != u:
        raise ValueError('Optimizer step mismatch')
    return u


def run(cfg,frozen,data,stop):
    import mlx.core as mx
    from sml_v1.checkpoint_bundle import save_bundle
    from sft.constraint_completion_1920 import engine
    from sft.conversation_completion_2048.data import batch_at
    from sft.conversation_completion_2048.evaluate import assess, review_template, retention_gate
    ep = OUTPUT/'evaluations';ep.mkdir(parents=True,exist_ok=True)
    mx.random.seed(cfg['seed']);b = engine.load(cfg);anchor = engine.load(cfg)
    opt = engine.optimizer(cfg);opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    u = 0;saved = -1;bundle = None
    if (OUTPUT/'latest.json').exists():
        u = restore(b,opt,OUTPUT/'latest.json',frozen,cfg)
        saved = u;bundle = OUTPUT/read_json(OUTPUT/'latest.json')['bundle']
        print('[resume]',u,'with exact optimizer and exposure cursors',flush=True)
    cancelled = lambda: stop['requested'] or (OUTPUT/'STOP').exists()
    def save():
        nonlocal saved,bundle
        bundle = Path(save_bundle(OUTPUT,b.model,opt,metadata(cfg,frozen,u),None,keep=1000000,reserve_gib=10.))
        saved = u
        print('[checkpoint]',2048+u,'retained for review',flush=True)
    def evaluate():
        dest = ep/f'update_{u:05d}.json'
        if dest.exists():
            result = read_json(dest)
            if result['contract'] != fingerprint(frozen) or result['bundle'] != bundle.name:
                raise ValueError('Existing evaluation identity changed')
        else:
            result = assess(b,data,cfg,'dev',cancelled)
            result.update(update=u,step=2048+u,bundle=bundle.name,contract=fingerprint(frozen))
            baseline = result if u == 0 else read_json(ep/'update_00000.json')
            result['retention_gate_passed'] = retention_gate(result,baseline,cfg)
            atomic_json(dest,result)
            print('[conversation-completion-eval]',u,json.dumps(dict(**result['metrics'],
                  retention=result['retention']['metrics'],retention_gate=result['retention_gate_passed'])),
                  '; actual-answer review required',flush=True)
        template = ep/f'review_{u:05d}.json'
        if not template.exists(): atomic_json(template,review_template(result))
    try:
        if saved < 0: save()
        if u in cfg['evaluation_updates'] and not cancelled(): evaluate()
        print(f'[ready] {u}/{cfg["updates"]}; one pass; 8 varied chat + 8 skill replay each update',flush=True)
        while u < cfg['updates'] and not cancelled():
            rows = batch_at(data,cfg,u);t = time.monotonic()
            result = engine.update(b,anchor,opt,rows,data['anchors'][u%len(data['anchors'])],cfg,u+1)
            result.update(seconds=time.monotonic()-t,batch_hash=fingerprint({f:[r['id'] for r in rr] for f,rr in rows.items()}))
            u += 1
            with (OUTPUT/'training.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
            if u == 1 or u%8 == 0:
                print(f'[conversation-completion {u}/{cfg["updates"]}] step={2048+u} chat_ce={result["complete_ce"]:.4f} KL={result["anchor_kl"]:.5f} lr={result["lr"]:.3e}',flush=True)
            if u%cfg['checkpoint_every'] == 0 or u in cfg['evaluation_updates']:save()
            if u in cfg['evaluation_updates'] and not cancelled():evaluate()
        if saved != u:save()
        status = 'stopped' if cancelled() else 'complete'
    except (KeyboardInterrupt,InterruptedError):
        stop['requested'] = True;(OUTPUT/'STOP').touch()
        if saved != u:save()
        status = 'stopped'
    except Exception as exc:
        atomic_json(OUTPUT/'report.json',dict(status='error',update=u,last_saved_update=saved,error=repr(exc)))
        raise
    finally:
        del b,anchor,opt;gc.collect();mx.clear_cache()
    verify(frozen)
    atomic_json(OUTPUT/'report.json',dict(status=status,update=u,step=2048+u,
                selected=None,automatic_promotion=False,reserved_test_evaluated=False,contract=fingerprint(frozen)))
    print('[finished]',status,'step',2048+u,'; inspect saved complete dialogues and retention answers before selection',flush=True)


def checks(cfg,frozen):
    import mlx.core as mx
    from sft.constraint_completion_1920 import engine
    from sft.conversation_completion_2048.data import batch_at
    from sft.reading_repair.generation import verify_generation
    data = rebuild(cfg)
    import tempfile
    with tempfile.TemporaryDirectory(prefix='opensml-conversation-snapshot-') as scratch:
        snapshot_inputs(scratch)
        # An identical retry must succeed; all archived inputs must match.
        snapshot_inputs(scratch)
        for src in input_files():
            copied=Path(scratch)/src.resolve().relative_to(ROOT.parent)
            if file_sha256(copied) != file_sha256(src):
                raise ValueError('Input snapshot byte mismatch')
    from sft.conversation_completion_2048.evaluate import surface, assess
    for split in ('train','dev','test'):
        for row in data[split]['complete']:
            if row['source']=='constructed' and not surface(row,dict(text=row['answer'],stop='end'))['surface_complete']:
                raise ValueError('Constructed gold answer fails its coverage/format checks')
    b = engine.load(cfg)
    ids = set();masks = 0
    for split in ('train','dev','test'):
        for family,rows in data[split].items():
            for row in rows:
                enc = engine.encode(b,row,cfg['context'])
                head = b.encode(engine.prefix(row['prompt']))
                if enc['y'][:len(head)-1] != [-100]*(len(head)-1) or enc['y'][-1] != b.eos:
                    raise ValueError('Assistant mask/EOS mismatch')
                if split == 'train' and family == 'complete':
                    if row['id'] in ids:raise ValueError('Repeated chat target')
                    ids.add(row['id'])
                masks += 1
    if len(ids) != 2048:raise ValueError('Incorrect distinct chat count')
    for u in range(cfg['updates']):
        rr = batch_at(data,cfg,u)
        if {f:len(r) for f,r in rr.items()} != cfg['per_update']:
            raise ValueError('Incomplete balanced batch')
    parity = verify_generation(b,['Hello!',data['train']['complete'][0]['prompt']],24)
    from evaluation.generation_comparison.run import playground
    pg=playground()
    from sft.conversation_completion_2048.dialogues import history
    first,follow=data['dialogues']['dev'][:2]
    req=pg.GenerationRequest(messages=[pg.Message(role='user',content=first['prompt']),
        pg.Message(role='assistant',content=first['answer']),pg.Message(role='user',content=follow['current'])],
        max_tokens=192,temperature=0.,repetition_penalty=1.)
    expected=b.encode(engine.prefix(history(first['prompt'],first['answer'],follow['current'])))
    if pg.prepare_plain_sft_prompt(b.tokenizer,req,2048)['ids']!=expected:
        raise ValueError('Training/playground history mismatch')
    # Exercise actual first-answer history and complete evaluation serialization.
    tiny=dict(data)
    tiny['human']={'dev':data['human']['dev'][:1]}
    tiny['dialogues']={'dev':data['dialogues']['dev'][:2]}
    tiny['dev']={f:rows[:1] for f,rows in data['dev'].items()}
    tiny['dev']['commonsense']=[next(r for r in data['dev']['commonsense'] if r['source']==src)
                              for src in ('commonsenseqa','socialiqa')]
    smoke=assess(b,tiny,cfg)
    if len(smoke['answers'])!=1 or len(smoke['probes'])!=2 or 'gold_history_generation' not in smoke['probes'][1]:
        raise ValueError('Dialogue evaluation smoke check failed')
    # One disposable update proves finite gradients and source immutability.
    anchor = engine.load(cfg);opt = engine.optimizer(cfg)
    opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    before = file_sha256(ROOT/cfg['source_bundle']/'model.safetensors')
    result = engine.update(b,anchor,opt,batch_at(data,cfg,0),data['anchors'][0],cfg,1)
    if not 0 < result['grad_norm'] or int(opt.state['step'].item()) != 1:
        raise ValueError('Disposable optimizer check failed')
    if file_sha256(ROOT/cfg['source_bundle']/'model.safetensors') != before:
        raise ValueError('Parent checkpoint changed')
    del b,anchor,opt;gc.collect();mx.clear_cache();verify(frozen)
    return dict(distinct_chat_targets=len(ids),masked_rows=masks,generation_parity=parity,playground_history_equal=True,
                input_snapshot_verified=True,input_snapshot_files=len(input_files()),
                disposable_grad_norm=result['grad_norm'],evaluation_smoke_completed=True,source_unchanged=True,production_updates=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    for mode in ('prepare','check','run'):g.add_argument('--'+mode,action='store_true')
    p.add_argument('--clear-stop',action='store_true')
    a = p.parse_args();cfg = read_json(DIR/'config.json');validate(cfg)
    mode = 'prepare' if a.prepare else 'check' if a.check else 'run' if a.run else 'plan'
    print(json.dumps(dict(mode=mode,source=str(ROOT/cfg['source_bundle']),
          additional_updates=cfg['updates'],final_step=2048+cfg['updates'],
          new_examples=cfg['new_counts'],per_update=cfg['per_update'],
          output=str(OUTPUT),automatic_promotion=False),indent=2),flush=True)
    if mode == 'plan':return
    preflight()
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
        if mode == 'prepare':
            if (OUTPUT/'contract.json').exists():raise ValueError('Cannot reprepare a started run')
            from sft.conversation_completion_2048.data import build
            data,receipt = build(cfg)
            atomic_json(DIR/'selection.json',receipt)
            atomic_json(DIR/'prepared.json',dict(data=data,receipt=receipt))
            sample=[]
            for family in cfg['human_quotas']:
                rows=[r for r in data['train']['complete'] if r['source']=='no-robots' and r['family']==family]
                # Keep accepted inspected IDs stable while replacing rejected rows.
                accepted=set(read_json(DIR/'review.json').get('accepted_sample_ids',[]))
                reviewed=sorted([r for r in rows if r['id'] in accepted],key=lambda r:r['id'])
                fresh=rows[::max(1,len(rows)//8)]
                chosen=[]
                for row in reviewed+fresh+rows:
                    if row['id'] not in {r['id'] for r in chosen}:chosen.append(row)
                    if len(chosen)==8:break
                sample.extend(chosen)
            # Inspect each authored dialogue family, including both gold replies.
            for family in sorted({r['family'] for r in data['train']['complete'] if r['source']=='constructed'}):
                rows=[r for r in data['train']['complete'] if r['source']=='constructed' and r['family']==family]
                groups=sorted({r['group'] for r in rows})[:2]
                sample.extend(r for r in rows if r['group'] in groups)
            atomic_json(DIR/'review_samples.json',sample)
            print('[prepared] 1536 human targets + 512 constructed targets; review the frozen sample before --check',flush=True)
            return
        frozen = contract(cfg)
        if mode == 'check':
            result = checks(cfg,frozen)
            if contract(cfg) != frozen:raise ValueError('Inputs changed during checks')
            atomic_json(DIR/'readiness.json',dict(status='passed',contract=frozen,checks=result))
            print('[check-passed] masks, EOS, generation parity, disposable update and parent integrity',flush=True)
            return
        receipt = read_json(DIR/'readiness.json')
        if receipt.get('status') != 'passed' or receipt['contract'] != frozen:
            raise ValueError('Readiness stale; run --check')
        if (OUTPUT/'contract.json').exists() and read_json(OUTPUT/'contract.json') != frozen:
            raise ValueError('Existing output belongs to different inputs')
        if (OUTPUT/'report.json').exists() and read_json(OUTPUT/'report.json').get('status') == 'complete':
            print('[already-complete] No automatic extension.',flush=True);return
        if shutil.disk_usage(ROOT).free < 35*1024**3:
            raise OSError('Need at least 35 GiB free for retained checkpoints and reserve')
        OUTPUT.mkdir(parents=True,exist_ok=True)
        if a.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
        if (OUTPUT/'STOP').exists():raise ValueError('STOP exists; use --clear-stop to resume')
        atomic_json(OUTPUT/'contract.json',frozen);atomic_json(OUTPUT/'config.json',cfg)
        snapshot_inputs(OUTPUT/'inputs')
        stop={'requested':False}
        def stop_now(*_):
            stop['requested']=True;(OUTPUT/'STOP').touch()
            print('[stop] Save after current operation; no automatic restart',flush=True)
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop_now)
        with (OUTPUT/'training.log').open('a',buffering=1) as log:
            previous=sys.stdout;sys.stdout=Tee(previous,log)
            try:
                data=rebuild(cfg,lambda:stop['requested'])
                if contract(cfg)!=frozen:raise ValueError('Inputs changed while rebuilding data')
                run(cfg,frozen,data,stop)
            finally:sys.stdout=previous


if __name__ == '__main__':main()

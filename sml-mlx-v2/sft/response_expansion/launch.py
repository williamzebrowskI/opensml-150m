"""Continue step 256 for 128 updates on broader short-answer tasks, OpenSML only."""
import argparse
import gc
import importlib.metadata
import math
from pathlib import Path
import shutil
import signal
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sml_v2.common import atomic_json, file_sha256, fingerprint, lock, read_json
from sft.transfer_control.data import records as old_records, digest
from sft.response_expansion.data import training_records, new_records, audit
from sft.response_expansion.rubrics import evaluate as evaluate_new, audit_references
from sft.transfer_control.launch import preflight, prose_texts, Tee

DIR = Path(__file__).resolve().parent
OUTPUT = ROOT/'runs/sft_response_expansion_v1'
ORIGINAL = ROOT/'runs/sft_transfer_second_pass_v1'


def code_files():
    return sorted(set(DIR.glob('*.py')) | set((ROOT/'sft/transfer_control').glob('*.py')) |
                  set((ROOT/'sft/transfer_continue').glob('*.py')) |
                  set((ROOT/'sml_v2').glob('*.py')) | set((ROOT/'evaluation/base_audit').glob('*.py')))


def verify_hashes(hashes):
    for path, expected in hashes.items():
        if file_sha256(path) != expected: raise ValueError(f'Protected file changed: {path}')


def validate(cfg):
    source = ROOT/cfg['source_bundle']
    meta = read_json(source/'model.safetensors.json')
    expected = {'model.safetensors': cfg['source_model_sha256'],
                'model.safetensors.optimizer.safetensors': cfg['source_optimizer_sha256'],
                'model.safetensors.json': cfg['source_metadata_sha256']}
    verify_hashes({str(source/k): v for k,v in expected.items()})
    original = read_json(ORIGINAL/'contract.json')
    if meta['continuation_contract'] != fingerprint(original) or meta['arm'] != 'opensml':
        raise ValueError('Source is not the verified OpenSML second-pass SFT')
    if meta['step'] != cfg['source_step'] or meta['training_format'] != 'plain-user-assistant-eos-v1':
        raise ValueError('Source step/format mismatch')
    if meta['next_example'] != 1024 or read_json(ORIGINAL/'report.json')['status'] != 'complete':
        raise ValueError('Source second pass is incomplete')
    verify_hashes({str(ROOT/p): sha for p,sha in original['code'].items()})
    old_cfg = read_json(ORIGINAL/'input/sft/transfer_continue/config.json')
    for k in ('batch','microbatch','weight_decay','betas','clip_norm','learning_rate'):
        if cfg[k] != old_cfg[k]: raise ValueError(f'Unexpected optimizer/recipe change: {k}')
    if cfg['legacy_max_new_tokens'] != old_cfg['max_new_tokens'] or cfg['max_new_tokens'] != 64:
        raise ValueError('Unexpected generation budgets')
    if cfg['additional_updates']*cfg['batch'] != len(training_records()):
        raise ValueError('Continuation must visit all 1,024 training examples exactly once')
    if cfg['source_step'] != 256 or cfg['additional_updates'] != 128 or cfg['learning_rate'] != 3e-6:
        raise ValueError('Only the agreed step-256, 128-update, constant-3e-6 continuation is supported')
    if cfg['evaluation_updates'] != [0,32,64,128]: raise ValueError('Unexpected evaluation schedule')
    for split in ('train','dev','test'): audit_references(new_records(split))
    return source, original


def contract(cfg):
    source, original = validate(cfg)
    protected = dict(original['protected'])
    # Inherit all protected base/comparison files and preserve the completed second pass.
    for p in ORIGINAL.rglob('*'):
        if p.is_file(): protected[str(p)] = file_sha256(p)
    verify_hashes(protected)
    return dict(config=fingerprint(cfg), source=str(source), protected=protected,
                data=audit(), prose=digest(prose_texts()),
                code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},
                runtime={p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','numpy','transformers','tokenizers')})


def restore(weights, cfg):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from sml_v2.precision import MasterAdamW
    from sft.transfer_control.engine import load
    backend = load('opensml')  # This launcher never loads/trains the reference.
    backend.model.load_weights(str(weights))
    optimizer = MasterAdamW(learning_rate=cfg['learning_rate'], betas=tuple(cfg['betas']), weight_decay=cfg['weight_decay'])
    optimizer.state = tree_unflatten(list(mx.load(str(weights)+'.optimizer.safetensors').items()))
    optimizer.learning_rate = cfg['learning_rate']
    mx.eval(backend.model.parameters(), optimizer.state)
    return backend, optimizer


def optimizer_step(optimizer):
    return int(optimizer.state['step'].item())


def ranking(result):
    new=result['new_tasks'];old=result['legacy']
    # Frozen development-only selection: equal weight to new response coverage
    # and the unchanged factual pair test. This is a proxy, not a chat score.
    return [(new['rule_macro']+old['both_correct'])/2,old['both_correct'],
            new['rule_macro'],old['exact'],-new['assistant_nll']]


def assess(backend, cfg, update, source_step, baseline_prose):
    from sft.transfer_control.engine import evaluate, prose_loss
    new=evaluate_new(backend,new_records('dev'),cfg['max_new_tokens'])
    legacy=evaluate(backend,old_records('dev'),cfg['legacy_max_new_tokens'],include_open=False)
    prose=prose_loss(backend,prose_texts())
    return dict(additional_updates=update,step=source_step+update,new_tasks=new,legacy=legacy,
                prose=prose,prose_change_from_parent=0. if baseline_prose is None else prose-baseline_prose)


def check(cfg, frozen):
    import mlx.core as mx
    from sml_v2.checkpoint_bundle import resolve_bundle
    from mlx.utils import tree_flatten
    from sft.transfer_control.engine import encode, arrays, objective, generate
    preflight()
    source = resolve_bundle(ROOT/cfg['source_bundle'])
    backend, optimizer = restore(source,cfg)
    if optimizer_step(optimizer) != cfg['source_step']: raise ValueError('Optimizer step was not restored')
    state = dict(tree_flatten(optimizer.state))
    masters = [v for k,v in state.items() if k.endswith('.master')]
    if not masters or any(v.dtype != mx.float32 for v in masters): raise ValueError('FP32 master weights missing')
    rows = training_records(); encoded = [encode(backend,r) for r in rows]
    heldout = [encode(backend,r) for split in ('dev','test') for r in new_records(split)]
    x,y = arrays(encoded[:2],backend.pad)
    loss = float(objective(backend.model,x,y,False).item())
    if not math.isfinite(loss): raise ValueError('Nonfinite source forward pass')
    probe = generate(backend,rows[0]['prompt'],limit=8)
    receipt = dict(status='passed',contract=frozen,source_optimizer_step=optimizer_step(optimizer),
                   fp32_master_tensors=len(masters),learning_rate=cfg['learning_rate'],forward_loss=loss,
                   examples=len(rows),assistant_targets=sum(r['targets'] for r in encoded),
                   heldout_examples=len(heldout),maximum_tokens=max(len(r['x']) for r in encoded+heldout),
                   generation_probe=probe,training_started=False,pretrained_updates=0,reference_loaded=False)
    del backend,optimizer;gc.collect();mx.clear_cache()
    verify_hashes(frozen['protected'])
    atomic_json(DIR/'readiness.json',receipt)
    print('[check-passed] OpenSML step 256 and FP32 optimizer restored; 768 new + 256 rehearsal examples verified; no training started.',flush=True)


def run(cfg, frozen, stop):
    import mlx.core as mx
    from sml_v2.checkpoint_bundle import resolve_bundle, save_bundle
    from sml_v2.pretrain import clip_gradients
    from sft.transfer_control.engine import encode, arrays, gradients
    report_path = OUTPUT/'report.json'
    if report_path.exists() and read_json(report_path)['status']=='complete':
        print('[already-complete] The additional pass has finished; no repeat.',flush=True);return
    original_weights = str(ROOT/cfg['source_bundle']/'model.safetensors')
    source_step = cfg['source_step']; update = 0
    latest = OUTPUT/'latest.json'
    if latest.exists():
        weights = resolve_bundle(latest); meta = read_json(weights+'.json')
        if meta['expansion_contract'] != fingerprint(frozen): raise ValueError('Expansion contract changed')
        if meta['next_example'] != meta['additional_updates']*cfg['batch']: raise ValueError('Data cursor mismatch')
        update, report, best_key = meta['additional_updates'], meta['report'], meta['best_key']
    else:
        weights = resolve_bundle(ROOT/cfg['source_bundle'])
        best_key = None
        report = dict(status='running',arm='opensml',source=original_weights,source_step=source_step,
                      baseline_prose=None,parent_key=None,training=[],evaluations=[],promoted=False)
    backend,optimizer = restore(weights,cfg)
    if optimizer_step(optimizer) != source_step+update: raise ValueError('Optimizer/cursor mismatch')
    report['status']='running'
    train = training_records()
    if not 0 <= update <= cfg['additional_updates']: raise ValueError('Invalid continuation cursor')
    if latest.exists() and (OUTPUT/'best.json').exists():
        best_weights = resolve_bundle(OUTPUT/'best.json')
        report['best']=dict(step=read_json(best_weights+'.json')['step'],path=str(Path(best_weights).parent))

    def save(best=False):
        report.update(step=source_step+update,additional_updates=update)
        if optimizer_step(optimizer) != source_step+update: raise ValueError('Optimizer count lost')
        meta=dict(step=source_step+update,source_step=source_step,additional_updates=update,
                  next_example=update*cfg['batch'],expansion_contract=fingerprint(frozen),
                  report=report,best_key=best_key,arm='opensml',experimental=True,
                  training_format='plain-user-assistant-eos-v1',playground_compatible=False)
        path=save_bundle(OUTPUT,backend.model,optimizer,meta,None,best=best,keep=1)
        report['latest']=path
        if best:report['best']=dict(step=source_step+update,path=path)
        atomic_json(report_path,report)
        print(f'[checkpoint {source_step+update}] '+('best + latest' if best else 'latest'),flush=True)

    def evaluation():
        result=assess(backend,cfg,update,source_step,report['baseline_prose'])
        atomic_json(OUTPUT/'evaluations'/f'step_{source_step+update:07d}.json',result)
        if report['baseline_prose'] is None: report['baseline_prose']=result['prose']
        row=dict(step=source_step+update,additional_updates=update,new_rule_macro=result['new_tasks']['rule_macro'],
                 legacy_both_correct=result['legacy']['both_correct'],legacy_exact=result['legacy']['exact'],
                 stopped=result['new_tasks']['stopped'],assistant_nll=result['new_tasks']['assistant_nll'],
                 prose_change_from_parent=result['prose_change_from_parent'])
        report['evaluations'].append(row)
        print(f'[eval {source_step+update}] new_rules={row["new_rule_macro"]:.1%} legacy_paired={row["legacy_both_correct"]:.1%} prose_change_from_parent={row["prose_change_from_parent"]:+.4f}',flush=True)
        return ranking(result)

    try:
        if not latest.exists():
            best_key=evaluation();report['parent_key']=best_key
            # Include the unchanged parent in selection, so an unsuccessful
            # expansion cannot silently replace it with a lower-scoring candidate.
            save(best=True)
        print(f'[ready] OpenSML only; total_step={source_step+update}; additional={update}/{cfg["additional_updates"]}; constant_lr={cfg["learning_rate"]:.3e}',flush=True)
        while update<cfg['additional_updates']:
            if (OUTPUT/'STOP').exists():stop['requested']=True
            if stop['requested']:break
            rows=train[update*cfg['batch']:(update+1)*cfg['batch']]
            if len(rows)!=cfg['batch']:raise ValueError('Unexpected end of response expansion')
            x,y=arrays([encode(backend,r) for r in rows],backend.pad)
            value,grad=gradients(backend.model,x,y,False,cfg['microbatch'])
            grad,norm=clip_gradients(grad,cfg['clip_norm']);mx.eval(value,grad,norm)
            if not math.isfinite(float(value.item())) or not math.isfinite(float(norm.item())):raise ValueError('Nonfinite update')
            optimizer.learning_rate=cfg['learning_rate']
            optimizer.update(backend.model,grad);mx.eval(backend.model.parameters(),optimizer.state)
            update+=1
            report['training'].append(dict(step=source_step+update,additional_update=update,loss=float(value.item()),
                                           lr=cfg['learning_rate'],grad_norm=float(norm.item()),
                                           examples=update*cfg['batch'],batch_hash=digest([r['id'] for r in rows])))
            if update==1 or update%8==0:print(f'[response-expansion {update}/{cfg["additional_updates"]}] total_step={source_step+update} loss={value.item():.4f} lr={cfg["learning_rate"]:.3e}',flush=True)
            if update in cfg['evaluation_updates']:
                key=evaluation();improved=key>best_key
                if improved:best_key=key
                save(improved)
            elif update%cfg['checkpoint_every']==0:save()
        if stop['requested']:
            report['status']='stopped';save();print('[stopped] OpenSML continuation saved.',flush=True);return
        selected=resolve_bundle(OUTPUT/'best.json')
        best_step=read_json(selected+'.json')['step']
        print('[fresh-test] Selection is frozen; evaluating unchanged step 256 and selected best on 156 new test prompts.',flush=True)
        backend.model.load_weights(original_weights);mx.eval(backend.model.parameters())
        parent_test=evaluate_new(backend,new_records('test'),cfg['max_new_tokens'])
        backend.model.load_weights(selected);mx.eval(backend.model.parameters())
        best_test=evaluate_new(backend,new_records('test'),cfg['max_new_tokens'])
        print(f'[fresh-test-result] parent_rules={parent_test["rule_macro"]:.1%} selected_rules={best_test["rule_macro"]:.1%}; inspect saved answers.',flush=True)
        result=dict(selected=selected,parent=original_weights,parent_evaluation=parent_test,
                    selected_evaluation=best_test,fresh_test=True,selection_unchanged=True,
                    limitation='Related synthetic task families; repeated paraphrases are not independent topics. Read saved answers.')
        atomic_json(OUTPUT/'fresh_test.json',result)
        report.update(status='complete',step=source_step+update,additional_updates=update,
                      best=dict(step=best_step,path=str(Path(selected).parent)),
                      improved_on_development=best_key>report['parent_key'],test=str(OUTPUT/'fresh_test.json'))
        atomic_json(report_path,report)
        print(f'[finished] OpenSML additional={update}; total_step={source_step+update}; selected_best_step={best_step}; no automatic restart.',flush=True)
    except Exception as exc:
        report.update(status='error',error=repr(exc));atomic_json(report_path,report);raise
    finally:
        del backend,optimizer;gc.collect();mx.clear_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--run',action='store_true');mode.add_argument('--check',action='store_true')
    parser.add_argument('--clear-stop',action='store_true');args=parser.parse_args()
    cfg=read_json(DIR/'config.json');frozen=contract(cfg)
    import json
    print(json.dumps(dict(mode='run' if args.run else 'check' if args.check else 'plan',model='OpenSML only',
                         source=frozen['source'],source_step=cfg['source_step'],additional_updates=cfg['additional_updates'],
                         final_step=cfg['source_step']+cfg['additional_updates'],learning_rate=cfg['learning_rate'],
                         optimizer='restored FP32 AdamW moments, master weights and step counter',
                         examples=1024,new_examples=768,rehearsal_examples=256,
                         data='Original local examples; no HF download or teacher',
                         output=str(OUTPUT),reference_training=False),indent=2),flush=True)
    if not args.run and not args.check:return
    with lock(ROOT/'sft/.experiment.lock'):
        if args.check:check(cfg,frozen);return
        ready=read_json(DIR/'readiness.json')
        if ready['status']!='passed' or ready['contract']!=frozen:raise ValueError('Setup changed; repeat --check')
        preflight()
        if (OUTPUT/'contract.json').exists():
            if read_json(OUTPUT/'contract.json')!=frozen:raise ValueError('Run contract changed')
        else:
            atomic_json(OUTPUT/'contract.json',frozen)
            for p in code_files()+[DIR/'config.json',DIR/'README.md']:
                dest=OUTPUT/'input'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
        if args.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
        if (OUTPUT/'STOP').exists():raise ValueError('STOP exists; pass --clear-stop to resume')
        stop=dict(requested=False)
        def handler(sig,frame):
            stop['requested']=True;(OUTPUT/'STOP').touch()
            print('[stop] Saving after the current update/evaluation; please wait.',flush=True)
        handlers={s:signal.signal(s,handler) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            with (OUTPUT/'console.log').open('a') as log:
                old=sys.stdout,sys.stderr;sys.stdout=Tee(old[0],log);sys.stderr=Tee(old[1],log)
                try:run(cfg,frozen,stop)
                finally:sys.stdout,sys.stderr=old
        finally:
            for s,h in handlers.items():signal.signal(s,h)
            verify_hashes(frozen['protected'])


if __name__=='__main__':main()

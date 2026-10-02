"""Long text-only reference DPO from 768, with conversation replay and retention KL."""
import argparse,gc,importlib.metadata,json,math,shutil,signal,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v1.common import atomic_json,read_json,file_sha256,fingerprint,lock
DIR=Path(__file__).resolve().parent;OUTPUT=ROOT/'runs/dpo_chat_768_v1'

def accepts_contract(digest,frozen):
    return digest==fingerprint(frozen)

def validate(cfg):
    assert cfg['source_step']==768 and cfg['updates']==1024 and cfg['context']==2048
    assert cfg['updates']*cfg['per_update']['preference']==cfg['preference_train_pairs']*cfg['preference_epochs']
    assert cfg['automatic_quality_stop'] is False and cfg['automatic_pruning'] is False
    source=ROOT/cfg['source_bundle'];m=read_json(source/'model.safetensors.json');manifest=read_json(source/'manifest.json')
    assert m['step']==768 and m['source_step']==512 and m['training_format']=='plain-user-assistant-eos-v1'
    assert file_sha256(source/'model.safetensors')==cfg['source_model_sha256']==manifest['files']['model.safetensors']['sha256']
    assert file_sha256(source/'model.safetensors.json')==manifest['files']['model.safetensors.json']['sha256']
    parent_cfg=read_json(ROOT/'sft/text_followup_512_v1/config.json')
    for k in ('context','validation_conversations','generation_conversations','max_new_tokens','training_counts'):
        assert cfg[k]==parent_cfg[k],('Retention evaluation drift',k)
    assert cfg['constraint_eval_conversations']==32

def prepared(cfg):
    p=read_json(DIR/'prepared.json');assert p['receipt']['config']==fingerprint(cfg)
    assert p['receipt']['data_sha256']==fingerprint(p['data'])
    for path,digest in p['receipt']['inputs'].items():
        if file_sha256(path)!=digest:raise ValueError('Prepared input changed: '+path)
    return p['data']

def contract(cfg):
    source=ROOT/cfg['source_bundle'];parent=read_json(source.parent/'contract.json');protected=dict(parent['protected'])
    for path,digest in protected.items():
        if file_sha256(path)!=digest:raise ValueError('Parent protected input changed: '+path)
    files=list(DIR.glob('*.py'))+[DIR/n for n in ('config.json','prepared.json','review_samples.json','review.json','rejections.json')]
    files += [Path(p) for p in read_json(DIR/'prepared.json')['receipt']['inputs']]
    files += [source/n for n in ('model.safetensors','model.safetensors.json','manifest.json')]+[source.parent/'contract.json']
    review=read_json(DIR/'review.json');assert review['status']=='sample-reviewed' and review['sample_sha256']==file_sha256(DIR/'review_samples.json')
    protected.update({str(p):file_sha256(p) for p in files})
    return dict(config=cfg,protected=protected,runtime={p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','numpy','tokenizers','pyarrow')})

def inspect_data(cfg,data):
    from collections import Counter
    from sft.dpo_chat_768_v1.data import encode_reply,turns,eligible,visible_prefix,PARENT,text_scope
    from sml_v1.tokenization import Tokenizer
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');seen=set();checked=0
    for split in ('preference','preference_dev','preference_test'):
        ids={r['id'] for r in data[split]};assert not seen&ids and len(ids)==len(data[split]);seen|=ids
        for row in data[split]:
            assert eligible(row['messages']+[dict(role='assistant',content=row['chosen'])])
            assert text_scope(row['messages']+[dict(role='assistant',content=row[k]) for k in ('chosen','rejected')])
            assert row['chosen'].strip()!=row['rejected'].strip()
            for field in ('chosen','rejected'):
                e=encode_reply(tok,row['messages'],row[field],cfg['context']);head=tok.encode(visible_prefix(row['messages']))
                assert e['y'][:len(head)-1]==[-100]*(len(head)-1) and e['y'][-1]==tok.eos
                assert e['targets']==sum(t!=-100 for t in e['y']);checked+=1
    original=read_json(PARENT)['data'];foundation=read_json(ROOT/'sft/conversation_foundation_original/prepared.json')['data']
    known={fingerprint(r['messages']) for r in original['train']+foundation['train'][:8192]}
    held={fingerprint(r['messages']) for s in ('dev','test') for r in original[s]}
    for row in data['replay']:
        f=fingerprint(row['messages']);assert f in known and f not in held and eligible(row['messages']) and text_scope(row['messages'])
        for e in turns(tok,row,cfg['context']):assert e['y'][-1]==tok.eos;checked+=1
    for family,counts in [('preference',1024),('replay',2048)]:
        assert len(data[family])==counts
        assert Counter(data['schedule'][family])==Counter({i:2 for i in range(counts)})
        assert len(data['schedule'][family])==cfg['updates']*cfg['per_update'][family]
    from sft.dpo_chat_768_v1.data import batch_at
    for u in range(cfg['updates']):
        rows=batch_at(data,cfg,u)
        assert len(rows['preference'])==2 and len(rows['replay'])==4
        assert sum(r['source']=='conversation' for r in rows['replay'])==3
    return dict(preference_pairs=1024,unique_replay_conversations=2048,epochs=2,assistant_sequences_checked=checked,holdout_groups_disjoint=True,replay_verified_against_parent_training=True,original_768_eval_preserved=True)

def check(cfg,data,frozen):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten,tree_unflatten
    from sml_v1.checkpoint_bundle import save_bundle,resolve_bundle
    from sft.dpo_chat_768_v1 import engine
    from sft.dpo_chat_768_v1.data import turns,visible_prefix
    from sft.transfer_control.engine import arrays
    from evaluation.generation_comparison.run import playground
    result=inspect_data(cfg,data);print('[check] data and masks',flush=True)
    b=engine.load(cfg);anchor=engine.load(cfg);pg=playground()
    # Frozen reference and trainable model must be separate tensors/models.
    assert b.model is not anchor.model
    toy=mx.array([-3.,-5.]);counts=mx.array([3.,5.]);dpo,ce,margin=engine.preference_terms(toy,counts,toy,cfg['dpo_beta'])
    mx.eval(dpo,ce,margin);assert abs(dpo.item()-math.log(2))<1e-6 and margin.item()==0
    grad=mx.grad(lambda lp:engine.preference_terms(lp,counts,toy,cfg['dpo_beta'])[0])(toy);mx.eval(grad)
    assert grad[0].item()<0 and grad[1].item()>0
    row=min(data['preference'],key=lambda r:len(r['chosen'])+len(r['rejected'])+sum(len(m['content']) for m in r['messages']))
    x,y=engine.pair_arrays(b,row,cfg);ref,n=engine.sequence_logps(anchor.model,x,y);lp,_=engine.sequence_logps(b.model,x,y)
    dpo,ce,margin=engine.preference_terms(lp,n,ref,cfg['dpo_beta']);mx.eval(dpo,ce,margin)
    assert abs(dpo.item()-math.log(2))<1e-5 and abs(margin.item())<1e-6
    # Independent per-token scoring checks masked sums, unequal-length padding and EOS.
    from sft.dpo_chat_768_v1.data import encode_reply
    for index,field in enumerate(('chosen','rejected')):
        e=encode_reply(b.tokenizer,row['messages'],row[field],cfg['context']);xx,yy=arrays([e],b.pad)
        pos=mx.array([i for i,t in enumerate(e['y']) if t!=-100]);z=b.model(xx)['logits'][0,pos]
        independent=-nn.losses.cross_entropy(z,yy[0,pos],reduction='sum');mx.eval(independent)
        assert abs(independent.item()-lp[index].item())<2e-4
    replay=min(data['replay'],key=lambda r:sum(len(m['content']) for m in r['messages']))
    multi=next(r for r in data['replay'] if len(r['messages'])>=4)
    for r in (replay,multi):
        for e in turns(b.tokenizer,r,cfg['context']):
            messages=r['messages'][:e['message_index']];instruction=messages[0]['content'] if messages[0]['role']=='system' else ''
            body=messages[1:] if instruction else messages
            opts=pg.GenerationRequest(messages=[pg.Message(**m) for m in body],instruction=instruction,max_tokens=1)
            actual=pg.prepare_plain_sft_prompt(b.tokenizer,opts,2048)
            assert actual['ids']==b.encode(visible_prefix(messages)) and not actual['trimmed_tokens']
    e=turns(b.tokenizer,replay,cfg['context'])[0];xx,yy=arrays([e],b.pad);pos=mx.array([i for i,t in enumerate(e['y']) if t!=-100])
    rr=engine.reference_logp(anchor,xx,pos);_,aux=engine.replay_objective(b.model,xx,yy,pos,rr,cfg);mx.eval(aux);assert abs(aux[1].item())<1e-6
    messages=[dict(role='user',content='Hello!')];assert engine.generate(b,messages,12)==engine.generate(b,messages,12,full=True)
    anchor_before=dict(tree_flatten(anchor.model.parameters()))
    print('[check] DPO, prompt parity, reference KL, decode parity',flush=True)
    opt=engine.optimizer(cfg);opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    batch=dict(preference=[row],replay=[replay,multi]);first=engine.update(b,anchor,opt,batch,cfg,1)
    assert first['grad_norm']>0 and int(opt.state['step'].item())==1
    assert all(bool(mx.array_equal(v,dict(tree_flatten(anchor.model.parameters()))[k]).item()) for k,v in anchor_before.items())
    with tempfile.TemporaryDirectory(prefix='opensml-dpo-check-') as scratch:
        bundle=save_bundle(scratch,b.model,opt,dict(step=1),None,keep=100,reserve_gib=1);weights=resolve_bundle(bundle)
        restored=engine.load(cfg,weights);opt2=engine.optimizer(cfg)
        opt2.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()));mx.eval(opt2.state)
        before=dict(tree_flatten(opt.state));after=dict(tree_flatten(opt2.state))
        assert before.keys()==after.keys() and all(bool(mx.array_equal(v,after[k]).item()) for k,v in before.items())
        engine.update(b,anchor,opt,batch,cfg,2);engine.update(restored,anchor,opt2,batch,cfg,2)
        a=dict(tree_flatten(b.model.parameters()));z=dict(tree_flatten(restored.model.parameters()))
        assert a.keys()==z.keys() and all(bool(mx.array_equal(v,z[k]).item()) for k,v in a.items())
        del restored,opt2,before,after,a,z
    print('[check] frozen reference and exact next-update resume',flush=True)
    tiny=dict(cfg,validation_conversations=2,generation_conversations=5,max_new_tokens=8,constraint_eval_conversations=1,preference_eval_pairs=2)
    smoke=engine.assess(b,anchor,data,tiny);assert smoke['metrics']['preference_dev_pairs']==2
    del b,anchor,opt;gc.collect();mx.clear_cache();assert contract(cfg)==frozen
    result.update(dpo_sign_and_initial_loss=True,answer_eos_mask_and_padding=True,native_history_parity=True,cached_decode_parity=True,zero_initial_kl=True,frozen_reference_unchanged=True,exact_optimizer_restore=True,exact_next_update_weights=True,evaluation_smoke=True,disposable_update=first,production_updates=0,source_unchanged=True)
    atomic_json(DIR/'readiness.json',dict(status='passed',contract=frozen,checks=result));print('[check-passed]',json.dumps(result),flush=True)


def train(cfg,data,frozen,stop):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from sml_v1.checkpoint_bundle import save_bundle,resolve_bundle
    from sft.dpo_chat_768_v1 import engine
    from sft.dpo_chat_768_v1.data import batch_at
    OUTPUT.mkdir(parents=True,exist_ok=True)
    path=OUTPUT/'contract.json'
    if path.exists() and read_json(path)!=frozen:raise ValueError('Existing run contract changed')
    if not path.exists():atomic_json(path,frozen)
    # Archive the small recipe files; prepared data remain fingerprinted at their original path.
    for p in list(DIR.glob('*.py'))+[DIR/'config.json',DIR/'review.json']:
        dest=OUTPUT/'inputs'/p.name;dest.parent.mkdir(exist_ok=True)
        if not dest.exists():shutil.copy2(p,dest)
    mx.random.seed(cfg['seed']);b=engine.load(cfg);anchor=engine.load(cfg);opt=engine.optimizer(cfg)
    opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    u=0;saved=0;bundle=None
    if (OUTPUT/'latest.json').exists():
        weights=resolve_bundle(OUTPUT/'latest.json');meta=read_json(weights+'.json');u=meta['additional_updates']
        assert accepts_contract(meta['contract'],frozen) and meta['source_step']==768 and 0<=u<=cfg['updates']
        b.model.load_weights(weights,strict=True)
        opt.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()));mx.eval(opt.state,b.model.parameters())
        assert int(opt.state['step'].item())==u and meta['preference_exposures']==u*cfg['per_update']['preference'] and meta['replay_exposures']==u*cfg['per_update']['replay']
        saved=u;bundle=Path(weights).parent
        print('[resume]',u,'with saved optimizer and data cursor',flush=True)
        log_path=OUTPUT/'training.jsonl'
        if log_path.exists():
            lines=log_path.read_text().splitlines(keepends=True);committed=[];tail=[]
            for line in lines:
                try:row=json.loads(line)
                except json.JSONDecodeError:tail.append(line);continue
                (committed if row['update']<=u else tail).append(line)
            if tail:
                (OUTPUT/f'training_uncommitted_{time.time_ns()}.jsonl').write_text(''.join(tail))
                log_path.write_text(''.join(committed))
    def cancelled():return stop['requested'] or (OUTPUT/'STOP').exists()
    def save():
        nonlocal saved,bundle
        if u==saved:return
        meta=dict(step=768+u,additional_updates=u,source_step=768,source_bundle=cfg['source_bundle'],
                  source_model_sha256=cfg['source_model_sha256'],preference_exposures=u*cfg['per_update']['preference'],replay_exposures=u*cfg['per_update']['replay'],method='DPO + conversation CE + reference KL',
                  contract=fingerprint(frozen),tokenizer=b.tokenizer.fingerprint,training_format='plain-user-assistant-eos-v1',
                  experimental=True,automatic_promotion=False,repetition_auto_stop=False)
        bundle=Path(save_bundle(OUTPUT,b.model,opt,meta,None,keep=1000000,reserve_gib=5));saved=u
        print('[checkpoint]',768+u,'retained; additional updates',u,flush=True)
    def evaluate():
        p=OUTPUT/f'evaluations/update_{u:05d}.json'
        if p.exists():
            old=read_json(p)
            assert accepts_contract(old['contract'],frozen)
            return
        result=engine.assess(b,anchor,data,cfg)
        result.update(update=u,contract=fingerprint(frozen),bundle=bundle.name if bundle else cfg['source_bundle'])
        atomic_json(p,result)
        lines=[f'# Text-only DPO from 768 — additional update {u}', '', 'Development responses; content review required.', '']
        for answer in result['answers']:
            lines += [f"## {answer['source']} — {answer['id'][:12]}", '']
            for t in answer['turns']:
                lines += ['**User**', '', t['prompt'], '', '**Model**', '', t['text'], '', f"Stop: {t['stop']}", '']
        p.with_suffix('.md').write_text('\n'.join(lines))
        print('[conversation-eval]',u,json.dumps(result['metrics']),'; review content; no automatic best',flush=True)
        if u:
            baseline=read_json(OUTPUT/'evaluations/update_00000.json')['metrics']
            changes={k:result['metrics'][k]-baseline[k] for k in ('chat_stopped','chat_repeated','text_followup_joint_proxy','validation_conversation_nll')}
            print('[change-from-768]',json.dumps(changes),'; reporting only; no quality-based automatic stop',flush=True)
    status='stopped'
    try:
        if u in cfg['evaluation_updates'] and not cancelled():evaluate()
        print('[ready]',u,'/',cfg['updates'],'updates; text-only DPO from 768; frozen reference and full-conversation replay',flush=True)
        while u<cfg['updates'] and not cancelled():
            batch=batch_at(data,cfg,u);started=time.monotonic()
            metrics=engine.update(b,anchor,opt,batch,cfg,u+1);u+=1
            metrics.update(seconds=time.monotonic()-started,batch_hash=fingerprint({f:[r['id'] for r in rr] for f,rr in batch.items()}))
            with (OUTPUT/'training.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
            if u==1 or u%8==0:print(f'[dpo {u}/{cfg["updates"]}] loss={metrics["loss"]:.4f} preference={metrics["dpo_loss"]:.4f} replay={metrics["replay_nll"]:.4f} chat_kl={metrics["chat_kl"]:.6f} lr={metrics["lr"]:.3e} seconds={metrics["seconds"]:.1f}',flush=True)
            if u in cfg['evaluation_updates']:
                save()
                if not cancelled():evaluate()
        save();status='complete' if u==cfg['updates'] else 'stopped'
    except Exception as exc:
        atomic_json(OUTPUT/'report.json',dict(status='error',update=u,last_saved_update=saved,error=repr(exc)))
        raise
    finally:
        del b,anchor,opt;gc.collect();mx.clear_cache()
    assert contract(cfg)==frozen
    atomic_json(OUTPUT/'report.json',dict(status=status,update=u,selected=None,source_step=768,
                automatic_promotion=False,all_checkpoints_retained=True,public_benchmarks_run=False))
    print('[finished]',status,u,'; review development responses before public benchmarks',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    for mode in ('prepare','inspect','check','run'):g.add_argument('--'+mode,action='store_true')
    p.add_argument('--clear-stop',action='store_true');a=p.parse_args()
    cfg=read_json(DIR/'config.json');validate(cfg)
    print(json.dumps(dict(source=cfg['source_bundle'],updates=cfg['updates'],preference_pairs=cfg['preference_train_pairs'],preference_epochs=2,
                         method='Reference DPO + chosen CE + full-conversation replay CE and KL',
                         per_update=cfg['per_update'],output=str(OUTPUT),training=bool(a.run)),indent=2),flush=True)
    if a.inspect:
        data=prepared(cfg);print(json.dumps(inspect_data(cfg,data),indent=2));return
    if not (a.prepare or a.check or a.run):return
    from sft.transfer_control.launch import preflight,Tee
    preflight()
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
        if a.prepare:
            if (OUTPUT/'contract.json').exists():raise ValueError('Cannot reprepare a started run')
            from sft.dpo_chat_768_v1.data import build
            build(cfg);return
        data=prepared(cfg);frozen=contract(cfg)
        if a.check:check(cfg,data,frozen);return
        if not (DIR/'readiness.json').exists() or read_json(DIR/'readiness.json').get('contract')!=frozen:check(cfg,data,frozen)
        ready=read_json(DIR/'readiness.json')
        if ready['status']!='passed' or ready['contract']!=frozen:raise ValueError('Run --check before training')
        if not (OUTPUT/'latest.json').exists() and shutil.disk_usage(ROOT).free<cfg['minimum_free_gib']*1024**3:
            raise OSError('Need 30 GiB free for retained checkpoint bundles')
        if a.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
        if (OUTPUT/'STOP').exists():raise ValueError('Stopped run: review saved responses, then use --clear-stop to explicitly resume')
        stop={'requested':False}
        def request_stop(*_):
            stop['requested']=True;print('[stop-requested] Saving after the current update/evaluation.',flush=True)
        signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
        OUTPUT.mkdir(parents=True,exist_ok=True)
        with (OUTPUT/'training.log').open('a',buffering=1) as log:
            old=sys.stdout;sys.stdout=Tee(old,log)
            try:train(cfg,data,frozen,stop)
            finally:sys.stdout=old

if __name__=='__main__':main()

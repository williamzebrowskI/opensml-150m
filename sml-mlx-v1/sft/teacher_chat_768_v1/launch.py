"""Teacher-assisted text SFT from 768. --run prepares data and compares three rates.

Then --continue NAME resumes the chosen comparison to 512 additional updates.
No quality-based automatic stopping, pruning, or automatic model promotion.
"""
import argparse
import gc
import importlib.metadata
import json
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sml_v1.common import atomic_json,file_sha256,fingerprint,read_json,lock
from sft.teacher_chat_768_v1.data import DIR,PARENT,FOUNDATION,make_plan,batch_at,turns,visible_prefix
WORK=ROOT/'runs/teacher_chat_768_v1'


def validate(cfg):
    assert cfg['source_step']==768 and cfg['context']==2048
    assert cfg['replay_records']==cfg['teacher_groups']['train']*2==2048
    assert cfg['updates']*16==2*(cfg['replay_records']+2*cfg['teacher_groups']['train'])
    source=ROOT/cfg['source_bundle'];meta=read_json(source/'model.safetensors.json')
    if meta['step']!=768 or meta['training_format']!='plain-user-assistant-eos-v1':raise ValueError('Wrong source format/step')
    if file_sha256(source/'model.safetensors')!=cfg['source_model_sha256']:raise ValueError('Parent 768 changed')


def frozen(cfg,prepared):
    files=list(DIR.glob('*.py'))+[DIR/'config.json',PARENT,FOUNDATION]
    for name in ('conversation_foundation_v1','text_followup_512_v1','skill_balance','transfer_control','intact_smoltalk_base_pilot','natural_control','complete_transfer_1024','response_repair'):
        files+=list((ROOT/'sft'/name).glob('*.py'))
    files+=list((ROOT/'sml_v1').glob('*.py'))+list((ROOT/'tokenizer/bytebpe32k_v1').glob('*'))
    return dict(config=cfg,prepared=fingerprint(prepared),
                code_and_inputs={str(p):file_sha256(p) for p in sorted(set(files)) if p.is_file()},
                runtime={p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','numpy','tokenizers')})


def arm_config(cfg,name):
    return dict(cfg,peak_lr=cfg['rates'][name],final_lr=cfg['rates'][name]/10)


def check_cpu(cfg,prepared):
    from collections import Counter
    from sml_v1.tokenization import Tokenizer
    tok=Tokenizer(ROOT/'tokenizer/bytebpe32k_v1');ids=set();groups=set();encoded=0;masked=0
    if prepared['data_hash']!=fingerprint(prepared['data']):raise ValueError('Prepared data changed')
    for split,rows in prepared['data'].items():
        current=set()
        for row in rows:
            if row['id'] in ids:raise ValueError('Duplicate row across splits')
            ids.add(row['id'])
            if row['source']=='teacher':current.add(row['group'])
            enc=turns(tok,row,cfg['context'])
            if row.get('last_only'):
                assert len(enc)==1 and enc[0]['message_index']==len(row['messages'])-1
                masked+=len(row['messages'])>2
            for e in enc:
                head=tok.encode(visible_prefix(row['messages'][:e['message_index']]))
                assert e['y'][:len(head)-1]==[-100]*(len(head)-1)
                assert e['y'][-1]==tok.eos and e['targets']==sum(t!=-100 for t in e['y'])
                encoded+=1
        assert not current&groups;groups|=current
    counts=Counter(r['id'] for u in range(cfg['updates']) for r in batch_at(prepared['data'],cfg,u))
    assert len(counts)==4096 and set(counts.values())=={2}
    return dict(assistant_targets_checked=encoded,student_history_masked=masked,unique_train_records=4096,
                exposures=8192,split_groups_disjoint=True,balanced_batches=True)


def train(cfg,prepared,name,until,stop):
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from sml_v1.checkpoint_bundle import save_bundle,resolve_bundle
    from sft.teacher_chat_768_v1 import engine
    c=arm_config(cfg,name);contract=frozen(c,prepared);digest=fingerprint(contract)
    out=WORK/name;out.mkdir(parents=True,exist_ok=True)
    cp=out/'contract.json'
    if cp.exists() and read_json(cp)!=contract:raise ValueError('Existing run inputs changed: '+name)
    if not cp.exists():atomic_json(cp,contract)
    mx.random.seed(cfg['seed']);b=engine.load(c);opt=engine.optimizer(c)
    opt.init(b.model.trainable_parameters());mx.eval(opt.state)
    u=0;saved=0;bundle=None
    if (out/'latest.json').exists():
        w=resolve_bundle(out/'latest.json');meta=read_json(w+'.json')
        if meta['contract']!=digest or meta['source_step']!=768:raise ValueError('Resume contract mismatch')
        u=meta['additional_updates'];saved=u;bundle=Path(w).parent
        b.model.load_weights(w,strict=True)
        opt.state=tree_unflatten(list(mx.load(w+'.optimizer.safetensors').items()));mx.eval(opt.state,b.model.parameters())
        assert int(opt.state['step'].item())==u and meta['conversation_exposures']==u*16
        log=out/'training.jsonl'
        if log.exists():
            lines=log.read_text().splitlines();committed=[];tail=[]
            for line in lines:
                try:valid=json.loads(line)['update']<=u
                except (ValueError,KeyError):valid=False
                (committed if valid else tail).append(line)
            if tail:
                (out/f'uncommitted_{time.time_ns()}.jsonl').write_text('\n'.join(tail)+'\n')
                log.write_text('\n'.join(committed)+'\n')
        print('[resume]',name,u,'saved optimizer and data cursor',flush=True)
    def save():
        nonlocal saved,bundle
        if u==saved:return
        meta=dict(step=768+u,additional_updates=u,source_step=768,source_bundle=cfg['source_bundle'],
                  source_model_sha256=cfg['source_model_sha256'],conversation_exposures=u*16,
                  contract=digest,tokenizer=b.tokenizer.fingerprint,training_format='plain-user-assistant-eos-v1',
                  experimental=True,automatic_promotion=False,arm=name)
        bundle=Path(save_bundle(out,b.model,opt,meta,None,keep=1000000,reserve_gib=5));saved=u
        print('[checkpoint]',name,768+u,'retained; additional updates',u,flush=True)
    def evaluate():
        # Same unmodified parent baseline shared across matched-rate trials.
        path=(WORK/'baseline.json') if u==0 else out/f'evaluations/update_{u:05d}.json'
        if path.exists():
            old=read_json(path)
            expected=fingerprint(frozen(cfg,prepared)) if u==0 else digest
            if old['contract']!=expected:raise ValueError('Evaluation contract mismatch')
            return
        result=engine.assess(b,prepared,c)
        result.update(update=u,bundle=str(bundle.relative_to(ROOT)) if bundle else cfg['source_bundle'],
                      contract=fingerprint(frozen(cfg,prepared)) if u==0 else digest)
        atomic_json(path,result)
        lines=['# '+name+f' — {u} additional updates','',result['note'],'']
        for answer in result['fresh_answers']:
            lines+=['## '+answer['category']+' / '+answer['group'][:12],'']
            for t in answer['turns']:lines += ['**User**',t['history'][-1]['content'],'','**Model**',t['text'],'',f"Stop: {t['stop']}",'']
        path.with_suffix('.md').write_text('\n'.join(lines))
        print('[retention-eval]',name,u,json.dumps(result['metrics']),flush=True)
        print('[fresh-conversation-eval]',json.dumps(result['fresh_metrics']),'; content scores follow after unloading student',flush=True)
    try:
        if u==0 or u in cfg['evaluation_updates']:evaluate()
        while u<until and not stop['requested']:
            start=time.monotonic();m=engine.update(b,opt,batch_at(prepared['data'],c,u),c,u+1);u+=1
            m.update(seconds=time.monotonic()-start)
            with (out/'training.jsonl').open('a') as f:f.write(json.dumps(m)+'\n')
            if u==1 or u%8==0:print(f'[teacher-sft {name} {u}/{cfg["updates"]}] loss={m["loss"]:.4f} lr={m["lr"]:.3e} seconds={m["seconds"]:.1f}',flush=True)
            if u in cfg['evaluation_updates']:
                save()
                if not stop['requested']:evaluate()
        save()
        atomic_json(out/'report.json',dict(status='complete' if u==cfg['updates'] else 'comparison-complete' if u>=until else 'stopped',
                                         additional_updates=u,total_step=768+u,all_checkpoints_retained=True,selected=False))
    finally:
        del b,opt;gc.collect();mx.clear_cache()
    if frozen(c,prepared)!=contract:raise ValueError('Inputs changed during training')


def judge_and_report(cfg,names):
    from sft.teacher_chat_768_v1.teacher import judge_answers
    paths=[WORK/'baseline.json']
    for name in names:
        paths+=sorted((WORK/name/'evaluations').glob('update_[0-9][0-9][0-9][0-9][0-9].json'))
    rows=[]
    for path in paths:
        if not path.exists():continue
        evaluation=read_json(path);score_path=path.with_suffix('.scores.json')
        key=fingerprint(evaluation)
        if score_path.exists():
            saved=read_json(score_path)
            if saved['evaluation']!=key:raise ValueError('Scoring input changed')
            scores=saved['scores']
        else:
            scores=judge_answers(cfg,evaluation,WORK/'judge_cache')
            atomic_json(score_path,dict(evaluation=key,scores=scores))
        m=evaluation['metrics'];row=dict(model='parent-768' if path.name=='baseline.json' else path.parent.parent.name,
              updates=evaluation['update'],teacher_mean=scores['mean'],valid=scores['valid'],expected=scores['expected'],
              chat_repeated=m['chat_repeated'],joint=m['text_followup_joint_proxy'],validation_nll=m['validation_token_nll'])
        rows.append(row);print('[comparison]',json.dumps(row),flush=True)
    atomic_json(WORK/'comparison.json',dict(rows=rows,automatic_selection=False,
                note='Choose with transcripts and retention; teacher scores are a same-teacher diagnostic. No public benchmarks used for this selection.'))
    lines=['# Learning-rate comparison','', 'Teacher scores are model-assisted 1–5 diagnostics, not an independent benchmark. Review transcripts and retained abilities before selecting.','',
           '| Model | Updates | Teacher mean | Scored | Chat repetition | Text joint | Validation NLL |','|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"| {r['model']} | {r['updates']} | {r['teacher_mean']} | {r['valid']}/{r['expected']} | {r['chat_repeated']:.1%} | {r['joint']:.1%} | {r['validation_nll']:.5f} |")
    (WORK/'comparison.md').write_text('\n'.join(lines)+'\n')
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group()
    g.add_argument('--run',action='store_true',help='Prepare teacher data and run three matched 64-update comparisons')
    g.add_argument('--prepare',action='store_true');g.add_argument('--check',action='store_true')
    g.add_argument('--continue',dest='candidate',choices=['conservative','moderate','faster'])
    a=p.parse_args();cfg=read_json(DIR/'config.json');validate(cfg)
    print(json.dumps(dict(parent=cfg['source_bundle'],teacher=cfg['teacher_repo'],
               teacher_groups=cfg['teacher_groups'],replay_records=cfg['replay_records'],
               comparison_updates=cfg['comparison_updates'],long_run_updates=cfg['updates'],
               rates=cfg['rates'],output=str(WORK),automatic_selection=False),indent=2),flush=True)
    if not (a.run or a.prepare or a.check or a.candidate):return
    if shutil.disk_usage(ROOT).free<cfg['minimum_free_gib']*1024**3:raise OSError('Need 30 GiB free')
    from sft.transfer_control.launch import preflight
    preflight()
    with lock(ROOT/'sft/.experiment.lock'),lock(DIR/'.launcher.lock'):
        plan=make_plan(cfg,DIR/'plan.json')
        if a.check:
            from sft.teacher_chat_768_v1.check import smoke_check
            smoke_check(cfg,plan);return
        ready=read_json(DIR/'readiness.json') if (DIR/'readiness.json').exists() else {}
        if (ready.get('status')!='passed' or ready.get('config')!=file_sha256(DIR/'config.json')
                or ready.get('code')!={p.name:file_sha256(p) for p in DIR.glob('*.py')}):
            from sft.teacher_chat_768_v1.check import smoke_check
            smoke_check(cfg,plan)
        from sft.teacher_chat_768_v1.teacher import prepare
        prepared=prepare(cfg,plan,WORK/'data')
        checks=check_cpu(cfg,prepared);atomic_json(WORK/'data_checks.json',checks)
        if a.prepare:return
        stop={'requested':False}
        def cancel(*_):stop['requested']=True;print('[stop-requested] Saving after the current update/evaluation.',flush=True)
        signal.signal(signal.SIGINT,cancel);signal.signal(signal.SIGTERM,cancel)
        names=[a.candidate] if a.candidate else list(cfg['rates'])
        if a.candidate and not (WORK/a.candidate/'latest.json').exists():
            raise ValueError('Run the comparison first with --run')
        for name in names:
            train(cfg,prepared,name,cfg['updates'] if a.candidate else cfg['comparison_updates'],stop)
            if stop['requested']:return
        judge_and_report(cfg,names)
        print('[finished]', 'long run complete' if a.candidate else 'comparison complete; review comparison.md, then use --continue conservative|moderate|faster',flush=True)

if __name__=='__main__':main()

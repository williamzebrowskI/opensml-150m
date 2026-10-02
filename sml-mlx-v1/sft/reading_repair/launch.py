"""Prepare or run one pass of reading-repair SFT from the preserved step 448."""
import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import signal
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from sml_v1.common import atomic_json,file_sha256,fingerprint,lock,read_json
from sft.transfer_control.launch import preflight,prose_texts,Tee
from sft.transfer_control.data import records as legacy_records,digest
from sft.response_repair.launch import restore,optimizer_step,verify_hashes
from sft.response_repair.data import new_records as repair_records
from sft.response_repair.rubrics import evaluate as repair_evaluate
from sft.reading_repair.source import build,evaluation_exclusions
from sft.reading_repair.evaluate import assess_new,eligible,ranking,parent_metrics,grade,verify_generation

DIR=Path(__file__).resolve().parent
OUTPUT=ROOT/'runs/sft_reading_repair_v1'
PARENT_RUN=ROOT/'runs/sft_response_repair_v1'


def code_files():
    paths=set(DIR.glob('*.py'))|{DIR/'challenge.json',DIR/'excluded_previous_prompts.json'}|set((ROOT/'sml_v1').glob('*.py'))
    for directory in ('transfer_control','transfer_continue','response_expansion','response_repair'):
        paths.update((ROOT/'sft'/directory).glob('*.py'))
    paths.update((ROOT/'evaluation/base_audit').glob('*.py'))
    return sorted(paths)


def learning_rate(update,cfg):
    if not 1<=update<=cfg['additional_updates']:raise ValueError('Invalid additional update')
    return cfg['learning_rate']


def restore_run(weights,cfg):
    return restore(weights,cfg)


def contract(cfg,selection):
    source=ROOT/cfg['source_bundle'];original=read_json(PARENT_RUN/'contract.json')
    meta=read_json(source/'model.safetensors.json')
    for name,key in (('model.safetensors','source_model_sha256'),('model.safetensors.optimizer.safetensors','source_optimizer_sha256'),('model.safetensors.json','source_metadata_sha256')):
        if file_sha256(source/name)!=cfg[key]:raise ValueError('Unexpected step-448 input: '+name)
    if meta['step']!=448 or meta['repair_contract']!=fingerprint(original) or meta['arm']!='opensml':raise ValueError('Wrong parent model')
    if meta['training_format']!='plain-user-assistant-eos-v1' or meta['next_example']!=512:raise ValueError('Wrong source format/cursor')
    parent_cfg=read_json(PARENT_RUN/'input/sft/response_repair/config.json')
    for key in ('batch','microbatch','weight_decay','betas','clip_norm'):
        if cfg[key]!=parent_cfg[key]:raise ValueError('Unexpected optimizer change: '+key)
    if cfg['source_step']!=448 or cfg['additional_updates']!=512 or cfg['batch']!=8:raise ValueError('Expected 512 additional updates from 448')
    if cfg['additional_updates']*cfg['batch']!=selection['splits']['train']['examples']:raise ValueError('One-pass size mismatch')
    protected=dict(original['protected'])
    for p in PARENT_RUN.rglob('*'):
        if p.is_file():protected[str(p)]=file_sha256(p)
    for p in (ROOT/'tokenizer/bytebpe32k_v1').iterdir():
        if p.is_file():protected[str(p)]=file_sha256(p)
    _,_,exclusions=evaluation_exclusions()
    for p in exclusions:protected[str(p)]=file_sha256(p)
    # The completed public benchmark evidence is preserved; never used for selection.
    for p in (ROOT/'diagnostics/full_benchmarks_v1/sft-448').rglob('*'):
        if p.is_file():protected[str(p)]=file_sha256(p)
    verify_hashes(protected)
    return dict(config=fingerprint(cfg),data=fingerprint(selection),source=str(source),protected=protected,
                prose=digest(prose_texts()),code={str(p.relative_to(ROOT)):file_sha256(p) for p in code_files()},
                runtime={p:importlib.metadata.version(p) for p in ('mlx','mlx-lm','numpy','transformers','tokenizers','requests','datasets','pyarrow')})


def assess(backend,cfg,data,update,baseline):
    from sft.transfer_control.engine import evaluate,prose_loss
    new=assess_new(backend,data,'dev',cfg['max_new_tokens'])
    legacy=evaluate(backend,legacy_records('dev'),cfg['legacy_max_new_tokens'],include_open=False)
    repaired=repair_evaluate(backend,repair_records('dev'),64)
    prose=prose_loss(backend,prose_texts())
    return dict(step=cfg['source_step']+update,additional_updates=update,new_tasks=new,legacy=legacy,
                repair_joint=repaired['fact_joint'],repair_evaluation=repaired,prose=prose,
                prose_change_from_parent=0. if baseline is None else prose-baseline)


def check(cfg,frozen,data):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from sml_v1.checkpoint_bundle import resolve_bundle
    from sft.transfer_control.engine import encode,arrays,gradients
    preflight();backend,optimizer=restore_run(resolve_bundle(ROOT/cfg['source_bundle']),cfg)
    if optimizer_step(optimizer)!=448:raise ValueError('Optimizer counter not restored')
    old=mx.load(str(ROOT/cfg['source_bundle']/'model.safetensors.optimizer.safetensors'))
    restored=dict(tree_flatten(optimizer.state))
    if set(old)!=set(restored):raise ValueError('Optimizer fields changed')
    for k in old:
        if k!='learning_rate' and not bool(mx.array_equal(old[k],restored[k]).item()):raise ValueError('Optimizer state changed: '+k)
    masters=[x for k,x in restored.items() if k.endswith('.master')]
    if not masters or any(x.dtype!=mx.float32 for x in masters):raise ValueError('FP32 masters missing')
    allrows=data['train']+[r for s in ('dev','test') for r in data['natural'][s]+data['authored'][s]]+data['challenge']
    encoded=[encode(backend,r) for r in allrows]
    for split in ('dev','test'):
        for r in data['authored'][split]+data['natural'][split]+(data['challenge'] if split=='dev' else []):
            if not grade(r,r['answer'],'end')['passed']:raise ValueError('Authored reference fails: '+r['id'])
    # Check forward/backward with real weights, without applying an optimizer update.
    human=max((r for r in data['train'] if r['origin']=='squad_v2'),key=lambda r:len(backend.encode(r['prompt'])))
    original=next(r for r in data['train'] if r['origin']=='authored')
    x,y=arrays([encode(backend,r) for r in (human,original)],backend.pad)
    loss,grads=gradients(backend.model,x,y,False,1)
    from sml_v1.pretrain import clip_gradients
    _,norm=clip_gradients(grads,cfg['clip_norm']);mx.eval(loss,norm)
    if not math.isfinite(float(loss.item())) or not math.isfinite(float(norm.item())):raise ValueError('Nonfinite forward/backward')
    probes=verify_generation(backend,['hello']+[r['prompt'] for r in data['train'][:3]],32)
    receipt=dict(status='passed',contract=frozen,optimizer_step=optimizer_step(optimizer),fp32_master_tensors=len(masters),
                 optimizer_state_exact=True,examples=len(data['train']),maximum_tokens=max(len(r['x']) for r in encoded),
                 forward_loss=float(loss.item()),gradient_norm=float(norm.item()),generation_checks=probes,
                 training_started=False,real_optimizer_updates=0)
    del backend,optimizer,old,restored,masters,grads;gc.collect();mx.clear_cache()
    verify_hashes(frozen['protected'])
    atomic_json(DIR/'readiness.json',receipt)
    print('[check-passed] Step 448 and exact optimizer state restored; streaming selection, masks, gradients and held-out references verified; no training started.',flush=True)


def run(cfg,frozen,data,stop):
    import mlx.core as mx
    from sml_v1.checkpoint_bundle import resolve_bundle,save_bundle
    from sml_v1.pretrain import clip_gradients
    from sft.transfer_control.engine import encode,arrays,gradients
    report_path=OUTPUT/'report.json';latest=OUTPUT/'latest.json'
    if report_path.exists() and read_json(report_path)['status']=='complete':
        print('[already-complete] One pass finished; no automatic repeat.',flush=True);return
    update=0;best_key=None
    report=dict(status='running',source=frozen.get('source'),source_step=cfg['source_step'],
                baseline_prose=None,parent_metrics=None,training=[],evaluations=[],promoted=False)
    if latest.exists():
        weights=resolve_bundle(latest);meta=read_json(weights+'.json')
        if meta['reading_contract']!=fingerprint(frozen):raise ValueError('Resume contract changed')
        update=meta['additional_updates'];report=meta['report'];best_key=meta['best_key']
        if meta['next_example']!=update*cfg['batch']:raise ValueError('Cursor mismatch')
    else:weights=resolve_bundle(ROOT/cfg['source_bundle'])
    if not 0<=update<=cfg['additional_updates']:raise ValueError('Invalid resume step')
    backend,optimizer=restore_run(weights,cfg)
    if optimizer_step(optimizer)!=cfg['source_step']+update:raise ValueError('Optimizer/step mismatch')
    report['status']='running'
    if (OUTPUT/'best.json').exists():
        best=resolve_bundle(OUTPUT/'best.json');report['best']=dict(step=read_json(best+'.json')['step'],path=str(Path(best).parent))

    def save(best=False):
        step=cfg['source_step']+update
        if optimizer_step(optimizer)!=step:raise ValueError('Optimizer counter lost')
        report.update(step=step,additional_updates=update)
        meta=dict(step=step,source_step=cfg['source_step'],additional_updates=update,next_example=update*cfg['batch'],
                  reading_contract=fingerprint(frozen),report=report,best_key=best_key,arm='opensml',experimental=True,
                  training_format='plain-user-assistant-eos-v1',playground_compatible=False)
        path=save_bundle(OUTPUT,backend.model,optimizer,meta,None,best=best,keep=2)
        report['latest']=path
        if best:report['best']=dict(step=step,path=path)
        atomic_json(report_path,report)
        print(f'[checkpoint {step}] '+('best + latest' if best else 'latest'),flush=True)

    def evaluation():
        result=assess(backend,cfg,data,update,report['baseline_prose'])
        if report['baseline_prose'] is None:report['baseline_prose']=result['prose']
        if report['parent_metrics'] is None:report['parent_metrics']=parent_metrics(result)
        accepted=update==0 or eligible(result,report['parent_metrics'],cfg)
        result['selection_eligible']=accepted
        atomic_json(OUTPUT/'evaluations'/f'step_{cfg["source_step"]+update:07d}.json',result)
        new=result['new_tasks']
        row=dict(step=cfg['source_step']+update,additional_updates=update,reading_exact=new['task_macro'],
                 paired=new['paired'],human_f1=new['natural_f1'],human_exact=new['natural_exact'],
                 challenge_correct=new['challenge_correct'],natural_nll=new['natural_nll'],
                 repeat_rate=new['repeat_rate'],cap_rate=new['cap_rate'],
                 legacy_paired=result['legacy']['both_correct'],repair_joint=result['repair_joint'],
                 prose_change_from_parent=result['prose_change_from_parent'],selection_eligible=accepted)
        report['evaluations'].append(row)
        print(f'[reading-eval {row["step"]}] reading={row["reading_exact"]:.1%} challenge={row["challenge_correct"]}/28 human_f1={row["human_f1"]:.1%} legacy_paired={row["legacy_paired"]:.1%} prose_change={row["prose_change_from_parent"]:+.4f} eligible={accepted}',flush=True)
        return ranking(result),accepted

    try:
        if not latest.exists():
            best_key,_=evaluation();report['parent_key']=best_key;save(best=True)
        print(f'[ready] Step {cfg["source_step"]+update}; additional={update}/{cfg["additional_updates"]}; one pass; main Mac only.',flush=True)
        while update<cfg['additional_updates']:
            if stop['requested'] or (OUTPUT/'STOP').exists():break
            batch=data['train'][update*cfg['batch']:(update+1)*cfg['batch']]
            if len(batch)!=cfg['batch']:raise ValueError('Unexpected end of examples')
            x,y=arrays([encode(backend,r) for r in batch],backend.pad)
            value,grad=gradients(backend.model,x,y,False,cfg['microbatch']);grad,norm=clip_gradients(grad,cfg['clip_norm'])
            mx.eval(value,grad,norm)
            if not math.isfinite(float(value.item())) or not math.isfinite(float(norm.item())):raise ValueError('Nonfinite update')
            lr=learning_rate(update+1,cfg);optimizer.learning_rate=lr
            optimizer.update(backend.model,grad);mx.eval(backend.model.parameters(),optimizer.state);update+=1
            report['training'].append(dict(step=cfg['source_step']+update,additional_update=update,loss=float(value.item()),
                                           lr=lr,grad_norm=float(norm.item()),examples=update*cfg['batch'],batch_hash=digest([r['id'] for r in batch])))
            if update==1 or update%8==0:print(f'[reading-repair {update}/{cfg["additional_updates"]}] step={cfg["source_step"]+update} loss={value.item():.4f} lr={lr:.3e}',flush=True)
            if update in cfg['evaluation_updates']:
                key,accepted=evaluation();improved=accepted and key>best_key
                if improved:best_key=key
                save(improved)
            elif update%cfg['checkpoint_every']==0:save()
        if stop['requested'] or (OUTPUT/'STOP').exists():
            report['status']='stopped';save();print('[stopped] Exact update and stream selection cursor saved.',flush=True);return
        selected=resolve_bundle(OUTPUT/'best.json');selected_step=read_json(selected+'.json')['step']
        print('[fresh-test] Selection frozen; compare parent, selected-best AND final checkpoint on the untouched test split.',flush=True)
        candidates={'parent':resolve_bundle(ROOT/cfg['source_bundle']), 'selected':selected,
                    'final':resolve_bundle(OUTPUT/'latest.json')}
        tests={};seen={}
        for name,weights in candidates.items():
            sha=file_sha256(weights)
            if sha not in seen:
                backend.model.load_weights(weights);mx.eval(backend.model.parameters())
                seen[sha]=assess_new(backend,data,'test',cfg['max_new_tokens'])
            tests[name]=dict(step=read_json(weights+'.json')['step'],model_sha256=sha,result=seen[sha])
        atomic_json(OUTPUT/'fresh_test.json',dict(selection_frozen=True,candidates=tests,
                    public_benchmarks_used=False,review_required=True))
        report.update(status='complete',step=cfg['source_step']+update,additional_updates=update,
                      best=dict(step=selected_step,path=str(Path(selected).parent)),improved_on_development=best_key>report['parent_key'])
        atomic_json(report_path,report)
        print(f'[finished] Additional={update}; final_step={cfg["source_step"]+update}; selected_best_step={selected_step}; inspect natural answers before promotion.',flush=True)
    except Exception as exc:
        report.update(status='error',error=repr(exc));atomic_json(report_path,report);raise
    finally:
        del backend,optimizer;gc.collect();mx.clear_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--check',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--clear-stop',action='store_true');args=parser.parse_args()
    cfg=read_json(DIR/'config.json')
    print(json.dumps(dict(mode='prepare' if args.prepare else 'check' if args.check else 'run' if args.run else 'plan',
                         source=str(ROOT/cfg['source_bundle']),source_step=448,additional_updates=cfg['additional_updates'],final_step=448+cfg['additional_updates'],
                         examples=4096,original_reading=2688,human_squad=1152,rehearsal=256,
                         learning_rate=cfg['learning_rate'],
                         optimizer='restored FP32 AdamW masters, moments and counter',main_mac_only=True,teacher=False,
                         data='Pinned SQuAD v2 train source streamed; bounded selected subset in RAM, original local examples and rehearsal; no raw dataset cache.',
                         output=str(OUTPUT)),indent=2),flush=True)
    if not (args.prepare or args.check or args.run):return
    from sml_v1.tokenization import Tokenizer
    with lock(ROOT/'sft/.experiment.lock'):
        preflight()
        data,selection=build(Tokenizer(ROOT/'tokenizer/bytebpe32k_v1'))
        if args.prepare:
            if (OUTPUT/'contract.json').exists():raise ValueError('Cannot re-prepare an existing run')
            atomic_json(DIR/'selection.json',selection);print('[prepared] Selection IDs, hashes and length statistics saved; no training.');return
        if selection!=read_json(DIR/'selection.json'):raise ValueError('Selection changed; review and repeat --prepare/--check')
        frozen=contract(cfg,selection)
        if args.check:check(cfg,frozen,data);return
        ready=read_json(DIR/'readiness.json')
        if ready['status']!='passed' or ready['contract']!=frozen:raise ValueError('Setup changed; repeat --check')
        if (OUTPUT/'contract.json').exists():
            if read_json(OUTPUT/'contract.json')!=frozen:raise ValueError('Cannot resume a changed contract')
        else:
            atomic_json(OUTPUT/'contract.json',frozen)
            for p in code_files()+[DIR/'config.json',DIR/'selection.json',DIR/'README.md',DIR/'DATA_LICENSE.md']:
                target=OUTPUT/'input'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        if args.clear_stop:(OUTPUT/'STOP').unlink(missing_ok=True)
        if (OUTPUT/'STOP').exists():raise ValueError('STOP exists; use --clear-stop to resume')
        stop=dict(requested=False)
        def handler(sig,frame):
            stop['requested']=True;(OUTPUT/'STOP').touch();print('[stop] Saving after the current update/evaluation; please wait.',flush=True)
        handlers={s:signal.signal(s,handler) for s in (signal.SIGINT,signal.SIGTERM)}
        try:
            with (OUTPUT/'console.log').open('a') as log:
                old=sys.stdout,sys.stderr;sys.stdout=Tee(old[0],log);sys.stderr=Tee(old[1],log)
                try:run(cfg,frozen,data,stop)
                finally:sys.stdout,sys.stderr=old
        finally:
            for s,h in handlers.items():signal.signal(s,h)
            verify_hashes(frozen['protected'])


if __name__=='__main__':main()

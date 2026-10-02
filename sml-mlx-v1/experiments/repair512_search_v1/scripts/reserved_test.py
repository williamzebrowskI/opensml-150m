"""One independent reserved public test comparison after selecting a candidate.

No optimizer is created. This does not tune the recipe or select test examples.
The exact same public evaluator runs on the reserved test split for both models.
"""
import gc
import sys
from pathlib import Path
from experiment import ROOT,PUBLIC,EXP,PARENT,public_module,verify_parent,verify_contract
from sml_v1.common import read_json,atomic_json,file_sha256

def run(name):
    import mlx.core as mx
    from sft.conversation_foundation_original import engine
    c=EXP/'candidates'/name;assert read_json(c/'comparison.json')['retain']
    original=read_json(c/'runs/sft/contract.json');verify_contract(original);verify_parent()
    complete=read_json(c/'runs/sft/finished.json');candidate=c/'runs/sft'/complete['bundle']
    selected_updates=read_json(c/'comparison.json')['updates']
    if read_json(candidate/'model.safetensors.json')['additional_updates']!=selected_updates:
        matches=[p.parent for p in (c/'runs/sft').glob('step_*/model.safetensors.json') if read_json(p)['additional_updates']==selected_updates]
        assert len(matches)==1;candidate=matches[0]
    models={'repair512':PARENT,'candidate':candidate}
    data=read_json(c/'data.json');assert not {r['group'] for r in data['train']}&{r['group'] for r in data['test']}
    frozen=dict(reserved_split='test',optimizer_updates=0,selection_tuning=False,
                prepared_sha256=file_sha256(c/'data.json'),script_sha256=file_sha256(__file__),
                models={k:dict(bundle=str(v),sha256=file_sha256(v/'model.safetensors')) for k,v in models.items()},
                original_contract=original)
    results={}
    for key,bundle in models.items():
        cfg=read_json(c/'config.json');cfg['source_bundle']=str(bundle.relative_to(ROOT))
        cfg['source_model_sha256']=file_sha256(bundle/'model.safetensors')
        module=public_module();module.RUN=c/'reserved_test'/key
        b=engine.load(cfg)
        module.evaluate(b,dict(dev=data['test']),cfg,0,str(bundle),frozen)
        p=module.RUN/'evaluations/update_00000.json';evaluation=read_json(p)
        public=evaluation.pop('public_development')
        public['note']='Reserved TEST split; named constraints only, not semantic quality. Same protocol for both models.'
        evaluation['public_reserved_test']=public;evaluation['split']='test'
        # Store under a separate filename to preserve evaluator resume metadata.
        atomic_json(module.RUN/'test_result.json',evaluation)
        results[key]=public['metrics']
        del b;gc.collect();mx.clear_cache();verify_parent()
    atomic_json(c/'reserved_test/comparison.json',dict(protocol=frozen,metrics=results,
        differences={k:results['candidate'][k]-results['repair512'][k] for k in results['candidate']
                     if isinstance(results['candidate'][k],(int,float))},
        interpretation='Independent check after exploratory benchmark selection; one small test does not establish statistical superiority.'))
    verify_contract(original);verify_parent()
    print('[reserved-test-complete]',name,flush=True)

if __name__=='__main__':run(sys.argv[1])

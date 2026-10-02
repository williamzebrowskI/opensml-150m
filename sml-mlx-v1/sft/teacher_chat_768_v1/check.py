"""Real teacher/MLX smoke check; all updated student weights are disposable."""
import gc
import tempfile
from pathlib import Path
from sml_v1.common import atomic_json,file_sha256,read_json
from .data import ROOT,DIR,turns,visible_prefix


def smoke_check(cfg,plan):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten,tree_unflatten
    from sml_v1.checkpoint_bundle import save_bundle,resolve_bundle
    from sft.transfer_control.engine import arrays
    from evaluation.generation_comparison.run import playground
    from .teacher import prepare,judge_answers
    from . import engine
    from .launch import arm_config
    source=ROOT/cfg['source_bundle']/'model.safetensors';before=file_sha256(source)
    with tempfile.TemporaryDirectory(prefix='teacher-chat-check-') as scratch:
        path=Path(scratch);small=dict(plan,groups={'train':plan['groups']['train'][:8]})
        prepared=prepare(cfg,small,path/'data',smoke=True)
        rows=[r for r in prepared['data']['train'] if r['source']=='teacher']
        assert len(rows)==2 and len(rows[1]['messages'])>=4
        b=engine.load(cfg);c=arm_config(cfg,'moderate');pg=playground()
        for row in rows:
            enc=turns(b.tokenizer,row,2048)
            assert len(enc)==1 and enc[0]['message_index']==len(row['messages'])-1
            messages=row['messages'][:-1]
            instruction=messages[0]['content'] if messages[0]['role']=='system' else ''
            body=messages[1:] if instruction else messages
            request=pg.GenerationRequest(messages=[pg.Message(**m) for m in body],instruction=instruction,max_tokens=1)
            assert pg.prepare_plain_sft_prompt(b.tokenizer,request,2048)['ids']==b.encode(visible_prefix(messages))
        e=turns(b.tokenizer,rows[1],2048)[0];x,y=arrays([e],b.pad)
        logits=b.model(x)['logits'].astype(mx.float32)
        pos=mx.array([i for i,t in enumerate(e['y']) if t!=-100])
        a=nn.losses.cross_entropy(logits[0,pos],y[0,pos],reduction='mean')
        z=(nn.losses.cross_entropy(logits,mx.maximum(y,0),reduction='none')*(y!=-100)).sum()/e['targets']
        mx.eval(a,z);assert abs(a.item()-z.item())<1e-5
        assert engine.generate(b,rows[1]['messages'][:-1],8)==engine.generate(b,rows[1]['messages'][:-1],8,full=True)
        del logits,a,z
        opt=engine.optimizer(c);opt.init(b.model.trainable_parameters());mx.eval(opt.state)
        first=engine.update(b,opt,rows,c,1);assert first['grad_norm']>0
        saved=save_bundle(path/'checkpoint',b.model,opt,dict(step=1),None,keep=10,reserve_gib=1)
        weights=resolve_bundle(saved);restored=engine.load(c,weights);opt2=engine.optimizer(c)
        opt2.state=tree_unflatten(list(mx.load(weights+'.optimizer.safetensors').items()));mx.eval(opt2.state)
        engine.update(b,opt,[rows[1]],c,2);engine.update(restored,opt2,[rows[1]],c,2)
        aa=dict(tree_flatten(b.model.parameters()));bb=dict(tree_flatten(restored.model.parameters()))
        assert aa.keys()==bb.keys() and all(bool(mx.array_equal(v,bb[k]).item()) for k,v in aa.items())
        prepared['scenarios']['dev']=prepared['scenarios']['train']
        evaluation=engine.assess(b,prepared,c,smoke=True)
        assert evaluation['fresh_metrics']['turns']>0
        del b,restored,opt,opt2,aa,bb;gc.collect();mx.clear_cache()
        scores=judge_answers(cfg,evaluation,path/'judgments')
        assert scores['valid']==scores['expected'] and not scores['unscored']
        if file_sha256(source)!=before:raise ValueError('Parent changed during check')
        report=dict(status='passed',teacher=prepared['teacher']['repo'],
                    machine_reviewed_sample=rows,teacher_review='all required dimensions passed',
                    native_prompt_parity=True,student_history_masked=True,masked_ce_parity=True,
                    cached_decode_parity=True,exact_next_update_after_restore=True,
                    evaluation_and_teacher_judge=True,source_unchanged=True,production_updates=0,
                    disposable_update=first,
                    code={p.name:file_sha256(p) for p in DIR.glob('*.py')},config=file_sha256(DIR/'config.json'))
        atomic_json(DIR/'readiness.json',report)
        print('[check-passed]',{k:v for k,v in report.items() if k not in ('machine_reviewed_sample','code')},flush=True)

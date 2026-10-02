"""Read-only reference evaluations on the identical new development suite."""
import gc
from pathlib import Path
from sml_v1.common import atomic_json,read_json,file_sha256,fingerprint
from .data import ROOT
REFERENCES={
 'foundation-512':('runs/sft_conversation_foundation_v1/step_0000512_156a108611bc','9f9574d7ba0cae82720e139f7889c1ea3a4b8ac81e9664fd59883839c669c5f2'),
 'text-followup-768':('runs/sft_text_followup_512_v1/step_0000768_10f5ea2a3354','309ad719dc8569e482444b47b05f3fd24e02e7b7470dad2af132a976bf11ca02')}

def run(cfg,data,frozen):
    import mlx.core as mx
    from .engine import load,assess
    out=ROOT/'diagnostics/conversation_foundation_v1_references'
    for name,(path,digest) in REFERENCES.items():
        weights=ROOT/path/'model.safetensors';assert file_sha256(weights)==digest
        target=out/(name+'.json')
        if target.exists():
            old=read_json(target)
            assert old['contract']==fingerprint(frozen) and old['weights_sha256']==digest
            print('[reference-already-complete]',name,flush=True);continue
        print('[reference-eval]',name,flush=True)
        b=load(cfg,str(weights));result=assess(b,data,cfg)
        result.update(model=name,weights_sha256=digest,contract=fingerprint(frozen),training=False)
        atomic_json(target,result)
        lines=['# Reference: '+name, '', 'Same fixed development suite; proxy checks are not semantic judgments.', '']
        for r in result['answers']:
            lines += ['## '+r['source']+' / '+r['id'][:12], '']
            for t in r['turns']:lines += ['**User:** '+t['prompt'], '', '**Model:** '+t['text'], '']
            if 'rubric' in r:lines += ['Review: '+r['rubric'], '', 'Proxy passed: '+str(r['proxy_passed']), '']
        target.with_suffix('.md').write_text('\n'.join(lines))
        del b;gc.collect();mx.clear_cache()
        print('[reference-finished]',name,result['metrics'],flush=True)

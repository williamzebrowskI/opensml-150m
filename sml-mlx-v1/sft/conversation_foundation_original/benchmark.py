"""Explicitly select an evaluated foundation checkpoint for the existing local MT-Bench."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from sml_v1.common import file_sha256,fingerprint,read_json
from evaluation.mt_bench_local import run as mt
from sft.conversation_foundation_original.launch import accepts_contract
RUN=ROOT/'runs/sft_conversation_foundation_v1'


def main():
    parser=argparse.ArgumentParser(description=__doc__,add_help=False)
    parser.add_argument('--step',required=True,type=int)
    args,rest=parser.parse_known_args()
    if '--model' in rest:parser.error('Select the checkpoint using --step only')
    if args.step<=0:raise ValueError('Select a saved post-training checkpoint')
    evaluation=read_json(RUN/f'evaluations/update_{args.step:05d}.json')
    bundle=RUN/evaluation['bundle'];meta=read_json(bundle/'model.safetensors.json')
    frozen=read_json(RUN/'contract.json')
    if meta['step']!=args.step or not accepts_contract(meta['contract'],frozen) or evaluation['contract']!=meta['contract']:
        raise ValueError('Selected checkpoint/evaluation contract mismatch')
    manifest=read_json(bundle/'manifest.json')
    digest=file_sha256(bundle/'model.safetensors')
    if digest!=manifest['files']['model.safetensors']['sha256']:
        raise ValueError('Checkpoint integrity mismatch')
    key=str(args.step);mt.MODELS[key]=(f'conversation-foundation-{key}',str(bundle.relative_to(ROOT)),digest)
    original=mt.selected
    def selected(k):
        result=original(k)
        result.update(selection_wrapper_sha256=file_sha256(__file__),training_contract=fingerprint(frozen),
                      development_evaluation_sha256=file_sha256(RUN/f'evaluations/update_{args.step:05d}.json'))
        return result
    mt.selected=selected
    sys.argv=[sys.argv[0],'--model',key,*rest];mt.main()

if __name__=='__main__':main()

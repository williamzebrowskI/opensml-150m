#!/usr/bin/env python3
"""Explicit five-Mac continuation of the retained adaptive run; plan-only by default."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sml_v2.common import ROOT, fingerprint, file_sha256, lock, read_json
from sml_v2.continuation import check_resume
from sml_v2.lr_scheduler import lower_floor_recipe, restore_scheduler
from sml_v2.recipe import learning_rate, tokens_per_update
from sml_v2.topology_transition import migrate_recipe

DEFAULT_NETWORK=ROOT/'diagnostics/m5_migration_20260922/network_20260922_115651'
DEFAULT_OUTPUT=ROOT/'runs/full_15b_five_mac_v1'
LOW_LR_OUTPUT=ROOT/'runs/full_15b_five_mac_low_lr_v1'
DATA_REVIEW_OUTPUT=ROOT/'runs/full_15b_five_mac_shuffled_v1'
TRIAL_PARENT=ROOT/'experiments/lr_trial_v1/parent/step_0073008_58d174e2c177'
TRIAL_OUTPUTS={arm:ROOT/f'runs/full_15b_five_mac_lr_{arm}_v1' for arm in ('rewarm','control')}
PARENT=ROOT/'runs/full_15b_plateau_v1'


def select_checkpoint(output, explicit=None, parent=PARENT):
    if explicit:
        return Path(explicit).resolve()
    checkpoint=(output if (output/'latest.json').exists() else parent)/'latest.json'
    if not checkpoint.is_file():
        raise ValueError('No latest checkpoint; fresh initialization is forbidden')
    return checkpoint


def verified_metadata(checkpoint):
    # Read/verify bytes without importing MLX or constructing the full model.
    from sml_v2.checkpoint_metadata import verify_bundle_metadata
    return verify_bundle_metadata(checkpoint)


def derive(metadata,batches,lr_floor=None,data_review=False,lr_trial=None):
    recipe=migrate_recipe(metadata,batches)
    manifest=None
    if data_review:
        if lr_floor is not None or recipe != metadata['recipe']:
            raise ValueError('Data review must not also change topology or the LR floor')
        from sml_v2.data_review import reviewed_recipe
        from sml_v2.common import load_corpus
        from sml_v2.stream import stream_manifest
        from sml_v2.tokenization import Tokenizer
        recipe=reviewed_recipe(metadata)
        manifest=stream_manifest(load_corpus(ROOT/'configs/corpus.json'),
                                 Tokenizer(ROOT/'tokenizer/bytebpe32k_v1'),recipe['stream'])
    if lr_floor is not None:
        if recipe != metadata['recipe']:
            raise ValueError('Complete the five-rank migration before changing the LR floor')
        recipe=lower_floor_recipe(metadata,lr_floor)
    if lr_trial is not None:
        if data_review or lr_floor is not None or recipe != metadata['recipe']:
            raise ValueError('LR trial must not also change topology, data, or floor')
        from sml_v2.lr_trial import trial_recipe
        recipe=trial_recipe(metadata,lr_trial)
    contract=dict(metadata['v2_contract'],recipe=fingerprint(recipe),
                  code=fingerprint({p.name:file_sha256(p) for p in (ROOT/'sml_v2').glob('*.py')}))
    if manifest is not None:contract['data']=fingerprint(manifest)
    transition=check_resume(metadata,contract,recipe,manifest)
    scheduler=restore_scheduler(recipe,metadata)
    if scheduler is None:raise ValueError('Expected the retained adaptive scheduler')
    return recipe,contract,transition,scheduler


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',action='store_true')
    p.add_argument('--network',type=Path,default=DEFAULT_NETWORK)
    p.add_argument('--output',type=Path,help='Default: original five-Mac run, or a separate low-LR run with --lr-floor')
    p.add_argument('--resume',type=Path)
    p.add_argument('--batches',default='4,3,2,2,2',help='Logical order: Mac-1,2,3,5,4; preserve sum 13')
    p.add_argument('--clear-stop',action='store_true')
    p.add_argument('--lr-floor',type=float,
                   help='Explicit lower floor plus one factor reduction; later resumes preserve saved scheduler state')
    p.add_argument('--data-review',action='store_true',
                   help='Separate continuation: 256-document/source shuffle, 4M-token validation; retain LR')
    p.add_argument('--lr-trial',choices=('rewarm','control'),
                   help='Separate 1000-update trial from the frozen best: ramp to 1e-5, or retain 5e-6')
    p.add_argument('--stop-after-steps',type=int,default=0)
    args=p.parse_args()
    batches=[int(x) for x in args.batches.split(',')]
    if args.stop_after_steps<0:p.error('--stop-after-steps must be nonnegative')
    if args.data_review and args.lr_floor is not None:p.error('Use --data-review without --lr-floor')
    if args.lr_trial and (args.data_review or args.lr_floor is not None):p.error('Use --lr-trial without other transition flags')
    output=(args.output or (TRIAL_OUTPUTS[args.lr_trial] if args.lr_trial else DATA_REVIEW_OUTPUT if args.data_review else LOW_LR_OUTPUT if args.lr_floor is not None else DEFAULT_OUTPUT)).resolve()
    if not output.is_relative_to(ROOT/'runs') or output in (PARENT,ROOT/'runs/full_15b_v1',ROOT/'runs/pilot_v1'):
        p.error('Use a separate output inside V2 runs; preserve all original runs')
    if args.lr_trial:
        checkpoint=Path(args.resume).resolve() if args.resume else (output/'latest.json' if (output/'latest.json').exists() else TRIAL_PARENT)
    else:
        checkpoint=select_checkpoint(output,args.resume,LOW_LR_OUTPUT if args.data_review else DEFAULT_OUTPUT if args.lr_floor is not None else PARENT)
    metadata=verified_metadata(checkpoint)
    recipe,contract,transition,scheduler=derive(metadata,batches,args.lr_floor,args.data_review,args.lr_trial)
    if transition in ('explicit-lr-floor-transition','explicit-data-review-transition','explicit-lr-trial-transition'):
        # Pointer and direct-bundle resumes must both preserve the parent run.
        if output==checkpoint.parent.resolve():
            p.error('Use a separate output for this experiment; preserve its source run')
    trial=recipe.get('lr_trial_transition')
    trial_end=trial['parent_step']+trial['trial_steps'] if trial else None
    if trial_end is not None and metadata['step'] >= trial_end:
        p.error('This bounded LR trial is already complete; refusing additional updates')
    rate=scheduler.rate(learning_rate(metadata['tokens']+tokens_per_update(recipe),recipe))
    print(json.dumps(dict(mode='run' if args.run else 'plan only',source=str(checkpoint),output=str(output),
        step=metadata['step'],tokens=metadata['tokens'],target_total_tokens=recipe['target_tokens'],
        physical_mac_order=[1,2,3,5,4],batches=batches,grad_accum=recipe['grad_accum'],
        context=recipe['model']['max_seq_len'],tokens_per_update=tokens_per_update(recipe),
        next_learning_rate=rate,lr_floor=recipe['plateau_scheduler']['min_lr'],
        scheduler=scheduler.state_dict(),resume_transition=transition,
        tokenizer=contract['tokenizer'],data=contract['data'],validation=metadata['validation_fingerprint'],
        document_shuffle_buffer=recipe['stream'].get('shuffle_buffer',0),
        validation_tokens=4*recipe['eval_batches_per_source']*recipe['eval_batch_size']*recipe['model']['max_seq_len'],
        new_validation_baseline_pending=transition=='explicit-data-review-transition'),indent=2),flush=True)
    if trial:
        print(json.dumps(dict(lr_trial=trial['arm'],start_learning_rate=trial['start_lr'],
            target_learning_rate=trial['target_lr'],warmup_updates=trial['warmup_steps'],
            absolute_stop_step=trial_end,remaining_updates=trial_end-metadata['step'],
            trial_tokens=trial['trial_steps']*tokens_per_update(recipe)),indent=2),flush=True)
    if not args.run:
        print('No model loaded. No SSH, downloads, training, or checkpoint writes. Add --run to start yourself.')
        return
    proof=read_json(args.network/'five_rank_correctness.json')
    modules={p.name:file_sha256(p) for p in (ROOT/'sml_v2').glob('*.py')}
    if proof.get('modules')!=modules or len(proof.get('receipts',[]))!=5 or any(r.get('status')!='passed' for r in proof['receipts']):
        raise ValueError('Five-rank correctness evidence does not match the current worker code')
    from five_mac_cluster import Cluster, PYTHON
    from sml_v2.launch import stage
    from sml_v2.common import load_corpus
    cluster=Cluster(args.network)
    with lock(output/'.launcher.lock'):
        if (output/'STOP').exists() and not args.clear_stop:
            raise ValueError('STOP exists; use --clear-stop when ready to resume')
        if (output/'latest.json').exists() and checkpoint.resolve()!= (output/'latest.json').resolve():
            raise ValueError('Existing five-Mac output must resume its own latest checkpoint')
        name=datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8]
        job=output/'jobs'/name;job.mkdir(parents=True)
        cluster.preflight(job)
        source=stage(job,recipe,ROOT/'tokenizer/bytebpe32k_v1',checkpoint,load_corpus(ROOT/'configs/corpus.json'))
        cluster.stage(source)
        if args.clear_stop:(output/'STOP').unlink(missing_ok=True)
        worker=['/usr/bin/caffeinate','-i',PYTHON,'-u','-m','sml_v2.pretrain',
                '--config',str(source/'recipe.json'),'--tokenizer',str(source/'tokenizer'),
                '--corpus',str(source/'corpus.json'),'--resume',str(source/'checkpoint'),
                '--save-dir',str(output),'--job-dir',str(job),'--ring-control','native',
                '--stop-after-steps',str(args.stop_after_steps)]
        # A bounded token experiment may legitimately run for hours. Workers
        # retain their no-progress watchdog; step limits end with a saved stop.
        cluster.launch(job,source,worker,timeout=None,stop_dir=output)
        result=read_json(job/'result.json')
        if result.get('status') not in ('stopped','complete'):raise RuntimeError('Missing clean completion')
        print(f'[verified-stop] {result}; log: {job/"train.log"}')


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Plan or launch the bounded, streaming five-Mac Stage B pilot."""
import argparse
from datetime import datetime
from pathlib import Path
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from sml_v1.common import atomic_json, file_sha256, fingerprint, load_corpus, lock, read_json
from sml_v1.checkpoint_metadata import verify_bundle_metadata
from sml_v1.continuation import check_resume
from sml_v1.recipe import learning_rate, tokens_per_update
from sml_v1.lr_scheduler import restore_scheduler
from sml_v1.stage_b import KEY, end_step, stage_recipe, training_manifest
from sml_v1.tokenization import Tokenizer

NETWORK = ROOT/'diagnostics/m5_migration_20260922/network_20260922_115651'
PARENT = ROOT/'experiments/stage_b_v1/parent/step_0073008_58d174e2c177'
OUTPUT = ROOT/'runs/stage_b_prose_v1'
CORPUS = ROOT/'stage_b/corpus.json'
AUDIT = ROOT/'stage_b/data_audit.json'
BUDGET = 250_000_000


def verify_readiness(modules, weights):
    audit = read_json(AUDIT)
    if (audit.get('corpus_sha256') != file_sha256(CORPUS)
            or audit.get('selection_sha256') != modules['stage_b_data.py']
            or audit.get('review_status') != 'passed'
            or set(audit.get('sources', {})) != set(weights)
            or any(v.get('accepted', 0) < 10 or v.get('scanned', 0) < 300
                   or not v.get('hf_cursor_restart') for v in audit['sources'].values())):
        raise ValueError('Stage B requires a current reviewed live-data audit')
    for record in audit['sources'].values():
        receipt = ROOT / record['receipt']
        if file_sha256(receipt) != record['receipt_sha256']:
            raise ValueError('Stage B data audit receipt changed')
    proof = read_json(NETWORK/'five_rank_correctness.json')
    if (proof.get('modules') != modules or len(proof.get('receipts', [])) != 5
            or any(r.get('status') != 'passed' for r in proof['receipts'])):
        raise ValueError('Five-rank correctness evidence differs from current Stage B worker code')


def prepare(output=OUTPUT):
    output = Path(output).resolve()
    if output != OUTPUT.resolve():
        raise ValueError('This reviewed pilot uses its own dedicated output directory')
    checkpoint = output/'latest.json' if (output/'latest.json').exists() else PARENT
    metadata = verify_bundle_metadata(checkpoint)
    corpus = load_corpus(CORPUS)
    tokenizer = Tokenizer(ROOT/'tokenizer/bytebpe32k_v1')
    manifest = training_manifest(corpus, tokenizer, metadata['recipe']['stream'])
    recipe = stage_recipe(metadata, manifest, BUDGET)
    modules = {p.name: file_sha256(p) for p in (ROOT/'sml_v1').glob('*.py')}
    contract = dict(recipe=fingerprint(recipe), data=fingerprint(manifest),
                    tokenizer=tokenizer.fingerprint, code=fingerprint(modules))
    transition = check_resume(metadata, contract, recipe, manifest)
    if metadata['step'] >= end_step(recipe):
        raise ValueError('Stage B pilot is complete; refusing to reset or extend its budget')
    scheduler = restore_scheduler(recipe, metadata)
    plan = dict(mode='plan only', source=str(checkpoint), output=str(output),
        step=metadata['step'], tokens=metadata['tokens'], starting_best=recipe[KEY]['baseline_loss'],
        additional_token_budget=BUDGET,
        actual_pilot_tokens=recipe[KEY]['trial_steps']*tokens_per_update(recipe),
        absolute_stop_step=end_step(recipe), remaining_updates=end_step(recipe)-metadata['step'],
        next_learning_rate=scheduler.rate(learning_rate(metadata['tokens']+tokens_per_update(recipe),recipe)),
        weights=manifest['weights'], batch_order='Mac-1, Mac-2, Mac-3, Mac-5, Mac-4',
        batches=recipe['batches'], context=recipe['model']['max_seq_len'],
        tokens_per_update=tokens_per_update(recipe), tokenizer=tokenizer.fingerprint,
        reference_validation=metadata['validation_fingerprint'],
        reference_weights=manifest['reference']['weights'], transition=transition,
        storage='HF streaming; bounded RAM buffers; pending input and selection state in checkpoints')
    return checkpoint, metadata, recipe, corpus, modules, plan


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--clear-stop', action='store_true')
    args = p.parse_args()
    checkpoint, metadata, recipe, corpus, modules, plan = prepare()
    plan['mode'] = 'run' if args.run else 'plan only'
    import json
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        print('No model loaded, network access, training, or checkpoint writes. Add --run to start yourself.')
        return
    verify_readiness(modules, plan['weights'])
    from five_mac_cluster import Cluster, PYTHON
    from sml_v1.launch import stage
    cluster = Cluster(NETWORK)
    with lock(OUTPUT/'.launcher.lock'):
        if (OUTPUT/'STOP').exists() and not args.clear_stop:
            raise ValueError('STOP exists; use --clear-stop to resume this pilot')
        job = OUTPUT/'jobs'/(datetime.now().strftime('%Y%m%d_%H%M%S_')+uuid.uuid4().hex[:8])
        job.mkdir(parents=True)
        cluster.preflight(job)
        source = stage(job, recipe, ROOT/'tokenizer/bytebpe32k_v1', checkpoint, corpus)
        cluster.stage(source)
        if args.clear_stop:
            (OUTPUT/'STOP').unlink(missing_ok=True)
        worker = ['/usr/bin/caffeinate','-i',PYTHON,'-u','-m','sml_v1.pretrain',
                  '--config',str(source/'recipe.json'),'--tokenizer',str(source/'tokenizer'),
                  '--corpus',str(source/'corpus.json'),'--resume',str(source/'checkpoint'),
                  '--save-dir',str(OUTPUT),'--job-dir',str(job),'--ring-control','native']
        cluster.launch(job, source, worker, timeout=None, stop_dir=OUTPUT)
        result = read_json(job/'result.json')
        if result.get('status') not in ('stopped', 'complete'):
            raise RuntimeError('Stage B did not stop cleanly')
        print(f'[verified-stop] {result}; log: {job/"train.log"}', flush=True)


if __name__ == '__main__':
    main()

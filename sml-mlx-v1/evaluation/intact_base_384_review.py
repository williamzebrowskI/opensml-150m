"""Summarize completed checkpoint-384 evaluations and a proposed next experiment."""
import collections
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sml_v1.common import atomic_json, file_sha256, read_json

DIAG = ROOT / 'diagnostics'
NEW = DIAG / 'intact_base_384_review_v1'
MODELS = {
    'pretrained': DIAG / 'full_benchmarks_v1/pretrained',
    'intact-base-384': NEW / 'benchmarks/intact-base-384',
    '1472': DIAG / 'explanation_transfer_1024_benchmarks_v1/explanation-transfer-1024-1472',
    '1920': DIAG / 'skill_balance_benchmarks_v1/skill-balance-1920',
}


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    summaries = {}; mc = {}; ifeval = {}; manifests = {}
    for model, path in MODELS.items():
        summaries[model] = {}
        for suite in ['multiple-choice'] + ([] if model == 'pretrained' else ['ifeval']):
            summary = read_json(path / suite / 'summary.json')
            if summary['completed'] != summary['expected'] or summary['status'] != 'complete':
                raise ValueError('Finish the full public suite first: ' + model + ' ' + suite)
            summaries[model][suite] = summary
            manifests[(model, suite)] = read_json(path / suite / 'manifest.json')
            rows = records(path / suite / 'records.jsonl')
            if suite == 'multiple-choice': mc[model] = {r['id']: r for r in rows}
            else: ifeval[model] = {r['id']: r for r in rows}
    protocols = {}
    for (model, suite), manifest in manifests.items():
        reference = manifests[('intact-base-384', suite)]
        equal = all(manifest[k] == reference[k] for k in ('protocol', 'prepared_sha256', 'versions',
                    'context', 'fp32', 'attention', 'ffn', 'batch_size', 'zero_shot'))
        protocols[model + '/' + suite] = equal
        if not equal: raise ValueError('Benchmark protocols differ: ' + model + '/' + suite)
    paired = {}
    for model in ('pretrained', '1472', '1920'):
        paired[model] = {}
        for task in ('arc_easy', 'arc_challenge', 'piqa', 'hellaswag'):
            selected = [r for r in mc['intact-base-384'].values() if r['task'] == task]
            paired[model][task] = {}
            for key in ('correct', 'correct_norm'):
                paired[model][task][key] = dict(
                    gained=sum(r[key] and not mc[model][r['id']][key] for r in selected),
                    lost=sum(not r[key] and mc[model][r['id']][key] for r in selected))
    families = {}
    for model, rows in ifeval.items():
        count = collections.defaultdict(lambda: dict(passed=0, total=0))
        for row in rows.values():
            for name, passed in zip(row['strict']['instruction_id_list'], row['strict']['follow_instruction_list']):
                count[name]['total'] += 1; count[name]['passed'] += int(passed)
        families[model] = dict(count)
    fresh = read_json(NEW / 'fresh_review.json')
    audit = read_json(NEW / 'training_weight_audit.json')
    decode = read_json(NEW / 'decode_audit.json')
    if decode['status'] != 'passed': raise ValueError('Resolve the generation-path audit before interpreting results')
    cleanup = read_json(ROOT / 'reports/INTACT_BASE_384_CHECKPOINT_CLEANUP_20260928.json')
    proposal = dict(
        status='recommendation_only_no_training_started', source='intact-base-384',
        objective='Family-balanced grounded SFT + gold choice ranking + complete-response replay; optional frozen-parent prose KL retention term',
        proposed_family_weights=dict(grounded_answers=.40, complete_instruction_outputs=.25,
                                     choice_ranking=.20, conversation_rewrite_replay=.15),
        proposed_budget='128-update pilot, evaluated every 32; extend only after useful held-out content improves',
        learning_rate='Provisional 1e-6 to 3e-6; validate with disposable updates rather than assuming optimality',
        selection='Correct supported content AND complete requested outputs on independent source/topic groups; retain choice recognition, follow-ups, prose and stopping',
        source_status='No new training corpus has been prepared or approved by this review. Reuse audited human reading sources and reviewed labeled knowledge sources only after overlap/quality checks.',
        excluded_from_training='All public benchmark questions/answers, reserved conversations, fresh diagnostic prompts, and other previously inspected development/test items',
        alternatives='DPO/GRPO may be explored later, after reliable correct/incorrect responses or rewards exist; current evidence prioritizes supervised grounding and task execution',
    )
    result = dict(status='complete', training=False, automatic_promotion=False,
                  benchmark_summaries=summaries, protocols_match=protocols,
                  paired_multiple_choice=paired, strict_instruction_families=families,
                  fresh_review=fresh, training_weight_audit=audit, decode_audit=decode,
                  cleanup_receipt=str(ROOT / 'reports/INTACT_BASE_384_CHECKPOINT_CLEANUP_20260928.json'),
                  proposed_next_experiment=proposal,
                  limitations='Repeatedly inspected public tests, one run per model, no causal ablation of objective/data/LR. Small diagnostic set and manual review are not representative assistant accuracy. Public benchmark items remain evaluation-only. Reserved conversations were not opened.')
    report = ROOT / 'reports/INTACT_BASE_384_REVIEW_20260928'
    atomic_json(report.with_suffix('.json'), result)
    lines = ['# V1 base SFT 384: evaluation and next fine-tuning direction', '',
             'Checkpoint 384 was explicitly selected by the user before these public evaluations. '
             'This is a new base-starting SFT lineage, distinct from the older response-expansion checkpoint also numbered 384.', '',
             '## Cleanup', '',
             f'Retained checkpoint 384, its optimizer and manifest unchanged. Removed {len(cleanup["removed"])} other checkpoint bundles '
             f'from the base pilot and its continuation ({cleanup["logical_bytes_removed"] / 1024**3:.2f} GiB logical bytes). '
             'Scripts, data preparation, logs, saved answers and reports remain. The latest pointer identifies retained 384; '
             'the historical report still records that training stopped at 448. Prior runs and pretrained weights remain unchanged.', '',
             '## Full public benchmarks', '',
             'All 15,428 multiple-choice questions and all 541 IFEval prompts were evaluated. '
             'Data, runtime, numerical settings and scoring/decoding protocols match the comparison models. '
             'Scores below are percentages; raw and character-normalized accuracy are separate metrics.', '',
             '| Task / metric | Pretrained base | This run: 384 | Older 1472 | Older 1920 |',
             '|---|---:|---:|---:|---:|']
    for task, label in [('arc_easy', 'ARC-Easy'), ('arc_challenge', 'ARC-Challenge'), ('piqa', 'PIQA'), ('hellaswag', 'HellaSwag')]:
        for metric, suffix in [('acc', 'raw'), ('acc_norm', 'normalized')]:
            values = [summaries[m]['multiple-choice']['tasks'][task][metric] * 100 for m in MODELS]
            lines.append('| ' + label + ' ' + suffix + ' | ' + ' | '.join(f'{v:.2f}%' for v in values) + ' |')
    for metric, label in [('prompt_accuracy', 'IFEval strict prompt'), ('instruction_accuracy', 'IFEval strict instruction')]:
        values = [summaries[m]['ifeval']['strict'][metric] * 100 for m in ('intact-base-384', '1472', '1920')]
        lines.append('| ' + label + ' | — | ' + ' | '.join(f'{v:.2f}%' for v in values) + ' |')
    s = summaries['intact-base-384']['ifeval']
    lines += ['', f'IFEval natural EOS: {s["stop_counts"]["eos"]}/541; token-limit hits: {s["stop_counts"]["length"]}/541. '
              'IFEval checks formal instructions rather than factual correctness or full content fulfillment. '
              'Inspected strict passes include repeated headings instead of a summary (1000), placeholder repetition '
              'instead of a resume (1005), and treating a requested word limit as an invented Japan travel policy (1092).', '',
              'The continuation improved stopping and local constraints, but full MC recognition is not broadly stronger '
              'than the pretrained base. ARC-Easy regressed; ARC-Challenge/PIQA/HellaSwag remain close. '
              'The older 1472/1920 models provide stronger ARC recognition under the matched protocol. '
              'Small score differences should not be treated as demonstrated general gains across seeds.', '',
              '## Fresh diagnostic answers', '',
              f'Twenty hand-authored prompts were frozen before inference. Manual review found '
              f'{fresh["counts"]["fully_fulfills_request"]}/20 fully fulfilled requests and '
              f'{fresh["counts"]["content_correct_without_unsupported_additions"]}/20 responses with correct content and no '
              'contradictory/unsupported additions. This targeted small set is not an estimate of general assistant accuracy.', '',
              'The model recognized 8/12 correct candidate answers with both scoring metrics, yet free responses often '
              'contradicted the passage or invented details. It confused owner/borrower, ignored a moved folder, '
              'invented a museum history, explained sorting instead of sorting, repeated JSON keys and ignored '
              'output-only rewrite requests. Supplying facts helped candidate recognition in one evaporation pair, '
              'but did not yield a grounded generated answer. One follow-up correctly updated a meeting to 4 pm.', '',
              'An additional decoding audit compared cached and independent full-prefix inference at every one of '
              'the first 128 decisions on two failing prompts. All 256 greedy tokens agreed; the logits were not '
              f'bit-identical (maximum absolute difference {decode["max_abs_logit_difference"]:.6f}). '
              'This limited check supports interpreting those failures as model behavior, without proving every serving path correct.', '',
              '## Training mix explains a useful next hypothesis', '',
              'The actual token-weighted CE allocation averaged over updates 1–384 was:', '',
              '| Source family | Mean loss weight |', '|---|---:|']
    for name, fraction in audit['mean_loss_weight_share'].items():
        lines.append(f'| {name} | {100 * fraction:.2f}% |')
    lines += ['', 'There was no dedicated passage-QA family. Long Magpie targets dominated despite a conversation-count '
              'mixture that looked more balanced. This is a plausible explanation and a testable intervention, not a causal ablation.', '',
              '## Recommended next experiment', '',
              '**Start one short pilot from preserved 384 using family-balanced grounded SFT plus gold choice ranking and replay.**', '',
              '1. Grounded answer targets: human passage QA, answerable/unanswerable cases, ownership/recipient/time/negation '
              'relations. Complete short answers with EOS, and no invented explanatory padding.',
              '2. Direct task execution: sorted lists, supported summaries, real rewrites, valid JSON, exact bullets/sentences '
              'and combinations. Targets must deliver requested content as well as formal constraints.',
              '3. Gold choice ranking for checked physical/common-sense/science questions from training-only sources. '
              'Track raw and normalized recognition separately. Do not teach all chat answers to become one-word labels.',
              '4. Retain complete conversation/rewrite replay and consider frozen-parent prose KL. Normalize examples within '
              'each family and assign explicit family weights so longer targets cannot consume nearly all of the objective.', '',
              'A provisional starting allocation is 40% grounded answers, 25% complete instruction outputs, '
              '20% choice ranking and 15% conversation/rewrite replay, with any prose KL as a separate retention term. '
              'These are proposed settings, not validated optima. Start with 128 updates, evaluations every 32, '
              'and a modest provisional peak LR around 1e-6–3e-6 after a disposable gradient/learning check. '
              'A matched replay-only control can distinguish the new objective from additional update budget.', '',
              'Review and partition genuinely distinct source/content groups before selecting a new training pool. '
              'The existing human-reading loader and prior Recovery ranking method are reusable precedents, '
              'not approval to reuse their development/test items. Check support, labels, licensing and overlap. '
              'Keep every public benchmark item, all 20 new diagnostic prompts and reserved conversations out of training. '
              'Select on correct complete held-out responses AND retained recognition/prose/follow-up behavior; '
              'freeze selection before the next public comparison. Confirm a promising result with another seed.', '',
              'DPO/GRPO are later alternatives once correct responses and reliable negative examples/rewards exist. '
              'The earlier small local DPO pilot did not establish useful natural-response gains. '
              'The official [SmolLM2 recipe](https://github.com/huggingface/alignment-handbook/blob/main/recipes/smollm2/README.md) '
              'uses SFT followed by DPO, but that is not evidence that DPO will repair this model\'s present deficits. '
              '[Research on new factual knowledge in SFT](https://arxiv.org/abs/2405.05904) found difficulty learning '
              'new facts and increased hallucination in its closed-book QA setup; it supports caution about expecting '
              'generic chat SFT to supply missing science knowledge, not a quantitative prediction for this model.', '',
              'No new training or playground promotion was performed. The reserved conversation test remains unused. '
              'This review recommends an experiment; it does not promise improvement on all benchmarks.']
    report.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(str(report.with_suffix('.md')))


if __name__ == '__main__':
    main()

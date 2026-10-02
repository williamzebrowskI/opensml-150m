"""Run the frozen checkpoint 384 benchmarks and show readable scores."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / 'diagnostics/intact_base_384_review_v1/benchmarks/intact-base-384'


def show_results(suite):
    path = RESULTS / suite / 'summary.json'
    if not path.exists():
        print(f'{suite}: no saved results yet', flush=True)
        return
    result = json.loads(path.read_text())
    print(f'\nCheckpoint 384 — {suite}: {result["status"]} '
          f'({result["completed"]:,}/{result["expected"]:,})', flush=True)
    if suite == 'multiple-choice':
        print(f'{"Benchmark":<16} {"Accuracy":>10} {"Normalized":>12}')
        for key, label in [('arc_easy', 'ARC-Easy'), ('arc_challenge', 'ARC-Challenge'),
                           ('piqa', 'PIQA'), ('hellaswag', 'HellaSwag')]:
            scores = result['tasks'][key]
            print(f'{label:<16} {scores["acc"]:>9.2%} {scores["acc_norm"]:>11.2%}')
    else:
        for kind in ('strict', 'loose'):
            scores = result[kind]
            print(f'{kind.capitalize()}: prompts {scores["prompt_accuracy"]:.2%}; '
                  f'instructions {scores["instruction_accuracy"]:.2%}')
        if result['status'] != 'complete':
            print('Partial results; rerun with --run to resume.')
    print(f'Results: {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=['all', 'multiple-choice', 'ifeval'], default='all')
    parser.add_argument('--run', action='store_true', help='Run or resume, then display scores')
    args = parser.parse_args()
    suites = ['multiple-choice', 'ifeval'] if args.suite == 'all' else [args.suite]
    for suite in suites:
        if args.run:
            outcome = subprocess.run([sys.executable, '-u', str(HERE / 'intact_base_384_benchmark.py'),
                                      '--suite', suite, '--run'])
            if outcome.returncode:
                return outcome.returncode
        show_results(suite)
        if args.run:
            result = json.loads((RESULTS / suite / 'summary.json').read_text())
            if result['status'] != 'complete':
                break
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

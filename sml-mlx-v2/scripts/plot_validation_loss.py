#!/usr/bin/env python3
"""Plot recorded V2 validation loss at step milestones; read-only to runs."""

import argparse
import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RUNS = ('pilot_v1', 'full_15b_v1', 'full_15b_plateau_v1', 'full_15b_five_mac_v1')
EVAL = re.compile(r'^\[eval\s+(\d+)\] val_loss=([\d.]+) val_ppl=([\d.]+) sources=(\{.*\})$')


def collect():
    weights = {s['label']: s['weight'] for s in json.loads((ROOT / 'configs/corpus.json').read_text())['sources']}
    rows, snapshots, contracts = {}, [], []
    previous_end = 0
    for index, name in enumerate(RUNS):
        run = ROOT / 'runs' / name
        pointer = json.loads((run / 'latest.json').read_text())
        metadata = json.loads((run / pointer['bundle'] / 'model.safetensors.json').read_text())
        recipe = metadata['recipe']
        update = sum(recipe['batches']) * recipe['grad_accum'] * recipe['model']['max_seq_len']
        if update != 106496 or metadata['tokens'] != metadata['step'] * update:
            raise ValueError('Unexpected training-token accounting')
        contracts.append((metadata['validation_fingerprint'], metadata['v2_contract']['tokenizer'], metadata['v2_contract']['data']))
        upper = metadata['step'] if index < len(RUNS) - 1 else float('inf')
        logs = [run / 'train.log'] if (run / 'train.log').exists() else sorted(run.glob('jobs/*/train.log'))
        if not logs:
            raise ValueError(f'No logs for {name}')
        for log in logs:
            data = log.read_bytes()
            snapshots.append(dict(path=str(log), bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            for number, line in enumerate(data.rsplit(b'\n', 1)[0].decode().splitlines(), 1):
                match = EVAL.fullmatch(line)
                if not match:
                    continue
                step, printed = int(match[1]), float(match[2])
                if not previous_end < step <= upper:
                    continue
                sources = ast.literal_eval(match[4])
                if sources.keys() != weights.keys():
                    raise ValueError('Validation source mismatch')
                loss = sum(sources[k] * weights[k] for k in weights) / sum(weights.values())
                if abs(loss - printed) > .000051:
                    raise ValueError(f'Weighted loss mismatch: {log}:{number}')
                row = dict(step=step, tokens=step * update, validation_loss=loss,
                           logged_loss=printed, run=name, log=str(log), line=number, **sources)
                if step in rows and abs(rows[step]['validation_loss'] - loss) > 1e-10:
                    raise ValueError(f'Conflicting validation records at step {step}')
                rows[step] = row
        previous_end = metadata['step']
    if len(set(contracts)) != 1:
        raise ValueError('Held-out validation, tokenizer or data contract changed')
    return sorted(rows.values(), key=lambda r: r['step']), snapshots, contracts[0][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=int, default=5000)
    parser.add_argument('--through-step', type=int, help='Keep a prior snapshot endpoint')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error('--interval must be positive')
    if args.output is None:
        args.output = ROOT / f'reports/loss_every_{args.interval}'
    interval_label = f'{args.interval / 1000:g}k'
    rows, snapshots, identity = collect()
    if args.through_step is not None:
        rows = [row for row in rows if row['step'] <= args.through_step]
    if not rows:
        raise ValueError('No validation records')
    latest = rows[-1]
    sampled = [dict(target_step='', kind='first recorded evaluation', **rows[0])]
    for target in range(args.interval, latest['step'] + 1, args.interval):
        row = min(rows, key=lambda r: (abs(r['step'] - target), r['step']))
        if abs(row['step'] - target) > 125:
            raise ValueError(f'Missing evaluation near milestone {target}')
        sampled.append(dict(target_step=target, kind=f'{interval_label} milestone', **row))
    if not sampled or sampled[-1]['step'] != latest['step']:
        sampled.append(dict(target_step='', kind='latest evaluation', **latest))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(sampled[0]))
        writer.writeheader()
        writer.writerows(sampled)
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    args.output.with_suffix('.json').write_text(json.dumps(dict(
        created_utc=timestamp, validation_fingerprint=identity,
        metric='Mixture-weighted validation cross-entropy (nats/token)',
        selection=f'First recorded evaluation, nearest recorded evaluation to each {args.interval:,}-step milestone, plus latest. No step-0 measurement or interpolation.',
        weights=dict(web=55, dclm=25, wiki=10, cosmopedia=10),
        recorded_evaluations=len(rows), selected=sampled, logs=snapshots), indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, FuncFormatter, FixedLocator

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(13.5, 6.8), dpi=170)
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#f8fafc')
    fig.subplots_adjust(left=.085, right=.96, bottom=.22, top=.78)
    fig.text(.085, .93, 'OpenSML V2 · Validation loss', fontsize=23, weight='bold', color='#16243a')
    fig.text(.085, .865, f'{args.interval:,}-step markers  |  150M parameters  |  Fixed held-out set', fontsize=12, color='#526276')
    visible = [r for r in rows if r['step'] >= sampled[0]['step']]
    ax.plot([r['step'] for r in visible], [r['validation_loss'] for r in visible],
            color='#2563eb', linewidth=2.4, label='All recorded evaluations', zorder=2)
    ax.plot([r['step'] for r in sampled], [r['validation_loss'] for r in sampled],
            color='#2563eb', linestyle='none', marker='o', markersize=6.5,
            markeredgecolor='#f8fafc', markeredgewidth=1.5, label=f'First + {interval_label}-step milestones', zorder=3)
    for index, row in enumerate(sampled[:-1]):
        label = (f"{row['validation_loss']:.3f}\nFirst measured · step {row['step']:,}"
                 if index == 0 else f"{row['validation_loss']:.3f}")
        ax.annotate(label, (row['step'], row['validation_loss']),
                    xytext=(8 if index == 0 else 0, 14), textcoords='offset points',
                    ha='left' if index == 0 else 'center', fontsize=10, color='#24466e')
    ax.scatter([latest['step']], [latest['validation_loss']], s=85, color='#e97826',
               edgecolor='#f8fafc', linewidth=1.5, zorder=5, label='Latest evaluation')
    ax.annotate(f"{latest['validation_loss']:.4f}\nStep {latest['step']:,}",
                (latest['step'], latest['validation_loss']), xytext=(0, 19),
                textcoords='offset points', ha='center', va='bottom', color='#a3470c',
                fontsize=10, weight='bold', bbox=dict(boxstyle='round,pad=.35', fc='#fff3e9', ec='none'))
    for at, label in [(38056, 'Adaptive LR'), (50096, 'Five Macs')]:
        if sampled[0]['step'] <= at <= latest['step']:
            ax.axvline(at, color='#9aa8ba', linewidth=1, linestyle=(0, (3, 4)), zorder=0)
            ax.text(at - 300, .97, label, transform=ax.get_xaxis_transform(),
                    rotation=90, ha='right', va='top', fontsize=9, color='#6b7788')
    ax.set_xlim(0, latest['step'] + 2400)
    upper_loss = max(r['validation_loss'] for r in visible) + .45
    ax.set_ylim(2.70, upper_loss)
    ax.xaxis.set_major_locator(MultipleLocator(args.interval))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: '0' if value == 0 else f'{value/1000:.0f}k'))
    ax.yaxis.set_major_locator(FixedLocator([2.7] + [n / 2 for n in range(6, int(upper_loss * 2) + 1)]))
    ax.set_xlabel('Optimizer step', labelpad=12, color='#526276')
    ax.set_ylabel('Validation loss · lower is better', labelpad=13, color='#526276')
    ax.grid(axis='y', color='#dce3ed', linewidth=.75)
    ax.tick_params(colors='#526276', length=0, pad=8)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_color('#cbd5e1')
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1.23), frameon=False, ncol=3, fontsize=9)
    fig.text(.085, .105, 'Milestones use the nearest measured evaluation; actual steps are listed in the CSV. No interpolated losses.',
             fontsize=9, color='#526276')
    fig.text(.085, .067, f"No step-0 loss was recorded; the starting point is the first evaluation at step {rows[0]['step']:,}. Loss axis starts at 2.70.",
             fontsize=9, color='#526276')
    fig.text(.085, .031, f'Snapshot {timestamp}  ·  Web / DCLM / Wiki / Cosmopedia weights: 55 / 25 / 10 / 10',
             fontsize=8, color='#738096')
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(args.output.with_suffix('.' + suffix), facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps(dict(output=str(args.output), observations=len(rows), selected=[
        {k: r[k] for k in ('target_step', 'step', 'validation_loss')} for r in sampled]), indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Continue the selected v2 checkpoint to 15B total tokens; never start fresh."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sml_v2.common import ROOT
from launch_pretrain import cli


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--from', dest='selection', choices=('latest', 'best'), default='latest')
    p.add_argument('--clear-stop', action='store_true')
    p.add_argument('--ring-control', choices=('all-sum', 'native'), default='all-sum')
    args = p.parse_args()
    output = ROOT / 'runs/full_15b_v1'
    # Once continuation exists, never silently go back to the pilot checkpoint.
    source = output if (output / 'latest.json').exists() else ROOT / 'runs/pilot_v1'
    checkpoint = source / (args.selection + '.json')
    if not checkpoint.is_file():
        p.error(f'{args.selection} checkpoint not found: {checkpoint}; no fallback or fresh start')
    print(f'[selected] {args.selection} from {source}; output={output}', flush=True)
    forwarded = ['launch_pretrain', '--resume', str(checkpoint), '--save-dir', str(output),
                 '--extend-target-tokens', '15000000000', '--ring-control', args.ring_control]
    if args.run:
        forwarded.append('--run')
    if args.clear_stop:
        forwarded.append('--clear-stop')
    sys.argv = forwarded
    return cli()


if __name__ == '__main__':
    sys.exit(main())

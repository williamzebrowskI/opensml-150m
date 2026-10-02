#!/usr/bin/env python3
"""Resume the validation-aware 15B experiment; plan only unless --run is given."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sml_v2.common import ROOT
from launch_pretrain import cli


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', action='store_true')
    p.add_argument('--clear-stop', action='store_true')
    p.add_argument('--ring-control', choices=('all-sum', 'native'), default='native')
    args = p.parse_args()
    output = ROOT / 'runs/full_15b_plateau_v1'
    source = output if (output / 'latest.json').exists() else ROOT / 'runs/full_15b_v1'
    checkpoint = source / 'latest.json'
    if not checkpoint.is_file():
        p.error(f'No latest checkpoint: {checkpoint}; fresh training is forbidden')
    print(f'[selected] latest from {source}; output={output}; parent run unchanged', flush=True)
    sys.argv = ['launch_pretrain', '--resume', str(checkpoint), '--save-dir', str(output),
                '--plateau-config', str(ROOT / 'configs/plateau.json'), '--ring-control', args.ring_control]
    if args.run:
        sys.argv.append('--run')
    if args.clear_stop:
        sys.argv.append('--clear-stop')
    return cli()


if __name__ == '__main__':
    sys.exit(main())

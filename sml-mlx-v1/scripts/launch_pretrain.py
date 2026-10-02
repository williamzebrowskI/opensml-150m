#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sml_v1.launch import main


def cli():
    try:
        main()
    except subprocess.CalledProcessError as exc:
        # Preflight captures stderr, which CalledProcessError's traceback omits.
        print(f'ERROR: Launch command exited with status {exc.returncode}.', file=sys.stderr)
        details = '\n'.join(part.strip() for part in (exc.stdout, exc.stderr) if part and part.strip())
        print(details or 'The failed command returned no diagnostic output.', file=sys.stderr)
        if 'Other training/benchmark workers active:' in details:
            print('Restart blocked by the idle safety check. The process IDs above may still be '
                  'shutting down after a saved stop. Wait for them to exit before retrying. '
                  'Do not start a second training job or kill unrelated processes.', file=sys.stderr)
        print('No automatic retry or restart was attempted.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(cli())

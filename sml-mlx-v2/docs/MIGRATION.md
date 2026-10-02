# Moving the training project to another Mac

This is a general migration checklist, not an automatic setup or training
command. The 2026-09-22 migration to the M5 coordinator and five-Mac ring has
dedicated tooling and small-model verification. Use the
[five-Mac training guide](FIVE_MAC_TRAINING.md) for that rack's current state,
resume command and remaining full-model validation.

## What Git provides

- V1 and V2 model/trainer source, tests, launchers, recipes, and documentation.
- Dashboard and playground source.
- Tokenizer-training code, but not fitted tokenizer artifacts or sample data.
- Existing network tools and the four-node ring map. Some helpers retain
  rack-specific paths and physical identity checks; they are not generic
  plug-and-play cluster provisioning tools.

From the home directory on the destination Mac, if the destination does not
already exist:

```bash
git clone https://github.com/williamzebrowskI/sml-mlx.git
cd sml-mlx
```

If a checkout already exists, inspect its local changes before updating it.
Do not overwrite a working directory or reset its changes blindly.

## Prepare the environment separately

Do not copy a virtual environment: recreate it using the matching Python
version and verify installed package versions against the source machine.
The verified rack runtime uses CPython 3.11 and the custom MLX / mlx-metal
`0.32.3.dev20260920+7241f12` build. The dependency files retain the stable
`0.32.2` baseline; installing those pins alone does not reproduce the native
ring runtime. Follow [the build provenance](MLX_NATIVE_BUILD.md), including
the wheel checksums, and validate compatibility on new hardware before use.

The current orchestration references both the repository-root `.venv` and
`sml-mlx-v2/.venv`. Configure and verify those interpreter paths explicitly.
Do not reinstall shared packages on the existing Macs while training runs.

Configure Hugging Face authentication and SSH access locally. Credentials,
private keys, trusted-host files, and `.env` files do not belong in Git.

## Transfer the exact training artifacts

Leave production running during source/environment preparation. Before the
final state transfer, request a clean saved stop and verify that all workers
have exited. Keep the original files as the rollback copy.

Transfer and checksum-verify:

- The frozen `tokenizer/bytebpe32k_v1` directory. Do not retrain a replacement
  tokenizer for an existing model.
- The complete selected checkpoint bundles and their `best.json` and
  `latest.json` pointers. Include weights, optimizer state, metadata,
  scheduler state, and compressed data-cursor state, not just model weights.
- The source run's recipes, logs, and job provenance needed to audit the
  continuation. The currently active branch is `runs/full_15b_plateau_v1`.
- Required local SSH/host configuration, reviewed for the new identities and
  topology rather than blindly reused. Those live configuration files are
  intentionally excluded from Git.

The dataset is streamed from Hugging Face; migration does not require a full
dataset download. Preserve the saved cursor and validation contract instead.

## Validate before resuming

Do not rename machines, change Thunderbolt wiring, or run distributed GPU
tests while the current training job is active. For a new coordinator or
fifth Mac, first update the physical-identity guards, routes, SSH host-key
mapping, rank order, batch allocation, and explicit resume compatibility.

Verify communication, native collectives, gradient weighting, checkpoint
loading, and exact data/scheduler continuity before a production launch.
Compare throughput at the same 106,496 tokens per update first. Do not reset
the optimizer, learning schedule, or dataset position merely to accommodate
the new topology. The user starts the production run after these checks.

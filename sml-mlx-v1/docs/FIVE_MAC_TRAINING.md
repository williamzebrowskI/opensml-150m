# Five-Mac continuation

Prepared 2026-09-22. The user starts the full model. Development validation
uses small synthetic models and communication probes only; the 150M model
has not been run or benchmarked as part of this migration.

## Machines and contribution

The Macs have been renamed using the original handoff identities. The user
accepted the discovered cable order **Mac-1 → Mac-2 → Mac-3 → Mac-5 → Mac-4 → Mac-1**.

| Logical rank | Physical label | Chip | Microbatch | Tokens per update |
| --- | --- | --- | ---: | ---: |
| 0 | Mac-1 | M5 Ultra | 4 | 32,768 |
| 1 | Mac-2 | M3 Ultra | 3 | 24,576 |
| 2 | Mac-3 | M4 Max | 2 | 16,384 |
| 3 | Mac-5 | M4 Max | 2 | 16,384 |
| 4 | Mac-4 | M4 Max | 2 | 16,384 |

This is replicated data parallelism. Each Mac holds the complete model and
FP32 optimizer, receives disjoint examples, computes gradients, combines the
gradients with sample-count weighting, and applies the same update. Mac-1 also
streams/tokenizes the corpus, supplies global batches, evaluates, and saves.
The model architecture and 32K tokenizer remain unchanged.

Context is 2,048 and accumulation is four. The effective update remains
`(4 + 3 + 2 + 2 + 2) * 4 * 2048 = 106496` tokens. This initial allocation
preserves the previous optimization geometry; it has not been tuned by a
full-model benchmark, and no speedup is claimed.

Ethernet carries pinned SSH and job-scoped reverse tunnels to a loopback-only
coordinator. Native JACCL over Thunderbolt RDMA carries tensors. The launcher
requires the native ring and five ranks; it cannot silently select local or
TCP-ring execution. The active network plan is:

`diagnostics/m5_migration_20260922/network_20260922_115651`.

## Start or resume yourself

### Data review continuation (2026-09-23, after step 70,710)

The optional `--data-review` continuation adds resumable document shuffling and
a 4.19M-token validation set, retains the 5e-6 LR, and preserves the stopped
parent run. See [the experiment guide and bounded resume command](DATA_REVIEW_CONTINUATION.md).
Its new validation baseline and the original legacy metric are tracked separately.

### Lower-LR continuation (2026-09-23)

The stopped pretraining run is at step **60,263**, **6,417,768,448** tokens.
Its LR cap reached the old **3e-5** floor. To continue with one factor-of-two
reduction to **1.5e-5** and a new **5e-6** floor:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/scripts/launch_five_mac.py \
  --lr-floor 5e-6 --run --clear-stop
```

This writes to `runs/full_15b_five_mac_low_lr_v1`. The first launch selects
the original five-Mac run's latest checkpoint; later launches with the same
command select the new run's own latest. It preserves weights, FP32 optimizer,
stream cursor, validation identity, scheduler history and global batch size.
The initial reduction starts the usual two-evaluation cooldown; later reductions
retain patience 8, factor 0.5 and minimum improvement 0.002. Repeating the flag
does not replay the reduction or reset the saved controller. The 15B total
token target remains unchanged. The original pretraining latest/best are retained.
The prior SFT experiment and its planning/audit files were subsequently removed
at the user's request.

Omitting `--lr-floor` still selects the original five-Mac run and its old LR.
The launcher's matching five-rank correctness evidence and live network checks
remain required. Live preflight accepts an additional macOS link-local /16
address and an active detached Thunderbolt bridge, while still requiring the
planned /30 addresses, unbridged training ports, correct interface routes and
all ten directed link pings. No network settings are changed by these checks.

### Original five-Mac continuation

Preview the exact checkpoint/recipe transition, without loading a model:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/scripts/launch_five_mac.py
```

Start or resume:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v1/scripts/launch_five_mac.py \
  --run --clear-stop
```

For a bounded first run, the user can add `--stop-after-steps 2`. The same
command without that option later resumes its saved latest checkpoint.
Ctrl-C requests a coordinated save and stop. Wait for all five `[finished]`
lines. Failed jobs are not automatically restarted.

The first launch reads `runs/full_15b_plateau_v1/latest.json`, step **50,096**,
**5,335,023,616** tokens. Its next LR is **0.00005**, with all scheduler
counters and AdamW state retained. The target remains **15B total tokens**.

New output is `runs/full_15b_five_mac_v1`; future invocations select that
output's own latest checkpoint. The original adaptive latest/best, STOP,
tokenizer, logs and earlier runs are preserved. The new run inherits the
validation best of 2.8575931057333945 and may have no local `best.json` until
it improves. Keep the parent run.

The new run's per-job log is `runs/full_15b_five_mac_v1/jobs/<job>/train.log`.
Existing four-Mac dashboard/playground defaults have not been redirected.

## Verification and boundaries

- Verified all five hardware/SSH identities, enabled RDMA and idle state.
- Verified ten directed link pings and interface routes around the ring.
- Applied isolated /30 training addresses; retained Ethernet, Wi-Fi and
  default routes. Original interface state and rollback are in the plan.
- Native five-rank int32/float32 all-gather, int32/float32/BF16 all-sum through
  32 MiB, repeated framed control exchanges, and unequal-batch autodiff
  gradient comparisons passed.
- Tiny BF16 model / FP32 optimizer checkpoint roundtrips passed on all ranks.
- Resume-contract tests verify that only batch distribution and explicit
  topology provenance change. Token pools produce identical global batches
  and consumed cursors when split across four versus five ranks.
- The launcher verifies the complete source bundle and saved cursor counts
  without constructing the full model in plan mode.
- The complete five-Mac 150M training path, live held-out reconstruction and
  sustained throughput remain for the user's first run. The full pipeline
  test module was excluded because it constructs the full model for a
  parameter-count test; focused migration tests use small fixtures.

The launcher requires matching five-rank correctness evidence and reruns
identity, idle, package, cable and IPv4 preflight before staging. Runtime
network settings may change after a reboot; failed preflight requires a
fresh network audit. Do not use the old four-Mac repair/launch commands for
this topology.

The native implementation was inspected from the retained source at
`7241f12e631440387a2156ec1018d1c9fe8e56b9`. Group sizes and neighbor arithmetic
are dynamic; live five-rank checks establish the tested behavior on this rack.

The first tiny checkpoint diagnostic used an order-dependent digest. The
failure was isolated to dictionary order after safetensors loading; all tensors
matched exactly. The probe now sorts tensor names and verifies metadata and
cursor contents. Production checkpoint hashing was not silently rewritten.

# Five-Mac migration progress

Updated 2026-09-22. See [the training guide](FIVE_MAC_TRAINING.md) for the
user-run resume command. Production remains stopped. The full 150M model
has not been run or benchmarked during this migration.

## Completed

- All 312 transferred files, 25,295,851,419 bytes, still match the donor SHA-256
  manifest exactly. No missing, changed, or extra files.
- Latest remains step 50,096 / 5,335,023,616 tokens. Best remains step 50,002.
  The parent adaptive run's STOP marker is intact.
- All five hardware and SSH identities were verified. The user enabled M5
  RDMA, authorized the dedicated M5 SSH public key, and renamed the machines
  Mac-1 through Mac-5 using the original physical-machine mapping.
- The user accepted the discovered cable order
  **Mac-1 → Mac-2 → Mac-3 → Mac-5 → Mac-4 → Mac-1**.
- The user applied the reviewed per-cable /30 network plan. All five
  configurations passed verification; Ethernet, Wi-Fi and default routes
  were preserved. All ten directed Thunderbolt link checks passed.
- Five-rank launch, pinned Ethernet SSH, loopback coordinator tunnels and
  native JACCL RDMA support are implemented. The legacy four-Mac launcher
  rejects five-rank recipes.
- Explicit checkpoint migration changes only batch distribution and topology
  provenance. Initial logical-rank microbatches are `[4,3,2,2,2]`, retaining
  106,496 tokens per update, context 2,048 and accumulation four. The saved
  optimizer, stream cursor and adaptive scheduler continue; next LR is 5e-5.
- Native five-rank collectives, unequal-batch gradient comparisons, and tiny
  BF16 model / FP32 optimizer checkpoint roundtrips passed on all ranks. The
  final proof matches the current worker source hashes.
- The plan-only production launcher verified the source checkpoint bytes and
  continuation contract without loading a model. New output will be
  `runs/full_15b_five_mac_v1`; no five-Mac production checkpoint exists yet.

## Audit records

Paths below are relative to `diagnostics/m5_migration_20260922/`:

- SSH authorization: `ssh_setup_20260922_114559`.
- Original names and rename rollback: `rename_20260922_115340`.
- Network plan, applied verification and rollback: `network_20260922_115651`.
- Final five-rank correctness proof: `collectives_20260922_121027/verified.json`;
  the launcher checks the copy at
  `network_20260922_115651/five_rank_correctness.json`.
- Original artifact manifest and checksum verifier: `destination_artifacts.json`
  and `verify_migration.py`.

## Remaining for the user's first run

The user starts the full model using the command in the training guide. Its
five-Mac training execution, live held-out reconstruction and sustained
throughput have not been exercised. The initial allocation is unbenchmarked.
The launcher rechecks identities, idle state, packages, cables and network
addresses before staging the run. Network settings are runtime-only; a reboot
requires rechecking and possibly reapplying them.

The original checkpoints, tokenizer, logs and earlier runs are retained.
Existing dashboard/playground defaults still reference the earlier run.

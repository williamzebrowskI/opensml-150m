# Five-Mac migration handoff

Prepared 2026-09-22 on the original M3 Ultra coordinator. This document and
the read-only inventory script were copied directly to the new M5 checkout.
They are not yet committed or pushed to Git. Read this before changing the
cluster. No five-rank training launch has been validated.

## User intent and safety boundaries

Move the coordinator and daily workstation to the new M5 Ultra, retain the
M3 Ultra and all three M4 Max machines, and use a five-cable Thunderbolt RDMA
ring. The user will wire the machines and start production training after
validation. Keep production stopped; do not auto-start or auto-restart it.
Use separate disposable outputs for tests and preserve all migration sources.

No hostnames, routes, Thunderbolt settings, SSH identities, rank mappings, or
production runtime code were changed by the artifact transfer or inventory.
Logical names below are the proposed names, not current macOS hostnames.

## Machine identity

The old Mac numbers mean the physical identities in the original project's
four-Mac configuration. They do NOT reliably match current macOS hostnames.
Use the Ethernet identity and pinned SSH fingerprint to disambiguate.
All addresses below were read live on September 22; DHCP can change them.

| New label / rank | Physical machine | Former project label | Ethernet IPv4 | Hardware Ethernet MAC |
| --- | --- | --- | --- | --- |
| Mac-1 / 0 | New M5 Ultra | Not previously in cluster | 192.168.12.210 | 7c:d6:2c:00:09:df |
| Mac-2 / 1 | M3 Ultra, original coordinator | Mac-1 | 192.168.12.117 | 1c:1d:d3:dd:fa:84 |
| Mac-3 / 2 | M4 Max | Mac-2 | 192.168.12.102 | 1c:1d:d3:db:7b:63 |
| Mac-4 / 3 | M4 Max | Mac-3 | 192.168.12.167 | 1c:1d:d3:d8:19:b6 |
| Mac-5 / 4 | M4 Max | Mac-4 | 192.168.12.193 | 1c:1d:d3:d7:b4:da |

New M5 SSH: `williamzebrowski@williamacstudio.lan`.
All five login accounts are `williamzebrowski`.
Hardware MACs above are from `networksetup -listallhardwareports`. On the M5,
`ifconfig en0` instead reported `02:00:00:00:00:00`; retain that discrepancy
in the audit and do not use this runtime value as a unique identity. SSH
fingerprints and independent hardware evidence must also match.

Verified ED25519 host-key fingerprints, obtained over the existing trusted
connections (do not blindly accept replacement keys):

```text
Mac-1 SHA256:ixB3h7qjaTrCzEn9p0Va1m2tLFb4T37jLVYc9c6yCFk
Mac-2 SHA256:v45WFZM77qEGF9ZuVXV2Fj7llE/XdXUOqTwDCVDeAhE
Mac-3 SHA256:oLLqAAJ679URqDBNzcrF7ebhPAduCHlIRa7lmlcOM2I
Mac-4 SHA256:8wdYLC64yOk+4wt9YQIf5U8eH+sUH2c0hKFBF4+1Z1s
Mac-5 SHA256:tty3pv5axM9NMBgu1Wz5BablZBwv/XUHffi+UqcdkPY
```

## Physical cable plan

Face the BACK of each Mac, looking directly at its sockets. Define two
physical labels consistently:

- **A:** the leftmost rear Thunderbolt USB-C socket.
- **B:** the next rear Thunderbolt USB-C socket to its right.

These are chosen socket positions, NOT `enX` names, RDMA device numbers, or
Apple's numbered controller labels. Use Thunderbolt-capable rear sockets and
Thunderbolt cables, not generic USB-C charging cables. Do not use USB-A, HDMI,
or front sockets for this plan. If a selected socket is occupied by a display
or another essential peripheral, resolve that deliberately before rewiring.
Unused sockets and gaps are fine. Nothing needs a terminator or cover.

| Cable label | Endpoint 1 | Endpoint 2 |
| --- | --- | --- |
| 1-2 | Mac-1 B | Mac-2 A |
| 2-3 | Mac-2 B | Mac-3 A |
| 3-4 | Mac-3 B | Mac-4 A |
| 4-5 | Mac-4 B | Mac-5 A |
| 5-1 | Mac-5 B | Mac-1 A |

This is `1 -> 2 -> 3 -> 4 -> 5 -> 1`: five cables, two cluster connections
per Mac. The arrows specify order, not one-way communication; every cable is
bidirectional. Remove obsolete Mac-to-Mac Thunderbolt links so they do not
create extra edges. Keep all Ethernet connections and the switch connected.
Only change the cluster Thunderbolt cables after confirming workers are idle.

Do not assign old `en4/en5` mappings to these sockets. After wiring, collect
fresh inventories and match reciprocal Thunderbolt peer identities to learn
the actual `enX` / `rdma_enX` mapping. Before wiring, the M5 enumerated its
six Thunderbolt interfaces as `en2,en13,en3,en4,en5,en6`; this alone does not
prove which physical socket is A or B.

## Immediate M5 prerequisite: RDMA

Read-only checks found RDMA **disabled on the M5** and **enabled on the four
existing Macs**. The M5 also had an UP default `bridge0` containing
`en2,en13,en3,en4,en5,en6`. Do not run the old four-Mac bridge repair script
to fix it. The new agent must audit and prepare an explicit five-node plan.
The fresh inventories also found UP `bridge0` on new Mac-4 and Mac-5;
new Mac-3's bridge was down and new Mac-2 had none. Re-audit after wiring or
reboot; do not assume old network repairs are still in effect.

The user must enable RDMA on the M5 locally through macOS Recovery. This
cannot be enabled through an ordinary remote SSH session, even with sudo.
Shut down the M5, hold its power button until startup options appear, choose
Options, enter Recovery, and open Utilities > Terminal. Run:

```bash
rdma_ctl enable
```

Then restart normally. In normal macOS, verify:

```bash
/usr/bin/rdma_ctl status
ibv_devices
```

Do not change SIP, FileVault, or boot security policy as part of this task.
Source: [MLX RDMA setup instructions](https://ml-explore.github.io/mlx/build/html/usage/distributed.html#enabling-rdma).
Those published instructions also describe the ordinary mesh backend; our
installed custom build and repository contain the separate JACCL ring path.
Do not replace it with the TCP-only `ring` backend by accident.

## Artifacts and environment already on M5

Repository: `<SOURCE_WORKSPACE>`.
Git base: `91eb73b742b088b495a94e14437e538dbdb32ca4` on `main`.
Private remote: `https://github.com/williamzebrowskI/sml-mlx.git`.

The complete retained V2 `tokenizer/` and `runs/` trees were transferred:
312 files, 25,295,851,419 bytes. SHA-256 checks found no missing, changed, or
extra files. These include weights, FP32 optimizer, stream cursors, scheduler
state, best/latest pointers, run logs, and staged job provenance. Originals
remain on the M3 Ultra. No dataset download or credential copy was performed.

The M5 has a recreated CPython 3.11.13 arm64 environment at root `.venv`,
with `sml-mlx-v1/.venv -> ../.venv`. All 105 package versions match the source.
MLX and mlx-metal are **0.32.3.dev20260920+7241f12**, the existing native
all-gather fix build, not stable 0.32.2. Verified wheels are retained in
`/Users/williamzebrowski/.cache/opensml/mlx-pr4443/wheels/`.
Do not run a blind `uv sync` that replaces this custom build with stable pins.

Already passed on M5:

- 75 offline V2 tests and dependency compatibility checks.
- Latest and best checkpoint manifest and model/optimizer digest checks.
- Tokenizer identity, stream cursor accounting, scheduler restoration, and
  unchanged resume-contract checks.
- Tiny GPU packed-Metal FFN / Metal CE forward/backward and synthetic update.
- Real latest-model GPU forward at 2048 tokens and short generation.

None of those proves five-node RDMA operation. No distributed or production
optimizer updates were run on the M5.

Full audit, verification scripts, environment lock, and before-wiring
inventories for all five Macs are under:

```text
<SOURCE_WORKSPACE>/sml-mlx-v2/diagnostics/m5_migration_20260922/
```

See that directory's `README.md`, `artifact_manifest.json`,
`runtime_validation.json`, `requirements-source.txt`, and
`before_wiring_new_mac1.json` through `before_wiring_new_mac5.json`.
The inventories found no matching training/launcher processes on any Mac.

No private SSH keys, authentication tokens, or trusted-host files were copied.
Working old-coordinator -> M5 SSH does NOT establish M5 -> peers SSH.
Provision the M5's own outgoing key and verify peer fingerprints over Ethernet;
the user may need to authorize the public key interactively. Check Hugging
Face access without printing tokens or copying the old Mac's credentials.

## Exact stopped training state

Run to continue:
`<SOURCE_WORKSPACE>/sml-mlx-v2/runs/full_15b_plateau_v1`.

| Item | Saved value |
| --- | --- |
| Latest | `step_0050096_92102f67c5ca` |
| Latest step / tokens | 50,096 / 5,335,023,616 |
| Best | `step_0050002_666b002c4e31` |
| Best step / tokens | 50,002 / 5,325,012,992 |
| Best validation loss | 2.8575931057333945 |
| Effective learning rate | 0.00005 |
| Absolute total token target | 15,000,000,000, not 15B additional |
| Previous batches / accumulation / context | 4,3,3,3 / 4 / 2048 |
| Previous tokens per optimizer update | 106,496 |

The STOP marker is preserved. All four original workers logged finished
after the latest save. Resume the latest full adaptive run, not a fresh model,
an earlier best, the 500M initial run, or an SFT checkpoint.

Preserve the 32K tokenizer, model, BF16 compute weights, FP32 master weights
and optimizer states, FP32 gradient path, AdamW settings,
HF streaming corpus, saved consumed cursor (not prefetched-ahead position),
validation fingerprint, and plateau/cosine scheduler state. Do not rewarm or
reset scheduler counters merely because rank 0 moves.

## Commands safe to run before configuration

After the user enables RDMA and wires the ring, run this on the M5. It only
collects local evidence and writes a new report; `--label` does not rename it.

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/scripts/inspect_mac_network.py \
  --label new-mac-1-m5 \
  --output <SOURCE_WORKSPACE>/sml-mlx-v2/diagnostics/m5_migration_20260922/after_wiring_new_mac1.json
```

The script refuses overwriting an existing report; use another filename for
a repeat, or `--stdout` instead of `--output PATH`. Copy this public script
to peers after outgoing SSH is configured and collect each with the proper
new label. A successful inventory exit is NOT a cable, RDMA, or readiness
pass: inspect every probe's return code and content.

Offline local tests, already passed before migration changes:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python -m unittest discover \
  -s <SOURCE_WORKSPACE>/sml-mlx-v2/tests -v
```

Artifact verification can be repeated before any intentional run changes:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/diagnostics/m5_migration_20260922/verify_migration.py verify
```

This recomputes checksums and refreshes its audit result, not checkpoints.
Do not run that verifier's `snapshot` action on M5: retain the donor manifest
as the reference. None of these commands starts production training.

## Engineering still required on M5

There is no validated five-Mac production command yet. Do not just append a
host or change four to five everywhere. In particular:

- `sml_v2/recipe.py` currently accepts only one or four ranks and requires
  equal peer microbatches in its four-rank case.
- `sml_v2/launch.py` and `sml_v2/pretrain.py` use `len(batches) == 4` to choose
  the distributed path. A fifth batch must never silently fall into local mode.
- V1 `benchmark_jaccl_ring.py`, `check_ring_thunderbolt.py`,
  `configure_rdma_mesh.py`, `configure_ring_ipv4.py`, and related repair
  scripts are pinned to four original identities and interfaces.
- The root `scripts/`, `cluster/`, and `train/` paths are V1 symlinks, not new
  generalized five-node tooling. The old ring is 1-2-4-3-1 by old labels.

Required sequence:

1. Read this handoff and the migration audit. Recheck idle state and saved
   STOP/latest/best before touching anything. Keep original artifacts intact.
2. Establish M5 outgoing SSH over Ethernet with the M5's own key and verified
   host identities. Do not disable host-key checks or use old Wi-Fi IPs.
3. Confirm user-completed RDMA enablement and fresh reciprocal physical
   connectivity: five unique Macs, five cables, degree two, one closed ring.
   Map actual interfaces and RDMA devices; never infer them from A/B labels.
4. Prepare a dry-run, identity-pinned five-node network plan with audit and
   rollback. Address Thunderbolt bridges, per-link non-overlapping IPv4
   subnets, stale aliases/routes, and coordinator reachability deliberately.
   Preserve Ethernet/default routes, remote access, and the user's Wi-Fi
   settings. Arrange explicit user approval/password entry for changes.
5. Keep the intended control plane wired. Prefer Ethernet for initial
   bootstrap; if retaining Thunderbolt-only SSH/coordinator control, design
   and test the necessary hops/tunnels for the new ring. No silent Wi-Fi
   fallback, no blanket IP forwarding, no open coordinator listener unless
   explicitly reviewed. Native all-gather is independent of SSH reachability.
6. Generalize rank count, staging, launch, stop/cleanup, monitoring, identity
   checks, and hostfile construction together. Inspect the installed custom
   MLX ring implementation rather than assuming odd world size five works.
7. Implement an explicit, reviewed topology/batch resume transition. Existing
   code/recipe fingerprints reject unexpected changes by design; do not remove
   the guards or edit historical checkpoint metadata to fake compatibility.
   Preserve model/optimizer/tokenizer/data/scheduler/validation identities.
8. Add tests for five ranks, disjoint batch slicing, token-weighted unequal
   microbatches, consumed-cursor continuity, and restart/stop behavior. Prefer
   retaining 106,496 tokens/update for the first comparison; choose a five-way
   allocation by measurement rather than scaling LR or assuming chip speed.
9. Run bounded five-rank collective correctness tests with explicit timeouts:
   native all-gather and all-sum, relevant dtypes/sizes, repeated payloads,
   numerical weighted-gradient checks, and synchronized checkpoint/resume.
   Ensure RDMA ring is actually selected and no rank silently falls back.
10. Only after correctness, benchmark isolated scratch runs at the same
    context/precision/effective batch. Never write benchmark updates into
    production. Record per-rank timing/memory and end-to-end throughput.
11. Leave production stopped and give the user one verified resume command
    from the retained full adaptive latest checkpoint. Do not claim an
    expected speedup before measurement or start the run on their behalf.

The four-Mac setup remains the rollback reference, not a topology to apply
blindly after the user has moved the cables.

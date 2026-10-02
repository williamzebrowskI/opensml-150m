# Verified MLX native ring build

On 2026-09-20, all four Mac Studios' shared project `.venv` environments were
upgraded from MLX / mlx-metal 0.32.2 to **0.32.3.dev10260920+7241f12**.
This is a locally built development snapshot, not a published stable release.

## Provenance

- Source: https://github.com/ml-explore/mlx/commit/7241f12e631440387a2156ec1018d1c9fe8e56b9
- Includes the native JACCL ring `all_gather` fix: https://github.com/ml-explore/mlx/pull/4443
- Built the complete snapshot at that commit, not an isolated backport.
- Build host: Mac 1, CPython 3.11.13, arm64, macOS 26.6.2.
- Xcode 26.6 (17F113), SDK 26.5; Apple Metal toolchain 17F109.
- `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` was scoped to build commands.
- `MACOSX_DEPLOYMENT_TARGET=26.2`, `CMAKE_ARGS=-DCMAKE_OSX_DEPLOYMENT_TARGET=26.2`,
  and `CMAKE_BUILD_PARALLEL_LEVEL=8`.
- Built backend first (`MLX_BUILD_BACKEND_PACKAGE=1`), then frontend
  (`MLX_BUILD_FRONTEND_PACKAGE=1`) using `uv build --wheel`.

Wheel SHA-256 values:

- `mlx_metal-0.32.3.dev10260920+7241f12-py3-none-macosx_26_0_arm64.whl`: `32616c4c1089d86e33e9a8be39f8e1bb2f601e58e04c01c5c8b31d37bebc22a1`
- `mlx-0.32.3.dev10260920+7241f12-cp311-cp311-macosx_26_0_arm64.whl`: `81f93bfc15543cb561d0607da7c8543e85d8092e9a877da60323bcc88f866741`

## Verification

Before installation, identical wheels were tested in isolated environments on
all four Macs. After installation, the checks passed again in the actual shared
training environments:

- Four identities, wired SSH routes, ring cabling and RDMA preflight.
- Loopback-only coordinator tunnels, including Mac 4 through Mac 2.
- Tiny v1 packed-Metal FFN / Metal CE forward and backward pass on each GPU;
  finite gradients and loss 6.221843 (also matched the old build on Mac 1).
- Rank-distinct int32 and float32 native gathers, sizes 1/17/4096/262147,
  two rounds each.
- 100 framed native-control exchanges on every rank.
- Correct float32 all-sum results for 17/8193/8388608 elements (up to 32 MiB).

These are correctness checks, not an end-to-end throughput benchmark or a
long-run numerical equivalence claim. Gradient reduction still uses `all_sum`.

## Runtime override and recovery

The public `pyproject.toml` pins remain the reproducible **stable 0.32.2 baseline**.
The rack currently uses this explicitly installed local build instead. Installing
those baseline requirements again will downgrade MLX and remove this fix.
Do not synchronize or modify shared packages while workers are running.

On each Mac, build wheels and original 0.32.2 rollback wheels are retained under
`~/.cache/opensml/mlx-pr4443/wheels/` and `~/.cache/opensml/mlx-pr4443/rollback/`.
With training and MLX services stopped, reinstall the verified custom pair with:

```bash
~/.local/bin/uv pip install --python <SOURCE_WORKSPACE>/.venv/bin/python \
  --no-deps ~/.cache/opensml/mlx-pr4443/wheels/*.whl
```

To roll back, use `rollback/*.whl` instead on **all four Macs**, and resume with
`--ring-control all-sum`. Do not mix package versions across ranks.

Mac 1 retains `source.json`, `wheel_hashes.json`, `installation.json`, build logs,
`rack-validation.log`, and `installed-validation.log` in the same cache directory.

## Resume

Training was left stopped at **step 31,463 / 3,350,683,648 tokens**;
`runs/full_15b_v1/latest.json` still selects `step_0031463_35d2a3f089d3`.
The STOP marker remains in place. From the repository root on Mac 1:

```bash
.venv/bin/python sml-mlx-v1/scripts/launch_full.py \
  --run --clear-stop --ring-control native
```

This resumes the existing 15B-token run. It does not initialize a fresh model.
Each native-mode launch performs its own correctness probe before training.
The playground server was restarted after the shared-runtime upgrade.

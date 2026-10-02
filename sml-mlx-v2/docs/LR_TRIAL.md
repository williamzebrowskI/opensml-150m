# Bounded learning-rate comparison

Prepared 2026-09-23. Production is stopped; the user starts either arm manually.

The completed shuffled run stopped normally at step 73,060. Both arms start from
a verified, independent filesystem copy of its best evaluated bundle, step
73,008 (7,775,059,968 tokens), rather than the unevaluated final weights. Parent
validation losses are 2.767114285717253 expanded and 2.818589221686125 legacy.

The frozen bundle is
`experiments/lr_trial_v1/parent/step_0073008_58d174e2c177`.
It preserves weights, FP32 optimizer/master weights, token counters, pending
shuffle documents, data cursor, tokenizer identity, and both validation histories.
The original run remains intact. The frozen copy stays available independently
of its original run's checkpoint retention.

## Higher-rate arm

Run on the M5:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/scripts/launch_five_mac.py \
  --lr-trial rewarm --run --clear-stop
```

- Output: `runs/full_15b_five_mac_lr_rewarm_v1`.
- Linear ramp from 5e-6 to 1e-5 across the first 100 updates.
- The first actual update uses 5.05e-6; update 100 reaches 1e-5.
- Afterward, the existing plateau controller may lower the cap, down to 5e-6.
  Its validation best, patience/cooldown state, and lifetime counters are retained;
  only its cap is explicitly raised when creating this arm.
- Stop after 1,000 total trial updates, at absolute step **74,008** and
  **7,881,555,968 tokens** (106,496,000 additional tokens).
- Evaluate both unchanged validation sets at the endpoint and save before exiting.

The same command resumes an interrupted arm from its own latest checkpoint. The
ramp and absolute update budget do not restart. Repeating it after completion
refuses additional training. A restored cap that has already been reduced stays
reduced. The parent is saved as the arm's initial best, so an unsuccessful trial
cannot replace its starting best with a worse evaluated checkpoint.

## Matched control

After the rewarm arm has stopped and all five workers are idle:

```bash
<SOURCE_WORKSPACE>/.venv/bin/python \
  <SOURCE_WORKSPACE>/sml-mlx-v2/scripts/launch_five_mac.py \
  --lr-trial control --run --clear-stop
```

This uses `runs/full_15b_five_mac_lr_control_v1`, the identical frozen parent and
data cursor, 5e-6 without a ramp, and the same 1,000-update budget. It is not
automatically launched. Run the arms sequentially; both use the five-Mac cluster.

Compare their endpoint losses at equal consumed-token counts, their per-source
losses, and their best checkpoints. A temporary loss increase after rewarming is
possible. A higher rate is an experiment, not an established improvement.

## Unchanged configuration and checks

The model, tokenizer, 256-document shuffle buffers, corpus mixture, 2,048 context,
microbatches [4,3,2,2,2], accumulation 4, optimizer settings, LR floor, and both
validation fingerprints stay unchanged. No SFT data or code is involved.

The original 15B-token target remains in the inherited recipe; the trial's
additional absolute stop is enforced in the worker and reported in the plan.
An optional earlier `--stop-after-steps` still permits a shorter interruption,
but cannot extend the trial beyond its absolute limit.

Omit `--run` for a read-only plan that loads no model. The launcher checks the
current native five-rank correctness evidence, verifies the frozen checkpoint,
and stages/verifies the updated worker code and inputs on all five machines
before training. No manual copying to the peers is required.

Validation evidence is in `diagnostics/lr_trial_20260923/`: scheduler/transition
regressions, a tiny one-layer CPU test of weights/optimizer/cursor equivalence
across restart and endpoint evaluation, and native five-rank RDMA/gradient/tiny
checkpoint checks. No production-model training was started during preparation.

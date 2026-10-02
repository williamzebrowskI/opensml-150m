Repair512 autonomous short experiment search
===========================================
Authorized by the user on 2026-10-01: run multiple 128/256-update experiments,
starting from repair512, evaluate each, and delete unsuccessful experiment
checkpoints and artifacts before starting the next candidate.

Protected parents: repair512, unified384, original768, pretrainedV1.
All runs are independent continuations of repair512, never chained failures.
No benchmark answer or held-out answer enters gradients or rollout collection.
Early trials used 50% previously consumed and 50% unused public conversations.
Later trials replayed consumed parent conversations, with separate reasoning
questions from official training splits. No failed candidate is used as a source.
Whole original reference answers, unchanged role format, assistant masks and EOS.

Candidate001: conversation-balanced CE plus 0.1 repeated-span unlikelihood;
128 updates, warmup16, LR2e-6 ->2e-7, batch16. Negatives come from frozen
repair512 greedy continuations of TRAIN prompts. This is offline sequence
unlikelihood, not on-policy RL. Only later repeated 4-token spans are penalized;
EOS and prompt/history are never negative labels. Original public instruction
selection excluded deliberately repetitive targets. This is a tested local
adaptation of the earlier repetition objective, not a paper reproduction.

Baseline must reproduce all saved repair512 development metrics exactly.
Evaluate with existing legacy retention and public-development prompts, limits
and greedy decoder. Formal instruction and lexical content checks are proxies.
Full public benchmarks use the unchanged MC and official IFEval protocol.

A useful overall candidate must improve generation and benchmark scores, with
no material loss of ordinary follow-up answers. Public benchmark selection
across repeated candidates is exploratory and cannot establish a statistically
independent final result. Report all evaluated candidates and avoid selecting
on lower training or validation loss alone.

Development screen: public repetition <=parent, legacy repetition no more
than one additional repeated turn, follow-up joint within one conversation,
public formal constraint passes >=parent, public EOS within two turns,
chat repetition no worse than parent. Screened candidates receive full public
benchmarks; benchmark results determine retention, not the screen alone.

Benchmark retention gate: strict AND loose prompt AND instruction accuracy at
least parent, MC acc_norm macro mean higher, no individual MC acc_norm decrease
larger than 0.5 percentage points, and generation screen passed. Also inspect
loose instruction scores and saved outputs. Small changes must be described
as exploratory, not demonstrated statistically reliable superiority.

Unsuccessful candidate directories are deleted in full. A compact configuration
and score/cleanup receipt remains under comparisons so the user can see what
was tried; no failed weights, optimizer, data, rollouts, or raw responses remain.
The controller script remains as reusable experiment infrastructure.

Executed variants
-----------------
001: offline repetition UL0.1 + conversation CE; LR2e-6;128updates.
002: DPO0.1 preferring clean references over repetitive parent rollouts + CE;
     beta0.1, LR2e-6;128updates. Reference sequence log probabilities frozen.
003: consumed-parent replay + character-normalized reasoning ranking0.2 +
     frozen-parent forward KL0.2; LR1e-6;128updates.
004: live on-policy repetition UL0.3 + consumed-parent replay + KL0.5;
     LR3e-6;128updates. Passed dev screen; failed broad benchmark gate.
005: live selective reference-aware UL0.3 + ranking0.3 + replay + KL0.5;
     LR2e-6;256updates. Two passes through2048 consumed parent conversations;
    1024 unique public TRAIN reasoning questions. Negative labels mark only
    third/later occurrences beyond the repetition count in the clean reference.
    This rule preserves examples requesting repetition present in their target.
All failed candidate folders are deleted before the next trial is prepared.

006: same selective UL/ranking recipe as005, with LR1e-6;256updates.
     Failed legacy/chat repetition development checks; removed.
007: same as005 with instruction CE weight2 (40% of CE);256updates.
     Better chat repetition, but lost a public formal pass; removed.
008: same as005 with frozen-parent KL2 instead of0.5;256updates.
     Passed development screen. All four normalized MC accuracies improved;
     IFEval lost six strict and seven loose instruction passes; removed.

Stopped after trial008 at the user's explicit request. Eight trials completed.
No all-around candidate passed the unchanged development/benchmark gates.
All candidate directories, weights, optimizer files, raw responses and data
were removed. Protected repair512, unified384, old768 and pretrainedV1 weights
were verified unchanged. Compact score/cleanup receipts and reusable scripts
remain. The full score history is in results.json.


Search resumed at the user's request.
009: top four transformer blocks only; 128 updates, LR8e-6; failed chat repetition.
010: top four blocks; 256 updates with stronger selective repetition penalty;
     failed chat repetition. Both removed; frozen parameters verified unchanged.
011: balanced, complete public instruction targets up to768 tokens; equal named
     family exposure, first/last16 target tokens weighted4; full model256 updates,
     LR3e-6, selective UL0.3, ranking0.3, KL2. Passed development and increased all
     four normalized MC scores. Lost two strict and two loose IFEval instruction
     passes. Removed with protected parents unchanged.
012: same recipe plus TRAIN-only correct/incorrect formatting pairs, preserving
     all answer words. Two pairs/update: requested-format pair and plain-control
     pair. Reference-relative character-normalized pairwise logistic, beta5,
     weight0.1; this is a custom contrast objective, not standard sequence-sum DPO.
     Named formatting checks verify chosen and rejected; public source factual
     content is not independently verified. Selection order is now deterministic.

012 completed. Passed development screen and improved all four normalized MC
scores. IFEval declined in all four metrics, so it was not the best overall and
was removed under the user's failed-trial cleanup rule. Compact paired family
changes are retained in its comparison receipt.
013: same full-model supervised/UL/ranking/KL recipe as011, with complete public
word-count targets up to400 requested words (previous ceiling120). Original
conservative ambiguity checks remain, targets pass all named verifiers, no
truncation, no format-preference objective. 256 updates independently from512.

013 completed. Development passed and normalized MC average increased by1.70pp,
while IFEval strict instructions lost8 passes and loose instructions lost7.
Removed before014, all protected parents unchanged.
014: identical selected training data and objective to013 plus forward KL4 on
frozen repair512 greedy response prefixes (up to64 exact generated tokens), one
chat and one instruction training prompt per update. Existing reference KL2 is
retained. No benchmark responses are used. First-step policy KL must be zero;
input/EOS boundaries are checked and the anchor has no trainable parameters.

014: development repetition21/73 exceeded parent20/73; other development checks
passed, including restored sentence and no-comma counts. Removed before015.
015: consumed-parent replay only, CEweight0.25, reasoning ranking0.3, reference
KL2, parent greedy-prefix KL8, selective repetition UL0.1, LR2e-6,256 updates.
No new instruction templates. Tests preserving generative behavior while
concentrating improvement on TRAIN-only reasoning likelihood ranking.

015: public repetition23/73 and follow-up joint20/32 failed the unchanged screen.
Removed before016; no public benchmarks run.
016: same anchored reasoning objective as015, but only the final normalization
layer is trainable; embeddings and all transformer blocks are exactly frozen.
256 updates, LR1e-4. Tests limited feature rescaling rather than changing learned
transformer representations. Frozen parameters checked after first and last
update. Original source bundles remain protected throughout.

Trial 16 removed: all four normalized MC scores exactly unchanged; one extra raw HellaSwag correct. Remaining IFEval stopped after decisive MC failure. Trial 17: 256 independent updates from repair512, all RMS normalization weights only, exact frozen-matrix checks, LR 5e-5, replay CE .25, selective UL .1, MC training ranking .3, reference KL 2, generated-policy KL 8.

Trial 17 removed before benchmarks: public development repetition 21/73 versus parent 20/73; all other declared development gates passed. Exact frozen matrices preserved. Trial 18 independently restarts the same 256-update normalization-only recipe with peak LR 2e-5 (40% of trial 17).

Trial 18 removed: public repetition 21/73 versus 20/73; paired review identified an additional third-turn roleplay echo loop. Trial 19: 256 updates from repair512, final transformer block and final norm only, LR 1e-6, replay CE .25, selective UL .15, MC train ranking .3, reference KL 2, policy KL 8. Live negatives and policy anchors now use the corresponding model's generated prior replies (64-token prior-turn cap), rather than reference assistant history. Development and benchmark answers excluded from training.

Trial 19 removed: all development gates pass, MC normalized macro mean -0.0055694 pp, no IFEval run. Trial 20: 128 independent updates from repair512, final block+final norm, LR 8e-6, CE .25, UL .3, ranking .3, ref KL 2, generated-history policy KL 8. Reference-aware UL now penalizes second/later repeated eight-token spans beyond clean reference occurrence counts; requested repetition preserved. Prior model-generated turns capped at 64, current negative at 192, policy anchor at 64. Native gradient and requested-repeat checks run before first update.

Trial 20 removed: normalized MC macro +0.0226557 pp, strict IFEval prompts 75/541 (-3), instructions 201/834 (-5), loose 77/541 (-4),203/834 (-6). Paired family breakdown saved, sentence count net -3, title -1, no-comma -1, forbidden word -1, word-count +1. Trial 21: 256 independent updates from repair512, all transformer blocks but frozen tied input/output embeddings; LR 1e-6; ordinary public replay CE 1, ranking .3, reference KL 2, generated-history policy KL 8; no unlikelihood or pair-preference loss. Exact embedding check first and final update.

Trial 21 redirected by user: investigate repair512 failures and change training data. The frozen-vocabulary method trial was cancelled at 55 updates and removed. 021-data-focused uses 3072 fresh public TRAIN examples plus 1024 parent replay, 256 ordinary CE+EOS updates, LR2e-6. Fresh public constraints, unused SQuAD TRAIN passages and SciQ TRAIN labels; benchmark overlap excluded, no generated teacher answers. Fresh dev96/test96 reserved independently.

Trial 21 complete and removed: 256 ordinary SFT updates on 1536 SmolTalk constraints, 1024 unused SQuAD TRAIN passages, 512 SciQ TRAIN labels, 1024 repair512 replay. Full MC15428 and IFEval541 complete. All public benchmark gates passed (normalized MC macro +0.5179pp; strict IFEval78 to82 prompts and206 to211 instructions). Retention failed: public repetition20 to24/73, chat repetition3 to6/18, follow-upjoint22 to21/32. Largest instruction-family gain highlighted sections6 to18; paragraph checks0 unchanged. Parent protected hashes verified before/after cleanup. Detailed receipt comparisons/021-data-focused.json.

Current baseline (user selected, 2026-10-01): Repair512 +256, trial28.
Checkpoint: <SOURCE_WORKSPACE>/sml-mlx-v1/experiments/repair512_search_v1/candidates/028-broad-public-long-stable/runs/sft/step_0000768_4980617cb301
Total step768 is distinct from the earlier Text Follow-up768 branch. The full trial completed512 additional updates. Only its +256 checkpoint is retained; +128, +384 and +512 bundles were removed. Weights, optimizer state, provenance, evaluations and full benchmark results are preserved. current_baseline.json and retained_baseline.json identify the current baseline; baseline.json remains the historical Repair512 comparison reference.

# Pre-registration: v13, a Qwen3.5-2B-Base torso

Committed before training. Same discipline as v9-v12.

## Why

Read without any training (SemIf's frozen readout), the torso generation moved JevBench
more than anything trained here:

| frozen, read SemIf's way | JevBench |
| --- | --- |
| Qwen3-1.7B | 0.558 |
| **Qwen3.5-2B** | **0.658** |
| Qwen3-8B | 0.684 |
| Qwen3.5-4B | 0.805 |

At matched size, Qwen3.5 is +0.10 over Qwen3. Training v7's recipe on Qwen3 added about
+0.11 over its frozen reading (0.558 -> 0.667). If training adds something similar on
Qwen3.5-2B, a trained torso lands well above v7. That extrapolation is the hypothesis
under test; v8 showed fine-tuning can also erode what a model already does well.

Frozen Qwen3.5-2B and v7 get different tasks right: 33 gained against 35 lost, strongest
on `ambiguous`, `temporal_numeric`, `trap` and `judge_hard`, weakest on `policy`,
`intent`, `multi_hop` and `long_policy`.

v12's teacher did not pass its pre-registration, so v13 uses v7's KL anchor to the
frozen torso, not the teacher.

## What is being changed

One thing against v7: the torso, `Qwen/Qwen3.5-2B-Base` (revision `b1485b2`, 1.90B
parameters, Apache-2.0) in place of `Qwen3-1.7B-Base`. Base to base, as v7. Same corpus
(`train_v5.jsonl`), pointer readout (dim 256), KL anchor 0.3, 3072-token window, one
epoch, length grouping off. Config: `configs/experiments/v13.yaml`.

What the torso change carries with it, necessarily:

- **LoRA targets** include the Gated DeltaNet projections (`in_proj_qkv`, `in_proj_z`,
  `in_proj_a`, `in_proj_b`, `out_proj`); otherwise LoRA reaches 6 of 24 layers. Verified:
  all 24 layers carry LoRA, 0.94% of parameters trainable (v7: 1.06%).
- **Environment**: trained under WSL2 (Ubuntu 24.04, torch 2.7.1, Triton 3.3.1,
  `flash-linear-attention` 0.5.2), because the fused linear-attention kernels do not run
  on Windows. Fused and reference kernels agree to cosine 0.9999 on hidden states.
  `causal_conv1d` uses its PyTorch path (a depthwise convolution).
- **Serving** uses batched encoding, not the shared-prefix cache, which cannot yet
  broadcast recurrent state (`HobsonModel.is_hybrid`). Exact, but latency is not
  comparable to v7's.

Measured: 0.26 steps/s alone on the GPU (v7 0.28-0.34), 5.6 GiB peak, ~3.2 h.

## Baselines (v7)

JevBench 154/231, ECE 0.068, macro 0.660. Held-out tasks, real questions: 0.653.

## Predictions

1. **JevBench at least 162/231 (0.701)**, eight tasks over v7.
2. **Held-out real-question accuracy at least 0.633** (v7 − 0.02).
3. **JevBench ECE at most 0.08.**

Exploratory, no numeric prediction: the four families where frozen Qwen3.5-2B beats v7
(`ambiguous`, `temporal_numeric`, `trap`, `judge_hard`; v7 15/47), and question
sensitivity on `tests/question_sensitivity.py`.

## What would count as failure

- **1 fails and v13 lands near frozen Qwen3.5-2B (about 0.66)**: training adds nothing on
  this torso — the extrapolation was wrong.
- **1 fails and v13 lands below v7**: fine-tuning erodes what the torso brings, as
  instruction tuning's gains eroded in v8.
- **2 fails**: the new torso generalises worse to unseen tasks under this recipe.

## If it passes

v13 would replace v7 only after a second seed confirms it (every run since v7 has
scored below v7's 154; training noise is unmeasured), and after the shared-prefix cache
handles recurrent state, so serving keeps its many-questions-per-state property.

## Power

231 tasks cannot resolve an overall difference below about +0.043; prediction 1 is
therefore suggestive rather than conclusive.

## Outcome (added after the run)

Nothing above this section was edited after training. **v13 does not qualify; v7
stays** — but it is the first model since v7 to score above it, and the result is best
read as a tie.

| | prediction | v7 | v13 | |
| --- | --- | --- | --- | --- |
| 1 | JevBench >= 162/231 | 154 | **157** (0.6797) | FAIL |
| 2 | held-out real-question accuracy >= 0.633 | 0.653 | 0.646 | pass |
| 3 | JevBench ECE <= 0.08 | 0.068 | 0.085 | FAIL, narrowly |

Against v7: 17 gained, 14 lost, McNemar p = 0.72 — two different models with the same
score, not a better one. Macro accuracy 0.685 is the highest any model here has
recorded (v7 0.660); held-out ECE 0.056 (v7 0.077); ordinal MAE worse (0.621 vs 0.592).

**The named failure, softened: training added little on this torso.** Frozen Qwen3.5-2B
reads 152/231; trained, 157 — +5 tasks (27 gained, 22 lost vs frozen). On Qwen3 the same
recipe added about +25 over the frozen reading. The extrapolation behind this
pre-registration did not hold.

The four exploratory families, where frozen Qwen3.5-2B beats v7, went from 15/47 to
20/47: `ambiguous` 1 -> 4 and `trap` 5 -> 7, with `temporal_numeric` (2) and `judge_hard`
(7) unchanged — so the frozen torso's lead on those two did not survive training.
Outside them, `probability` 3 -> 6 and `adversarial` 4 -> 5 (where the frozen torso is
no better than v7), while `tradeoff` fell 4 -> 1 and `long_policy` 8 -> 6. The question-sensitivity probe is unchanged from v7 (same answer to
a changed question 0.96 of the time on held-out choice tasks): that property comes from
the corpus, not the torso.

**Latency** p50 0.106 s / p95 0.291 s against v7's 0.246 / 0.583 — faster even without
the shared-prefix cache, but measured on a different OS (WSL2 against Windows) and so
not a clean comparison.

Seven runs since v7 now span 143-157 around its 154. Whether v13 and v7 differ at all is
a question a seed replicate answers and a benchmark of 231 tasks does not.

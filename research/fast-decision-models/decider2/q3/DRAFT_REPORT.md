# Q3 (B4): decision-level rounding, decision-level scales and quantization-aware distillation for 4-bit hobson-v19

Labels: [M] measured, [V] verified, [A] arithmetic. Box q3 (A10G). Code `~/decider2/q3/code/`; notes, predictions, scores in `~/decider2/q3/`.

## Summary
- **No B4 variant meets the fidelity bar** [M]. Through the deployed kernels, on all 3,227 questions:
  - best plain W4A4: 6.09% REAL flips (bar 0.70%; untrained GPTQ 9.23%);
  - row-role w4q8: 3.05% untrained (GPTQ), 3.60–3.69% after B4 training.
- **The one lever that worked was decision-level per-channel weight scales.** 0.6M parameters were trained on the decision KL for 1.5M tokens, with the GPTQ codes fixed [M].
  - Plain W4A4 flips fell from 9.23% to 6.56%: 64 decisions gained, 35 lost against GPTQ (p .005).
  - TV fell from .0820 to .0603.
- **Full-weight QAD added nothing measurable** [M]:
  - plain, 20M tokens over the scales: 43 gained / 38 lost, p .66;
  - w4q8, scales plus 8M tokens of QAD against GPTQ: 17 gained / 24 lost, p .35.
- **Decision-level rounding (AdaRound-style, all 1.37B weights) changed no rounding** in its 1M-token budget [M].

## What I built
- **`q3lib.py`:** a differentiable hobson-v19 student in H1/H2's rotated format, plus a bf16 teacher, in hobson's shared-prefix multi-question layout. Packed and single-question outputs are identical (TV 0.0) [V].
  - **Forward GEMM:** FORMATS.md F2 exactly. Per-token absmax int4 activations (clip 0.9), per-channel int4 weights, int32 accumulation through `torch._int_mm`, then (acc·s_t)·s_w → bf16.
  - **Backward:** a straight-through estimator. Its gradients match autograd on the STE reference to 0.2–0.3% [V].
  - **Three trainable parametrizations:**
    - per-channel log-scales, with codes fixed;
    - AdaRound soft rounding of every weight;
    - full-weight bf16 latents, trained with stochastic-rounding Adam (factored second moment, update clipping) and a decoupled L2 pull (λ 100) toward the seed.
  - **Row role w4q8** (FORMATS.md §8): state rows W4A4 and trainable; question rows W8A8 with fixed GPTQ8 codes.
- **Init:** H1's GPTQ recipe (64 train-split states, act-order). It also saves GPTQ's error-compensated weights.
- **Loss:** KL(bf16 hobson ‖ student) on the decision, plus 0.5 × the relative MSE of the residual at layers 5, 11, 17 and 23 on the answer and option rows.
- **Data:** `train_pool` (10,116 requests) plus 20% `train_v5`; dev = 366 questions from 15 held-out train-split tasks (selection by dev TV only).
- **Scoring:** exported codes run through H2's QRT kernels on q3 (plain), or Q2's QRT2C (w4q8). Results are against hobson's references; McNemar is against the bf16 QRT runtime on the same box (0.55% flips).
- **Training speed:** 1,050 → 1,850 tokens/s after code caching, ±1-matmul Hadamards and compiled glue [M].

## Results through the deployed kernels, all 3,227 questions [M]

| | bar | W4A4 GPTQ | + scales | + QAD 10M | + QAD 20M | w4q8 GPTQ | w4q8 + scales | w4q8 + scales + QAD 8M |
|---|---|---|---|---|---|---|---|---|
| REAL flips vs hobson | ≤ .70% | 9.23% | 6.56% | 6.74% | 6.09% | 3.05% | 3.60% | 3.69% |
| REAL TV (b8 .0048) | | .0820 | .0603 | .0586 | .0588 | .0312 | .0300 | .0289 |
| McNemar vs bf16 runtime (lost/gained) | p > .05 | 98/4 | 67/2 | 72/5 | 65/5 | 32/5 | 37/4 | 36/2 |
| CF retention | ≥ .99 | .780 | .844 | .817 | .789 | .826 | .908 | .872 |
| CF-probe retention | ≥ .95 | .676 | .695 | .638 | .648 | .667 | .762 | .714 |
| JB-hard (McNemar vs hobson, p) | n.s. | .546 (.66) | .508 (.84) | .462 (.12) | .500 (.66) | .523 (1.0) | .492 (.42) | .500 (.65) |
| REAL-label | ≥ .78 | .780 | .790 | .785 | .783 | .785 | .788 | .780 |

- Every McNemar vs bf16 is p < .001; CF-probe uses 105 tracked pairs (SE ~.05).

## Learning curves [M]
Dev is exact emulation on 366 questions. REAL (1,083 questions) and CF-probe (105 pairs) use fast emulation, whose glue is compiled; at the plain arm's start it gave the same REAL flips as the deployed kernels (6.56%).

| arm, tokens | dev TV | dev flips | codes changed | REAL TV | REAL flips | CF-probe |
|---|---|---|---|---|---|---|
| plain GPTQ | .0715 | 19 | – | | | |
| plain + scales 1.5M | .0535 | 23 | 0 | .0621 | 6.56% | .629 |
| plain QAD 10M | .0481 | 21 | 5.6e-5 | .0589 | 7.02% | .667 |
| plain QAD 20M | .0514 | 15 | 5.0e-5 | .0558 | 6.65% | .629 |
| w4q8 GPTQ | .0258 | 9 | – | | | |
| w4q8 + scales 2M | .0252 | 9 | 0 | .0296 | 3.51% | .686 |
| w4q8 QAD 4M | .0230 | 9 | 3.9e-5 | .0282 | 3.69% | .714 |
| w4q8 QAD 8M | .0229 | 5 | 2.9e-5 | .0285 | 3.51% | .724 |
| w4q8 QAD 12M | .0238 | 8 | 1.9e-5 | .0281 | 3.32% | .714 |

- **QAD lowers TV by about 5% per 10M tokens; flips and CF-probe move only within noise.** Extrapolating the plain arm to 50M tokens gives TV ≈ .048, about 10x b8 [A].
- Dev flips are noise-dominated: models differing in ~40 of 1.37B codes scored 20 and 23 [M].

## What failed, and why
- **Rounding.**
  - With the soft value started at GPTQ's choice (h = .95), lr 3e-3 and update clipping, nothing crossed h = .5 in 1M tokens. One flip needs ΔV ≈ 2, which takes ≥ 220 consistent steps [A].
  - At lr 1e-2 the soft model degraded (train KL .04 → .15 in 20 steps).
- **QAD instability.**
  - With latents started at GPTQ's compensated weights, QAD diverged within 30 steps (train KL .02 → .21) [M]. The likely cause is that every near-boundary weight flips on Adam's sign-like early steps [A]. Starting the latents on grid points fixed it [M].
  - lr 5e-5 diverged; lr 2e-5 was stable but changed only about 5e-5 of codes [M].
- **Mechanism.**
  - After rotation, state-row activations are near-Gaussian, so per-token int4 leaves about 13% error per row. Weights cannot cancel it, and Q1 found the state-row sensitivity is high-rank.
  - Per-weight STE gradients carry too little signal at 10–20M tokens; each of the 0.6M scales sees ~2,000x more gradient [A].
  - In w4q8 the question rows are already int8, and scales gave a non-significant 23 gained / 29 lost (p .49). This suggests the plain-arm scale gain came mainly from question-row error.

## Speed against W8A8-b8 (q3 A10G, Q2's q2bench, exclusive GPU, 20 reps; p95 within 0.3 ms of the median) [M]
Training changes codes, not kernels: plain = H2's W4A4 kernel, w4q8 = Q2's QRT2C (side stream).

| T, questions | b8 (ms) | W4A4 (ms) | W4A4 / b8 | w4q8 (ms) | w4q8 / b8 |
|---|---|---|---|---|---|
| 64, 1 | 8.38 | 5.93 | .71 | 9.44 | 1.13 |
| 256, 1 | 14.38 | 9.65 | .67 | 12.06 | .84 |
| 1000, 1 | 36.40 | 23.46 | **.645** | 26.18 | **.719** |
| 4000, 1 | 142.65 | 85.59 | .60 | 89.48 | .63 |
| 1000, 15 | 166.22 | 102.95 | .62 | 155.24 | .93 |
| 4000, 15 | 293.62 | 194.08 | .66 | 231.09 | .79 |

- **GEMM kernel time at M = 1,125** (graph profiler):
  - b8 25.82 ms;
  - W4A4 12.99 ms (.503x, meets the ≤ .65 bar);
  - w4q8 20.17 ms (.781x, misses it).
- **End to end at T = 1000, one question (bar ≤ .75x, ≤ 27 ms):** W4A4 and w4q8 both pass.

## Projections, T = 1000, one question (H2's model: GEMMs scale with peak rate, other kernels with DRAM bandwidth, host constant) [A]
- **W4A4:** RTX 3090 13.5 ms, 4090 8.9, 5090 5.7.
  - The 5090 figure assumes NVFP4: Blackwell has no int4 tensor path, so it is a different format from the one scored here.
- **b8:** 3090 19.9, 4090 11.6, 5090 7.9.
- **w4q8** (Q2's projection): 3090 16.2, 4090 11.6, 5090 7.2.

## Conduct and deviations
- **Budget.** QAD ran 20M tokens on plain and 12M on w4q8, against the brief's 50–100M.
  - The A10G trained at 1,650–1,850 tokens/s.
  - I stopped the plain arm when its curve was flat and Q1 recommended w4q8 as the QAT base.
  - q3b never came up.
- **Timer.** Reset once, at 05:07 UTC (`shutdown -h +600`; new shutdown 15:07 UTC).
- **Errors** (self-matching `pkill`, a chain-script race and OOM, a compiled quantizer corrupting int8 rows in training) were caught before any reported number.
- **L2 anchor.** It is the seed latent (hobson after GPTQ and scales), not raw hobson, so the pull does not undo GPTQ.
- **Permissions.** No permission denials.
- **Weights.** The trained checkpoints and codes are only on q3 (`~/work/q3/ck_*`, `codes_*.pt`), which shuts down at 15:07 UTC.

## Most valuable next step
- **Stop spending tokens on per-weight 4-bit training of hobson at this scale.** Rounding, QAD at 10–20M tokens and lr sweeps all left state-row activation rounding untouched, and that is the binding error in every 4-bit format scored.
- **Next, make a better state-row activation format deployable, for example Q1's B5 transform coding.** tc45rq8 reached 1.29% flips in emulation, against 3.69% for w4q8.
- **Then run B4.1a's 15-minute decision-level scale pass on it.** It is the only training step with a measured effect, though on w4q8 it was not significant. Gate it on a paired McNemar through the deployed kernels.

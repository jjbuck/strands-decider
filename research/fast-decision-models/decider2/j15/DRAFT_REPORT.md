# J15: speculative precision for decision prefill. Refuted; the depth version works

Box j15 (A10G). Files are in `~/decider2/j15/`; exit-head weights stay on the box. Tags: **[M]** measured, **[V]** arithmetic from measured anchors, **[S]** speculation.

## 1. Hypothesis, and the first-principles problem

- **The idea.** Run W4A4 as a draft, and re-run W8A8-b8 only where the draft's margin is below τ. The cost is D + f·V.
- **Why speculative decoding gains (arXiv 2211.17192).** A decode step is memory-bound, so k drafted tokens are verified in about one step.
- **Why decision prefill does not.**
  - It is compute-bound, so verifying means re-running in full.
  - Reusing the draft's partial products (W8 = 16·W4hi + W4lo) needs identical inputs to draft and verifier, which holds only at layer 0.
- **The ratio that matters, D/V [M].** W4A4/b8 is 0.70 at 16 tokens, 0.645 at 1000, 0.595 at 4000 and 0.65 at 9000. So the 0.65x bar needs f ≈ 0 at 1000 tokens [V].
- **Why it would matter.** 4-bit speed at 8-bit fidelity, without first making 4-bit accurate.

## 2. What I built

- **The deployed kernels, rebuilt:** H2's CUTLASS s4/s8 and EVT kernels, H1-recipe GPTQ codes (three W8 draws, one W4). b8 reproduces 36.32 / 142.1 ms at 1000 / 4000 tokens [M].
- **A runner on H2's QRT** with residual taps, layer-range execution and segment CUDA graphs.
- **Train-split sets, disjoint by tau task:** DEV (1,736 questions; τ, early stopping) and EXIT (4,205; training).
- **Uncertainty signals:** the draft margin, and an MLP on the draft's final hidden state and logits predicting W4A4 ≠ b8.
- **Exit heads at layers 4–20.** Each starts from hobson's pointer head and adds a rank-512 adapter. They read b8's residual after L layers and are trained by KL to b8's final distribution.

## 3. Speculative precision [M]

Setup: all 3,227 eval questions, deployed kernels, τ fitted on DEV. Cost is the mean over REAL+LONG+JevBench (1,785 questions), each interpolated at its exact row count.

**The draft.**
- W4A4 flips 10.0% of decisions against b8 (REAL 8.7% vs hobson, CF fgh .771, CF-probe .581). k64+qb flips 0.84% against b8 and passes fidelity alone (REAL 0.55%, CF-probe .981), but at 40.4 ms (H6's measurement) it is slower than b8: useless as a draft.
- Flips do sit near the boundary: the median draft margin is .10 for flipped decisions and .44 for the rest.
- But the tail is long: the 95th percentile of flipped margins is .35.

| cascade | f | REAL flips vs hobson | paired vs b8 | CF fgh | CF-probe fgh | JB-hard | cost (x b8) |
|---|---|---|---|---|---|---|---|
| W4A4→b8, DEV eps 0 | .71 | 1.11% | 0/0 | 1.000 | .962 | .538 | **1.28** (e2e **1.27**) |
| W4A4→b8, DEV eps .3% | .40 | 2.03% | 1/11, p .006 | .982 | .971 | .531 | 0.97 |
| W4A4→b8, learned uncertainty | .60 | 1.20% | 0/1 | .982 | .962 | – | ≈1.2 [V] |
| W4A4→b8, **oracle** deferral | .085 | = b8 | 0/0 | = b8 | = b8 | = b8 | **0.71** |
| k48→b8, eps 0 | .12 | 1.11% | 0/0 | 1.000 | .962 | .531 | 0.92 |

- **The learned uncertainty does not beat the margin.** It ties on DEV and is worse on eval: .0112 residual flips against .0068 at f = .4.
  - The 4-bit perturbation is noise the draft cannot observe, so f is set by the noise scale.
- **Cheaper verifiers fail.**
  - **Re-running only the uncertain questions:** 98% of 4-question requests defer at least one question, and the state must be re-run.
  - **Refining only the later layers:** W4A4 for layers 0–11 plus b8 for layers 12–23 still flips 9.1% against b8, against 10.0% for the bare draft. The error is made early.
  - **Multi-level cascades:** at least .63 + .65 × .79, which is above 1.
- **End to end, 120 real requests [M]:** cascade 101.2 / 87.3 / 306 ms (mean / p50 / p95), against b8 79.7 / 64.5 / 217.
- **Projections [V, H2's model].** D/V is .68 on a 3090, .76 on a 4090 and .73 on a 5090 (NVFP4). The non-GEMM floor does not shrink with precision, so newer cards make the cascade worse.

**Verdict: speculative precision is refuted [M].** Matching b8 costs 1.27x b8 end to end. Even a perfect uncertainty oracle gives 0.71x, against the 0.65x bar.

## 4. The variant that works: depth speculation [M]

**How it works.**
- **Draft:** b8's own first L layers plus an exit head.
- **Verify:** continue layers L to 23 from the saved residual.
- The prefix costs L/24 within 1% at every length, and prefix plus suffix equals the full run within 0.5%. Nothing the draft computes is wasted.
- This is LayerSkip-style self-speculation (arXiv 2404.16710) and CALM-style early exit (arXiv 2207.07061) for a decision readout.

**Finding [M].** The exit head alone agrees with full b8 on 80.0% of eval decisions at L8, 89.5% at L12 and 98.8% at L16. Its DEV KL to the full model:

| layer | DEV KL to full b8 |
|---|---|
| L14 | .0016 |
| L16 | .0004 |
| L20 | .0001 |
| for scale: bf16 vs b8 | .00011 |

hobson's last third moves real decisions about as little as 8-bit rounding does.

**Cascades.** τ at zero DEV residual; base = b8 (my draw); eval.

| exits | exit share | REAL flips | paired vs base | CF fgh | CF-probe fgh (base .962) | JB-hard | e2e mean (x b8) |
|---|---|---|---|---|---|---|---|
| {16} | .93 | 1.11% | 0/0 | 1.000 | .962 | .538 | **0.688** |
| {8, 16} | .20 + .73 | 1.20% | 0/1 | 1.000 | .962 | .538 | **0.603** |
| DEV-optimal {4, 10, 14, 18} | .98 | 1.39% | 2/5 | 1.000 | .933 | .523 | 0.519 (interpolated) |

**Replication over four bases** (b8 draws s0, s1 and s2, plus bf16), with τ re-fitted on each base's DEV, over REAL+LONG (6,216 decisions):
- {16} changes **0** decisions of its base. CF fgh is 1.000 on every base, and CF-probe is within one pair of the base.
- {8, 16} changes 9, all of them losses (sign test p .004).

**On the bf16 base, {16} passes the full fidelity bar:**
- REAL flips 0.46% (paired against bf16, 2/2);
- LONG 0.42%;
- CF 1.000;
- CF-probe .971;
- JB-hard .538.

**Exits at L ≤ 14 are confidently wrong on JSON probes deep in the context.** DEV is real traffic with no such probes, so τ is blind to them.

**{16} was chosen post hoc** (after eval singles). A DEV-only rule selects it: exit only where the head's DEV KL is within about 4x of the verifier's quantization KL.

## 5. Latency [M]

A10G, CUDA graph, fresh ids, median of 20 reps (p95 within 0.2 ms); 1 question, rows = T + 125.

| T | 16 | 256 | 1000 | 2000 | 4000 | 9000 |
|---|---|---|---|---|---|---|
| bf16 | 15.5 | 25.5 | 57.1 | 107.6 | 203.5 | 458.2 |
| W8A8-b8 | 8.20 | 14.36 | 36.32 | 67.23 | 142.1 | 337.1 |
| W4A4 | 5.78 | 9.65 | 23.42 | 42.81 | 84.57 | 220.0 |
| b8 to layer 16 + exit head | – | – | 24.08 | 44.94 | 94.50 | 224.5 |

**End to end on 120 real requests at exact lengths** (54 REAL, 18 LONG, 48 JevBench), mean / p50 / p95 in ms:

| system | REAL | LONG | JevBench | all |
|---|---|---|---|---|
| b8 | 88.1 / 76.8 / 156.8 | 200.4 | 24.9 / 8.1 | 79.7 / 64.3 / 216.9 |
| depth {16} | 58.7 / 51.1 / 104.6 | 141.7 | 17.7 / 5.6 | 54.8 / 42.8 / 146.1 |
| depth {8, 16} | 50.8 / 45.3 / 91.0 | 121.3 | 17.3 / 5.7 | 48.0 / 39.6 / 131.0 |

- Every e2e exit matched the offline prediction. By interpolation over the full 1,785-question mix: {16} 0.694x, {8, 16} 0.604x.

**Projections at 1000 tokens [V].** Depth savings are a layer fraction, so they carry over to every card.

| | 3090 | 4090 | 5090 |
|---|---|---|---|
| b8 | 19.9 | 11.6 | 7.9 |
| {16} | 13.8 | 8.0 | 5.5 |
| {8, 16} | 12.0 | 7.0 | 4.8 |

## 6. Caveats

- **"b8 passes at 0.18%" was a favourable GPTQ draw.** My three draws flip 1.11%, 1.02% and 0.65% of REAL decisions (CF-probe .962, .943, .971).
  - TV is constant at .0048.
  - On DEV the draws are indistinguishable (4 against 6 flips; KL .00011), so no train-split selection is possible.
- **The depth cascade inherits its base's fidelity.**
- **Calibration.** JevBench is outside the exit heads' training traffic. Its Brier worsens .349 → .354 with {16} and → .374 with {8, 16}; REAL-label Brier is unchanged.
- **Resolution.** JB-hard cannot resolve under 4 points; one CF-probe pair is 0.95 points.

## 7. Verdict

- **Speculative precision is refuted [M].** The draft cannot see its own noise, and verification is a full re-run. Matching b8 costs 1.27x b8, an oracle gives 0.71x, and newer GPUs make it worse.
- **Depth speculation is the version of the idea that survives [M].**
  - The layer-16 exit preserves its base's decisions exactly on 6,216 real decisions across four bases.
  - CF is identical, and CF-probe stays within one pair.
  - It runs at 0.688x of b8 end to end.
- **The combined bar is missed narrowly:** {16} on b8 is 0.688x (bar 0.65x); {8, 16} reaches 0.603x but drifts by 0.15% of decisions; on bf16 fidelity passes, but bf16 is 1.47x b8.
- **What it means.** About 93% of decisions do not need hobson's last third, and the saving composes with 8-bit, schema and fp16 accumulation.

## 8. The single decisive next step

Make one exit below layer 16 CF-safe. That would move the cascade from 0.69x to about 0.52x of its base [S]: the DEV-optimal sets, which today fail only CF-probe.

1. Build a train-split capacity DEV: edits to amounts, identities and record fields in train-split states, from a generator distinct from evalkit's CF templates.
2. Train the L12 and L14 exits with it as negatives.
3. Fit τ on it.

**Pass:** 0 decision changes on REAL+LONG across the four bases, and CF-probe within one pair of each base.

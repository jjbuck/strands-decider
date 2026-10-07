# Brief 10: faster matrix multiplies for a prefill-only decision model; 4-bit activations as safe as QuaRot made 8-bit

You are one of three agents (Q1, Q2, Q3) working for a principal engineer who wants the matrix multiplies of a 2B decision model to run faster on the GPU, without changing its decisions. Read this brief in full, then `~/code/jit-eval/docs/IDEAS_EXPLORED.md` (rows P1–P9 and "The floor on wall time") and section 5 of `~/code/jit-eval/docs/FAST_DECISION_MODEL.md`.

## Why this is the lever

hobson-v19 is a Qwen3.5-2B GDN hybrid with a pointer head. A decision is one forward pass over the state plus its questions, with no generation, so every request above about 100–150 rows is limited by tensor-core arithmetic, not by memory.

**At 1000 tokens (A10G):**
- 87% of the bf16 time is matrix multiplies, at 86–91% of cuBLAS's rate.
- W8A8 with rotation and GPTQ (P1) takes 36.3 ms against 57.0 for bf16, with decisions at the noise floor.

The tensor-core rate follows **activation** precision, so the next 2x has to come from 4-bit activations. This is where decision models differ from LLM serving, whose generation is memory-bound and favours weight-only quantization.

**What we measured before (H1, H2, H5, H6, J5, J10, J15), full 24 layers, deployed kernels:**
- **Kernel speed:** int4 tensor-core GEMMs run 2.75–4.3x faster than bf16 at hobson's shapes; int8 runs 1.6–2.4x. Plain W4A4 takes 23.4 ms at 1000 tokens.
- **W4A4 accuracy:** every configuration with most multiplies in 4 bits changes 2–13% of real decisions.
  - With rotation and GPTQ: 8.6–9.2% flips.
  - The best mix found: 64 GEMMs in W8A8, question rows in bf16, GDN gates in bf16. That is 38% of the work in 4 bits, at 0.46% flips, but it is no faster than W8A8.
- **Cause:** local rounding error. Per-token max/rms at the GEMM inputs is 15–40. After a Hadamard rotation it is 3.2–3.8, which still leaves 13–16% relative error per token at int4. Error does not grow through depth.
- **Most sensitive GEMMs, in order:** layer 0's output projection; the attention input projections at layers 7 and 11; layer 23. Layers 12–22 are nearly inert, and the GDN gate projections are least sensitive.
- **Question rows carry much of the decision error:** running them at higher precision brought total variation to the bf16 floor.
- **Tried and not enough:**
  - learned rotations (SpinQuant-style, a 14–25% error cut);
  - LoRA quantization-aware training (four runs, no gain, small budgets);
  - NVFP4 block scales in emulation (halved the error, still 4.2–5.2% flips);
  - speculative precision (slower);
  - native ternary BitNet (worse).
- **Kernel code:**
  - H2's CUTLASS s4/s8 kernels and fused runtime (`~/decider2/h2/code/`: `qrt.py`, `qgemm.py`, `h2mix.cu`, `g2s4.cu`, `h2evt.cu`);
  - J5's short-input kernels (`~/decider2/j5/code/`);
  - J15's rebuild of both (`~/decider2/j15/code/`).
- **Quantization code:** H1's GPTQ, sensitivity and precision maps (`~/decider2/h1/code/`).

Prior art to know (you must go beyond it, not reproduce it), verifying each ID by fetching `https://arxiv.org/abs/<id>` (cite only what you fetched):
- QuaRot 2404.00456;
- SpinQuant 2405.16406;
- FlatQuant 2410.09426;
- ResQ 2412.14363;
- DuQuant 2406.01721;
- QUIK 2310.09259;
- Atom 2310.19102;
- OSTQuant 2501.13987;
- QServe 2405.04532;
- microscaling formats 2310.10537;
- BitNet v2 2504.18415;
- EfficientQAT 2407.11062.

If an ID is wrong, search for the title.

## Why format tricks alone cannot reach 8-bit fidelity, and what can

After a Hadamard rotation, GEMM input activations are roughly Gaussian.

**The Gaussian rate-distortion bound:** at R bits per value, the relative rms error is at least 2^-R.
- 4 bits: ≥ 6.25%.
- 8 bits: 0.39%.

Practical 4-bit formats are worse: we measured 13–16% after rotation and 7–9% for NVFP4 block scales. No 4-bit format (ResQ, FlatQuant, NVFP4, …) can bring per-value error near W8A8's, which is why every 4-bit attempt so far changed 2–13% of decisions.

A 4-bit model at the noise floor must therefore come from **the decision not depending on most of that error.** That is plausible here, because the output is a few probabilities, not a distribution over 250k tokens. The program tests four ways this could happen, each with a decisive cheap measurement first:

**B1. Decision-sufficient precision.**
1. **Measure the spectrum.** For each of the 96 GEMMs, compute the second moment of the decision loss's gradient with respect to the GEMM output, G = E[gᵀg] over output channels. Do the same for the input side. Separate state rows from question rows. Use ≥ 200 train-split requests and bf16 backward through hobson, with the KL between the decision distribution and itself perturbed as the loss, or a sum over option logits.
   - Report the effective rank: the directions holding 90% and 99% of the trace.
2. **If the spectrum is concentrated** (r ≪ N): run the GEMM in int4 and add an **exact correction of the rounding error inside the top-r sensitive subspace P**: Y = Y_int4 + ((X − deq(Q(X))) · (W·P)) · Pᵀ. That costs ~2r/N extra work, so to first order the decision-relevant error vanishes.
   - Also correct the weight-rounding error the same way, with W − deq(Q(W)) in the subspace.
   - Variants: P per GEMM; P split by row role; input-side versus output-side subspaces.
3. **Validate a first-order flip predictor** (Δdecision ≈ Σ⟨g, error⟩) against real flips, so formats can be ranked without full evaluations.

**B2. Noise averaging through aggregation.**
- **The idea:** state rows influence the decision only through attention and GDN sums over many rows. Round-to-nearest error is a deterministic function of the input and is correlated across repetitive content (JSON keys, boilerplate), so it accumulates like a bias.
- **What to test:** unbiased, decorrelated rounding (dither or stochastic rounding with per-request seeds) on state rows, with question and answer rows in int8.
- **The check:** flips should fall with the number of state rows aggregated if this works.

**B3. Prototype subtraction.**
- **The idea:** per GEMM input, a codebook of K_c prototypes (k-means on calibration activations, K_c = 64–1024). For each row, subtract the nearest prototype c; quantize only the residual x − c to 4 bits. Add c·W from a precomputed table in the epilogue: exact and nearly free. Systematic outliers, the cause of the 15–40 max/rms, live in the prototypes.
- **First measurement:** residual max/rms and 4-bit error against plain rotated int4, by layer.
- **Also report:** the cost of finding the nearest prototype (one skinny GEMM).

**B4. Decision-level rounding and training.**
1. Choose each weight's rounding (up or down) to minimise the end-to-end decision KL on train-split requests, AdaRound-style but with the decision as the objective rather than GPTQ's per-layer output error.
2. Then run full-weight quantization-aware distillation, seeded from the best of B1–B3, with the exact kernel arithmetic in the forward pass.

**Published formats as baselines.** Run the published methods only as comparison points, in exact emulation:
- rotated per-token int4 (H2);
- ResQ-style int8/int4 subspace split (r = 128);
- group-scaled int4 (g = 64);
- precision by row role (question rows int8).

Every B result is compared with these on equal 4-bit work share.

## The bar

**Fidelity**, against hobson-v19's references, on all 3,227 evalkit questions, scored with `~/decider2/evalkit`, through the deployed kernels or a bit-exact emulation of them:
- REAL flips ≤ 0.7%, with McNemar against the bf16 runtime not significant (p > .05);
- CF retention ≥ .99;
- CF-probe retention ≥ .95;
- JB-hard McNemar against hobson not significant;
- REAL-label ≥ .78.

Also report total variation; W8A8-b8 is at .0048.

**Speed:**
- matrix multiply time ≤ 0.65x W8A8-b8's at hobson's shapes for M = 1,125;
- end-to-end ≤ 0.75x W8A8-b8 at a 1000-token state with one question (≤ 27 ms on the A10G);
- also report T = 64, 256, 4000 and 15 questions.

**Projections** to RTX 3090, 4090 and 5090 by the doc's method: on GeForce, int8 runs at 2x and int4 at 4x the fp16-accumulation rate; Blackwell has native FP4.

## The three agents

### Q1: decision sensitivity, B1–B3 in exact emulation (box `q1`)
1. Do B1.1 first; it decides the program's direction. Publish `~/decider2/q1/sensitivity.json` (effective ranks per GEMM and row role) as soon as it exists.
2. Then B1.2–B1.3, B2 and B3, plus the baselines. Use Q2's `~/decider2/q2/FORMATS.md` arithmetic once it exists.
3. **Report:** flips, CF and CF-probe retention, JB-hard, REAL-label, total variation, the share of GEMM work in 4 bits, and the extra work of any correction.
4. Publish `~/decider2/q1/best_formats.json` and keep it current.

### Q2: kernels (box `q2`)
1. Within about 2 hours, publish `~/decider2/q2/FORMATS.md` with the exact arithmetic of each kernel: scale granularity, rounding, accumulation, order of operations.
2. Build Ampere (SM86) kernels, extending H2's CUTLASS s4/s8 work:
   1. **int4 GEMM with a fused low-rank error-correction path (B1).** The prologue computes ΔX = X − deq(Q(X)) in bf16. Warp-specialized CUDA-core or tensor-core warps compute ΔX·(W·P) (K×r) concurrently with the int4 main loop, and the epilogue adds (·)Pᵀ. Measure the overhead for r = 32, 64, 128 and 256.
   2. **Dither or stochastic rounding inside the fused quantize prologue (B2),** with per-request seeds.
   3. **Prototype subtraction (B3):** nearest-prototype search plus residual quantization in the prologue, and the c·W table gather in the epilogue.
   4. **The baselines:** int8/int4 inner-dimension split; group-scaled int4; a row-role grouped GEMM (state rows 4-bit, question rows int8) with one weight copy where possible.
3. **Measure:**
   - GEMM time at hobson's shapes for M = 140, 400, 1,125 and 4,125, against H2's W8A8-b8 and W4A4;
   - then integrate the best format into H2's QRT runtime and measure end to end at T = 64, 256, 1000 and 4000, with 1 and 15 questions.
4. **Measure the fewer-multiplications route.** One level, and if promising two levels, of Strassen–Winograd on tensor cores at hobson's shapes (K = 2048, N = 2048–12288, M = 140–4,125). The user asked about fast matrix-multiplication algorithms, and this is the only finite-size member of that family. Strassen adds input blocks before multiplying, which widens integers by one bit, so test:
   - bf16/fp16;
   - int8 with 7-bit inputs (sums fit int8);
   - int4 with 3-bit inputs.

   Report the speedup over the same-precision dense kernel and the accuracy effect of the lost bit. This must be a measurement, not an argument.
5. **Look for any other measured way to make these GEMMs faster on this GPU at equal accuracy**, for example 2:4 sparse int8/int4 tensor cores at small M (kernel speed first), or tile and split-K choices for small M. Report only measurements.

### Q3: decision-level rounding and quantization-aware distillation, B4 (box `q3`)
1. **Decision-level rounding.** Rotated weights with GPTQ initialization, then optimise rounding (and, separately, scales) against the end-to-end decision KL to bf16 hobson on train-split requests.
2. **Full-weight quantization-aware distillation.**
   - **Fake quantization:** a straight-through estimator implementing the exact `FORMATS.md` arithmetic. Start with plain per-token W4A4 with rotation, the fastest kernel. Add Q1's best format when `best_formats.json` appears.
   - **Loss:** KL to the teacher's decision distribution on `evalkit/train_pool.jsonl` and `training/data/train_v5.jsonl`; relative MSE of hidden states at layers 5, 11, 17 and 23 on answer and option rows; an L2 pull toward hobson's weights (J14's full fine-tune forgot without one).
   - **Optimizer that fits in 24 GB:** bf16 weights with stochastic rounding and factored or 8-bit Adam (see J14's arm B, `~/decider2/j14/code/j14train.py`).
   - **Budget:** 50–100M tokens.
   - **Checkpoints:** every ~10M tokens, scored on a fixed dev subset of train-split requests. Never train on evalkit eval items.
3. **Score and report.** Score the best checkpoint on all 3,227 questions, through Q2's kernels if available, otherwise through exact emulation. Report the learning curve: flips and CF-probe retention against tokens.

## Infrastructure and rules

- **Boxes.** Use only `~/decider2/box.sh <BOX> run "<cmd>" | put <local> <remote dir under ~/work> | get <remote path under ~/work> <local> | status`.
  - Wait for `~/decider2/boxes/<BOX>.ready`, checking every ~5 minutes with `sleep 290`.
  - Each box has `~/work/sd` (strands-decider), `~/venv`, hobson-v19 and Qwen3.5-2B-Base in the HF cache, and the bundle (systems, tokens, shrink, training, d1, evalkit, h2/code, h4, h7/code).
  - Put any other code you need (h1, j5, j14, j15) with `box.sh put`. CUTLASS and nvcc are on the box; check with `nvcc --version`.
- **Box lifetime.** Each box auto-terminates 10 hours after launch. You may reset your own box's timer once (`sudo shutdown -c; sudo shutdown -h +N`) if needed; note it in NOTES.md.
- **Commands.** One command must finish in under 30 minutes. Run long jobs with nohup and make them resumable. GPU memory must stay under 22 GB. Latency needs exclusive use of the GPU, fresh inputs and ≥ 12 warm reps (median and p95).
- **No models on the laptop.** Don't run torch or any model locally, and keep weights and checkpoints on the boxes. Small JSON outputs come back to the laptop.
- **No other AWS resources.** Don't touch any and don't launch instances.
- **Writing.** Write only under `~/decider2/<you>/`:
  - `NOTES.md` with timestamps every ~15 minutes;
  - `DRAFT_REPORT.md` (the harness blocks files named REPORT.md);
  - results JSON.

  Copy results off the box at least every 30 minutes. You may read the other two agents' folders but never write to them.
- **Permissions.** If a permission is denied, don't work around it; report it.
- **Language.** Plain and specific. Every number gets its metric, baseline and unit. No verdict labels ("survives", "dominated", "cost-class"); say what was measured and what it means. Mark each claim as measured, verified or arithmetic.

## Report

At most 1,500 words. Return it as your final message and also write it to `DRAFT_REPORT.md`. Cover:
- what you built;
- the measured results against the bar;
- speed against W8A8-b8;
- projections;
- the single most valuable next step.

## Addendum, 2026-10-06 ~20:30: hypotheses B5–B10, added to Q1's and Q2's queues

All six rest on the same decision-specific premise as B1: the output is a few thresholded probabilities over a narrow task, so most of the network's numerical error can be invisible to the decision. Each builds on Q1's sensitivity measurement (B1.1).

**B5. Decision-weighted transform coding: fewer multiplications, not just cheaper ones.**
- **Allocation:** for each GEMM input, find a basis that diagonalises both the activation covariance Σ_x and the input-side decision sensitivity Λ (for example the eigenvectors of Λ^½ Σ_x Λ^½). Allocate bits per direction by reverse water-filling on variance × sensitivity, b_i = max(0, ½·log2(λ_i σ_i² / θ)), at an average of 4 bits.
- **0-bit directions** are replaced by their mean (a bias), so the GEMM's inner dimension K shrinks.
- **The basis change** folds into the previous weights (any invertible map works: xW = (xR)(R⁻¹W)).
- **Q1:** report the allocation per GEMM, the K reduction, and flips in emulation, rounding each direction group to the nearest hardware format (8, 4 or 0 bits).
- **Q2:** contiguous K slices at int8 and int4 with the 0-bit slice skipped. This is the inner-dimension split kernel with a smaller K; measure the speed.
- **Why it differs from J13's failed low-rank cuts:** those dropped high-variance directions blind to sensitivity. Here only directions with negligible variance × sensitivity are dropped, which keeps the change first-order small.

**B6. Self-certifying 4-bit decisions.**
- **Mechanism:** the fused quantize step holds both X and Q(X), so the rounding error E is known exactly at runtime. With the static sensitive subspaces from B1, compute a cheap certificate statistic per request, for example Σ_ℓ ||P_ℓᵀ E_ℓ|| weighted by sensitivity. Calibrate it on train-split requests to a bound on the decision-margin change (conformal, target 99.9%).
- **Fallback:** accept the int4 decision when the bound is below the request's own margin. Otherwise re-run in int8, or re-run only the most sensitive layers.
- **Q1:** report the correlation of predicted against actual margin change, the share of requests certified, residual flips and the expected cost.
- **Q2:** compute the statistic inside the prologue and epilogue (one reduction per GEMM); measure its cost.
- **Why it differs from J15:** J15's speculative precision failed because the draft could not see its own noise. Here the noise is measured.

**B7. Per-deployment and per-question calibration.**
- **Q1:** calibrate rotations, GPTQ codes, sensitive subspaces and B5 allocations on the banking deployment's own train-split traffic and its 5 question specs. Compare against generic calibration on banking eval items and on the whole kit.
- **Also:** measure the per-question sensitive subspace's effective rank against the all-questions rank.

**B8. More bits for value-bearing tokens.**
- **Q1:** measure per-row decision sensitivity by token class: digits, amounts, dates, ids, emails and phones, JSON keys, JSON punctuation, prose, and so on. Then emulate value-bearing tokens in int8 with everything else in int4.
- **Report:** the share of rows in int8 and flips.
- **Q2:** the row-role grouped GEMM must accept an arbitrary row partition.

**B9. Unbiased sampled matrix multiplies for state rows.**
- **Mechanism:** per output tile, keep a random fraction f of inner-dimension tiles (tile = 64 or 128 along K, a per-request seed) and rescale by 1/f. State rows only; question rows exact. Combine with B1's correction.
- **Q1:** emulate f = 0.75 and 0.5. Report flips against the number of state rows aggregated, and check whether SwiGLU turns the variance into bias.
- **Q2:** a tile-skipping GEMM driven by a per-request mask; measure speed against f.

**B10. Spend precision only on differences between options.**
- **Insight:** softmax ignores a shift shared by all options, so error common to every option row cancels.
- **Q1:** split the option rows' sensitivity into a common component and a difference component at the late layers. Measure whether the difference subspace is smaller, and use it in B1 and B5.

**Order of work:**
- **Q1:** B1.1 first, then B5 and B6, then B8, B7, B9 and B10, interleaved with B1.2–B1.3, B2, B3 and the baselines as results suggest.
- **Q2:** the B1 correction kernel, the B5 K-slice split, the row-role grouped GEMM (B8), tile skipping (B9) and the B6 statistic, then the rest of the original list including Strassen.
- **Q3:** unchanged.

## Addendum 2, 2026-10-06 ~21:00: hypotheses B11–B14 (fewer or better-utilised multiplies), added to Q1's and Q2's queues

**B11. Decision-aware 2:4 structured sparsity.**
- **The lever:** Ampere sparse tensor cores (`mma.sp`) double the math rate when each group of 4 weights along K holds at most 2 non-zeros. int8 2:4 runs at int4's dense rate with 8-bit values; int4 2:4 at 2x int4.
- **The decision-specific twist:** choose which 2 of 4 weights to keep by the decision's own sensitivity on train-split traffic. Use OBS or SparseGPT-style updates driven by the decision Fisher or Hessian rather than by magnitude or per-layer error, and calibrate per deployment (banking) as well as generically.
- **Q1:**
  - emulate 2:4 on the least sensitive layers first (12–22 were nearly inert to rounding in H1), then everywhere;
  - test int8 2:4 and int4 2:4, with and without B1's correction;
  - report flips, CF and CF-probe retention, and the share of GEMM work made sparse.
- **Q2:** a custom `mma.sp` kernel, int8 and int4 2:4, in the q2gemm family, at hobson's shapes for M = 140–4,125, against the dense kernels. S9's "2:4 is slower than dense below 2048 rows" came from the cuSPARSELt library, not the hardware; measure your own.
- **If Q1 finds a near-miss,** Q3 may add 2:4-aware distillation.

**B12. Per-deployment dead MLP neurons.**
- **Mechanism:** a narrow deployment may never fire many of the 6,144 SwiGLU neurons per layer, or fire them without effect on the decision. Removing neurons shrinks W_gate and W_up's output width and W_down's inner width, which are the largest matrices.
- **Q1:**
  - per layer and per neuron, measure firing statistics and variance × decision sensitivity on banking train-split traffic and on generic traffic;
  - remove exactly-dead neurons (never active on the deployment's traffic beyond a tolerance) and then ranked neurons;
  - report the FLOP reduction against flips on banking eval and on the whole kit.

**B13. Exact dead-computation elimination.**
- **Layer 23:** state rows need only their K/V projections. Their output projection, gate/up and down are never read (Q1 measured exactly zero gradient), about 3% of GEMM work.
- **Layer 0:** the input projections depend only on the token id, so precompute them per token of the deployment's vocabulary and replace them with a gather (about 1.5%).
- **Q2:** implement both in the runtime and measure.
- **Q1:** confirm both are exact, with zero flips.

**B14. A persistent kernel for short decisions.**
- **Mechanism:** for requests of ≤ 512 rows, run the whole decision in one persistent kernel that streams each layer's weights exactly once while all rows stay on chip, overlaps the next layer's weight prefetch with the current layer's compute, writes no KV cache, and computes the last layer only for the rows that are read.
- **Baseline:** W8A8-b8 at 140 rows is 7.4 ms on the A10G against a ~3.2 ms floor (J5). J5 bounded the gain at ~35%; measure it.
- **Q2:** lowest priority; scope it first (W8A8, M ≤ 512) and report what was reached.

**Updated order:**
- **Q1:** B1.1 → B5 → B6 → B11 → B12 → B13 checks → B8, B7, B9, B10 → B1.2–B1.3, B2, B3 and the baselines.
- **Q2:** B1 correction kernel → B5 K-slice split → B11 sparse kernel → B8 row partition → B13 → B9 tile skipping → B6 statistic → Strassen measurement → B14 → the rest.

## Addendum 3, 2026-10-06 ~21:10: two more agents; work redistributed

| agent | box | owns | gives up |
|---|---|---|---|
| **Q1**, precision accuracy | q1 | B1 (spectrum, correction, predictor), B5, B6, B8, B9, B10, B2, B3, published-format baselines | B7, B11, B12, B13 → Q5 |
| **Q2**, precision kernels | q2 | `FORMATS.md`, the q2gemm format kernels for B1, B5, B8, B9 and B6, group-scaled and split baselines, integration of the accurate formats into QRT, end-to-end latency | B11 sparse kernel, B13, Strassen, B14 → Q4 |
| **Q3**, decision-level training | q3 (q3b if a bigger GPU arrives) | B4 | unchanged |
| **Q4**, multiply count and utilisation kernels (new) | q4 | the B11 `mma.sp` 2:4 kernels (int8, int4); B13 (layer 23's state rows compute only K/V; the layer-0 table); the Strassen–Winograd measurement (bf16, int8 on 7-bit inputs, int4 on 3-bit inputs); B14 (the persistent kernel for ≤ 512-row decisions) | — |
| **Q5**, structure accuracy (new) | q5 | B11 (2:4 pattern selection by decision sensitivity), B12 (per-deployment dead neurons), B13 exactness checks, B7 (per-deployment and per-question calibration for both precision and structure) | — |

**Sharing code:**
- Q4 builds on Q2's `~/decider2/q2/code/` (copy it; never edit Q2's files) and follows `FORMATS.md` for any shared arithmetic.
- Q5 builds on Q1's `~/decider2/q1/code/` (`q1lib.py`, `q1spec.py`, `q1fmt.py`; copy, never edit) and reads `~/decider2/q1/sensitivity.json`.
- Q5 publishes `~/decider2/q5/best_structure.json`, listing the accurate sparsity and neuron-removal settings with flips. Q4 reads it to time the right configurations, and Q2 integrates the winners.

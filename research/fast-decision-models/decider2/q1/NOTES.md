# Q1 notes (box q1): decision sensitivity, B1-B3 in exact emulation

## Summary (kept at the top; updated as results land)
**B1.1, the decisive measurement (MEASURED, 256 train-split requests, 437k tokens; `sensitivity.json`).** For each of the 96 GEMMs, the
eigen-spectrum of G = E[g^T g], g = gradient of the decision KL (softmax-Fisher of the pointer head) wrt the GEMM output and input, split
by state rows and question rows. "r90/r99" = number of directions holding 90%/99% of the trace.
- **Where the state matters (layers 0-12), the sensitive subspace is NOT low-rank.** Input side, K = 2048: state rows r90 = 350-1250,
  r99 = 965-1875; question rows r90 = 250-1020. State rows hold 18-39% of the trace in layers 0-8, 3-25% in 9-12.
  => B1's exact low-rank correction cannot remove most of the decision-relevant int4 error there: removing 99% of it needs r ~ 0.5-0.9 K.
- **Late layers (13-23): question rows hold 96-100% of the trace, in very few directions** (Wo/Wd/Wgu question rows r90 = 2-60,
  r99 = 4-700; layers 18-23 r90 = 2-5). Option rows (the head's keys) carry 59-99% of that at layers 19-23; the readout row 1-35%.
- **State-row sensitivity is concentrated in ROWS, not directions:** the top 1% of state rows hold 24-73% of the state-row gradient
  energy and the top 10% hold 69-95%; the last tenth of the state holds 17-72%, the first tenth (attention sinks) 10-33%.
- Layer 23's Wo/Wgu/Wd get exactly zero gradient on state rows (exact skip, 3% of GEMM MACs).
- Fisher weight is spread over requests (top 10% of requests hold 14% of the Fisher trace).
- Held-out check (MEASURED, set B: 64 requests from tau tasks disjoint from A): A's top eigenvectors capture of B's trace:
  layer 0 Wo state rows 16% / 33% / 54% / 88% at r = 8 / 64 / 256 / 1024; 11.Win 40% / 71% / 92% / 99.5%;
  late question rows transfer (20.Wd, 23.Win, 23.Wd question rows: 98-99.6% at r = 8).
- Correction to the late-layer line above: Wo/Wd/Wgu question rows r90 = 2-63 at layers 13-23; Win (GDN in_proj / attn qkv) r90 = 8-225.
- FINAL sensitivity.json published 21:07 (all 96 GEMMs, held-out capture included). Table: res/spec_table.txt.


**Program result so far (03:20 PDT, MEASURED, emulation in Q2 FORMATS F2 arithmetic):** no format passes the fidelity bar.
- Full kit (REAL flips / CF / CF-probe): bf16 0.55% / 1.000 / .952; row role w4q8 3.69% / .798 / .676; ResQ-128 + row role
  2.40% / .945 / .848; deployable B5 3.14% / .872 / .800; per-GEMM B5 4.5 bits (not deployable) 1.29% / .890 / .914.
- DEV screen (rms margin change; b8 0.030): row role 0.200; B5 per-GEMM 0.088-0.110; + question-attention row routing 0.050.
- Refuted: B1 low-rank correction for state rows (spectra broad), B2 dither (kappa ~ 1, SR worse), B6 certificate (corr ~ 0),
  B8 value tokens (less sensitive), B9 tile sampling (27% flips), noise shaping along rows (new; worse), decision-weighted GPTQ.

## Log

### 2026-10-06 19:55 PDT start
- Read BRIEF10 in full, IDEAS_EXPLORED (P1-P9, floor on wall time), FAST_DECISION_MODEL sec 2-5, H1 code (h1lib, h1sens, h1score, h1eval),
  H1 NOTES/REPORT/FORMAT, H2 REPORT/NOTES, g2lib (recovered copy ~/decider2/recovered/g2/g2/g2lib.py; H1 imports it from ~/work/g2).
- Box q1 not ready yet (q1.ready absent). Planning the B1.1 script meanwhile.

### 20:20 PDT box q1 ready (20:12); B1.1 running
- Box: A10G, torch 2.14.1+cu130, CUDA 12.8 nvcc, 124 GB RAM, 32 vCPU, 201 GB disk free.
- Put on box: ~/work/g2 (g2lib.py, subsets.json from ~/decider2/recovered/g2/g2), ~/work/h1 (H1 code), ~/work/q1 (my code).
- Code (laptop copies in ~/decider2/q1/code):
  - q1lib.py: differentiable forward of H1's lean hobson (fla GDN, SDPA, pointer head), per-GEMM output hooks, pluggable GEMM emulations,
    train-split request sets A (256 req) / B (64 req, disjoint tau tasks), softmax-Fisher directions.
  - q1spec.py: B1.1. G = sum over Fisher eigen-directions (lam_j, u_j) of lam_j g_j^T g_j (exact second-order KL for binary questions),
    G_out [N,N] and G_in [K,K] (input = what the quantized kernel sees), state vs question rows, accumulated in 8 groups of 3 layers.
  - q1fmt.py: format emulation (H1 FORMAT v1 base; B1 correction, B2 dither/SR, B3 prototypes, ResQ-style split, group-scaled int4).
- Timing check (MEASURED): differentiable fwd+bwd at T=1711: 0.14 + 0.23 s once Triton kernels are compiled; each new length bucket
  costs a one-time compile of 5-100 s.
- H1 calibration Hessians (h1calib --n 64, train-split, as H1/J15) done in 53 s. Spectrum chain (groups 7..0, then held-out B) running.
- Coordinator added B5-B10 (BRIEF10 addendum, ~20:30). Plan after B1.1: B5 (water-filling on variance x sensitivity), B6 (certificate),
  then B8, B7, B9, B10 with B1.2-B1.3, B2, B3 and baselines.

### 20:45 PDT B1.1 first groups (layers 18-23), MEASURED, 256 train-split requests (437k tokens), Fisher-weighted decision KL
- Question rows carry almost all of the decision-gradient trace at layers 18-23: state-row share of tr(G_in) 0.0-1.7% per GEMM
  (e.g. 21.Wd: state 2.6e-6 vs question 1.8e-2 per request). State rows feed layers 21-23 only through 6 GDN/attention mixes.
- At layers 21-23, option rows (the rows the pointer head reads as keys) carry 59-99% of the output-side energy; the readout row 1-4%.
  Layer 23's Wo/Wgu/Wd get exactly zero gradient on state rows (only Win's K/V of layer 23 are read) -> those 3 GEMMs need only the
  question rows (an exact 3% saving of GEMM MACs, independent of quantization).
- Question-row spectra are very concentrated: r90 = 2-5 directions (Win 7-20), r99 = 4-34 (Win 15-220), of K = 2048-6144.
- State-row spectra (input side, K=2048/6144): r90 = 20-475, r99 = 136-1422.
- GPTQ codes (H1 recipe, act-order, 64 calibration states) for all 96 GEMMs at 4 and 8 bits: done in ~2 min, cached on the box.

### 21:10 PDT B1.1 complete; queue running
- Group 5 OOM'd in a 12288-dim eigh while my GPTQ-code job shared the GPU (my error); rerun alone, fine.
- sensitivity.json final. Direction for the program: B1's low-rank subspace correction is not the lever for the state (layers 0-12);
  it only fits question rows in late layers, which row-role int8 already covers. The lever the data points to is ROW concentration
  (few state rows, near the end of the state and the sinks, hold most state sensitivity) and the question-row/state-row split.
- Coordinator addendum 2 (B11-B14) read. Order: B5 -> B6 -> B11 -> B12 -> B13 -> B8, B7, B9, B10 -> B1.2-B1.3, B2, B3, baselines.
- Queue on box (runq.sh, queue.txt): q1cal (cal stats) -> q1b5 (B5 plans) -> q1fo w4a4 / w4q8 (first-order) -> q1hd (decision-weighted
  Hessians + neuron saliency for B11/B12) -> q1b13 (exactness).
- arXiv fetched: 2412.14363 ResQ (PCA top-1/8 subspace at 8-bit, rest 4-bit, random rotation within subspaces), 2404.00456 QuaRot,
  2310.09259 QUIK (some outlier weights/activations at higher precision).

### 21:30 PDT B5 planner (model, not yet validated), q1hd done, first-order pass running
- Coordinator addendum 3: B7, B11, B12, B13 moved to Q5 (B13 check dropped from my queue). q1hd (decision-weighted Hessians,
  MLP-neuron saliency by banking/retail) had already started and finished (3 min); outputs left on the box at ~/work/q1/hd for Q5.
  API notes for Q5: ~/decider2/q1/code/README.md.
- q1cal: per-GEMM input mean/second moment by role on 128 set-A requests (1.7 min).
- B5 planner b4s (ARITHMETIC from the distortion model, per GEMM, state rows, average 4 bits with {8,4,0}-bit K slices, eps4 = .0225):
  modelled first-order decision distortion relative to rotated per-token int4 = 0.003-0.47 per GEMM; 22.8% of state-row MACs dropped
  (0-bit). Best basis: 'mix' (eig of C/trC + Lam/trLam) for 51 GEMMs, 'sens' (eig Lam) for 41.
  Two caveats found: (1) for Wd (K=6144) Lam was rank-1024 (top eigenvectors only), so the planner dropped the complement for free;
  held-out tasks put 12-16% of their trace there. Fixed: complement filled isotropically + shrinkage 0.15 toward isotropic (plans b4sr,
  b4sir). (2) dense per-GEMM bases are not deployable for Wo/Wd (an online K x K transform); the deployable variant is the channel basis
  (permutation + Hadamard inside each slice; plan b4sir). Residual readers (Win/Wgu) could use one shared offline basis; per-GEMM bases
  in emulation are an upper bound.
- First-order pass (W4A4 all rows, 300 DEV requests from tau tasks disjoint from A): first 75 requests corr(pred dm, actual dm) = 0.23.
  Queued sanity checks: W8A8 (small errors) and W4A4 at layer 0 only.

### 21:45 PDT first-order pass, W4A4 all rows (MEASURED, 300 train-split DEV requests, tau tasks disjoint from set A)
- W4A4 (rotated, GPTQ, per-token int4) flips 26/300 = 8.7% against the in-runtime bf16; mean TV .0717. (Deployed: 8.6-9.2%.)
- **B1.3 first-order predictor**: per request, dm_pred = sum over GEMMs and rows of <d margin/dY, local error on the dense trajectory>.
  rms dm_pred 0.46 vs rms actual dm 0.48 (logit units), but corr(dm_pred, dm) = 0.12 and flip AUC 0.46: it predicts the SIZE of the
  margin change, not its value. Reason: the quantized run's inputs differ, so each GEMM's rounding error is a different realization.
  As a statistical predictor it works: expected flips sum_r Phi(-m0_r / sigma) = 24.0 (global sigma) or 22.2 (per-request layer
  sum of squares) against 26 actual, and it matches per margin bin (17.5 vs 17 in the lowest fifth). => formats can be ranked by the rms
  margin change on DEV, which needs only a quantized forward (q1dm.py, ~1.5 min per config).
- **B2 coherence** (MEASURED): kappa = E[(sum_t a_t)^2] / E[sum_t a_t^2] per GEMM, rows of one role: state rows 1.10 (layers 0-8), 1.28
  (9-12), 0.95 (13-23); question rows 1.03 / 1.11 / 0.39. kappa = 1 means independent errors. Round-to-nearest errors do NOT accumulate
  like a bias across rows, so dither/stochastic rounding (which adds variance) cannot reduce flips. B2's premise is refuted at first order.
- **B6 certificate** (MEASURED, half the requests calibrate alpha = max |dm|/stat, the other half test): sensitivity-weighted error energy
  (top-64 G_out eigenpairs per GEMM, errors the runtime knows): corr(|dm|, sqrt(stat)) = -0.04, certifies 18% of test requests
  (0 residual flips of 10); raw error energy: 17%; even the oracle first-order |dm_pred| (needs a backward pass): 2.7%.
  The margin alone (J15's rule) certifies 69% with 1 residual flip. The error energy hardly varies between requests; |dm| is set by
  the request's own sensitivity, which the runtime cannot see without a backward pass. B6 as specified does not work at W4A4.
- **Rows**: question rows hold 79% of the first-order decision variance (state 6.4e-3, question 2.4e-2 logit^2). State rows: last 64 rows
  (4% of state rows) hold 21%, last 256 (16%) 45%, first 4 (sinks) 3.5%; oracle top 1% / 5% / 10% / 25% of state rows hold
  35% / 63% / 76% / 90%.
- **B8 token classes** (per-row first-order energy relative to the state-row mean): newline 3.06x (4.7% of rows), email 1.68x, other
  (punctuation/specials) 1.43x, prose 0.97x, json punctuation 0.95x, amount 0.52x, digit 0.32x, id 0.24x, date 0.22x, phone 0.15x,
  json key 0.08x. Value-bearing tokens are LESS sensitive than average to rounding; B8's premise is refuted at first order.
- Layer share of first-order variance: layers 0-12 3-9% each (sum 85%), 13-22 < 2% each, layer 23 9.8%.

### 21:55 PDT aligned with Q2 FORMATS.md v1 (21:08)
- My emulation (q1fmt.Fmt) follows Q2's F2 convention: f = fp32(fp32(acc * s_t) * sw_n), extra terms added in fp32, bf16 output.
  Fixed: activation scale now s = fp32(fp32(amax / qmax) * clip) with IEEE division by a tensor (FORMATS sec 0 pitfall); runs before
  21:55 (fo w4a4, fo w4q8) used the scalar division (1-ulp scale differences on some rows; immaterial to first-order statistics).
- Known differences (to tell Q2/Q3; I cannot write to their folders):
  - B5 (FORMATS 10.1): my plans subtract the calibration mean from ALL columns (producer subtracts a per-column constant; bias = mu W over
    all K), not only from the 0-bit columns; and spread each coded slice with a Haar rotation (a deployable kernel needs a fast
    transform there, e.g. block Hadamard, or the rotation folded offline for residual readers with a shared basis).
  - B2 SR uses torch.rand, not Q2's counter hash (same distribution, not bit-exact).
  - B9: my tile mask is per output-column block only; Q2's varies per 128-row block too (more decorrelation).
  - The H1/H2 Hadamard of the activations is fp32 in my emulation (H2 uses fp16 dots).

### 22:00 PDT first-order pass, row role w4q8 (state rows W4A4, question rows W8A8), MEASURED, same 300 DEV requests
- flips vs in-runtime bf16 7/300 = 2.33% (W4A4: 26/300); mean TV .0294 (W4A4 .0717); rms margin change 0.216 (W4A4 0.484) logit units.
- First-order variance: question rows 1.2e-4 (W4A4: 2.4e-2, 200x lower), state rows 6.35e-3 (unchanged) -> state rows hold 98%.
  By layer (state rows): layer 0 18%, layers 1-11 5-10% each, 12-23 < 1% each.
- Predictor corr 0.25 (W4A4 0.12), flip AUC 0.75: closer to linear at the smaller error, still not a per-request predictor.
- B6 at w4q8: corr(|dm|, sqrt(cert)) = 0.06; certificate certifies 42% of test requests (0 residual flips), margin-only rule 83%
  (1 residual flip of 5). B6 still loses to the margin alone.
- **B10 (MEASURED, input side, option rows, layers 16-23, Fisher-weighted)**: common-mode (mean over options) trace is 0.1-2% of the
  differential trace (e.g. 22.Wd 0.0054 vs 4.35; 23.Wgu 0.074 vs 3.84); differential subspace r90 = 1-2, r99 = 3-10 directions;
  common r99 = 4-65. Confirms the softmax shift cancellation; the option rows are a handful of rows already run at int8 in the row-role
  format, so B10 saves no measurable MACs.
- B2 coherence unchanged (state rows kappa 1.10 / 1.29 / 0.95 by layer band).

### 22:25 PDT fast screen q1dm (MEASURED, 300 DEV requests, vs in-runtime bf16; rms dm = rms change of the dense top-2 logit margin)
| config | flips /300 | rms dm | mean TV | predicted flips sum Phi(-m0/rms) |
|---|---|---|---|---|
| b8 (W8A8 GPTQ8, 8 GEMMs bf16) | 0 | 0.030 | .0039 | 1.0 |
| W8A8 all 96 (fo pass, 100 req) | 0/100 | 0.038 | .0051 | - |
| W4A4 | 26 | 0.479 | .0739 | 24.9 |
| w4q8 (state W4A4, question W8A8) | 9 | 0.200 | .0267 | 10.3 |
| nsq8: + noise shaping along rows, c = 1 | 13 | 0.276 | .0373 | 14.5 |
| ns5q8: + noise shaping, c = 0.5 | 11 | 0.219 | .0268 | 11.4 |
- **New idea tested and refuted: sigma-delta noise shaping of activation rounding along the token axis** (error of row t fed into row t+1
  before rounding, Triton kernel q1ns.py, exact vs a loop reference). It cuts the 64-row sum of rounding errors 7.4x on random data but
  raises the per-element error 1.41x; on hobson it raises rms dm 0.200 -> 0.276 (c=1) / 0.219 (c=.5): the decision gradient is not smooth
  enough from token to token for the shaped noise to cancel.
- Scale of the target: W8A8-class fidelity is rms dm 0.03-0.04; w4q8 is 0.20, so state-row error must fall ~5x in std (~25x variance).
- W8A8 first-order pass: rms pred 0.032 vs rms dm 0.038 but corr 0.29 (re-rolled bf16 rounding floor dominates at 8 bits;
  H1 measured that floor). The predictor ranks formats by magnitude, not individual decisions.

### 22:58 PDT screen continued (MEASURED, DEV 300)
| config | flips | rms dm | TV | state-row MAC shares |
|---|---|---|---|---|
| w4q8d (GPTQ on decision-weighted Hessian) | 10 | 0.2025 | .0265 | int4 1.0 |
| w4bq (question rows bf16) | 10 | 0.1999 | .0265 | int4 1.0 |
| w4q8_wonly (state rows: only weights 4-bit) | 2 | 0.0846 | .0112 | - |
| w4q8_aonly (state rows: only activations 4-bit) | 11 | 0.1956 | .0257 | - |
| w4q8 + first 4 and last 64 state rows int8 | 10 | 0.1935 | .0251 | int8 rows 4% |
| w4q8 + first 4 and last 256 state rows int8 | 5 | 0.1698 | .0213 | int8 rows 16% |
| **tc4rq8 (B5, per-GEMM basis, shrink .15)** | **4** | **0.1100** | **.0141** | int4 .770, int8 .115, dropped .115 |
| tc4irq8 (B5, channel basis) | 7 | 0.1602 | .0215 | int4 .982, int8 .009 |
- State-row error split: activation rounding 0.196, weight rounding 0.085, together 0.200 (quadrature). Activations are the problem.
- B5 is the first lever that moves the state-row error at int4 speed: 0.200 -> 0.110 (3.3x lower variance) with the MAC time of W4A4
  (0.770 + 2 x 0.115 = 1.0 int4 units). The planner's model predicted a median 5.5x per GEMM.
- Screen crashed at resq128q8 (bf16 Wo/Wd input into an fp32 matmul in the split path); fixed (Fmt casts inputs to fp32), re-queued.
- Building the deployable B5: one shared offline basis for all residual readers (Win/Wgu) replacing R1, channel basis for Wo/Wd,
  block Hadamard 64 inside slices (plans b4shr / b45shr).

### 23:30 PDT B5 variants (MEASURED, DEV 300) and Q2's deployed-kernel cross-check
| config | flips | rms dm | TV | state-row MAC shares | MAC time vs W4A4 (int8 = 2 units) |
|---|---|---|---|---|---|
| tc45rq8 (B5 at 4.5 bits) | 2 | 0.0882 | .0117 | int4 .818, int8 .152, dropped .030 | 1.12 |
| tc4rq8_l0 (B5 4 bits + layer-0 state rows W8A8) | 4 | 0.0958 | .0131 | int4 .740, int8 .151 | 1.04 |
| tc45rq8_l0 | 6 | 0.0906 | .0116 | int4 .787, int8 .186 | 1.16 |
| w4q8_l0 (row role + layer-0 state rows W8A8) | 6 | 0.1754 | .0236 | int4 .957, int8 .043 | 1.04 |
- Q2 (their NOTES 22:09-22:17, deployed kernels, all 3227 questions): row role w4q8 = REAL flips 3.88%, TV .0320, CF retention .835,
  CF-probe .762 -> consistent with my DEV screen (w4q8 rms dm 0.200 = 6.7x b8's 0.030). Their b8 draw: 1.11% REAL flips, McNemar vs
  bf16 p .039. Q2 kernel times at M=1125 (sum of 96 GEMMs vs b8): W4A4 CUTLASS .535, q2 s4 .614, B5 slices (int8-256 + int4-1280) .617,
  ResQ int8-128 + int4 .685, group g64 .891, row role .801; e2e row role 26.34 ms = 0.727x b8 at T=1000/1q.
- The baseline screen crashed again (GPTQ Cholesky on the projected rank-deficient Hessian of the K=6144 split); fixed (symmetrize,
  retry with 10% damping), re-queued after the row-router test.
- New test queued (q1router.py): which state rows to run at int8 (Q2's B8 partition kernel makes extra int8 rows nearly free once the
  int8 weight copy is streamed): oracle by backward-pass sensitivity vs question->state attention (layers 3, 7), recency, random.

### 00:20 PDT deployable B5 and the row router (MEASURED, DEV)
| B5 variant (state rows, question rows W8A8) | flips /300 | rms dm | state-row MAC shares |
|---|---|---|---|
| per-GEMM basis + Haar in slices (tc4rq8; online K x K transforms, not deployable) | 4 | 0.110 | int4 .770, int8 .115, drop .115 |
| shared residual basis (Win/Wgu) + channel basis (Wo/Wd), Haar in slices (tc4shHq8) | 11 | 0.164 | int4 .939, int8 .031 |
| same, block Hadamard 64 in slices (tc4shq8) | 13 | 0.259 | int4 .939, int8 .031 |
| same at 4.5 bits, Hadamard 64 (tc45shq8) | 6 | 0.177 | int4 .873, int8 .124 |
| channel basis everywhere, Haar (tc4irq8) | 7 | 0.160 | int4 .982, int8 .009 |
| channel basis everywhere, Hadamard 64 (tc4ihq8) | 14 | 0.302 | int4 .982, int8 .009 |
| reference: row role w4q8 | 9 | 0.200 | int4 1.0 |
- The spreading rotation inside each slice decides it: Haar (dense) keeps the gain, 64-point block Hadamards lose it (outlier channels
  are not spread). A fast deployable version needs full-slice Hadamards (orders 2^a or 12*2^a): planner option --rot hadfull, queued.
- **Row router (q1router, 130 of 150 DEV requests, base w4q8; 4 sink rows + f of the state rows at W8A8):**
  rms dm: none 0.204 | oracle (backward-pass first-order int4 energy per row) 5% 0.131, 10% 0.122, 20% 0.086 |
  question->state attention at layer 7 (dense pass) 5% 0.161, 10% 0.140, 20% 0.095 | layer 3: 0.170 / 0.172 / 0.145 |
  most recent rows 0.198 / 0.203 / 0.186 | random 0.190 / 0.214 / 0.173.
  Layer-7 question attention nearly matches the oracle at 20% (variance 4.6x vs 5.7x lower). But it is only known after layer 7, and
  layers 0-7 hold ~73% of the state-row variance, so using it needs a pre-pass (8/24 of an int4 forward) -> GEMM time ~0.77x b8
  (arithmetic), over the 0.65x bar. Recency and token class (B8) are weak selectors.

### 00:35 PDT router final + first baselines (MEASURED, DEV)
- Router final (150 requests, base w4q8, rms dm; none 0.202): oracle 5/10/20% = 0.128 / 0.118 / 0.085 (captures 62 / 75 / 86% of the
  first-order energy; 86% predicts 2.6x lower std, measured 2.4x); layer-7 question attention 0.158 / 0.137 / 0.092; layer-3 0.166 /
  0.165 / 0.140; most recent 0.192 / 0.193 / 0.176; random 0.188 / 0.205 / 0.167.
- resq128q8 (ResQ-style: top-128 PCA directions int8, rest int4; question rows int8): 5/300 flips, rms dm 0.154, TV .0190, state-row
  int4 share .947 (+ 5.5% extra bf16 MACs for the projections in emulation form).
- sens128q8 (B1 variant: top-128 G_in directions int8 instead of PCA, same work): 11/300, rms dm 0.204 -> no gain over w4q8 (0.200).
  At equal work the variance subspace beats the decision-sensitivity subspace, as B1.1 predicted (128 directions hold 15-50% of the
  early-layer sensitivity).

### 01:20 PDT more baselines (MEASURED, DEV 300); dense full-kit eval running in parallel
- g64q8 (group-scaled int4 g=64 on state rows, group GPTQ without act-order; question rows int8): 11/300, rms dm 0.178, TV .0225
  (w4q8 0.200). Q2's kernel: GEMM time .891x b8 at M=1125.
- srq8 (B2: stochastic rounding on state rows, question rows int8): 20/300, rms dm 0.322 (RTN 0.200): worse, as kappa ~ 1 predicted
  (SR adds variance, there is no coherent bias to remove). B2 refuted directly.

### 01:45 PDT (MEASURED)
- corr128q8 (B1.2 exact low-rank correction of state-row int4 activation error in the top-128 G_in(state) subspace, bf16 operands;
  question rows int8): 7/300, rms dm 0.177 (w4q8 0.200): 22% lower variance for 8.1% extra bf16 MACs (Q2: tail r=128 .717x b8 GEMM
  time without the Z GEMM, .874 with it).
- b9f75q8 (B9: 75% of K tiles kept per output tile, rescaled 1/f, state rows): 81/300 flips, rms dm 1.50, TV .228. Refuted: unbiased
  tile sampling adds ~(1-f)/f relative variance per output, ~10x the int4 rounding variance, and nothing averages it away (kappa ~ 1).
- Full-kit dense (bf16 runtime emulation, 3227 questions, 1659 s): REAL flips 6/1083 = 0.55% vs hobson, TV .0032, CF retention 1.000,
  CF-probe .952, JB-hard .538 (McNemar vs hobson p .5), REAL-label .783, LONG flips 0.42%. Same floor as H1 (5/1083) and H2 (4/1083).
  NB the CF-probe bar (.95) is at the bf16 runtime's own level.

### 00:25 PDT B3 first measurement, deployable B5, full-kit evals running (MEASURED)
- B3 (q1b3.py; k-means codebooks on 80% of the q1cal state-row samples, tested on the other 20%; per-token int4 clip .9 after the
  GEMM's rotation): plain rotated int4 relative error 12.2-14.3% (median by GEMM type and layer band). Residual coding with Kc = 1
  (mean only) / 64 / 256: error ratio vs plain 0.75-0.98 / 0.63-0.90 / 0.59-0.85 (Wo layers 9-23 best: 0.590; Wd layers 0-8 worst:
  0.845). Residual crest after rotation unchanged (3.6); the gain is the residual's smaller norm (0.58-0.84 of |x|). Search cost Kc/N of
  the GEMM MACs (int8): 2.1% (Wgu) to 12.5% (Wo, Wd) at Kc = 256. Codebooks for Kc 64/256 being built; DEV screen queued.
- Deployable B5 (tc4shFq8: shared residual basis + channel basis for Wo/Wd + full-slice Kronecker Hadamard inside slices, orders 2^a or
  12*2^a): 8/300, rms dm 0.180 (w4q8 0.200; same with Haar slices 0.164; per-GEMM bases 0.110). At 4.5 bits (tc45shFq8): 11/300, 0.229
  (the Hadamard-order constraint forces poor slice sizes at that budget).
- w4q8w4 (row role with ONE weight copy: question rows W4A8): 11/300, rms dm 0.246 (two copies 0.200).
- Full-kit evals (3227 questions) running: tc45rq8 (0.66 s/question), w4q8; then tc4shFq8 + resq128q8; then tc4rq8 + w4a4.
- best_formats.json updated with every DEV row (flips, rms dm, TV, MAC shares, Q2 GEMM times).

### 01:45 PDT FULL KIT tc45rq8 (MEASURED, emulation, 3227 questions, vs hobson refs; paired vs my bf16 emulation)
- tc45rq8 (B5 per-GEMM decision-weighted transform coding of state rows at 4.5 bits + question rows W8A8; not deployable as is):
  REAL flips 14/1083 = 1.29% (bar <= 0.70%; bf16 6/1083); paired vs bf16 12 lost / 4 gained, p .077; TV .0145 (bf16 .0032; b8 .0048
  per brief); CF retention .890 (bar .99); CF-probe retention .914 (bar .95; bf16 .952); JB-hard .531, McNemar vs hobson p 1.0;
  REAL-label .790; LONG flips 1.49%. FAILS flips, CF, CF-probe. Q2's deployed row role w4q8: 3.88% / CF .835 / CF-probe .762.
- Eval cost: tc45rq8 2,900 s for 3,227 questions (LONG and CF states are 3-8k tokens).
- tc45rq8 CF losses by kind (pairs hobson tracks): human_insert 10/54 lost, amount_insert 1/9, insists_3v4 1/1, wrapup_replace 0/45.
  CF-probe: id_match_distract 3/16, amount_vs_limit_distract 3/9, status_equal_distract 1/11, amount_vs_limit 1/14, date_order 1/3,
  status_equal 0/31, id_match 0/20. The losses are single inserted sentences deep in long states and distractor records.

### 02:20 PDT FULL KIT w4q8 (MEASURED, emulation)
- w4q8 (row role: state rows W4A4, question rows W8A8): REAL flips 40/1083 = 3.69% (Q2 deployed kernels: 3.88%); paired vs bf16 38 lost /
  4 gained (p < .001); TV .0325; CF retention .798; CF-probe .676; JB-hard .585 (McNemar vs hobson p .057); REAL-label .7775; LONG 3.18%.
- B5 per-GEMM (tc45rq8) vs row role: REAL flips 1.29% vs 3.69% (2.9x fewer), TV .0145 vs .0325, CF .890 vs .798, CF-probe .914 vs .676.

### 02:55 PDT status
- B3 decision screen was wrong (94/300 flips): bug, the exact c W table was added after converting Wo/Wd outputs out of the R1-rotated
  basis. Fixed (table added before unrotation); re-queued as proto256fq8 / proto64fq8 after the running full-kit evals.
- Running: full kit tc4shFq8 (deployable B5) then resq128q8 (ResQ baseline). Then B3 screen, router on B5 (tc45rq8 base), full kit w4a4.
- DRAFT_REPORT.md drafted (DEV table, speed from Q2, projections, next step); full-kit table to be filled.

### 03:25 PDT FULL KIT tc4shFq8 (deployable B5; MEASURED, emulation)
- REAL flips 34/1083 = 3.14%; paired vs bf16 30 lost / 2 gained (p < .001); TV .0275; CF .872; CF-probe .800; JB-hard .538 (p .79);
  REAL-label .7825; LONG 2.34%. Better than row role (3.69% / .798 / .676) on every metric, far from the bar; the non-deployable per-GEMM
  version at 4.5 bits gets 1.29% / .890 / .914.

### 02:50 PDT FULL KIT resq128q8 (ResQ-style baseline; MEASURED, emulation)
- REAL flips 26/1083 = 2.40%; paired vs bf16 23 lost / 3 gained (p < .001); TV .0193; CF .945; CF-probe .848; JB-hard .531 (p 1.0);
  REAL-label .780; LONG 3.18%. State-row MACs: int4 .947, int8 .053 (deployable; Q2 split kernel .685x b8 GEMM time w/o row role).
- At about equal 4-bit share, ResQ (PCA subspace) beats my deployable B5 (tc4shFq8: 3.14%, CF .872, CF-probe .800); only the
  non-deployable per-GEMM B5 at 4.5 bits does better on REAL flips and CF-probe (1.29%, .914), and ResQ is better on CF (.945 vs .890).
- B3 decision screen (fixed; MEASURED, DEV 300, question rows int8): proto256fq8 8/300, rms dm 0.152, TV .0194 (w4q8 0.200, ResQ
  0.154); proto64fq8 6/300, rms dm 0.179. All state-row GEMM MACs stay int4; nearest-prototype search costs Kc*K MACs per row
  (5.5% / 1.4% of GEMM MACs at Kc 256 / 64). GPTQ codes here use the Hessian of x, not of the residual (a refinement left untried).
  Full kit of proto256fq8 queued after w4a4.

### 03:10 PDT row router on top of B5 (MEASURED, 150 DEV requests, base tc45rq8)
- rms dm: none 0.086 | oracle 10% / 20% of state rows W8A8: 0.051 / 0.045 | layer-7 question attention 10% / 20%: 0.063 / 0.050.
  Reference: b8 0.030, W8A8 all-96 0.038, w4q8 0.200. This stack (per-GEMM B5 at 4.5 bits + question rows int8 + 20% of state rows int8
  chosen by layer-7 question attention) is the closest any combination gets (1.3x W8A8's rms, 1.7x b8's), but it needs per-GEMM dense
  transforms and an 8-layer pre-pass for the attention, so it is not deployable at the target speed. Full kit not run (no time).
- Running: full kit w4a4, proto256fq8, tc4rq8 (box TTL 05:36 PDT).

### 03:30 PDT FULL KIT w4a4 (MEASURED, emulation)
- W4A4 all rows: REAL flips 95/1083 = 8.77% (deployed H2/J15: 8.6-9.2%); paired vs bf16 93 lost / 4 gained; TV .0818; CF .651; CF-probe
  .648; JB-hard .569 (p .21); REAL-label .7575; LONG 5.73%.

### 04:03 PDT FULL KIT proto256fq8 (B3; MEASURED, emulation)
- B3 (256 k-means prototypes per GEMM input on state rows, residual int4, exact c W table; question rows int8): REAL flips 41/1083 =
  3.79%; paired vs bf16 37 lost / 2 gained; TV .0234; CF .927; CF-probe .829; JB-hard .508 (p .815); REAL-label .795; LONG 4.25%.
  Better than row role on TV / CF / CF-probe, no better on REAL flips; worse than ResQ on flips. Eval: 3,604 s.
- tc4rq8 full kit running (may not finish before the box TTL at 05:36).

### 04:15 PDT wrap-up
- Results copied to ~/decider2/q1/res: preds/ (dense, base = tc45rq8 + w4q8, b2 = tc4shFq8 + resq128q8, b3 = w4a4 + proto256fq8 [+ tc4rq8
  if it finishes]), dm/ (every DEV screen; proto*q8 files without 'f' are the buggy B3 run, renamed *_BUGGY_superseded), fo/ (first-order
  passes w4a4, w4q8, w8a8), b5/ (planner summaries), spec/ (B1.1), router_*.json, res_b3.json, cal/summary.json, logs/.
- Box-only (large): ~/work/q1/spec/L*.pt (eigenvectors), cal/ (S_, X_ stats), hd/ (decision-weighted Hessians, neuron saliency for Q5),
  b5/*/P_*.pt (plans), codes/ (GPTQ codes), proto/ (codebooks). The box terminates at 05:36 PDT; I did not reset its timer.
- DRAFT_REPORT.md written (1,289 words excluding table syntax). No permission denials this session.

### 04:52 PDT FULL KIT tc4rq8 (B5 per-GEMM, 4 bits; MEASURED, emulation) - last run
- REAL flips 29/1083 = 2.68%; paired vs bf16 26 lost / 3 gained; TV .0172; CF .936; CF-probe .810; JB-hard .492 (p .42); REAL-label .790;
  LONG 2.76%. Fails. All results copied off the box; report final.

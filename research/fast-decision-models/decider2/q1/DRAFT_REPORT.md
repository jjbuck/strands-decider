# Q1 report: decision sensitivity and 4-bit formats for hobson-v19 (box q1, exact emulation)

**Bottom line: no format I tested meets the fidelity bar.**

Tags: [M] measured, [A] arithmetic from measured numbers. Definitions:
- **DEV:** 300 train-split requests from tau tasks disjoint from the calibration set, scored against my bf16 runtime emulation.
- **rms dm:** rms change of the bf16 top-2 logit margin, in logit units. On DEV, b8 scores 0/300 flips and rms dm 0.030, and W8A8 with all 96 GEMMs scores 0.038.
- **Full kit:** all 3,227 evalkit questions against hobson's references.
- **Arithmetic:** Q2's FORMATS.md F2, on H1's rotations and GPTQ codes.

## 1. What I built
In `~/decider2/q1/code` (APIs in its README): a differentiable hobson with per-GEMM hooks (`q1lib`), the spectra (`q1spec`), a first-order pass for B1.3, B2, B6, B8, B10 (`q1fo`), all formats (`q1fmt`), the B5 planner, a row router, a Triton noise-shaping quantizer, a 1.5-minute screen (`q1dm`) and full-kit scoring. Published: `sensitivity.json`, `best_formats.json`, `NOTES.md`.

## 2. B1.1, the decisive measurement [M]
For all 96 GEMMs: the eigen-spectrum of G = E[gᵀg], g = gradient of the decision KL (pointer-head softmax Fisher) wrt the GEMM's output and input, state and question rows separate; 256 requests (437k tokens), 64 held out from other tau tasks.

**Where the state matters, sensitivity is not low-rank.**
- In layers 0–8, state rows hold 18–39% of the trace.
- Input-side r90 is 349–1,257 and r99 is 965–1,875 directions, of K = 2,048.
- Held out, 64 directions capture only 22–34% at layer 0.

**It is low-rank only for question rows in layers 13–23.**
- Question rows hold 96–100% of the trace there.
- Wo/Wd/Wgu r90 is 2–63 (2–5 in layers 18–23).
- These subspaces transfer: 8 directions capture 98–99.6% held out.

**State sensitivity sits in rows, not directions.** The top 1% of state rows hold 24–73% of state-row energy.

So B1's low-rank correction cannot carry the state rows; question rows are cheaper in int8.

## 3. Where the 4-bit error comes from [M, DEV]
- **W4A4:** 26/300 flips, rms dm 0.479. Question rows hold 79% of the first-order variance.
- **Row role** (question rows W8A8, state rows W4A4): 9/300, rms dm 0.200.
- **What remains is state-row activation rounding.** A4 alone gives 0.196, W4 alone 0.085, both 0.200.
- **B1.3 predictor:** right size (rms 0.46 vs 0.48), wrong value (correlation 0.12 at W4A4, 0.29 at W8A8). Statistically, Σ Φ(−m0/σ) predicts 24 flips vs 26 actual, so rms dm is the screen.
- **B2:** per-row first-order terms add independently (κ = 0.95–1.29; 1 = independent), so dither has no bias to remove.

## 4. DEV screen [M]
State-row MAC shares are int4 / int8 / dropped. GEMM time is Q2's measured ratio to b8 at M = 1,125.

| format (question rows W8A8 unless noted) | flips | rms dm | state-row MACs | GEMM time / b8 |
|---|---|---|---|---|
| W4A4, all rows | 26 | 0.479 | 1/0/0 | .535 |
| **row role** (w4q8) | 9 | 0.200 | 1/0/0 | .801 |
| ResQ-style: PCA-128 int8 + int4 | 5 | 0.154 | .947/.053/0 | .685, excluding row role |
| B1: G_in top-128 int8 + int4 | 11 | 0.204 | .947/.053/0 | – |
| B1.2: exact correction, r = 128 | 7 | 0.177 | 1/0/0, +8.1% bf16 | .717–.874, excluding row role |
| group-scaled int4, g = 64 | 11 | 0.178 | 1/0/0 | .891, excluding row role |
| B2: stochastic rounding | 20 | 0.322 | 1/0/0 | – |
| B9: 75% of K tiles kept | 81 | 1.504 | 1/0/0 | .531, excluding row role |
| B3: 256 prototypes | 8 | 0.152 | 1/0/0, +5.5% search | – |
| B8: last 256 state rows + 4 sinks int8 | 5 | 0.170 | 16% of rows int8 | – |
| new: noise shaping along rows | 11–13 | 0.219–0.276 | 1/0/0 | – |
| **B5, per-GEMM basis, 4 bits** (not deployable) | 4 | 0.110 | .770/.115/.115 | – |
| **B5, per-GEMM basis, 4.5 bits** (not deployable) | 2 | 0.088 | .818/.152/.030 | – |
| B5, deployable: shared basis, full-slice Hadamard | 8 | 0.180 | .930/.035/.035 | – |
| B5 4.5 bits + 20% of state rows int8 by layer-7 question attention | – | 0.050 (n = 150) | + 20% of rows int8 | – |

- **B5 is the only lever at int4 MAC time, and only with dense per-GEMM bases** (online K×K transforms for Wo/Wd). With a shared basis most of the gain goes: 0.164 (Haar slices), 0.180 (full-slice Hadamard), 0.259 (64-point blocks).
- **Row routing needs a pre-pass.** At 20% of state rows the oracle gives 0.085 on row role (0.045 on B5), and layer-7 question attention nearly matches it; but layers 0–7 hold ~73% of the state-row variance, so [A] the 8-layer pre-pass puts GEMM time near 0.77x b8.
- **B6:** the known error energy correlates with |dm| at −0.04 to 0.06 and certifies 18–42% of requests, against 69–83% for the plain margin; |dm| depends on the request's own sensitivity, which needs a backward pass.
- **B8:** value tokens are less sensitive than average (amounts 0.52x, ids 0.24x); newlines are 3.06x.
- **B10:** option rows' common-mode sensitivity is 0.1–2% of the differential (r99 = 3–10), but they are already int8, so no MACs are saved.
- **Also tried:** a one-weight-copy row role (0.246) and GPTQ on a decision-weighted Hessian (0.203); neither helped.

## 5. Full kit against the bar [M, emulation]
Bar: REAL ≤ 0.70%, McNemar against bf16 p > .05, CF retention ≥ .99, CF-probe retention ≥ .95, JB-hard McNemar not significant, REAL-label ≥ .78. 4-bit share = share of all GEMM MACs in int4, from the state-row shares and DEV's row mix (88% state rows, 1 question).

| config | 4-bit share [A] | REAL flips | lost / gained vs bf16 (p) | TV | CF | CF-probe | JB-hard (p) | REAL-label |
|---|---|---|---|---|---|---|---|---|
| bf16 runtime (floor) | 0 | 0.55% | – | .0032 | 1.000 | .952 | .538 (.5) | .783 |
| row role (w4q8) | .88 | 3.69% | 38 / 4 (<.001) | .0325 | .798 | .676 | .585 (.057) | .778 |
| ResQ-style (r = 128) | .83 | 2.40% | 23 / 3 (<.001) | .0193 | .945 | .848 | .531 (1.0) | .780 |
| B5 deployable | .82 | 3.14% | 30 / 2 (<.001) | .0275 | .872 | .800 | .538 (.79) | .783 |
| **B5 per-GEMM, 4.5 bits** | .72 | **1.29%** | 12 / 4 (.077) | .0145 | .890 | .914 | .531 (1.0) | .790 |
| B5 per-GEMM, 4 bits (W4A4 MAC time) | .68 | 2.68% | 26 / 3 (<.001) | .0172 | .936 | .810 | .492 (.42) | .790 |
| W4A4, all rows | 1.00 | 8.77% | 93 / 4 (<.001) | .0818 | .651 | .648 | .569 (.21) | .758 |
| B3, 256 prototypes | .88 (+5% search) | 3.79% | 37 / 2 (<.001) | .0234 | .927 | .829 | .508 (.82) | .795 |

None passes. The best (B5 per-GEMM) loses CF pairs with one inserted request for a human in a long state (10 of 54) and CF-probe distractor pairs (6 of 25). Q2's deployed row role gives 3.88% / .835 / .762, matching my emulation.

## 6. Speed against W8A8-b8 [M by Q2, A10G]
Row role end to end (CUTLASS int4 for state rows, int8 question rows on a side stream), median ms:

| request | row role | b8 | ratio |
|---|---|---|---|
| T = 1000, 1 question | 26.34 | 36.21 | 0.727x (bar 0.75x) |
| T = 4000 | 88.75 | 142.6 | 0.62x |
| T = 256 / 64 | 11.61 / 9.92 | 14.38 / 8.58 | 0.81x / 1.16x |

- **GEMM time at M = 1,125:** .801 against the ≤ .65 bar. The second (int8) weight copy costs about 2.3 ms.
- Only all-int4 formats meet the GEMM bar (W4A4 .535), and they all fail accuracy. B5's slices cost .617 in Q2's kernel; no B5 format has been timed end to end. With 15 questions row role is 0.78–1.02x b8.

## 7. Projections [A]
Carrying the A10G ratio over (int4 is 2x int8 on GeForce too): J15's b8 at T = 1000 / 1q is 19.9 / 11.6 / 7.9 ms on the 3090 / 4090 / 5090, so row role ≈ 14.5 / 8.5 / 5.8 ms. Below ~330 rows the 4090 is weight-bound, so the second weight copy costs more there; the 5090's FP4 is a different error model.

## 8. The single most valuable next step
**Quantization-aware distillation in the row-role format, seeded with B5's per-GEMM transforms recast as Kronecker factors, stopping on DEV rms dm ≤ 0.04.**

Row role is already 0.727x b8 end to end; Q3 is starting w4q8 distillation on q3b.
- **Formats alone stop at 1.29% flips;** the rest is state-row activation rounding spread over ~1,000 directions, which only training can change.
- **B5 shows decision-weighted transforms halve the rms margin change.** FlatQuant (arXiv 2410.09426) fuses per-layer Kronecker-factored transforms into one kernel; [A] about 1–8% of a GEMM's MACs (K = 2,048 as 32 x 64), which would make B5 deployable.
- **The DEV screen tracks flips** and gives the stopping signal.

Conduct: no permission denials; box q1's timer was not reset (TTL 05:36 PDT). Large artifacts (eigenvectors, plans, codes, Hessians) are on the box only; every JSON result is in `~/decider2/q1/res/`.

# Q2 draft report: kernels for 4-bit decision GEMMs on the A10G (boxes q2, q2b)

Everything is in `~/decider2/q2/`: `code/` (with a `README`), `NOTES.md`, `FORMATS.md`, `res/` and `preds/`.

**[M]** measured on the A10G with exclusive GPU use; **[A]** arithmetic from measurements. Latency: CUDA-graph replay, fresh state ids, 20 warm reps, median ms (p95 within 0.5 ms); hobson layout, 1 question = details_match (125 tokens), 15 questions = 3,720 tokens. **b8** = H2's deployed W8A8-GPTQ-b8. **row-role** = state rows W4A4, question rows W8A8. **kNNrr** = H6's NN most sensitive GEMMs W8A8 on all rows, the other 96 − NN row-role.

## 0. Update, 2026-10-07: k64rr with J15's layer-16 exit passes fidelity and both 1-question speed bars (box q2b)

**What was done.**
- **Rebuild.** I rebuilt the kernels, QRT2C and the H1-recipe GPTQ codes. k64rr reproduced its q2-box scores exactly [M].
- **Exit head.** QRT2 now runs a layer range from a saved residual. Splitting at layer 16 is bit-identical to the full pass [M].
  - Recipe (J15's): hobson's head plus a rank-512 adapter, trained by KL to k64rr's final distribution on EXIT (4,205 train-split questions) and early-stopped on DEV (1,736).
  - Result: DEV KL .00044, DEV agreement .990 [M]. J15's b8 head had .0004 / .994.
- **Threshold.** τ = .054 gives zero DEV residual changes, with 96.1% of DEV exiting. It was applied unchanged to eval.
- **Latency setup.** Two CUDA graphs with one host read of the margin between them: layers 0–15 plus the exit head, then layers 16–23 plus the hobson head. Torch does not expose conditional graph nodes, so a single graph cannot skip layers 16–23 at run time.

**Fidelity, all 3,227 questions [M].**

| config | REAL flips | vs bf16 runtime (model-only/base-only, p) | TV | CF ret | CF-probe ret | JB-hard (McNemar vs hobson) | REAL-label |
|---|---|---|---|---|---|---|---|
| b8 | 1.11% | 10/2, p .039 | .0048 | 1.000 | .962 | .538 (0/2) | .792 |
| k64rr | 0.65% | 6/3, p .51 | .0058 | 1.000 | .962 | .523 (0/0) | .790 |
| **k64rr + exit16** | **0.65%** | **6/3, p .51** | **.0097** | **1.000** | **.952** | **.531 (0/1)** | **.790** |

- **Exit share:** 94.2% of eval questions exit (CF-probe 89.5%).
- **Decisions changed by the exits:** 4 of 3,227 against full k64rr, none on REAL or LONG.
- **Margin to the bar:** CF-probe passes by one pair.

**Latency [M].**

| | b8 | k64rr | b8 + exit16 | **k64rr + exit16** |
|---|---|---|---|---|
| J15's 120 real requests, same box: mean / median / p95, ms | 79.7 / 63.9 / 217.3 | 69.3 / 55.6 / 186.6 | 54.9 / 42.7 / 146.8 | **50.6 / 40.0 / 137.7 (0.635x)** |
| T = 1000, 1 question: exit path / suffix, ms | 24.33 / 12.27 | 23.20 / 9.80 | – | 23.77 expected (0.654x) [A] |

- **b8 + exit16** is timed on this box with J15's recorded exit decisions. It reproduces J15's measurement (54.8 / 42.8 / 146.1 ms).
- **Expected latency** at T = 64 / 256 / 1000 / 4000 with 1 question: 0.722 / 0.686 / 0.654 / 0.633x of b8 [A]. These multiply the measured segment times by the measured exit share.
- **Over all 1,785 REAL+LONG+JevBench eval questions** (interpolated): 0.635x [A].
- **With 15 questions** all branches must exit. 4-question DEV requests all exit 86% of the time, close to independence, so 15 questions should all exit about 55% of the time [A]. That gives 0.76–0.82x.
- **GEMM kernel time** at T = 1000: 16.4 ms against b8's 25.8, or 0.636x [A] (bar 0.65x).
- **Projected at T = 1000** (H2's model): 13.1 / 7.8 / 5.3 ms on RTX 3090 / 4090 / 5090, against b8's 19.9 / 11.6 / 7.9 [A].

**How to read it.**
- Most of the gain is the exit. k64rr's 32 row-role GEMMs are in layers 13–23, and 25 of them are skipped by the exit. Before the exit, k64rr is all-int8 apart from 7 GEMMs, and runs b8's 7 bf16 projections in int8.
- So the stack is 0.922x of b8 + exit16.
- B12 and B13 were not built. Inside the cascade they only touch layers 12–15 and the 6% of requests that continue, worth about 1–2% [A], and each would need the exit head re-validated.

## 1. What I built (2026-10-06)

- **`FORMATS.md` (v2).** The exact arithmetic of every kernel, with torch references and one pitfall: `x / 7.0` on CUDA multiplies by the reciprocal.
- **`q2gemm.cu`, raw-PTX tensor-core GEMMs for SM86.**
  - int4/int8 main loop with a bf16 or int8 tail (B1, ResQ, B5); group scales; B9 sampled K tiles; split-K.
  - Two problems per launch with row scatter (row-role, B8); SwiGLU, B3-gather and B6 epilogues.
  - Integer paths are bit-exact against torch [M].
- **Triton prologues (`q2pro.py`)** for stochastic rounding, dX, group scales, K slices, row-role, B6 and the B3 search; all bit-exact [M].
- **Runtime (`q2rt.py`).**
  - `QRT2C`: CUTLASS W4A4 state rows, q2gemm int8 question rows on a side stream, CUTLASS for all-int8 GEMMs.
  - `QRT2`: Q2 kernels only, arbitrary row partitions.
  - Both support per-GEMM maps, layer ranges and taps.
- **Moved to Q4:** B11, B13, Strassen and B14.

## 2. Kernel speed: sums over the 96 GEMMs at M = 1,125, relative to b8 (27.5 ms) [M]

| format | GEMM time / b8 |
|---|---|
| all-int8, CUTLASS / W4A4, CUTLASS / W4A4, q2gemm | .968 / .535 / .600 |
| B1 bf16 tail, r = 32 / 64 / 128 / 256 (Z excluded) | .644 / .666 / .717 / .811 |
| group-scaled int4, g = 64 / 128 | .891 / .846 |
| B9 sampled K tiles, f = .75 / .5 | .525 / .433 |
| ResQ (128 int8 columns) / B5 (256 int8 + 1,280 int4, rest skipped) | .685 / .613 |
| row-role (125 question rows int8), one launch | .725 |

**Notes.**
- **q2gemm** runs at 0.85–0.95x CUTLASS's speed.
- **B1's Z** costs another 20–90 µs per GEMM input, so B1 on every GEMM adds 3.5–5.5 ms per request [M/A].
- **Prologue extras:** SR +4% (+32% at K = 6,144), B6 +4% [M].
- **B3's search** costs 0.4–6x the GEMM it feeds [M].
- **B5 as emulated** (a dense per-GEMM basis) costs 46 ms per request in bf16 [M/A]; only shared, block-diagonal or channel bases are affordable.
- **Question-row GEMMs** (125 rows) are latency-bound. W4A8 and split-K were both slower [M].

## 3. End to end, no exit (ms, T = 1000 / 4000 with 1 question; 1000 with 15) [M]

| b8 | W4A4 (fails accuracy) | row-role | k48rr | k56rr | k64rr |
|---|---|---|---|---|---|
| 36.4 / 142.6 / 165.6 | 23.6 / 85.9 / 103.1 | 26.3 / 88.8 / 151.6 | 30.9 / 112.8 / 158.8 | 31.1 / 116.5 / 159.4 | 31.9 / 119.8 / 159.6 |

## 4. Other formats through the deployed kernels (all 3,227 questions) [M]

| format | REAL flips | McNemar vs bf16 runtime | CF ret | CF-probe ret |
|---|---|---|---|---|
| row-role | 3.88% | p < .001 | .835 | .762 |
| k48rr | 1.29% | p .021 | .982 | .914 |
| k48rr + digit rows int8 (B8) | 1.66% | p .001 | .972 | .952 |
| k56rr | 0.65% | p .375 | .991 | .943 |

**Caveat.** The k-maps are prefixes of H6's ranking, and I chose k56 after k48 failed and k64 passed.

## 5. Against the bar

| bar | k64rr + exit16 |
|---|---|
| REAL flips ≤ 0.7%, McNemar vs bf16 runtime n.s. | 0.65%, p .51 [M] |
| CF ret ≥ .99 / CF-probe ret ≥ .95 | 1.000 / .952 [M] |
| JB-hard n.s. / REAL-label ≥ .78 | 0/1 / .790 [M] |
| GEMM time ≤ 0.65x b8, M = 1,125 | 0.636x [A] |
| end to end ≤ 0.75x b8, T = 1000, 1 question | 0.654x [A]; 0.635x on 120 real requests [M] |
| 15 questions | 0.76–0.82x [A]: not met |

Without the exit, k64rr passes fidelity at 0.877x, and row-role reaches 0.724x but fails fidelity [M].

## 6. Projections [A]

Using H2's model at T = 1000 with 1 question (ms, RTX 3090 / 4090 / 5090):

| config | ms | ratio to b8 |
|---|---|---|
| b8 | 19.9 / 11.6 / 7.9 | – |
| k64rr + exit16 | 13.1 / 7.8 / 5.3 | 0.66–0.67x |

## 7. The single most valuable next step

**Make layers 0–12 cheaper at the same fidelity.** With the exit in place, layers 0–15 are about 98% of the GEMM time [A], and k64rr keeps layers 0–12 entirely in int8 because they are the sensitive ones. The two candidates:
- B4-style distillation (Q3) on FORMATS §8.1's row-role arithmetic, restricted to layers 0–12's state rows and scored through QRT2C with the exit;
- B11 2:4 int8 sparsity there (Q4/Q5).

Either one acts directly on the remaining 0.635x.

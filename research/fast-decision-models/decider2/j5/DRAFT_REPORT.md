# J5: millisecond decisions for short inputs with the full 2B

Box j5 (A10G). Code, notes and results are in `~/decider2/j5/`. Tags: **[M]** measured, **[V]** verified by arithmetic, **[S]** speculation.

## Verdict

**The ≤2–3 ms target at ≤256 state tokens on a 3090 cannot be met with the full 2B at hobson's fidelity.**

- **The premise does not hold.** Short decisions are not bound by streaming weights: the question adds about 105 rows.
- **Weight-only 4-bit fails twice.** It costs 2.2–3.2% flips and is no faster on Ampere.
- **What works is int8 compute with short-M kernels.** W8A8-b8 decides the median JevBench input in 7.4 ms on the A10G, indistinguishable from the bf16 runtime (McNemar p .38) [M], against 15.5 ms for the prior fused bf16 runtime.
  - That projects to 4.8 ms on a 3090 (about 13x below Brooker's 60–65 ms, not 20–30x), 3.6 on a 4090 and 2.5 on a 5090 [S].
  - The floors are 1.8, 1.65 and 0.93 ms [V].
- **A megakernel can recover at most about 35%.** The 2 ms band exists only on 4090/5090-class cards, near 150 rows, with a dataflow executor.

## 1. The arithmetic that reframes the hypothesis

**Measured input sizes [M].**
- A decision's rows include its question. JevBench's median input is 143 rows: 39 state tokens plus 103 question tokens.
- 58% of JB-all inputs are ≤256 rows.
- Four questions add about 430 rows.

**3090 arithmetic [V].** Each row costs 2.745 GFLOP. Compute equals weight-streaming time at:

| weights / compute | crossover |
|---|---|
| bf16, fp16 accumulation | 152 rows |
| W4A16 | 38 rows |
| W8A8 | 152 rows |

- **W4A16 at the median decision:** compute-bound, with a 2.8 ms floor against 1.5 ms for W8A8.
- **256 state tokens plus one question (364 rows):** the floors at 100% of peak are 7.0 ms (fp16), 3.5 ms (int8) and 1.8 ms (int4).

So only int4 compute fits under 2–3 ms, and W4A4 fails accuracy (H2: 9% flips).

## 2. What I built [M]

- **Weight-only GPTQ formats** (Hessians over 107 train-split requests): W8, W4 g64 and g128, W3 g128, RTN, and a mix of W4 MLP with W8 mixers. Each is scored on all 3,227 evalkit questions with dequantized weights in the fold runtime, which is exact for weight-only formats.
- **Short-M GEMMs**, all exact to output rounding:
  - a Triton template covering bf16, W8A16, W4A16, W8A8 and W4A8, with fused fold epilogues and deterministic split-K;
  - a multi-M-tile skinny kernel;
  - 16/32-row tiles and serial split-K added to H2's CUTLASS int8 kernels;
  - a per-shape selector inside H2's W8A8-b8 runtime.
- **Measurement:** exclusive GPU, fresh ids, CUDA graph, 20–30 reps, p95 within 0.15 ms of the median; real JevBench questions (1 question = 108 tokens, 4 = 429).

## 3. Accuracy (3,227 questions) [M]

| | bf16 runtime | W8 | W4 g64 | W4 g128 | W4 g128 RTN | W3 g128 | W4 MLP + W8 mixers | **W8A8-b8, J5 kernels** |
|---|---|---|---|---|---|---|---|---|
| flips vs hobson, all / REAL | 0.50 / 0.28% | 0.46 / 0.37% | 3.16 / 2.59% | 3.19 / 2.49% | 10.8 / 9.6% | 7.9 / 6.9% | 2.23 / 1.29% | **0.65 / 0.74%** |
| McNemar vs bf16 runtime (bf16-only / format-only) | – | 9/10, p 1.0 | 94/8 | 95/8 | 338/7 | 247/9 | 62/6 (4-bit: all p<1e-3) | **13/8, p .38** |
| REAL / LONG agree_sd | .997 / .988 | .991 / 1.000 | .945 / .933 | .960 / .927 | .873 / .891 | .873 / .921 | .977 / .958 | .986 / .994 |
| CF / CF-probe pair accuracy (hobson .268 / .328) | .271 / .328 | .268 / .328 | .264 / .350 | .259 / .269 | .200 / .303 | .234 / .259 | .273 / .297 | .276 / .322 |
| CF / CF-probe fgh | 1.000 / .981 | .991 / .971 | .963 / .905 | .927 / .771 | .587 / .648 | .771 / .695 | .991 / .810 | 1.000 / .971 |
| JB-all / JB-hard (hobson .723 / .523) | .736 / .546 | .727 / .531 | .727 / .546 | .745 / .569 | .693 / .477 | .701 / .500 | .745 / .562 | .732 / .538 |
| REAL-label (hobson .785) | .785 | .787 | .765 | .790 | .765 | .775 | .795 | .787 |
| Brier JB-all / REAL-label (hobson .348 / .347) | .348 / .347 | .347 / .346 | .351 / .350 | .351 / .345 | .376 / .366 | .388 / .361 | .348 / .344 | .349 / .347 |

- **W8 weight-only and the deployed W8A8-b8 sit at the noise floor.** For W8A8-b8 I regenerated the GPTQ codes; my kernels match H2's on 3,224 of 3,227 decisions.
- **Every 4-bit weight-only format fails by 3–7x.** GPTQ matters (RTN flips 10.8%), halving the group size does not help, and 8-bit mixers still leave 2.2% flips.
- **W4 g128 also fails the new-architecture bar** on CF-probe pair accuracy (.269 against .328).

## 4. Latency on the A10G (ms, median) [M]

Cells are listed by state length T = 32 / 64 / 128 / 256 / 400.

| runtime | 1 question (140–508 rows) | 4 questions (461–829 rows) |
|---|---|---|
| prior fused bf16 (d1/H2) | 15.5 / 15.8 / 16.1 / 25.5 / 33.0 | 34.4 / 34.7 / 38.7 / 45.3 / 50.2 |
| bf16, short-M GEMMs | 10.7 / 12.1 / 14.7 / 21.2 / 27.7 | 28.2 / 28.9 / 32.8 / 38.7 / 45.6 |
| W4A16 | 11.6 / 14.1 / 17.6 / 25.3 / 33.1 | 31.3 / 34.5 / 37.5 / 45.8 / 53.7 |
| **W8A8-b8, J5 kernels** | **7.38 / 8.01 / 9.34 / 13.3 / 17.0** | **18.1 / 18.5 / 20.0 / 24.8 / 28.8** |
| W4A4 (fails accuracy) | 5.58 / 5.94 / 6.51 / 9.57 / 11.5 | 12.8 / 13.1 / 14.0 / 17.3 / 19.8 |
| W8A8-b8, compiled schema (latency only) | 5.38 / 5.95 / 7.49 / 10.9 / 15.5 | 5.44 / 6.16 / 7.67 / 11.0 / 15.5 |

- **H2's kernels** take 8.15 ms at 140 rows. **Anchors at 1000 tokens:** 56.9 ms (bf16) and 36.1 ms (W8A8-b8), matching the document's 57.0 and 36.3.
- **The prior runtime's 128-row tiles made 140 and 236 rows cost the same.**
- **W4A16 runs at 60–210 GB/s in my kernel at M=16.** Above about 120 rows no kernel can beat bf16 with it [V].
- **The schema layout's gain comes from question rows,** which are 72–93% of a short request. It needs a schema-first model at fidelity, which does not exist yet (H7: .69).

## 5. Where 7.4 ms goes at 140 rows, and the megakernel bound [M]

| component | ms |
|---|---|
| roofline floor (512 GB/s achievable; int8 at 85% of peak) | ~3.2 |
| GEMM steady-state loss (best int8 GEMMs reach ~55% of roofline) | ~1.4 |
| fixed cost of 96 GEMM kernels (5–14 µs each, from n kernels of depth K against one of depth nK) | ~0.6 |
| GDN, 90 launches (about 50 µs per layer for about 5 µs of data) | 0.88 |
| glue, 121 launches | 1.0 |
| attention and other | 0.3 |

- **Launch gaps inside the graph are only 0.02–0.2 ms.** A persistent executor therefore wins only by removing fixed costs and overlapping weight streams: at most about 2.7 ms, or 35% [V].
- **fla's fused_recurrent GDN is slower than chunk** at this length (9.9 against 7.8 ms) [M].
- **Published megakernels are all decode:** Hazy's Llama-1B (78% of H100 bandwidth, 1.5x over SGLang; blog), MPK (arXiv 2512.22219, up to 1.7x), Ada-MK (arXiv 2605.11581, up to 23.6% on an L20) and AutoMegaKernel (arXiv 2606.09682, up to 1.08x on the A10G).

## 6. Projections, W8A8-b8 (ms) [S]

**Method.** GEMMs use the A10G's additive compute/bandwidth split, scaled by each card's int8 rate and bandwidth, so the measured inefficiency is kept. Other kernels are divided by 1.2, 1.5 and 1.8. The floor in brackets is a perfect GEMM dataflow.

| input | A10G [M] | 3090 | 4090 | 5090 |
|---|---|---|---|---|
| 32 + 1q (median JevBench) | 7.4 | 4.8 [1.8] | 3.6 [1.65] | 2.5 [0.93] |
| 256 + 1q | 13.3 | 8.3 [4.1] | 5.6 [1.8] | 4.2 [1.4] |
| 32 + 4q | 18.1 | 11.4 [5.2] | 7.7 [2.3] | 5.9 [1.8] |
| 32 + 4q, schema | 5.4 | 3.7 [1.8] | 3.1 [1.65] | 2.0 [0.93] |

## 7. The decisive next step

**Build an int8 layer-persistent executor for 16 to 512 rows and measure it on a 4090.**
- **Design:** prefetch the next GEMM's weights during the current op, and fold GDN and glue into tasks.
- **What it settles:** whether 3.6 ms closes toward the 1.65 ms floor.

**In parallel:** a schema-first model at fidelity is the multi-question lever.

# Q4 draft report: fewer and better-used multiplies (B11, B13, Strassen–Winograd, B14)

Box q4 (A10G). Code: `~/decider2/q4/code/`; results: `q4/res/`, `q4/preds/`; log: `NOTES.md`.

Tags: **[M]** measured, **[V]** verified bit-exact, **[A]** arithmetic over measured inputs. Every latency used an exclusive GPU and at least 30 timed reps; I quote medians (p95 is within 1–3%).

## What I built

- **`q4sp.cu`: 2:4 sparse int8 and int4 GEMMs** (raw `mma.sp` PTX, weights as the sparse operand, bf16 / SwiGLU / fp16 epilogues per FORMATS.md, plus a same-loop dense control). Every config is bit-exact [V].
- **A fact for Q5** [M]. A one-instruction probe pinned the metadata layout. Ampere's int4 "2:4" keeps 2 of every 4 adjacent nibble *pairs*, so int4 masks must be chosen pair by pair; int8 is true 2:4.
- **`q4rt.py`:** H2's QRT plus the B13 eliminations and 2:4 GEMMs for all rows or for state rows only.
- **`q4st.cu`: one-level Strassen fused into one kernel** (7 products through one pipeline, combined in registers), in int8 on 7-bit codes, int4 on 3-bit codes and bf16; integer results bit-exact [V]. Two levels run with the outer level unfused.
- **`q4pk*.cu`:** a persistent W8A8 kernel running a forward's 96 GEMMs in one launch (grid barrier per GEMM, L2 prefetch of the next weights).

## B11: 2:4 sparse kernels

Per-GEMM speedup (dense time / sparse time) over hobson's five shapes [M]:

| M | int8 2:4 vs deployed dense int8 | vs the same loop run dense | int4 2:4 vs deployed dense int4 |
|---|---|---|---|
| 140 | 1.05–1.35x | 1.23–1.46x | 1.08–1.52x |
| 400 | 1.15–1.42x | 1.46–1.60x | 0.98–1.49x |
| 1,125 | 1.32–1.71x | 1.57–1.81x | 1.06–1.60x |
| 4,125 | 1.43–1.73x | 1.54–1.82x | 1.06–1.55x |

- **The claim "2:4 is slower than dense below 2,048 rows" is not a hardware limit.**
  - My kernel beats dense int8 at every M ≥ 140.
  - CUTLASS's sparse GEMM also wins on most shapes once its tiles fit the A10G's 99 KB of shared memory. Its A100-sized configs do not launch here.
- **int8 2:4 runs at 0.70–1.09x of dense int4's speed,** not at the full int4 rate.
- **End to end, b8 + B13 + 2:4 int8 weights, as a ratio of b8** [M]. Masks are magnitude-chosen, since speed does not depend on which weights are kept. Q5's dev screen (layers 12–22, decision-weighted masks):
  - state rows only: KL 2.8e-4, 4/160 flips;
  - all rows: KL 1.1e-3, 3/160 flips;
  - b8 alone: KL 2.0e-4.

  Layers 0–11 are untested for accuracy.

| T (rows), 1 question | b8, ms | state rows, layers 12–22 | state rows, layers 0–22 | all rows, layers 12–22 | all rows, layers 0–22 |
|---|---|---|---|---|---|
| 16 (141) | 8.18 | – | – | 0.884x | 0.757x |
| 64 (189) | 8.39 | 1.118x | 1.231x | 0.892x | 0.794x |
| 256 (381) | 14.40 | 0.958x | 0.939x | 0.854x | 0.729x |
| 1,000 (1,125) | 36.35 | **0.890x** | 0.818x | **0.843x** | 0.726x (26.4 ms) |
| 4,000 (4,125) | 142.63 | 0.824x | 0.696x | – | – |

- **Without B13** at T = 1,000: state rows at layers 12–22 give 0.931x; all rows give 0.886x.
- **With 15 questions** (state rows + B13): 0.939x at layers 12–22 and 0.915x at layers 0–22.
- **State rows only needs two launches per GEMM** (state rows on 2:4, question rows dense). That loses below about 256 state rows; there, all rows through one weight copy is faster.
- **GEMM time per forward at 1,000 state + 125 question rows** [A over M]. b8 takes 27.0 ms; every row on int8 2:4 takes 19.2 ms (0.71x). With only the state rows sparse:

| state rows on 2:4 | layers 12–22 | layers 0–22 |
|---|---|---|
| int8 | 25.0 ms (0.92x) | 22.8 ms (0.84x) |
| int4 (Q5's dev KL 3.9e-4 at layers 12–22) | 22.4 ms (0.83x) | 17.5 ms (0.65x; accuracy unknown) |

## B13: exact eliminations

**Exactness:**
- **With layer 23 on row-count-invariant kernels,** the layer-0 table plus layer 23 for the read rows only is bit-identical to its baseline in probabilities and logits on all 3,227 questions [V].
- **Against H2's runtime as deployed:**
  - the layer-0 table is bit-identical [V];
  - the layer-23 change alters 2 of 3,227 decisions (max |Δp| 0.004) [M]. cuBLAS and SDPA pick kernels by row count. Swapping layer 23's kernels with no elimination at all moves probabilities by as much (max |Δp| 0.003, 1 decision).

**End to end, b8 [M]:**

| request | b8, ms | B13, ms | ratio |
|---|---|---|---|
| T = 64, 1 question | 8.40 | 8.43 | 1.005 |
| T = 256, 1 question | 14.39 | 14.04 | 0.975 |
| T = 1,000, 1 question | 36.30 | 34.80 | 0.958 |
| T = 4,000, 1 question | 142.57 | 134.92 | 0.946 |
| T = 1,000, 15 questions | 166.3 | 159.7 | 0.960 |

- **The layer-0 table** saves 0.6%: its gather costs a third of the GEMM it replaces. The full-vocabulary table is 4.09 GB.
- **Layer 23** gives the rest.

## Strassen–Winograd

**Form.** Winograd's variant sums four blocks (two extra bits), so the integer runs use Strassen's original two-term form (one bit).

**Speed [M]:**
- **One level, against the identical kernel run dense** (excluding the producer's operand sums): 0.86–1.21x at M = 1,125 and 0.34–0.95x at M = 140.
- **Including the sums, against the best dense kernel:** never faster, 0.32–0.98x; at M = 1,125, int8 0.67–0.86x, int4 0.61–0.81x, bf16 0.75–0.92x.
- **Two levels:** 0.25–1.04x, without counting the producer.
- **The seven products alone** reach only 0.66–1.07x at M = 1,125.

**Error from the lost bit**, per-GEMM output error (median of 96 GEMMs, 28,706 real rows) [M]:

| | dense | fewer bits, per-token scales | one level (pair-shared scales) | two levels |
|---|---|---|---|---|
| int8 | 1.04% | 7-bit: 2.12% | 2.29% | 6-bit: 4.99% |
| int4 | 15.6% | 3-bit: 34.1% | 36.6% | – |
| bf16 | 0.22% | – | 0.38% | 0.75% |

**Decisions**, all 3,227 questions, deployed kernels, RTN weights [M]:
- 8-bit: REAL flips 0.83%, CF 1.000.
- **7-bit: REAL flips 1.48%, total variation 1.7x, CF .945, McNemar against the bf16 runtime p .007.** This fails the bar even before the pair-shared scales are applied.
- 3-bit: 61% flips.

## B14: persistent kernel for short decisions

**Scope:**
- **Where the time goes** [M, J5]: at 140 rows W8A8-b8 takes 7.38 ms. GEMMs take 5.17 ms, glue 1.0, GDN 0.88 and other work 0.32.
- **Floors** [A]:
  - streaming the weights: 1.37 GB / 600 GB/s = 2.29 ms;
  - int8 compute: 2.9 ms at peak.

**Built** [M]: the 96 GEMMs in one persistent launch.
- 140 rows: 5.26 ms, against 4.68 ms for a CUDA graph of 96 tuned J5 kernels (1.12x slower).
- 256 and 512 rows: 1.15x slower.
- The 95 grid barriers cost 0.22 ms.
- Prefetching the next GEMM's weights changes the time by less than 1%.

The launches and the dependency are not where the time goes. At ≤ 512 rows, per-tile efficiency is: J5's kernels run gate_up at about 50% of DRAM bandwidth and 55–60% of the tensor rate.

**Not built:** GDN and glue inside the kernel. They take 1.9 ms in 211 launches at 140 rows. Run as phases (about 2 µs of data plus a 2.3 µs barrier each), they would save about 1 ms, roughly 13% [A].

## Against the bar

- **Matrix-multiply time ≤ 0.65x b8 at M = 1,125:** not reached with any structure known to be accurate.
  - int4 2:4 state rows at layers 12–22: 0.83x.
  - int8 2:4: 0.92x.
  - int4 2:4 state rows in every layer reaches 0.65x, but its accuracy is unknown.
- **End to end ≤ 0.75x at 1,000 tokens (≤ 27 ms):**
  - B13 alone: 0.958x, exact.
  - B13 + 2:4 at layers 12–22: 0.890x (state rows) and 0.843x (all rows).
  - Only all rows sparse at layers 0–22 + B13 (26.4 ms, 0.726x) meets the speed, and its accuracy is untested.
- **Fidelity:**
  - B13 is exact.
  - Sparse fidelity is Q5's to measure.
  - Strassen at 7 bits fails (CF .945, p .007).

## Projections [A]

**Method:** the doc's. GEMMs scale with each card's peak, and memory-bound kernels with bandwidth.

- **B13 and the sparse savings** are fractions of the work, and Ampere, Ada and Blackwell GeForce all have 2x sparse tensor cores.
  - b8 at 1,000 tokens (J15's projection): 3090 19.9 ms, 4090 11.6, 5090 7.9.
  - With B13 plus sparse state rows in layers 12–22: about 17.7, 10.3 and 7.0 ms.
- **At about 150 rows on a 4090,** GEMMs are weight-bound and 2:4 cuts weight bytes to 0.56x.
- **Strassen gains on no card:** its sums and combination are memory passes.

## Most valuable next step

Have Q5 score decision-weighted 2:4 masks on all rows, for layers 0–22, on the full kit. Then run the masks that hold through `q4sp` in one weight copy.

- **The speed is already measured:** 0.726x b8 at 1,000 tokens with B13 (26.4 ms, which meets the speed bar) and 0.757x at 141 rows.
- **Layers 12–22 alone give 0.843x,** at 5x b8's dev KL with all rows sparse, or 1.4x with state rows only.
- **Fallback:** state rows only, or int4 2:4 state rows (0.83x GEMM time at layers 12–22).

**Kernel work** that follows:
- one launch for the state/question row split;
- closing the 10–30% gap between int8 2:4 and dense int4.

**Permission note:** `ncu` returns ERR_NVGPUCTRPERM on q4; I did not work around it, so I have no stall profile. Each ratio is against b8 measured in the same session.

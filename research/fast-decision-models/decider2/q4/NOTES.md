# Q4 notes (box q4): multiply count and utilisation kernels (B11 2:4 sparse, B13 exact eliminations, Strassen-Winograd, B14 persistent)

## 21:22 PDT start
- Read BRIEF10 in full incl. addenda 1-3 (Q4 = B11 mma.sp kernels, B13, Strassen measurement, B14), IDEAS_EXPLORED "floor on wall time",
  FAST_DECISION_MODEL 2.2 / 5 / 8, Q2 FORMATS.md + NOTES + code (q2gemm.cu, q2k.py, bench_gemm.py, q2sp.cu = Q2's CUTLASS-sparse start),
  H2 REPORT + qrt.py/qgemm.py/bench.py, J5 and J15 reports.
- Box q4 = i-REDACTED (g5.2xlarge, A10G), launched 21:18:38 PDT; waiting for boxes/q4.ready. TTL ~07:18 PDT.
- Hardware fact for B11 (arithmetic from CUTLASS 3.5.1 arch/mma_sparse_sm80.h): s4 sparse mma is m16n8k128 with 2-bit metadata, 32 bits
  per thread = 1024 bits per 16x128 A tile = 256 groups of 8 nibbles with 2 indices each, so int4 "2:4" on Ampere is PAIR-wise
  (keep 2 of 4 adjacent nibble pairs = 4:8 with pairs). int8 m16n8k64: 256 groups of 4 bytes, 2 indices each = true 2:4. To be
  confirmed empirically on the GPU; matters for Q5's pattern selection.
- Strassen note (arithmetic): Winograd's variant has 4-block operand sums (S4 = A11+A12-A21-A22, T4 likewise), 2 extra bits; only
  Strassen's original form keeps every operand sum to 2 blocks (1 extra bit). So "7-bit inputs fit int8" holds for original Strassen;
  Winograd's form would need 6-bit (int8) / 2-bit (int4) inputs. Plan: integer runs use the original form, bf16 runs both.

## 21:58 PDT box q4 ready (A10G 1710 MHz, 8 vCPU); custom mma.sp kernel bit-exact
- Box up since ~21:19 PDT, so the 10 h TTL ends ~07:19 PDT. Put h2/h1/j5/j15/q2 code + g2lib; setup_q4.sh (CUTLASS 3.5.1 + main, H2/J5/Q2
  libs, Q2's CUTLASS-sparse copy, bundles, H1-recipe GPTQ w8 codes) running with nohup.
- q4sp.cu (raw mma.sp / ldmatrix / cp.async, weights as the sparse A operand, D = W X^T, epilogue staged through smem to Y[t, n]):
  int8 m16n8k64 and int4 m16n8k128 share one smem pipeline (per k-step: 16 ch x 32 kept bytes, 8 tok x 64 bytes, 1 u32 metadata/lane).
- [MEASURED] metadata layout pinned by a one-instruction probe (random 2:4 A, random B, 3 trials each, exact int32 match): lane (g=lane>>2,
  t=lane&3) holds row g (t even) or row g+8 (t odd), groups 8*(t>>1) .. +7, 4 bits each (idx0 | idx1<<2). The same layout holds for int4,
  where the probe only matches with PAIR semantics: int4 "2:4" on Ampere = keep 2 of each 4 adjacent nibble pairs (4:8 with pairs).
  -> Q5 must select int4 patterns at pair granularity (told in DRAFT_REPORT; I cannot write to q5/).
- [MEASURED] test_sp.py: every config that fits smem is bit-exact vs the torch fp64 reference (int32, bf16 dequant, fp16a, SwiGLU
  epilogues) for sp8, sp4 and the same-loop dense controls d8/d4, at (512,136,1024), (8224,77,2048), (2048,300,6144), (256,1,512).

## 22:35 PDT B11 speed [MEASURED] (res/res_sparse.jsonl; A10G exclusive, CUDA events, best config by median of 30, p95 within ~1-3%)
- q4sp sparse int8 (2:4, bf16-dequant or SwiGLU epilogue) vs the deployed dense int8 (H2 CUTLASS fp16a / J5 s8 / H2 EVT SwiGLU for gate_up),
  speedup = dense time / sparse time: M=140 1.05-1.35x, M=400 1.15-1.42x, M=1125 1.32-1.71x, M=4125 1.43-1.73x.
  Against the identical main loop run dense (q4 d8): 1.23-1.82x (the hardware effect with everything else equal).
- q4sp sparse int4 (pair-2:4) vs deployed dense int4 (H2 CUTLASS s4 / EVT): M=140 1.08-1.52x, 400 0.98-1.49x, 1125 1.06-1.60x, 4125 1.06-1.55x.
- sparse int8 is 0.70-1.07x as fast as DENSE int4 (i.e. mostly 10-30% slower than dense int4), so 2:4 int8 sits between W8A8 and W4A4.
- CUTLASS 2.x GemmSparseUniversal (Q2's q2sp.cu, copy + configs sized for 99 KB smem): only configs <= 99 KB run; res/res_sparse_cs.jsonl.
- The S9 claim "2:4 slower than dense below 2048 rows" (cuSPARSELt) does not hold for these kernels: sparse is faster at every M >= 140 for int8.

## 22:40 PDT Strassen [MEASURED] (res/res_strassen.jsonl): q4st.cu fused one-level kernel (original Strassen, 7 products through one cp.async
pipeline, 3 accumulator sets in registers, combination in registers), bit-exact for int8 (7-bit codes) and int4 (3-bit codes) vs dense on the
same codes; bf16 rel. error vs fp32 4.7e-3 (dense bf16 kernel 1.65e-3) on Gaussian inputs.
- One level, fused, excluding the producer sums, vs the SAME kernel dense: M=140 0.34-0.95x, 400 0.85-1.16x, 1125 0.86-1.21x, 4125 0.94-1.19x.
- Including the producer (five activation sums: 22-264 us) and against the best dense kernel available (H2 CUTLASS int / Q2 / cuBLAS):
  0.32-0.98x everywhere; at M=1125: int8 0.67-0.86x, int4 0.61-0.81x, bf16 0.75-0.92x. Never faster.
- Bound, the 7 half-size products alone (batched, no sums, no combination) vs best dense: M=1125 0.66-1.07x; M=4125 0.81-1.17x.
- Two levels: real (outer level unfused, int32 products to memory + combine kernel, producer NOT counted) 0.25-1.04x; bound (49 quarter
  products alone) 0.35-1.24x (best: bf16 at M=4125).
- Why: dense kernels already run at ~80% of peak; half-size products lose efficiency (fewer CTAs, short K); 3 live accumulator sets halve
  occupancy; sums are a memory pass. The 12.5% (one level) / 23% (two levels) multiply saving is smaller than these losses at M <= 4125.

## 22:36 PDT B13 exactness [MEASURED] (res/res_b13_exact.json; q4rt.py QRT4 = H2 QRT + eliminations, b8 + my GPTQ draw)
- 300 random evalkit questions (all suites, hobson single layout) + 12 packed requests (4q / 15q bundles, T = 64 / 1000).
- layer-0 table (fp16 C16 + per-token scale for all 248,320 ids = 4.09 GB; a deployment vocabulary needs only its ids): hidden states and
  probabilities bit-identical to H2's runtime on 312/312.
- layer 23 'kv' (state rows K/V only) and 'read' (also only the read rows past attention): vs H2's runtime as deployed, identical on 8/312 and
  6/312, max |dp| 0.0062 / 0.0053, 0 decision flips. Cause: cuBLAS bf16 GEMMs and SDPA choose kernels by row count (measured: X[q0:] @ W^T
  rows differ from (X @ W^T)[q0:] by up to 0.5 at bf16; SDPA subset queries differ by ~1e-3). Same size as swapping layer 23's kernels with
  no elimination (inv vs h2: 33/300 identical, max |dp| 0.0030).
- With layer 23 on row-count-invariant kernels (Q2's bf16 q2gemm; baseline's layer-23 attention issued as state / question calls), every
  elimination is bit-identical to its baseline on 300/300 (hidden and probabilities).

## 22:45 PDT B13 end to end [MEASURED] (res/res_b13_e2e.jsonl; H2 bench machinery: CUDA graph, fresh ids, H2D + replay + D2H, 30 reps)
- b8 1q (hobson layout), median ms base -> l0 + read: T=64 8.40 -> 8.43 (1.005x), 256 14.39 -> 14.04 (0.975x), 1000 36.30 -> 34.80
  (0.958x, -1.50 ms), 4000 142.57 -> 134.92 (0.946x). 15q packed: 64 129.0 -> 124.1 (0.962x), 1000 166.3 -> 159.7 (0.960x), 4000 304.1 -> 290.5 (0.955x).
- Split at T=1000 1q: layer-0 table alone 0.994x (-0.22 ms; the two index_select gathers cost ~0.09 ms of the ~0.3 ms saved; fusing the
  gather into the conv kernel would recover them [arithmetic]); layer 23 'kv' 0.965x; 'read' adds nothing for 1q (125 question rows) but
  0.967x vs 0.991x for 15q (3,720 question rows, 113 read).
- Row-invariant layer-23 kernels (bit-exact variant) cost +0.3% on the baseline and nothing on the eliminated runtime (l0+read_inv 34.83 ms).

## 22:50 PDT Strassen's lost bit at GEMM level [MEASURED] (res/res_bit.json; q4bit.py: 12 train-split requests, 28,706 rows, all 96 GEMMs,
rotated H1 basis, relative Frobenius error of the GEMM output vs fp32; median over GEMMs [min..max ratios])
- int8: W8A8 1.04%; 7-bit per token 2.12% (2.02x); one-level Strassen format (7-bit, rows t/t+M/2 and channels n/n+N/2 share scales)
  2.29% (2.18x [2.12-2.31]); channel pairs sorted by absmax offline 2.19% (recovers 4%); two-level (6-bit, quads share) 4.99% (4.77x).
- int4: W4A4 15.6%; 3-bit 34.1% (2.19x); one-level 3-bit shared 36.6% (2.36x). (two-level ternary: NaN in the scale search; not reported)
- bf16: dense (bf16 operands) 0.22%; Strassen one level 0.38% (1.82x), two levels 0.75% (3.5x) with every operand sum rounded once.

## 22:48 PDT B14 prototype v1 [MEASURED] (res_pk.jsonl on box; q4pk.cu + bench_pk.py): 96 W8A8 GEMMs of one forward, fixed int8 inputs
- graph of 96 tuned kernels (best of H2 CUTLASS / J5 per shape; J5 wins every shape): M=140 4.68 ms (sum of isolated 5.27), 256 6.41, 512 12.15.
- one persistent launch (resident CTAs, grid barrier per GEMM, L2 prefetch of next GEMM's first K stages), 6 tile configs x 4 prefetch depths:
  best M=140 5.26 ms (1.12x SLOWER), 256 7.40 (1.15x slower), 512 14.0 (1.15x slower). Prefetch changes < 1%.
- Floors [arithmetic]: weights 1.373 GB / 600 GB/s = 2.29 ms; int8 compute at M=140 (padded to 160) ~2.9 ms at 100% of ~150 TOPS.
  The graph already hides launches (0.89x of the isolated sum); what is left is per-tile efficiency at small M (J5's kernels: ~50% of
  DRAM bandwidth and ~56-64% of the tensor rate on gate_up at 140 rows), not launch or dependency gaps. v2: all rows in one M tile
  (BM 160/256), wide N tiles, split-K with int32 atomics, so each weight byte is read once and activations are not re-read per narrow tile.

## 22:57 PDT B14 prototype v2 [MEASURED] (q4pk2.cu: generic warp grid, all rows in one M tile, wide N tiles, split-K by int32 atomics,
zeroing of the next GEMM's output inside the previous phase; 7 tile configs x 3 split-K policies x 2 prefetch depths; all results exact)
- best M=140 5.97 ms (64x128 tiles, no split-K) vs graph of 96 J5 kernels 4.73 ms; 256: 7.08 vs 6.45; 512: 13.59 vs 12.16. All-rows tiles
  (BM 160/256) are the slowest (8-21 ms at 140 rows): few CTAs per GEMM, and split-K atomics do not recover them.
- Conclusion so far: on the A10G the per-GEMM tile efficiency at <= 512 rows, not launches or the dependency, sets the GEMM time; a
  persistent kernel built from a generic tile loses to J5's tuned kernels. The barrier-only cost is being measured (pk_barrier.py).

## 23:00 PDT B13 at scale [MEASURED] (all 3,227 evalkit questions, deployed kernels, my GPTQ draw; laptop/score.py with J15's scorer)
- b8 base: REAL flips 1.11% (12/1083), LONG 1.06%, TV .0048, CF fgh 1.000, CF-probe fgh .962, JB-hard .538 (McNemar vs hobson 2/0, p .5),
  REAL-label .793 (same draw quality as J15's: 1.11%; H2's draw 0.18%).
- b8 + layer-0 table + layer-23 read-rows-only: identical scores except CF-probe acc .328 -> .325; 2 of 3,227 decisions differ, max |dp|
  0.0042. Both differences come from the row-count-dependent cuBLAS/SDPA kernels at layer 23 (bit-exact with invariant kernels on 300/300;
  the full-kit invariant-kernel pair is queued).

## 23:08 PDT B11 row-role timing (Q5's most accurate structure: state rows sparse, question rows dense) [MEASURED] (res/res_rr.jsonl)
- two launches per GEMM (q4sp sp8 on state rows + deployed dense int8 on question rows) vs dense on all rows, per GEMM:
  T=1000+125: 1.07-1.28x; 4000+125: 1.33-1.59x; 1000+3720 (15q): 1.04-1.10x; 256+125: 0.81-0.98x (slower); 64+125: 0.62-0.83x (slower).
  At <= 256 state rows the extra small-M launch for the question rows costs more than sparsity saves; there one weight copy for all rows
  (all sparse or all dense) is faster.
- CUTLASS 2.x sparse (copy of Q2's q2sp.cu + configs that fit 99 KB): with configs that fit, it also beats dense int8 at every M >= 140
  except gate_up (0.90-1.07x); best-of-both vs dense int8 at M=1125: 1.39-1.78x on attn_in / out / down. The A100-sized CUTLASS configs
  (Q2's 0-9) do not launch on the A10G (smem > 99 KB), which is one way a library gets "sparse slower than dense".

## 23:14 PDT Strassen's lost bit at decision level [MEASURED] (all 3,227 questions, deployed kernels, RTN weights for both; score.py)
- b8 RTN 8-bit: REAL flips 0.83%, TV .0067, CF fgh 1.000, CF-probe .952, JB-hard .538, REAL-label .788, vs bf16 runtime 2/6 (p .29).
- b8 RTN 7-bit (qmax 63, per-token scales, i.e. BEFORE Strassen's pair-shared scales): REAL flips 1.48%, TV .0113 (1.7x), CF fgh .945
  (bar .99), CF-probe .962, JB-hard .538, REAL-label .793, vs bf16 runtime 2/13 (McNemar p .007). Fails CF retention and the McNemar test.
- int4 width [MEASURED, all 3,227, deployed kernels, RTN, all 96 GEMMs]: W4A4 REAL flips 14.9%, TV .124, CF fgh .670, CF-probe .524, JB-all .684;
  W3A3 (Strassen-int4 width, per-token): REAL flips 61.4%, TV .369, CF fgh .239, CF-probe .210, JB-all .303, REAL-label .448 (model broken).

## 23:22 PDT B14 barrier cost [MEASURED] (pk_barrier.py): 95 grid barriers with all CTAs resident and no work = 217 us (2.3 us each).
- So the persistent v1 kernel's 5.26 ms at 140 rows is ~0.22 ms of synchronisation + ~5.0 ms of tile work, against 4.68 ms for the graph of
  96 J5 kernels. The dependency structure costs little either way; what decides is tile efficiency at <= 512 rows. L2 prefetch of the next
  GEMM's weights across the barrier gains < 1% (prefetch depth 0 vs 2/4/8 stages in v1 and v2).

## 23:30 PDT B13 bit-exact at scale [VERIFIED] (all 3,227 questions): with layer 23 on row-count-invariant kernels, b8 + layer-0 table +
layer-23 read-rows-only gives probabilities AND logits identical to its baseline on 3,227 / 3,227 (0 decisions differ, max |dp| 0.0).
The invariant baseline itself vs H2's deployed runtime: 1 decision differs, max |dp| 0.0031, 403/3227 bit-identical (kernel-choice noise).
- PERMISSION DENIED (reported, not worked around): ncu on box q4 fails with ERR_NVGPUCTRPERM (no access to GPU performance counters as
  user ubuntu). I did not use sudo. Consequence: no stall-reason profile of the sparse kernel.

## 23:38 PDT B11 end to end [MEASURED] (res/res_b11_e2e.jsonl; QRT4: state rows (t < q0) through q4sp 2:4 int8 weights, question rows
through the dense int8 weights, two launches per GEMM, magnitude masks: speed only, accuracy is Q5's; b8 map otherwise; 30 reps)
| T, questions | b8 | b8+B13 | sp12 (L12-22) | sp12+B13 | spall (L0-22) | spall+B13 |
| 64, 1q | 8.39 | 1.007x | 1.135x | 1.118x | 1.250x | 1.231x |
| 256, 1q | 14.40 | 0.977 | 0.983 | 0.958 | 0.963 | 0.939 |
| 1000, 1q | 36.35 | 0.958 | 0.931 | 0.890 (32.34 ms) | 0.859 | 0.818 (29.75 ms) |
| 4000, 1q | 142.63 | 0.947 | 0.878 | 0.824 | 0.747 | 0.696 |
| 1000, 15q | 165.77 | 0.963 | 0.978 | 0.939 | 0.953 | 0.915 |
- At T=64 the split into two small launches loses (state rows 64): dispatch rule = sparse only above a few hundred state rows.
- GEMM time at 1000+125 rows [A over res_rr]: b8 27.0 ms, sp12 25.0 (0.92x), spall 22.8 (0.84x); all rows sparse 19.2 (0.71x).
- Smoke test: finite outputs; magnitude 2:4 on state rows of layers 12-22 moves the 6 tested REAL decisions' probabilities by <= .007.
(timestamps corrected 23:42 PDT: several headers had been written ~30-50 min ahead of the clock)

## 23:55 PDT sparse kernel tuning + int4 state rows [MEASURED]
- Tall-warp configs 12-15 (warp 128x32 / 64x64 with 256 tokens, single fragment buffer), bit-exact: no gain (best 0-11 vs 12-15 within
  +-2% on the large shapes, worse at 140 rows; res_tune_sp.json on box). B-fragment reuse is not what limits sp8; my dense int4 control
  is also only 0.67-0.85x of CUTLASS s4, so the main loop's feed rate at int4-equivalent math density is the limit (no ncu: permission).
- Q5 dev (23:45): int8 2:4 state rows L12-22 rotated, decision-weighted: bank KL 2.8e-4 (b8 alone 2.0e-4), 4/160 flips; int4 2:4 state rows
  L12-22: KL 3.9e-4, 2/160. So I timed int4 2:4 state rows + int8 dense question rows (res/res_rr_sp4.jsonl), per GEMM vs dense int8 all rows:
  1000+125 1.38-1.88x; 4000+125 1.93-2.61x; 256+125 0.94-1.21x.
- GEMM time per forward [A over M], 1000 state + 125 question rows: b8 27.0 ms; int8 2:4 state L12-22 25.0 (0.924x); int4 2:4 state L12-22
  22.4 (0.828x); int8 2:4 state L0-22 22.8 (0.843x); int4 2:4 state L0-22 17.5 (0.648x, accuracy unknown). At 4000+125: 0.858 / 0.755 /
  0.707 / 0.497x. The int4-state-row formats need row-role quantization in the glue (Q2's B8 prologue) for an end-to-end run.
- B14 arithmetic for the non-GEMM part at 140 rows (J5: glue 121 launches 1.0 ms, GDN 90 launches 0.88 ms): each op moves ~1 MB (~2 us at
  600 GB/s); as phases of a persistent kernel with 2.3 us barriers they would cost ~0.5-0.7 ms, i.e. ~1.2-1.4 ms (16-19%) of 7.4 ms [A].

## 00:15 PDT B11 all rows through 2:4 int8 (one launch, one weight copy) end to end [MEASURED] (res/res_b11_short.jsonl; magnitude masks)
| T (rows), 1q | b8 ms | B13 | all rows L12-22 | + B13 | all rows L0-22 | + B13 |
| 16 (141) | 8.18 | 1.007x | 0.898x | 0.884x | 0.768x | 0.757x (6.19 ms) |
| 64 (189) | 8.57 | 0.984 | 0.909 | 0.892 | 0.811 | 0.794 |
| 256 (381) | 14.39 | 0.975 | 0.878 | 0.854 | 0.752 | 0.729 |
| 1000 (1125) | 36.32 | 0.958 | 0.886 | 0.843 | 0.765 | 0.726 (26.38 ms) |
- Speed-only: the end-to-end bar's speed (<= 27 ms, 0.75x) is met by all rows 2:4 in layers 0-22 + B13 (26.4 ms), but Q5 has accuracy
  only for layers 12-22 (dev, all rows, int8 decision-weighted: bank KL 1.1e-3, 3/160 flips; b8 alone 2.0e-4) and layers 0-11 are the
  sensitive ones (H1). For short decisions (141 rows) all-rows L12-22 gives 0.898x (7.35 ms vs 8.18).

## 00:20 PDT wrap-up
- DRAFT_REPORT.md written (1,499 words excluding table markup). All results copied to ~/decider2/q4/res (+ res/logs) and q4/preds.
- Box q4 left idle (no jobs running); it auto-terminates at its 10 h TTL (~07:19 PDT). I did not reset the timer.
- Open items for others: Q5 to score decision-weighted 2:4 masks for all rows / state rows on layers 0-22 (speed already measured here);
  Q2 to integrate int4 2:4 state rows (needs row-role quantization in the glue); int4 masks must be pair-wise (hardware).

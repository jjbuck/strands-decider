# H2 notes (box g1): end-to-end low-bit decision runtime

## 21:55 PDT start
- Read BRIEF7, F7_REPORT, evalkit README, d1/lean2.py (+ systems/g/lean.py), G2's g2s4.cu / kbench.py / res_prec.json / g2qat.py / g2lib.py.
- Fetched arXiv abs+html 2609.23886: schema-first layout = [questions+options P][state S][N one-token answer slots]; P's attention K/V and
  delta-rule recurrent state cached once; per request only |S|+N tokens. Conv tails not mentioned by the paper (we need them: 4-tap conv).
- Coordinator correction (21:57): skip hob12; full 24-layer hobson only; formats from H1 (h1/FORMAT.md) and H5 (h5/FORMAT.md, learned rotations).
- G2 format (g2lib/g2qat): Win/Wgu input = unweighted RMSNorm(x) rotated (1+w folded into W), Wo/Wd input rotated online; R = sign-randomized
  Kronecker Hadamard (2048 = H32xH64 = Sylvester H2048; 6144 = Paley H12 x H16 x H32); per-channel W absmax w/ MSE clip search, per-token A absmax
  (clip 0.9 for 4-bit). Plan: rotate the residual stream offline (fold R into embed / writers / readers), online FWHT fused into gnorm/gate/SwiGLU kernels.
- Plan: CUTLASS s4/s8 GEMMs (fp16 out with safe scalar alpha; per-token x per-channel dequant applied by the consumer Triton kernel, which
  also does the nonlinearity, online Hadamard, absmax and quantization of the next GEMM input).

## 22:20 PDT CUTLASS GEMMs on g1 (MEASURED)
- Built G2's g2s4.cu (CUTLASS 3.5.1) and H2's h2mix.cu: W4A8 = s8 x s4 mixed-input (CUTLASS 4.8 OpMultiplyAddMixedInputUpcast; 3.5.1 lacks it).
  All correct (max rel err 2-4e-4 = fp16 output rounding). res_gemm.json. Speedup over cuBLAS bf16 at M=1000 / 4000:
  | shape | s8 | s8s4 (W4A8) | s4 |
  | gdn_in | 2.34 / 1.89 | 2.19 / 1.95 | 4.17 / 3.82 |
  | attn_in | 1.86 / 1.88 | 1.77 / 1.94 | 3.19 / 3.75 |
  | out | 1.73 / 2.21 | 1.65 / 2.17 | 2.77 / 4.22 |
  | gate_up | 1.97 / 1.93 | 1.89 / 1.98 | 3.64 / 4.02 |
  | down | 1.63 / 2.19 | 1.55 / 2.17 | 2.96 / 4.29 |
  W4A8 runs at int8 tensor-core rate (as expected: prefill is compute-bound; it only halves weight memory).
- H1 FORMAT v0 read (22:15): residual stored rotated (R1 seed 1234 folded: E R1, W diag(1+g) R1, R1^T Wo R2, R1^T Wd R4); online R2 (seed 1235,
  2048 or per-head) before Wo and R4 (seed 1236, P12 x H16 x H32) before Wd; per-token absmax A (int4 clip 0.9), per-channel W; epilogue fp32 -> bf16.
  => I implement H1's format directly on real hobson weights (RTN now; GPTQ codes from H1 later). No random weights needed.
- Real question lengths (REAL-agree+LONG, 38 distinct specs): median 110 tok, mean 267, p90 1014 (needed_procedure 1201, procedure 1014).

## 22:35 PDT runtime qrt.py working (MEASURED, g1 exclusive, T=1000 random ids, single sequence, graph replay median of 12)
- Files on g1 ~/work/h2 (copied to ~/decider2/h2/code): qgemm.py (ctypes CUTLASS), qk.py (Triton glue), rot.py (H1 Rot), qrt.py (runtime + layouts + compiled prefix).
- Glue kernels verified vs torch refs of H1 format: code mismatches 1e-5..1e-3 (rounding-boundary cases, max 1 code step).
- Bug found+fixed: SDPA output is [B,T,H,D]-strided (query layout), kernel now takes strides.
- Online Hadamard kernels were slow (swiglu 300 us/call): permute-heavy formulation. Rewrote as one row/program, left-multiply 2D dots + one batched dot,
  fp16 dots on the row-prescaled vector (code mismatch vs fp32 emulation 6e-4 int4 / 1e-2 int8 +-1 step; tf32x3 exact but 3x slower).
- Final-hidden cos vs d1 bf16 fold (random ids, RTN, untrained): bf16 0.99999, W8A8 0.99993, W4A8 0.9949, W4A4 0.9872.
- Latency T=1000: bf16 52.26 | W8A8 33.65 | W4A8 34.99 | W4A4 21.88 ms.
- W4A4 kernel breakdown before the Hadamard rewrite: GEMM 12.9 ms, glue dominated by swiglu+R4 7.2, gnorm+R2 3.2; fla 2.7; flash 0.7.
- Compiled-schema and packed-branch layouts verified EXACT vs a full single pass over the same tokens (bf16 cos 0.99999; W8A8 0.99995).

## 22:55 PDT full latency matrix v1 (MEASURED, res_bench.jsonl; g1 exclusive; 3 warm + 20 timed, fresh random state ids each rep, H2D+graph+D2H)
- Real question bundles (exact hobson token ids via plib.P.prep): 1q = details_match (125 tok); 4q = the most common real 4-question
  request (details_match, failure_cause, needed_procedure, rule_bound_values: 1613 tok); 15q = 15 most frequent banking questions (3720 tok).
- plain = hobson layout (1q: one sequence state+question; 4q/15q: shared state + one branch per question, varlen fla + per-branch SDPA).
- schema = compiled prefix: P = bundle cached (GDN states, conv tails, K/V); per request state + N one-token slots.
- median ms at T=1000 (bf16 / W8A8 / W4A8 / W4A4):
  1q plain 56.9/36.9/38.4/24.1; 1q schema 52.7/34.2/35.6/22.4; 4q plain 138.2/87.9/91.9/58.2; 4q schema 54.3/35.8/37.2/23.9;
  15q plain 237.1/163.4/166.9/106.8; 15q schema 56.7/38.1/39.6/26.3. p95 within 0.05 ms of the median everywhere (one at 0.4).
- at T=4000: 1q plain 203.5/138.1/138.1/86.9; 1q schema 199.3/134.7/134.9/84.6; 4q plain 277.9/195.3/199.3/133.9; 4q schema 204.6/140.0/140.4/89.9;
  15q plain 398.1/281.4/289.4/193.4; 15q schema 212.1/147.6/149.0/97.6.
- NOTE the v1 kernel split mislabelled flash attention as GEMM (its template names contain 'cutlass'); fixed in bench.py, re-profiled:
  15q schema W4A4 T=1000: GEMM 12.1 | attention 4.7 (3720-token prefix) | glue 5.6 | fla GDN 3.4 | other 0.3 ms.
  T=4000: GEMM 43.6 | attn 21.1 | glue 20.0 | GDN 12.3. GEMMs run at ~81% of A10G peak at every precision (bf16 58 TF, int8 113 TOPS, int4 227 TOPS at T=1000).
- W4A8 (s8 x s4 mixed input) is no faster than W8A8 on A10G prefill (compute-bound; it only halves weight bytes).
- Glue kernels were L2-bound on per-column scale/sign vectors (reloaded per row): now 4 rows/program with hoisted vectors (gnorm 75->40 us).
- Eval runs (all evalkit questions x 4 precisions, H1 v0 format, RTN weights) in progress on g1.

## 23:00 PDT accuracy of the H2 runtime (MEASURED: all 3227 evalkit questions, hobson layout, H1 FORMAT v0 RTN weights; scored on laptop)
| metric | hobson ref | H2 bf16 | H2 W8A8 RTN |
| REAL flips vs hobson | 0 | 0.37% | 1.20% |
| REAL agree_sd | 1 | 0.994 | 0.988 |
| LONG agree_sd | 1 | 0.988 | 0.970 |
| CF fgh | 1 | 1.000 | 0.991 |
| CF-probe fgh | 1 | 0.971 | 0.924 |
| JB-hard acc | .523 | .538 | .531 (McNemar p 1.0) |
- H2 bf16 runtime sits at the known runtime floor (merged_full 0.4% flips, CF-probe fgh 0.971) => the runtime is faithful.
- W8A8 RTN fails the low-bit bar (flips 1.20% > 0.5%, CF-probe fgh 0.924 < 0.97); same flip rate as G2's rotated W8A8 emulation (1.20-1.29%).
- Chained on g1: H1-emulation cross-check (h1x.py), bench v2 (correct attention labelling, hoisted glue, H1 precmaps k24/k48), map evals.
- H1 precmaps (W4A4 with 24 / 48 most sensitive GEMMs at W8A8) copied from h1/code; runtime supports any per-GEMM map, dense learned R1 (H5),
  H1 rotated-space QAT LoRA rule, and external (codes, scales) per GEMM.

## 23:15 PDT accuracy, all four precisions (MEASURED, H1 FORMAT v0 with RTN weights, all evalkit suites)
| metric | hobson | H2 bf16 | W8A8 | W4A8 | W4A4 |
| REAL flips vs hobson | 0 | 0.37% | 1.20% | 9.51% | 13.48% |
| REAL agree_sd | 1 | .994 | .988 | .899 | .855 |
| LONG agree_sd | 1 | .988 | .970 | .897 | .879 |
| CF fgh | 1 | 1.000 | .991 | .844 | .670 |
| CF-probe fgh | 1 | .971 | .924 | .590 | .514 |
| JB-hard acc (McNemar p) | .523 | .538 | .531 (1.0) | .500 (.65) | .508 (.84) |
| REAL-label acc | .785 | .785 | .790 | .743 | .728 |
- Every low-bit RTN config fails the pre-registered low-bit bar; W4 weights are the dominant error (W4A8 ~ W4A4).
- Queued on g1: H1-emulation cross-check -> bench v2 -> evals of H1 precmaps k24/k48 -> H1-recipe GPTQ (calib 64 states) W8 and W4 codes + evals.

## 23:20 PDT cross-checks, a failed optimization, bench v3 launched
- H2 runtime vs H1's emulation (h1lib, same RTN codes, 150 REAL-agree questions): W8A8 decision agreement 0.98, median |dp| 0.004 (p90 0.012);
  W4A4 0.913, median |dp| 0.054. W4A4 RTN is chaotic (my own runtime vs itself in two numerically-equivalent layouts: final-hidden cos 0.992),
  so a layer-by-layer divergence check (h1layer.py) is queued to separate implementation error from amplification.
- bench v2 (res_bench_v2.jsonl) used 4 rows/program for the glue kernels: SLOWER in graphs (swiglu 2.33 -> 3.81 ms at T=1000): one row per
  program keeps more loads in flight; the earlier standalone timing was taken under contention. Reverted to 1 row/program (v1 behaviour).
  v2 still gives correct per-category splits and the H1 precmap latencies (k24/k48 cost +1.8/+5.4 ms at 1q-schema T=1000 vs pure W4A4).
- Glue kernels in a graph loop: swiglu+R4 93 us @T=1000 (297 GB/s), 312 us @4000 (354 GB/s); addq 25/112 us; gnorm+R2 31/102 us.
  Remaining lever = bytes, not kernel tuning: a SwiGLU GEMM epilogue would halve gate_up output traffic (~1.3 ms @1000, ~5 ms @4000).
- bench v3 (final timing, 1 row/program) running; then layerwise check, map evals, GPTQ chain.

## 23:40 PDT layerwise check vs H1 emulation; fused SwiGLU GEMM epilogue (MEASURED)
- h1layer.py (6 real items): W8A8 cos(H2, H1-emulation) of the normed residual = 0.99985 after layer 1, 0.9999 after 24 -> implementation matches.
  W4A4: 0.985 after layer 1 (0.987 after 24). dbg4.py isolates it: weight codes IDENTICAL, GEMM output with H1's activation codes matches to
  1e-4; activation codes differ (int4 0.3%, int8 5%) only because H2 stores the rotated residual in bf16 (FORMAT v0: residual stored as xR1),
  so per-token absmax scales differ <=0.34%. At int4 each flipped code is a full step: W4A4-RTN decisions are fragile to implementation
  rounding (H2 vs H2 in two equivalent layouts: final cos 0.992). => accuracy measured in emulation must be confirmed in the runtime.
- NEW: gate_up GEMM with fused dequant+SwiGLU epilogue (CUTLASS 4.8 Sm80 EVT: acc*ra[row]*cs[col] in fp32, custom visitor pairs interleaved
  g/u columns and stores bf16 m at half width). Bit-exact vs a torch reference; same GEMM time as the plain kernel while writing half the bytes.
  The SwiGLU kernel then only does R4 + quant (reads 12 KB/row, not 24). EVT=1 default.
  W4A4 1q-schema: T=1000 22.50 -> 21.65 ms; T=4000 85.46 -> 82.47. W8A8 1q-schema T=1000 34.28 -> 32.87.
- Residual-add epilogue fusion estimated at only ~0.2 ms @1000 / ~1.3 ms @4000 net (epilogue must read x): not done.
- Map evals (k24, k48) restarted cleanly with EVT numerics; then W4A4-EVT eval (implementation-noise estimate); then GPTQ chain.

## 00:05 PDT map evals + implementation-noise test (MEASURED, all 3227 questions, RTN, EVT runtime)
- precmap k24 (W4A4, 24 GEMMs W8A8): REAL flips 8.3%, agree_sd .853, CF fgh .835, CF-probe fgh .638, JB-hard .538.
- precmap k48 (48 GEMMs W8A8): REAL flips 3.9%, agree_sd .942, CF fgh .917, CF-probe fgh .810, JB-hard .515 (McNemar p 1.0).
- W4A4 RTN, two numerically-equivalent H2 runtimes (gate/up through fp16 then SwiGLU, vs fp32 dequant + SwiGLU in the GEMM epilogue):
  12.0% of all decisions differ between them; CF-probe fgh .514 vs .400, JB-hard .508 vs .469; REAL flips vs hobson 13.5% vs 13.7%.
  => at W4A4-RTN hobson's decisions are dominated by quantization noise; suite numbers at 4 bits move by ~0.1 under rounding-level changes.
- GPTQ chain (H1 v1 recipe: h1calib 64 states, act-order GPTQ8 + GPTQ4) running on g1; then b8 / k48 GPTQ maps; then final bench v4.
- 00:20 W8A8 GPTQ8 (H1 v1 recipe, regenerated on g1) in the H2 runtime, all suites: REAL flips 0.83% vs hobson (1.02% vs H2 bf16), agree_sd .991,
  LONG agree_sd .970, CF fgh 1.000, CF-probe fgh .990, JB-hard .531 (McNemar p 1.0), REAL-label .787. Fails only the <0.5% flip bar.
- 00:40 W4A4 GPTQ4 (all 96 GEMMs): REAL flips 8.96%, agree_sd .850, CF fgh .826, CF-probe fgh .657, JB-hard .523 -> FAIL.
- 00:40 **W8A8 GPTQ8 + H1 map b8 (8 GEMMs bf16: 23.Wd + 7 Wo) in the H2 runtime: REAL flips 0.18% vs hobson (2/1083; 0.55% vs H2 bf16),
  agree_sd .997, LONG agree_sd .982, CF fgh 1.000, CF-probe fgh .971, JB-hard .523 (McNemar 0/0, p 1.0), REAL-label .785 -> PASSES the
  pre-registered low-bit bar** (at the runtime's own noise floor: H2 bf16 vs hobson 0.37%).
- 00:15 W4A4 map k48 with GPTQ4/GPTQ8 codes: REAL flips 2.59%, agree_sd .965, LONG agree_sd .939, CF fgh .972, CF-probe fgh .848, JB-hard .538 -> FAIL
  (flips, CF-probe). Full table: ~/decider2/h2/scores.json (score_h2.py). Final bench v4 (EVT runtime incl. b8 and k48 maps) running.

## 00:30 PDT final latency matrix v4 (MEASURED, res_bench_v4.jsonl, EVT runtime; p95 <= 1.1 ms above median) + schema compile cost
- T=1000 medians (bf16 / W8A8 / W4A8 / W4A4 / W8A8-GPTQ-b8 / W4A4-k48): 1q plain 57.0/35.6/38.6/23.5/36.3/29.2; 1q schema 52.8/32.9/35.6/21.7/33.5/27.0;
  4q plain 138.3/90.2/92.1/56.6/93.2/72.4; 4q schema 54.4/34.5/37.2/23.3/35.1/28.5; 15q plain 237.4/163.6/166.5/103.4/166.4/131.5;
  15q schema 56.6/36.8/39.6/25.6/37.4/30.9.
- T=4000: 1q plain 203.6/138.6/139.1/84.7/142.1/110.0; 1q schema 199.3/135.3/135.9/82.8/138.8/107.8; 4q plain 277.9/194.8/198.3/127.6/198.3/159.2;
  4q schema 204.6/140.6/141.1/88.2/144.1/112.5; 15q plain 398.2/288.4/289.0/194.6/293.6/238.6; 15q schema 212.2/148.2/148.9/95.9/151.7/120.3.
- W4A4 1q schema split: T=1000 GEMM 12.1 | attn 1.0 | glue 4.8 | fla GDN 3.4 | other 0.2 (non-GEMM 44%); T=4000 43.7 | 8.5 | 17.8 | 12.3 | 0.6 (47%).
- Schema compile (once per deployment, 15q bundle 3720 tok): 186 ms bf16 / 81 ms W4A4 eager; cache 115 MB.
- Projections (proj.py; dense peaks, bandwidth for memory-bound kernels): W4A4 1q schema T=1000 -> 3090 12.5, 4090 8.1, 5090 (NVFP4) 5.3 ms;
  W8A8-GPTQ-b8 -> 18.3 / 10.7 / 7.3; bf16 (fp16 acc) -> 27.4 / 13.7 / 10.0. Non-GEMM share at W4A4: 52% / 69% / 62%.
- Report returned to the coordinator as text (the harness blocks subagents from writing report .md files, so no REPORT.md was written).

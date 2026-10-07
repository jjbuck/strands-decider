# Q2 notes (box q2): kernels for 4-bit decision GEMMs on the A10G

## 20:10 PDT start
- Read BRIEF10 (Q2 section), IDEAS_EXPLORED "The floor on wall time", FAST_DECISION_MODEL 2.2/5/9, H1 FORMAT.md, H2 REPORT/NOTES/code
  (qrt.py, qgemm.py, qk.py, g2s4.cu, h2mix.cu, h2evt.cu), J5 report + g2s4x.cu/j5rt.py, J15 report/setup/bench.
- Box q2 (i-REDACTED) pending; waiting for boxes/q2.ready.
- Deployed arithmetic (H2) read from code: glue quantizes per token s=div_rn(max(amax,1e-8),qmax)*clip, q=rint(div_rn(y,s)) clamp;
  CUTLASS s4/s8 output C16 = fp16(alpha*acc), alpha = power of 2 (s4: 0.5 @K2048, 0.125 @K6144; s8: 2^-10 @K2048, 2^-11 @K6144);
  consumer dequantizes ((C16*ra)*csa) in fp32, csa = s_w/alpha; EVT gate_up: ((acc*ra)*cs) fp32 then SwiGLU with __expf.
- Plan: (1) FORMATS.md v0 now; (2) one CUDA kernel family (raw mma.sync s4/s8/bf16 + cp.async, 64-byte K tiles so all three
  precisions share one smem pipeline) with K-segments (int4 main + bf16/int8 tail = B1 low-rank correction and ResQ split),
  per-group scaling, row-role grouping (two problems, one launch), batched launch (Strassen products), epilogues (bf16 store,
  SwiGLU, table gather for B3); (3) Triton prologue variants (SR, deltaX, group scales, prototype search); (4) Strassen
  measurement; (5) integration into QRT.
- Cost arithmetic before building [arithmetic]: B1 extra MACs per GEMM = M*K*r (Z = dX*A) + M*r*N (tail). At bf16 rate (1/4 of int4)
  the tail costs 4r/K of the int4 main loop: r=64 -> 12.5% at K=2048, 4.2% at K=6144. Computing Z inside every N-tile CTA (the
  brief's literal design) repeats M*K*r per N-tile: cost 4r/BN of the main loop (r=64, BN=128: +200%), so Z must be computed once
  per row block (separate skinny GEMM or prologue) and only Z*P^T fused into the int4 GEMM.

## 20:35 PDT box ready, H2 libs built, GPTQ codes done; q2gemm v1 bit-exact but slow
- Box q2: A10G 1710 MHz, nvcc 12.8, CUTLASS v3.5.1 + main (4.8.0). TTL poweroff 05:36 PDT. H1-recipe GPTQ8/GPTQ4 codes regenerated
  (~/work/h2/codes_gptq_w{8,4}.pt). Addendum (B5-B10) read: new Q2 priority B1 tail > B5 K-slice split > B8 arbitrary row partition >
  B9 tile skipping > B6 statistic > rest (Strassen etc.).
- q2gemm.cu v1 (raw mma.sync s4/s8/bf16, 64-byte K tiles, K segments, group scales, B9 skip, row-role two-problem launch, epilogues):
  test_q2.py: ALL variants BIT-EXACT vs the FORMATS.md torch reference (s4, s8, s4+s8 tail, g64, g128, skip, swiglu, fp16a, row-role
  with random rowmap) at (77,256,256), (300,520,384), (1125,2048,2048); bf16 / bf16-tail within fp32 rounding [MEASURED].
- v1 speed at M=1125 [MEASURED, us, median of 30]: Q2 s4 is 1.2-1.7x slower than H2's CUTLASS s4 (gdn_in 255 vs 173; gate_up 416
  (swiglu) vs 244/255(evt); down 177 vs 139); s8 423 vs 322. Cause: v1 loads all fragments of a tile then issues all mma (no
  ldmatrix/mma overlap) and syncs per 64-byte tile. v2 = CUTLASS MmaMultistage order (fragment double buffering across k-steps,
  one barrier before each tile's last k-step), 128-byte K-tile configs, precomputed ldmatrix offsets.

## 21:02 PDT q2gemm v2 speed + first variant sweep (res/res_gemm_base.jsonl, res/res_gemm_var.jsonl) [MEASURED]
- v2 (CUTLASS-order main loop, precomputed copy pointers) is 0.85-0.95x of H2's CUTLASS s4 speed and 0.85-0.92x of its s8 at hobson's
  shapes (us at M=1125, best config each: gdn_in Q2 s4 194 vs CUTLASS 172; gate_up swiglu 294 vs EVT 255; down 155 vs 139;
  out 74 vs 60). ncu: same instruction count as CUTLASS, but 3x more 'wait' + 7x more short-scoreboard stalls; not closed. Variant
  overheads below are against Q2's own s4 (same main loop); absolute times are compared with H2's CUTLASS where it matters.
- GEMM-time arithmetic at M=1125 from the standalone timings [arithmetic over MEASURED]: W8A8-b8 (deployed: 88 CUTLASS s8 incl. EVT
  gate_up + 8 cuBLAS bf16) = 27.5 ms; the 0.65x bar = 17.8 ms. Plain W4A4: CUTLASS 14.7 ms (0.535x), Q2 kernels 16.8 ms (0.61x).
- Overheads vs Q2 s4 at M=1125 (ratio of medians; Z GEMM separately):
  B1 bf16 tail r=32 1.04-1.07, r=64 1.04-1.11, r=128 1.08-1.21, r=256 1.15-1.41 (prediction 4r/K: 6/12/25/50% at K=2048).
  B1 int8 tail r=64 1.07-1.18, r=128 1.09-1.22, r=256 1.12-1.30 (cheaper than bf16 only at r=256).
  Z = dX*A as a separate GEMM: cuBLAS 37-42 us, Q2 bf16 57 us at M=1125 (both latency/occupancy bound for N=r) -> must be fused.
  group-scaled int4 g64 1.31-1.51, g128 1.28-1.44.  B9 skip f=.75 0.84-0.90, f=.5 0.69-0.81.
  B5/ResQ split int8 128 + int4 1920: 1.11-1.14; K-shrunk int8 256 + int4 1280: 0.98-1.04; int4 only 1536 of 2048: 0.81-0.87.
  row-role (1000 state rows int4 + 125 int8, one launch) 1.23-1.43x all-int4 (ideal 1.11); random 10% int8 partition: same cost.
- Next: int8 CTAs first in the row-role launch (they are the slow tail), packed SwiGLU stores, fused quantize+Z prologue, B6 stat,
  Strassen, B3, FORMATS v1.

## 21:30 PDT prologues, row-role, Q1 direction, reassignment [MEASURED unless marked]
- Coordinator: B11 sparse, B13, Strassen, B14 moved to Q4 (Addendum 3). q2sp.cu (CUTLASS sparse) left in code/ as an untested start
  for Q4: only s8 cfg8 passes an identity-metadata check; s4 wrong; other cfgs fail initialize. Not pursued further.
- Q1 best_formats.json (21:12): B1.1 says decision sensitivity is low-rank only for QUESTION rows at layers 13-23; state rows have
  r99 = 47-92% of K. Candidates: w4q8 (state rows W4A4, question rows W8A8, separate int8 weight copy), tc4q8 (B5 slices on state
  rows + int8 question rows). => integrate row-role first.
- torch pitfall found and documented in FORMATS section 0: x / python_scalar on CUDA multiplies by the reciprocal (1-ulp scale
  differences). With tensor division all Triton prologue modes are bit-exact vs torch (RTN, SR hash, dX, g64, split, row-role,
  B6 stat, B3 search idx/codes/scales).
- Prologue cost inside a CUDA graph (res/res_pro_graph.jsonl), H2-style residual-add + RMSNorm + quantize kernel, M=1125, K=2048:
  RTN 31.6 us; + SR 1.3 us (4%), + row-role 0.4, + B6 stat 1.3, + g64 0.3, + split 1.1, + dX bf16 write 9.9 (31%), + bf16 copy for B3 9.9.
  Pure quantize at K=6144, M=1125: RTN 39.3 us; SR +12.4 (32%), dX +26.1 (66%). (Event timing of single Triton launches is host-bound
  below ~1000 rows: 46-48 us per call; use graph timing.)
- B3 search + residual quantize (Triton, int8 scores), M=1125: K=2048 Kc=64/256/1024 = 111/166/486 us; K=6144 283/442/1372 us
  (event timing). For scale: the int4 GEMM it feeds is 74-297 us -> B3 costs 0.4-6x its GEMM; needs a fused/cheaper search to matter.
- Row-role GEMM time over 96 GEMMs, T=1000 + 125 question rows (res/res_rr.jsonl, rotating input buffers):
  all-W8A8 CUTLASS 26.50 ms; all-W4A4 CUTLASS 14.71 (0.56x); Q2 all-int4 16.78 (0.63x); row-role: Q2 one launch 20.92 (0.79x),
  CUTLASS s4 + s8 back to back 20.13 (0.76x), two streams 21.59 (0.81x). The 125 int8 rows cost ~5.4 ms, about 4x their share of
  work: each 125-row int8 GEMM streams a second (int8) weight copy, 1.37 GB per forward = 2.3 ms at 600 GB/s, and runs one wave.
  With a single input buffer (rr2.py) the one-launch row-role is 17.63 vs 16.24 ms all-int4 (+8.6%): L2 residency moves these by ~15%;
  the e2e graph measurement decides. Interleaving the int8 tiles through the launch (sched=1) was slower (18.60).
- At 15 questions (3720 question rows of 4720) row-role is 0.90-0.97x of all-int8: no gain, as expected.

## 21:50 PDT end-to-end runtime with row-role GEMMs (q2rt.py) [MEASURED, A10G exclusive, graph replay, fresh state ids, 20 reps]
- q2rt.QRT2 = H2 QRT + qk2 glue (row-role quantizer: rows >= q0 int8) + q2gemm two-problem GEMMs (bf16-direct outputs, unit scales
  in the consumers, SwiGLU epilogue). QRT2C = state rows on H2's CUTLASS W4A4 (bit-identical to H2 w4a4: final-hidden cos 1.00000)
  + question rows on q2gemm int8 written in the int4 GEMM's fp16 units (C16 = fp16(acc8*g_n), activation scale stored x32), optionally
  on a side stream (Q2SIDE=1). Sanity (300 random ids): QRT2 q0=0 vs H2 w8a8 cos .99995; q0=T vs bf16 .98899 (H2 w4a4 .98902);
  question rows at q0=200 vs bf16 .99916 (QRT2) / .99908 (QRT2C); state rows identical to all-int4 (causal).
- e2e medians (ms; p95 within 0.5 ms), 1 question = details_match 125 tok, 15 q = 3720 tok (hobson layout, packed branches):
  | T | q | b8 (H2) | w4a4 (H2, fails acc.) | w4q8 Q2 one launch | w4q8c seq | w4q8c side stream | w8 Q2 kernels | w4 Q2 kernels |
  | 1000 | 1 | 36.21 | 23.46 | 27.26 | 27.16 | 26.34 | 38.39 | 24.65 |
  | 64 | 1 | 8.58 | 5.92 | 9.92 | 10.00 | - | 8.89 | 6.37 |
  | 256 | 1 | 14.38 | 9.68 | 12.81 | 12.43 | 11.61 | 14.51 | - |
  | 4000 | 1 | 142.61 | 85.09 | 95.96 | 89.25 | 88.75 | 150.16 | 92.17 |
  | 1000 | 15 | 165.57 | 102.41 | 156.35 | 153.03 | - | 177.83 | - |
  | 64 | 15 | 128.92 | 77.89 | 132.77 | 130.92 | - | 132.72 | - |
  | 256 | 15 | 140.54 | 83.51 | 136.89 | 134.90 | - | 143.44 | - |
  | 4000 | 15 | 293.43 | 193.78 | 240.04 | 229.74 | - | 326.64 | - |
- => row-role with CUTLASS int4 + concurrent int8 question rows: 26.34 ms at T=1000/1q = 0.727x b8 (bar 0.75x); 0.62x at T=4000;
  0.81x at T=256; slower than b8 at T=64 (question rows are 66% of rows). With 15 questions it is 0.78-1.02x b8 (no lever).
- Kernel-time sums in the graph (profiler, overlapping streams counted separately): b8 GEMM 25.86 ms, w4a4 13.00, w4q8 one-launch 16.68.
- q2gemm main loop improvements tried today: CUTLASS-order copies, interleaved ldmatrix/cp.async between mma, serpentine n order,
  L2::128B prefetch: plain s4 at gdn_in M=1125 went 234 -> 196 us (fp16 out); no-store main loop 180 us vs CUTLASS 173 total.
- Evals queued (all 3227 questions, H1-recipe GPTQ codes of this box): q2_w4q8c (QRT2C), h2_b8, h2_bf16.

## 22:09 PDT accuracy through the deployed kernels (all 3227 questions; this box's H1-recipe GPTQ draw) [MEASURED]
- h2_b8 (H2 W8A8-GPTQ-b8): REAL flips 1.11% (12/1083) vs hobson, TV .0048, CF ret 1.000, CF-probe ret .962, JB-hard .538 (McNemar 0/2),
  REAL-label .792. (Same draw-dependence J15 reported: b8 itself is 0.65-1.11% across draws, so the 0.7% flip bar is at the edge of
  b8's own variance.)
- q2_w4q8c (row-role: state rows W4A4 via H2 CUTLASS, question rows W8A8 via q2gemm): REAL flips 3.88%, TV .0320, CF ret .835,
  CF-probe ret .762, JB-hard .538 (McNemar 8/10, p .81), REAL-label .782, LONG agree_sd .921. Fails flips / CF / CF-probe. Agrees with
  Q1's DEV fast screen (w4q8 9/300 flips, rms margin change .200 vs .03 for W8A8): state-row error must fall ~5x.
- Added: B8 arbitrary-partition runtime path (QRT2.set_rowmask; qk2 POS map; q2gemm rowmap scatter), per-GEMM maps
  (q2map_k24/k48/k64rr = H1/H6's 24/48/64 most sensitive GEMMs all-int8, rest row-role), fused B1 prologue (q2pro._qz_k: int4 codes +
  Z = dX A in one pass, optional B6 subspace stat), score_q2.py (bar check), proj_q2.py (H2's projection model).
- Queued: e2e for k24/k48/k64 row-role maps and B8 extra int8 state rows (5/10/25%), evals of ck48rr / ck64rr, kernel-split e2e for
  projections, fused-Z timing.

## 22:17 PDT more accuracy and speed through the deployed kernels [MEASURED]
- h2_bf16 runtime (floor): REAL flips 0.37%, TV .0032, CF 1.000, CF-probe .971, JB-hard .538, REAL-label .785 (passes, as before).
- h2_b8 vs bf16 runtime: 10 model-only / 2 base-only REAL flips, McNemar p .039 -> this box's b8 GPTQ draw itself misses the bar.
- q2_ck48rr (H1's 48 most sensitive GEMMs all-int8, the other 48 row-role): REAL flips 1.29% (13/3 vs bf16, p .021), TV .0106,
  CF ret .982, CF-probe .914, JB-hard .531, REAL-label .792. e2e 31.73 ms at T=1000/1q (0.876x b8; int8 GEMMs then on q2gemm).
- e2e (side stream) k24rr 28.24 / k48rr 31.73 / k64rr 33.16 ms at T=1000/1q; 98.4 / 114.7 / 122.8 at T=4000; 12.0 / 12.9 / 13.3 at 256.
- B8 (5% of state rows int8 at random positions + question rows, one q2gemm launch): 27.22 ms vs 27.26 contiguous row-role at
  T=1000 -> extra int8 rows are nearly free once the int8 weight copy is streamed for the question rows. Sweep 10/25/50% queued.
- QRT2C now routes all-int8 ('w8') GEMMs of a map to H2's CUTLASS s8 / EVT (deployed arithmetic, faster than q2gemm int8).
- Standalone GEMM sums vs b8 at M=1125 (res/gemm_tables.json): W4A4 CUTLASS .535, q2 s4 .614, B1 bf16 tail r=32/64/128/256 (no Z)
  .644/.666/.717/.811, group g64/g128 .891/.846, B9 f=.75/.5 .531/.441, ResQ int8-128 + int4 .685, B5 int8-256 + int4-1280 .617,
  row-role (125 int8 rows, one launch) .801.

## 22:24 PDT first format that passes the fidelity bar through the deployed kernels [MEASURED, all 3227 questions]
- q2_ck64rr = H6's 64 most sensitive GEMMs all-int8 (H2 CUTLASS s8 / EVT) + the other 32 GEMMs row-role (state rows W4A4 on H2
  CUTLASS, question rows W8A8 on q2gemm). REAL flips 0.65% (7/1083) vs hobson; vs the bf16 runtime 6 model-only / 3 base-only
  (McNemar p .51); TV .0058; CF retention 1.000; CF-probe retention .962; JB-hard .523 (McNemar vs hobson 0/0); REAL-label .790;
  LONG agree_sd .976 -> passes every fidelity criterion of BRIEF10 (b8 with the same GPTQ draw does not: 1.11%, p .039).
- Its 4-bit share: the 32 row-role GEMMs are 37.9% of GEMM MACs; state-row int4 = 33.7% of all MACs at T=1000/1q.
- Speed (q2gemm int8 for the 'w8' GEMMs, before routing them to CUTLASS): 33.16 ms at T=1000/1q = 0.916x b8; re-measuring with
  CUTLASS int8 now (run_bench5). It cannot reach the 0.75x e2e bar: with 66% of MACs at int8 the GEMM floor is ~0.83x of all-int8.

## 22:30 PDT k-map frontier through the deployed kernels; projections [MEASURED; projections arithmetic]
- q2_ck56rr (56 most sensitive GEMMs int8, 40 row-role): REAL flips 0.65% (4/1 vs bf16, p .375), TV .0068, CF ret .991, CF-probe
  ret .943 (bar .95), JB-hard .515 (1/0), REAL-label .782 -> misses only CF-probe. (k48 / k56 / k64 are prefixes of one fixed H6
  ranking; k56 was run after seeing k48 fail and k64 pass, i.e. a 1-D search on eval.)
- e2e with CUTLASS int8 for the 'w8' GEMMs + side-stream question rows (res_e2e_k64.jsonl), ms (b8 in brackets):
  k64rr: T=64 8.50 (8.57), 256 13.42 (14.39), 1000 31.90 (36.36) = 0.877x, 4000 119.8 (142.6) = 0.84x; 15 q at 1000 159.5 (165.6).
  k48rr: 8.84 / 13.00 / 30.91 (0.850x) / 112.8; 15 q at 1000 158.8.
- B8 extra int8 state rows (QRT2 one launch, random rows), T=1000/1q: 0/5/10/25/50% -> 26.33/26.90/28.62/29.02/31.68 ms;
  T=4000: 92.49/94.66/96.78/101.66/113.34.
- Kernel split of sequential row-role at T=1000/1q: CUTLASS int4 state rows 12.01 ms, q2gemm int8 question rows 4.81 ms (125 rows x
  96 GEMMs, weight-streaming bound: 1.37 GB of int8 weights at ~290 GB/s effective); side stream hides 1.06 ms of it.
- B1 Z cost in a CUDA graph (res_qz.jsonl), M=1125: dX write adds 10.6 us (K=2048) / 26.0 us (K=6144) to the quantize kernel, cuBLAS
  Z = dX A takes 9.9-23.8 us (K=2048, r=32-256) / 36.7-67.1 us (K=6144). The fused Triton quantize+Z kernel was slower (71-95 /
  210-274 us). B1 on all 96 GEMMs would add ~3-4 ms of Z work + 4-11% tails at T=1000: affordable only on a few GEMMs.
- Projections (H2 model; q2 int8 question rows scaled by DRAM bandwidth below 500 question rows), T=1000/1q, ms:
  b8 3090 19.9 / 4090 11.6 / 5090 7.9; w4a4 13.6 / 8.9 / 5.8 (5090 = NVFP4, other format); row-role w4q8c 16.2 / 11.6 / 7.2;
  k64rr 18.2 / 10.9 / 7.3 (all its GEMM time scaled at the int8 ratio). The int8 question rows' weight streaming grows relative
  to compute on Ada/Blackwell, so row-role gains shrink there (4090: row-role = b8).

## 22:38 PDT full rebuild with the v3 main loop (interleaved ldmatrix/cp.async, serpentine, L2::128B) [MEASURED]
- test_q2.py: ALL OK (every variant x config bit-exact). res/res_gemm_var2.jsonl, sums vs b8 at M=1125: q2 s4 .600 (v2 .614),
  row-role one launch .725 (v2 .801); but B1 tails got relatively worse (r=32/64/128/256 .656/.680/.731/.823 vs v2 .644/.666/.717/.811)
  and group scales regressed (g64 1.107 vs .891; g128 .871 vs .846): the group path does not use the interleaved loop and its copy
  order changed. The report quotes v2 for tails/group (each variant against its own version's s4) and v3 for plain/row-role.
- B8 digit test running: Qwen tokenizes numbers one digit per token; 28 vocab ids contain a digit.

## 22:47 PDT Q1's B5 (tc*) formats need a dense per-GEMM basis online [arithmetic over MEASURED]
- Q1 DEV screen (their NOTES): tc45rq8 2/300 flips, rms dm .088 (w4q8 .200, b8 .030), 'MAC time vs W4A4 1.12'. But q1fmt.qgemm_tc
  projects every GEMM input with dense K x (n8+n4) matrices (UR8, UR4): one extra ~K x K GEMM per input. At M=1125 a bf16 2048^2 GEMM is
  189 us [M]; 6144^2 ~1.45 ms [A]; per request ~48 ms in bf16 (~24 ms in int8) [A] > b8's 36 ms. Added to FORMATS 8.2 in bold so Q1
  sees it. Only shared (foldable into R1), block-diagonal (<= 256) or permutation/scale bases keep B5 fast.

## 22:50 PDT B8 digit rows, W4A8 question rows [MEASURED]
- B8 with an a-priori row class (state tokens that are digits run W8A8; Qwen tokenizes numbers one digit per token; 14.9% of state
  rows), QRT2 one-launch path, all 3227 questions:
  dk48rr (k48 map + digit rows): REAL flips 1.66% (16/2 vs bf16, p .001), TV .0101, CF ret .972, CF-probe ret .952, JB-hard .546,
  REAL-label .797. dw4q8 (row-role everywhere + digit rows): 3.32%, TV .0309, CF .725, CF-probe .781. Digits in int8 move CF-probe
  (k48: .914 -> .952) but not flips or TV; same-path k48 control (QRT2 without digits) queued to separate path noise from the digits.
- W4A8 question rows (H2's CUTLASS s8 x s4 mixed input, one int4 weight copy), 125 rows, standalone sums over 96 GEMMs: 8.18 ms vs
  7.33 ms for W8A8 question rows (T=1000; same at 256/4000). So the second weight copy is NOT what makes question rows expensive: at
  125 rows each GEMM is one wave of tiles (about 76 us average), occupancy/latency bound for int8 and int4 weights alike.

## 22:55 PDT controls and basis cost [MEASURED]
- Same-path control q2_k48rr_qrt2 (k48 map through QRT2, no digit rows): REAL flips 1.48%, TV .0102, CF ret .982, CF-probe .924,
  JB-hard .538, REAL-label .790. Against it, digit state rows in int8 (dk48rr) change CF-probe .924 -> .952, CF .982 -> .972,
  flips 1.48 -> 1.66%, TV .0102 -> .0101: CF-probe (value comparisons in JSON records) responds to value-bearing rows, the rest is noise.
  The two QRT2/QRT2C paths of the same k48 format differ by 1.29 vs 1.48% flips and .914 vs .924 CF-probe (path noise at this level).
- Dense basis change per GEMM input (Q1's tc* as emulated), CUDA graph, M=1125: 2048^2 bf16 170 us / int8 90 us; 6144^2 1417 / 712 us.
  Per request at T=1000: 46 ms (bf16) / 24 ms (int8) -> FORMATS 8.2 updated with the measured numbers.

## 23:05 PDT split-K for the 125 question rows: correct, slower [MEASURED]
- q2gemm split-K (int32 partials [splits][M][N], last CTA of a tile sums them in fixed order -> deterministic, bit-exact; counters
  self-reset so graph replays work): test_sk.py ALL OK at (125, 2048, 6144), (125, 2048, 2048), (77, 520, 1024), (300, 12288, 2048).
- Question-row int8 GEMMs, 125 rows, sum over 96 GEMMs in a CUDA graph: no split 4.55 ms, split 2 7.15, split 4 10.23; H2 CUTLASS s8
  4.95. Down (K=6144): 44.1 / 52.6 / 57.9 us. The partial write + reduction outweighs the extra parallelism. The 4.55 ms is ~2x the
  int8-weight streaming floor (1.37 GB at 600 GB/s = 2.3 ms).
- Final full build + all tests rerun on the box (build_final.log, test_final.log).

## 23:13 PDT wrap-up
- Final q2gemm source (v3 main loop + split-K + epi 6/9) built in full on the box: test_q2.py ALL OK (every variant x config
  bit-exact), test_sk.py ALL OK. Logs copied to res/box_logs/. DRAFT_REPORT.md written. Box left idle (TTL 05:36).

## 2026-10-07 06:05 PDT phase 2: k64rr + layer-16 exit (approved next step), box q2b
- Box q2 expired; q2b (g5.2xlarge, i-REDACTED) launched 06:14, waiting for .ready.
- Plan: rebuild (setup_q2b.sh: H2 libs, H1-recipe GPTQ codes, q2gemm, J15 DEV/EXIT suites, bundles) -> q2run.py k64rr through QRT2C on
  eval / DEV / EXIT with layer-16 taps -> q2exit.py (J15's EH recipe on k64rr residuals, KL to k64rr's final distribution on EXIT,
  early stop on DEV) -> q2exit_an.py (tau for zero DEV residual changes, applied unchanged to eval) -> score_q2.py ->
  q2xbench.py grid (full / pre16 / post16 at T=64..4000, 1q/15q; b8 and k64) and real (J15's 120 requests).
- Code added: QRT2.forward/_fwd_q take (x0, i0, i1) and taps like J15's QRTJ; q2run.py, q2exit.py, q2exit_an.py, q2xbench.py.
- "One graph" note: CUDA graphs cannot branch on a device value without conditional nodes (not exposed by torch), so the cascade is
  two captured graphs, [0,16)+exit head and [16,24)+head, with one host read of the margin between them (J15's dcasc design).
- B12/B13 arithmetic before building: both mainly remove work in layers 16-23 (B13: layer 23; B12: 60% of neurons in 12-23, mostly
  16-23: kept 2280..919 per layer vs 6144 at 12-13), which the exit already skips for the requests that exit. Within the cascade they
  can only act on layers 12-15 (B12: 13% / 35% of layer 14 / 15 MLPs) and on the non-exiting requests. Build only if time remains.

## 06:58 PDT q2b rebuilt; k64rr reproduces bit for bit [MEASURED]
- q2b ready 06:41; setup_q2b.sh: CUTLASS v3.5.1 + main (4.8.0), H2 libs rc 0, H1-recipe GPTQ8/GPTQ4 codes, q2gemm full build rc 0,
  test_q2 ALL OK, rtcheck identical to q2 (QRT2C q0=T vs H2 w4a4 cos 1.00000), bundles. TTL 16:15 PDT.
- xcheck.py: k64rr [0,16) then [16,24) from the saved residual == full forward (max |d| 0); layer-16 tap == residual.
- k64rr on all 3227 questions (this box's codes): REAL flips 0.65% (6/3 vs bf16 runtime, p .508), TV .0058, CF 1.000, CF-probe .962,
  JB-hard .523 (0/0), REAL-label .790 -> identical to the q2-box run (the H1 recipe regenerates the same codes).
- Running: k64rr DEV / EXIT with layer-16 taps, then the exit head, b8 and bf16 references.

## 07:20 PDT exit head on k64rr; cascade fidelity [MEASURED, all 3227 questions]
- Exit head (J15 EH: hobson's head + rank-512 adapter, LayerNorm input) on k64rr's layer-16 residual; trained by KL to k64rr's final
  distribution on EXIT (4205 train-split questions), early-stopped on DEV (1736): best DEV KL .00044, DEV agreement .9902 at epoch 35
  (J15's b8 head: .0004 / .9937). 53 s on the A10G.
- tau for zero DEV residual changes (eps 0): .0540; DEV exit share .961. Applied unchanged to eval: exit share .942 (JB-all .965, REAL
  .958, LONG .947, CF .950, CF-probe .895). Exits change 4 of 3227 decisions vs full k64rr (JB-all 1, CF 2, CF-probe 1; REAL 0, LONG 0).
- k64rr + exit16: REAL flips 0.65% (6/3 vs bf16 runtime, p .508), TV .0097 (k64rr .0058), CF ret 1.000, CF-probe ret .952 (k64rr .962;
  bar .95, one pair of margin), JB-hard .531 (McNemar vs hobson 0/1), REAL-label .790 -> passes the fidelity bar.
- Multi-question exits (all questions of a request must exit to skip layers 16-23): DEV 4-question requests .860 (per question .962;
  .962^4 = .857: near independence) -> 15 questions ~.96^15 = .55 [A].

## 07:58 PDT cascade latency [MEASURED; expected values = measured segments x measured exit share, ARITHMETIC]
- Segment graphs (q2xbench grid, 20 reps, p95 within 0.4 ms), ms, b8 / k64rr: full, exit path (layers 0-15 + exit head), suffix 16-23:
  T=1000/1q b8 36.34 / 24.33 / 12.27; k64 32.55 / 23.20 / 9.80. T=4000/1q b8 142.41 / 94.95 / 47.83; k64 120.30 / 88.27 / 32.33.
  T=256/1q b8 14.39 / 9.71 / 4.95; k64 13.99 / 9.61 / 4.61. T=64/1q b8 8.59 / 5.85 / 2.98; k64 9.31 / 6.00 / 3.54.
  15q (3720 question rows): T=1000 b8 165.75 / 110.77 / 55.74; k64 159.18 / 107.70 / 51.24.
- Expected k64rr+exit at the eval exit share .942 (1q): T=64 6.20 (0.722x b8), 256 9.87 (0.686x), 1000 23.77 (0.654x), 4000 90.14
  (0.633x). 15q with all-exit ~.55: 0.76-0.82x. GEMM kernel time at T=1000/1q: 16.02 + .058 x 6.20 = 16.38 ms = 0.636x b8's 25.78.
- J15's 120 real requests, this box, same items (res_xreal.jsonl), mean / median / p95 ms: b8 79.7 / 63.9 / 217.3 (J15's file 79.7 /
  64.3 / 216.6); k64rr 69.3 / 55.6 / 186.6 (0.869x); b8+exit16 timed here with J15's recorded exits 54.9 / 42.7 / 146.8 (0.688x; J15
  54.8 / 42.8 / 146.1); k64rr+exit16 (my head, tau .054) 50.6 / 40.0 / 137.7 = 0.635x b8 mean (0.922x b8+exit); exit share 117/120.
  By group (k64rr+exit vs b8 mean): REAL 54.9 vs 88.1 (0.623x), LONG 128.1 vs 201.0 (0.637x), JevBench 16.7 vs 24.8 (0.673x).
- Interpolated over all 1,785 REAL+LONG+JevBench eval questions at their exact lengths and exits: 0.635x b8 (b8+exit at the same
  exits 0.685x).
- Why the stack gains little over b8+exit: k64's 32 row-role GEMMs sit in layers 13-23 (7 in 13-15, 25 in 16-23) and the exit skips
  16-23; in the prefix k64rr is all-int8 except 7 GEMMs, and its gain over b8 there is mostly b8's 7 bf16 Wo GEMMs running in int8.
- bf16 runtime evaluated again on this box: 9 of 3227 decisions differ from the q2-box run (max |dp| .017): GPU-level nondeterminism
  of the floor itself.
- Projections (H2 model, all GEMM time at the int8 ratio), T=1000/1q: k64rr+exit 3090 13.1, 4090 7.8, 5090 5.3 ms (b8 19.9 / 11.6 / 7.9).
- B12 / B13 not built: inside the cascade they act only on layers 12-15 and on the 6% of requests that continue. B12's removed neurons
  in layers 13-15 are ~2% of the prefix's GEMM MACs; B13's layer-0 table saves 0.6% (Q4) and its layer-23 cut only applies to the
  suffix -> ~1-2% end to end together [A], each needing re-validation with the exit head (B12 changes the layer-16 residual).

## 08:05 PDT report updated (section 0 + condensed sections 1-7); results copied (res_xgrid, res_xreal, exit_q2b_k64x16.json,
   proj_cascade.json, preds/k64_*, preds/exit_k64_L16_*, preds/preds_q2b_*). Exit-head weights stay on q2b (preds/exithead_k64_L16.pt).

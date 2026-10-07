# J5 NOTES

## 2026-10-06 09:00 start
- Read BRIEF8, FAST_DECISION_MODEL.md. Box j5 launched 08:57 (i-REDACTED, g5.2xlarge), waiting for .ready.


## 2026-10-06 09:35 box j5 ready (09:17), setup done
- Box: A10G 1710 MHz, torch 2.14.1+cu130, triton 3.8, fla 0.5.2, nvcc 12.8. Built h2 CUTLASS libs (g2s4 / h2mix / h2evt) OK.
- MEASURED token lengths (exact hobson ids, prep_b.py -> lenstats.json):
  JB-all: state median 39 tok, question median 103, total median 143 (p75 565). 58% of JB-all inputs <= 256 total tokens, 69% <= 400.
  JB-hard total median 502. REAL-agree total median 2043 (only 0.4% <= 256). Question tokens are ~72% of a median JevBench input.
- Bundles: jb1 (108 tok), jb4 (429 tok, 4 JB questions near median), bk1 details_match (125), bk4 (392, 4 REAL questions near median).
- FIRST-PRINCIPLES (arithmetic): hobson streams 1.3726e9 GEMM params = 2.745 GB bf16; 2.745 GFLOP per row.
  Row count M = state + question rows (~108 per question). Roofline crossover (compute = weight bytes) per card:
  3090 (936 GB/s; fp16-acc 142 TF, int8 284 TOPS): bf16 weights M*=152, W8A16 76 (fp16 MMA), W4A16 38, W8A8 152, W4A8 76.
  A10G (600 GB/s; 70 TF bf16, ~140 int8): bf16 M*=117, W4A16 29, W8A8 117.
  => for a median JevBench decision (M~143) on a 3090, W4A16 is compute-bound at the fp16 rate (2.8 ms floor at peak) and is SLOWER
     than W8A8 (max(1.38 compute, 1.47 bytes) ms). Weight-only 4-bit only pays on Ampere below ~40 rows.
  => target "<=2-3 ms at <=256 state tokens" on a 3090: M = 256+108 = 364 -> W8A8 compute floor 3.5 ms at 100% of int8 peak;
     fp16-acc floor 7.0 ms. Unreachable with 8-bit or 16-bit activations; only int4 compute (1.76 ms floor) fits, and W4A4 fails accuracy.

## 10:00 bf16 short baseline + kernel work
- MEASURED (A10G, h2 QRT fold bf16 runtime = d1 fused, graph replay, fresh ids, 20 reps): state T + jb1 (108-tok question) / jb4 (429):
  T=32: 15.47 / 34.36 ms; 64: 15.75 / 34.68; 128: 16.10 / 38.72; 256: 25.50 / 45.29; 400: 32.96 / 50.19. p95 within 0.03 ms.
  GEMM = 14.0 ms of 15.5 at 140 rows: the fold Triton GEMM uses BM=128 for M<=384, so 140 and 236 rows both cost 2 row tiles (~7 ms per
  128-row tile, ~50 TF). Tile quantization at short M is the dominant waste: roofline for 140 rows = max(5.5 compute, 4.6 bytes) ms.
  GDN core 0.87 ms (90 kernels), glue 0.27, attn 0.11, other 0.27; in-graph gaps 0.02 ms (403 kernels). Launch gaps are NOT the cost.
- sk.py: one Triton GEMM template (bf16 / W8A16 / W4A16 g64|g128 / W8A8 / W4A8) with fold epilogues (row-scale, SwiGLU, residual+sumsq) and
  h2's int interfaces; SPLIT=1 verified to bf16 output rounding (rel 1.7e-3) on all modes/epilogues. The in-kernel split-K fix-up is racy
  on large grids (random 1-7% errors with >~400 tiles, exact on small grids, even with data-dependent tickets) -> disabled.

## 10:50 GEMM microbench v1 (MEASURED, A10G, eval paused via SIGSTOP, DRAM-fresh weights: cycled copies > 48 MB) res_gbench_v1.jsonl
- Sum of the 96 GEMMs per forward (ms), best kernel per family:
  M=16:  cuBLAS bf16 7.64 | sk W4A16 4.97 | CUTLASS W8A8 5.21   (floors: bf16 4.6, W8 2.3, W4 1.2 at 600 GB/s)
  M=140: cuBLAS 12.2 | sk bf16 11.0 | sk W4A16 10.5 | CUTLASS W8A8 6.27 (compute floors: bf16 5.5 @70TF, int8 2.75 @140TOPS)
  M=364: sk bf16 20.1 | CUTLASS W8A8 10.9 ; M=508: 25.8 | 12.7
- => no kernel reaches DRAM bandwidth at small M: skinny N=2048 shapes (out, down) launch too few CTAs (down bf16 at M=16: 157 GB/s);
  W4A16 at M=16 only 141 GB/s effective. Weight-only 4-bit is already compute-bound (bf16 MMA) from M~100 and loses to W8A8 at M>=64.
  W8A8 above M~200 is compute-proportional to padded rows (~28 us per row-forward = ~98 TOPS incl. padding).
- Fixed split-K: replaced the in-kernel fix-up with a deterministic 2-kernel split (partials + reduce/epilogue kernel); exact everywhere.

## 11:50 e2e short-length matrix v1 (MEASURED A10G, exclusive, 20 reps, fresh ids; res_sbenche2e.jsonl). ms median (p95 within 0.05)
| rows (T+q) | h2 fold bf16 | j5 bf16 (sk GEMMs) | j5 W4A16 | h2 W8A8-b8 (CUTLASS) | j5 W8A8-b8 (best-of CUTLASS/sk) |
| 140 (32+jb1) | 15.47 | 10.71 | 11.59 | 8.15 | 7.82 |
| 172 (64+jb1) | 15.75 | 12.08 | 14.13 | 8.67 | 8.47 |
| 236 (128+jb1) | 16.10 | 14.68 | 17.63 | 9.70 | 9.69 |
| 364 (256+jb1) | 25.50 | 21.17 | 25.25 | 14.28 | 14.22 |
| 508 (400+jb1) | 32.96 | 27.66 | 33.07 | 17.28 | 17.27 |
| 461 (32+jb4) | 34.36 | 28.16 | 31.32 | 18.44 | 18.45 |
| 685 (256+jb4) | 45.29 | 38.71 | 45.84 | 25.85 | 25.85 |
- j5 W8A8 at 140 rows: GEMM 5.70, glue 0.98, GDN 0.87, other 0.19, attn 0.11 ms; 430 kernels, gaps 0.02. At 461 rows: GEMM 12.5, GDN 2.5, glue 2.2.
- W4A16 is slower than bf16 at every measured length on the A10G: compute-bound at the bf16 MMA rate from ~100 rows, and my Triton
  dequant kernel reaches only 60-210 GB/s at M=16 (smem round trip for the unpacked operand). Not worth more kernel work on Ampere.
- fla fused_recurrent instead of chunk for GDN: SLOWER (9.9 vs 7.8 ms at 140 rows). Chunk stays.

## 13:15 kernel limits at short M (MEASURED, A10G, eval paused)
- Achievable DRAM bandwidth: 512 GB/s read-only (85% of 600), 482 GB/s copy, 463 GB/s for 25 MB reads incl. per-call cost.
- Extended h2's CUTLASS int8 GEMM with small-M tiles (32x128, 32x256, 64x64, 16x128, 32x64x128) + serial split-K variant (exact):
  per-forward int8 GEMM sum: M=16 5.21 -> 4.24 ms; 140: 6.27 -> 5.17; 364: 10.88 -> 9.48; 508: 12.73 -> 12.13. Split-K adds nothing.
  Floors at 140 rows: int8 compute 2.75 ms @140 TOPS, W8 bytes 2.77 ms @510 GB/s -> GEMMs at ~55% of roofline.
- My Triton multi-M-tile skinny kernel (all rows per CTA, weights read once): exact, but no faster than CUTLASS (Triton int8 MMA efficiency).
- Per-kernel fixed cost of an int8 GEMM (n GEMMs of depth K vs one of depth nK): 5-14 us at M=16..364 -> ~0.6 ms per forward (96 GEMMs).
- Decomposition of the 7.8 ms W8A8-b8 decision at 140 rows: roofline floor ~3.2 ms (85% of peak) + GEMM steady-state inefficiency ~1.4
  + GEMM per-kernel fixed costs ~0.6 + non-GEMM kernels 2.1 (GDN 0.87 in 90 launches, glue 0.98 in 121, misc 0.3).
  => what a persistent megakernel could remove at most: ~2.7 ms (35%); in-graph launch gaps are only 0.02 ms.
- W8A16 (weight-only int8 GPTQ) accuracy: flips vs hobson 0.46% all / 0.37% REAL (bf16 runtime 0.50 / 0.28), vs in-runtime bf16 9/10 McNemar
  p=1.0; CF fgh .991, CF-probe fgh .971, JB-hard .531, REAL-label .787. Passes the fidelity bar.

## 14:35 final latency (MEASURED A10G exclusive, 30 reps; res_sbenchfinal.jsonl) + weight-only accuracy (scores_wq.json)
- j5 W8A8-b8 (CUTLASS small-M tiles + split-K + sk, best per shape), hobson layout (exact): jb1 T=32/64/128/256/400 -> 7.38 / 8.01 / 9.34 /
  13.31 / 16.98 ms; jb4 -> 18.05 / 18.48 / 19.96 / 24.84 / 28.81. T=1000 jb1 36.11 (h2: 36.3). bf16 fold at 1000: 56.88 (doc 57.0).
- Schema layout (latency only; needs a schema-first model that does not yet exist at fidelity): jb1 5.38 / 5.95 / 7.49 / 10.88 / 15.50;
  jb4 5.44 / 6.16 / 7.67 / 11.03 / 15.46. At short states question rows are 72-93% of all rows, so this layout is the 4-question lever (3.3x).
- W4A4 (h2, fails accuracy) reference: jb1 5.58 / 5.94 / 6.51 / 9.57 / 11.53; jb4 12.76 / 13.14 / 14.04 / 17.31 / 19.76.
- Weight-only accuracy (all 3227 questions, through the fold runtime with dequantized GPTQ weights): flips vs hobson all / REAL:
  bf16 0.50/0.28%, W8 0.46/0.37% (PASS, McNemar vs bf16 9/10), W4 g64 3.16/2.59%, W4 g128 3.19/2.49% (McNemar vs bf16 95/8 p<1e-3),
  CF-probe fgh W4g128 .771, W4g64 .905. 4-bit weight-only FAILS the function-preserving bar by ~6x.

## 15:35 deployed-kernel fidelity + remaining weight-only formats (MEASURED)
- Weight-only: W4g128 RTN 10.75% flips; W3g128 7.87%; MIX (W4 MLP + W8 mixers) 2.23% (REAL 1.29%), CF fgh .991, CF-probe fgh .810. All fail.
- Own GPTQ codes for H1 FORMAT v0 W8A8-b8 (rotated Hessians from my calibration), H2's CUTLASS kernels: flips vs hobson 0.62% all / 0.65% REAL,
  McNemar vs bf16 runtime 13/9 p=.52, CF fgh 1.000, CF-probe fgh .971, JB-hard .538, REAL-label .787 (H2 reported 0.18% REAL with H1's codes).
- j5 int8 kernels vs h2 kernels, same RTN codes, random ids T=32..400: final-hidden cos 0.99997, max|dp| 0.003-0.010 (runtime-noise level).

## 16:10 wrap-up
- Deployed j5-kernel W8A8-b8 with my GPTQ codes: flips 0.65% all / 0.74% REAL, McNemar vs bf16 runtime 13/8 p=.38, CF fgh 1.000,
  CF-probe fgh .971, JB-hard .538, REAL-label .787; h2-kernels vs j5-kernels on the same codes differ on 3 of 3227 decisions (TV .0016).
- Files: DRAFT_REPORT.md (final), scores_wq_table.md + scores_wq.json (all formats), res_sbenchfinal.jsonl / res_sbenche2e.jsonl /
  res_sbenchbase.jsonl (latency), res_gbench*.jsonl (GEMM microbench), res_ramp.json, lenstats.json, preds/ (all suites per format), code/.
- Box j5 left idle (no jobs running); it auto-terminates at its TTL.

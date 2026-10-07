# d1 NOTES — framework tax on the LONG compute-bound prefill path (box a8)
## [05:10 UTC] start
- Read BRIEF4/2/3 + FAST_DECISION_MODEL s1,2,4. Box a8 fresh (A10G, torch 2.14.1+cu130, triton 3.8.0, fla 0.5.2, nvcc 12.8). Prior systems agent's prof3 numbers were contended (useless at >=1000 tok); I own a8 so all my CUPTI numbers are uncontended.
- Plan: (1) eager-annotated CUPTI profile (per-stage kernel time) + graph-replay profile (busy, gaps, wall) at 256/1000/4000; bare-GEMM microbench on the same shapes; (2) fusion steps each timed; (3) megakernel bubble estimate + persistent Triton prototype (MLP block with row-block dependency counters); (4) host path; (5) projections.
## [05:40 UTC] STEP 1 profile — MEASURED, exclusive a8, lean 2B forward only (graph replay, fresh ids, n=20 reps; no host/head; T exact, no bucket pad)
| T | wall med/p95 ms | graph busy | in-graph gaps | GEMM ms (TFLOPS) | non-GEMM ms | bare-GEMM skeleton (graphed, same 96 GEMMs) | frac of skeleton |
| 256 | 19.44/19.46 | 19.51 | 0.11 | 13.51 (52) | 6.00 | 13.46 (52.2 TF) | 0.69 |
| 1000 | 74.18/74.20 | 74.46 | 0.12 | 46.96 (58.5) | 27.49 | 46.71 (58.8 TF) | 0.63 |
| 4000 | 296.17/296.24 | 296.1 | 0.02 | 183.6 (59.8) | 112.5 | 183.2 (59.9 TF) | 0.62 |
- Non-GEMM at 1000 (ms): add_rms 9.30 (480 kernels = 10 per norm call, fp32 unfused), gdn_gnorm 5.47 (13 kernels/layer), fla gdn_chunk 5.13 (11 kernels/layer incl 1.08 of .contiguous copies of q/k/v, 0.61 l2norm), silu_mul 3.28, attn_prep 1.84 (36 kernels/layer), conv 0.97, sdpa 0.88, glue 0.29, embed/rope 0.20, attn_gate 0.13.
- At 4000: add_rms 37.9, gnorm 21.7, gdn_chunk 19.7 (o 4.1, copies 3.9, h 3.7, w_u 3.0, l2norm 2.5, kkt 2.2), silu_mul 12.6, sdpa 8.0 (49 TFLOPS, already flash), attn_prep 7.3, conv 3.6.
- Per-GEMM TFLOPS at 1000: gate_up 61.4, down 62.7, in_attn 63.6, in_gdn 51.5 (8224/128=64.25 -> 65 N-tiles: wave quantization), out 51.8 (64 CTAs < 80 SMs). At 256: all 44-53.
- KEY: launch/inter-kernel bubbles inside the CUDA graph total only 0.02-0.12 ms (<0.6%). The 35-40% "framework tax" on the long path is NOT launches; it is unfused memory-bound fp32 glue (norms 15 ms of 27.5 at 1000) + fla GDN (5) + SwiGLU (3.3). Fusion, not persistence, is the lever. Tile/wave quantization on in_gdn/out/down = the only "bubble" a megakernel could address: ~2 ms at 256 (GEMMs at 52 vs ~61 TF), ~2.3 ms at 1000, ~7 ms at 4000.
## [06:15 UTC] STEP 2 fusion ladder — MEASURED (a8 exclusive, forward graph replay wall, n=20, median; p95 within 0.1 ms). Accuracy = final hidden vs base Lean at T=1000 (cos mean / min, rel L2 err)
| step (cumulative) | 256 | 1000 | 4000 | TFLOPS@1000 | acc |
| base lean+graph | 19.28 | 74.14 | 296.2 | 37.0 | - |
| + fused add+RMSNorm (Triton, 1 kernel vs 10) | 18.09 | 66.41 | - | 41.3 | cos .999991/.99993 rel .0043 |
| + fused gated RMSNorm (GDN out) | 17.18 | 61.35 | - | 44.7 | same |
| + fused SiLU*mul kernel | 16.78 | 59.94 | - | 45.8 | same |
| + fused attn prep (q/k norm+RoPE+sigmoid gate, 1 kernel vs 36) | 16.28 | 58.39 | - | 47.0 | same |
| + conv+SiLU+l2norm+split kernel (contiguous q/k/v, kills fla copies+l2norm) + fla use_gate_in_kernel/beta_sigmoid | 15.84 | 56.65 | 227.0 | 48.5 | cos .99999/.99991 rel .0044 |
| + Triton gate_up GEMM with SwiGLU epilogue (interleaved W rows) | 15.73 | 54.08 | 215.8 | 50.8 | rel .0046 |
| ALL-Triton GEMMs + RMSNorm folded into next GEMM (W'=W*(1+w), per-row rsqrt applied in epilogue; residual add + row sum-of-squares atomics in out/down epilogue) | 16.10 | 52.68 | 200.5 | 52.1 (54.8@4000) | cos .99999/.99987 rel .0045 |
- vs cuBLAS bare-GEMM skeleton (13.46/46.71/183.2): 256 -> 0.856 (swiglu variant), 1000 -> 0.887, 4000 -> 0.914 (fold). TARGET >=85% MET at all three.
- Triton GEMM (simple, grouped tile order) vs cuBLAS on A10G: matches/beats at M>=1000 for in_gdn (56.6 vs 49.8 TF @1000), gate_up (65.6 vs 62.3 @4000), down/out @4000 (60.4 vs 53.0); loses at M=256 small-N and down@1000 (49.3 vs 58.8: 128 tiles on 80 SMs = 1.6 waves). SwiGLU / row-scale / residual+sumsq epilogues cost ~0.
- Reference noise: rel err .0044 is bf16 re-association noise (the prior HF-vs-lean check was cos .9999).
## [06:45 UTC] compilers + megakernel prototype — MEASURED
- torch.compile(whole Lean.forward, static shapes, our CUDA graph around it): default 16.18 / 58.34 / 233.1 ms (256/1000/4000); mode max-autotune-no-cudagraphs 16.21 / 58.33 / 232.5 (no gain: GEMMs stayed ATen/cuBLAS; inductor cannot fuse SwiGLU across the split gate_up output nor fold RMSNorm into the next GEMM). = my hand step "+prep" level; my fold path is 10-14% faster (52.7 / 200.5). compile time 5-48 s per shape. acc rel .0046.
- MEGAKERNEL prototype (mega.py): one persistent Triton launch over L=4 consecutive folded-RMSNorm SwiGLU MLP blocks; on-GPU ticket scheduler (atomic counter) over tiles A(l,r,n)=gate_up+SwiGLU, B(l,r,n)=down+residual+row-sumsq; per-(layer,row-block) acquire/release counters, so layer l+1 tiles of row-block r start when B(l,r,*) is done. Deadlock-free by ticket order. Correct (rel err 6-9e-4 vs sequential = fp32 atomic order noise).
  speedup vs the SAME Triton tiles as 8 separate launches in a CUDA graph (n=20, median):
  T=256: 1.007 (group-8 order) / 0.984 (row-major order);  T=1000: 1.006 / 0.870;  T=4000: 0.878 (g8), 0.954 (g2) / 0.798 (row-major).
  => bubble/tail removal on the MLP chain is <=0.7%; row-major task order (needed for cross-layer row-block pipelining) loses 13-20% to L2 weight-reuse loss. Batch-1 MLP chain has no parallel slack: B(l,r) needs ALL 96 A tiles of row r, A(l+1,r) needs all 16 B tiles of row r.
## [07:20 UTC] fused-path profile, host side, e2e — MEASURED
- Best fused (fold) graph at 1000: 382 kernels (from 1487), gaps 0.007 ms; GEMM 46.0 ms (59.6 TF) + SDPA 0.9 + fla GDN 3.4 + conv_l2 1.2 + misc 0.7 + norms 0.4 + attn_prep 0.24 = 52.7. At 4000: GEMM 170 (64.6 TF) + SDPA 8.0 + fla 12.5 + conv 5.0 + misc 2.2 + norms 1.8 + prep 1.1 = 200.5. At 256: GEMM 14.2 (48.5 TF) + fla 1.0 + conv .3 + misc .4.
- Remaining "megakernel-addressable" time: kernels with fewer CTAs than SMs: down GEMM at M~1000 (64 CTAs of 256x128 on 80 SMs) 11.4 ms -> stream-K/split-K could recover ~2.3 ms; fla chunk_h recurrence (32 CTAs) 1.0 ms@1000, 3.7@4000 (overlappable only by cross-op wavefront).
- Latency is NOT data dependent (random/real/pad/same-token ids: 74.74-74.77 base, 52.77-52.84 fold at T=1024; 1710 MHz, 210-245 W of 300 W, no throttle).
- Host (real banking states, n=24, 8 vCPU): tokenize 1000 tok HF 1.61 ms -> Rust encode_batch over 8 line-aligned chunks 0.82 ms (24/24 identical ids); 4000 tok 6.23 -> 2.14 ms (identical). H2D of ids: torch.tensor(list) int64 0.095/0.31 ms -> numpy int32 pinned 0.049/0.127 ms.
- Padding trap: the runner's buckets (1024,1280,...,4096) put a "1000-token state + question" at 1280 (20% waste; explains the doc's 94 ms "at 1000"). 64-token buckets fix it.
- E2E request path (FastEngine.ask: render+tokenize+offsets+H2D+graph(fwd+pointer head)+D2H+answer), real states, n=14, 1 question:
  ~1000 tok: 76.86 (p95 77.01) -> 54.36 (54.43) ms, host prep 1.74 -> 1.18;  ~4000 tok: 305.5 (305.8) -> 203.5 (203.9), host prep 6.32 -> 2.42. noul p: .1638 vs .1655, .2178 vs .2183.
## [07:50 UTC] exact-length baselines (coordinator request) + 0.8B + fp16-acc — MEASURED (a8 exclusive, forward graph replay, exact T, n=20 median, p95 within 0.05 ms)
| step (2B) | 256 | 512 | 1000 | 2000 | 4000 |
| base lean+graph | 19.28 | 36.41 | 74.14 | 149.43 | 296.17 |
| +addrms | 18.09 | 33.62 | 66.41 | 133.60 | - |
| +gnorm | 17.18 | 31.77 | 61.35 | 123.37 | - |
| +silu | 16.78 | 30.98 | 59.94 | 120.76 | - |
| +prep | 16.28 | 30.17 | 58.39 | 117.68 | - |
| +conv/l2/gate-in-kernel | 15.84 | 29.20 | 56.65 | 114.56 | 227.0 |
| +SwiGLU GEMM epilogue | 15.73 | 28.26 | 54.08 | 109.31 | 215.8 |
| fold (all-Triton GEMMs, norm folded) | 16.10 | 32.95 | 52.68 | 104.08 | 200.5 |
| best-of | 15.73 | 28.26 | 52.68 | 104.08 | 200.5 |
- slope 512->4000: base 0.0745 ms/tok, best 0.0494 ms/tok. fold loses below ~1000 tokens (tile config for M=512 gives 32 CTAs on down) -> dispatch by T.
- A10G fp16-input GEMM, Triton: fp32-acc 67.5 TF, fp16-acc 67.5 TF (no 2x on A10G, confirms brief); rel err 2.1e-4 vs 2.35e-3 (K=2048). A10G peak ~70 TF => best Triton GEMM at 96% of peak; A10G ~= RTX 3090 with fp32 accumulate.
- Qwen3.5-0.8B (498M body) through the same kernels: base 10.40/39.10/155.0 -> swiglu path 7.38/25.79/97.46 -> fold 9.04/23.89/95.13 ms (256/1000/4000); rel vs HF .0136 (lean-vs-HF itself .0130). Fixed two 0.8B-only bugs (silu kernel I%1024, fold Kd=hidden).
- torch.export/AOTInductor: blocked ("Raw Triton kernel calls are not supported by non-strict torch.export"; every fla + own kernel must be wrapped as torch.library.triton_op). TensorRT-LLM 1.2.1 on PyPI; not installed (pins its own torch; Qwen3.5 hybrid support unverified).

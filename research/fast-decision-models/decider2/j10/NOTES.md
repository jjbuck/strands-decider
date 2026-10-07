# J10 notes: natively low-bit (BitNet b1.58 2B4T) decider

## 09:08 start
- Read BRIEF8, FAST_DECISION_MODEL.md, F7_REPORT, evalkit README, Brooker posts (system-one, encoders), BitNet 2B4T card + config, arXiv 2504.12285 + 2510.13998 abstracts.
- BitNet 2B4T config (verified, HF config.json): 30 layers, d 2560, FFN 6912 (ReLU^2 GLU), 20 heads / 5 KV heads, all full attention, RoPE theta 5e5,
  vocab 128256 (LLaMA-3 tokenizer, tied), ctx 4096. W1.58 (absmean ternary, per-tensor) A8 (per-token absmax).
- First-principles (arithmetic, before any run):
  - non-embedding params: per layer attn 16.4M + FFN 53.1M = 69.5M; x30 = 2.085B -> 4.17 GFLOP/token vs hobson ~2.77 (1.5x hobson's GEMM FLOPs).
  - no ternary tensor core exists on Ampere/Ada: W1.58A8 executes as int8 MMA (2x bf16). hobson W8A8 already runs int8 MMA and passes fidelity.
    => at compute-bound lengths BitNet-A8 is ~1.5x hobson-W8A8 GEMM time, plus 30 full-attention layers (quadratic) vs 6.
  - weight bytes: 2.085B * 2 bit = 0.52 GB vs hobson bf16 ~2.8 GB / int8 1.4 GB. A10G 600 GB/s: 0.87 ms vs 4.6 / 2.3 ms streaming floor.
  - roofline ridge on A10G int8 (~140 TOPS / 600 GB/s): 2-bit weights become compute-bound at T ~ 30 tokens; int8 weights at ~117.
    => the weight-byte advantage only matters below ~100-150 tokens. Short-prompt win is bounded (~1-3 ms), not 10x.
  - the transformative variant is W1.58A4: ternary weights are exact in s4 / FP4(E2M1), so ONLY activation rounding remains,
    and int4 MMA is 4x bf16 on Ampere/Ada (FP4 on Blackwell). Test: crest factors of BitNet activations (doc: hobson 15-40).

## 09:21 box j10 ready (auto-terminates ~18:58). torch 2.14.1+cu130, transformers 5.18, triton 3.8, nvcc 12.8
## 09:35 torso validated + crest factors (measured)
- Own BitNet torso (bitnet_j10.py) vs HF BitNetForCausalLM (online quant): final-hidden cosine 0.9988-0.9991, LM loss 2.313/2.308, 2.412/2.410,
  2.167/2.169, 2.448/2.448 on 4 train-split states (738-2001 tok). Equivalent up to bf16 tie noise.  (ref_j10.json)
- Teacher logits: reusing F7's hobson-v19 teacher logits (recovered g3 rows_*_q.pt) -- regenerate the same rendered rows (prep_g3, same seeds)
  with the BitNet tokenizer, check Qwen ids identical, copy logits. Saves ~65 GPU-min of labelling.
- Crest factor (per-token max/rms) at BitLinear inputs, 64 real train-split states, base BitNet 2B4T (crest_j10.json):
  | input | crest median | p99 | max | int8 rel err | int4/token | int4 block16 | block32 |
  | qkv (after RMSNorm) | 5.2 | 10.3 | 24 | 1.3% | 23% | 8.9% | 10.4% |
  | o (after attn_sub_norm) | 6.1 | 11.2 | 31 | 1.4% | 25% | 8.8% | 10.1% |
  | gate/up (after RMSNorm) | 4.1 | 9.2 | 24 | 1.0% | 18% | 8.7% | 10.1% |
  | down (relu^2*up, ffn_sub_norm) | 30.7 | 70.8 | 83 | 4.0% | 39% | 6.2% | 8.5% |
  vs hobson (doc s5): crest 15-40, int4/token 50-70%, Hadamard 13-16%, NVFP4 7-9%.
  => native A8 training lowers crest on 3 of 4 inputs to 4-6 but NOT to int4-safe levels (per-token int4 still 18-25%), and the ReLU^2
     down-proj input is as peaked as hobson's worst. "Born low-bit avoids outliers by construction" holds only for the weights.

## 09:55 decider training launched (ck_a8)
- Rows: F7 corpus (train_v5 minus 3% val + v19 synthetic docs, 102,547 rows) + 3,600 of F7's 6,000 KL-only real train-split states (frac 0.6,
  time budget), 106,147 rows, 30.2M BitNet tokens (LLaMA-3 tokenizer is ~6% denser than Qwen's). Regenerated twin ids == F7 ids (0 mismatches).
- Recipe: F7 (pointer head 256, LoRA r16/a32 all 7 projections x 30 layers, lr 1e-4 / head 1e-3, 32 rows/step, cosine, CE + KL(hobson) + KL real)
  but LoRA lives in the LATENT weight: deployed W = ternary(W0 + 2 B A) per tensor (absmean), activations int8 per token as pretrained. Norm gains trainable.
- Throughput: 5.1k tok/s at 4k-token microbatches (42 TFLOPS effective, 75% GEMM); refresh of 210 ternary matrices per step 131 ms.
  3,317 steps, ETA ~1.7-2 h. Step 60: CE 0.98 KL 0.50 (G3 SmallThinker same step: CE 0.91 KL 0.40).

## 10:20 runtime kernels written + checked (4-layer base model, 30 real questions; check4.log)
- rt_j10.py: Triton glue (add+RMSNorm+absmax quant, RoPE split, attn_sub_norm+quant, ReLU^2*up+ffn_sub_norm+quant), Triton int8 GEMM (codes [K,N],
  fused per-token x per-tensor dequant), Triton W2A8 GEMM (2-bit packed weights, 4/byte, unpacked in registers to int8 MMA), CUTLASS s4xs4 (G2's g2s4.cu,
  built here vs CUTLASS 3.5.1) for A4; torch SDPA flash (GQA); CUDA graph.
- vs emulated torso (bf16 fake quant): bf16 runtime cos 0.9991; i8 and w2 identical to each other (exact integer math), cos 0.9991, argmax agree 30/30.
  A4 per-token on all GEMMs: cos 0.876 after only 4 layers; A4 on qkv+gate/up only ("mix"): cos 0.983. (measured)
- LC eval at 25% of training (step 829): val_v5 .805 / gen_v16 .669 / gen_v18 .569 / adequacy .609  vs hobson on same rows .874 / .880 / .781 / .808.
  Large gap on document-reading synthetic sets (-20 pts) at 25% of the schedule.
- Verified (arXiv 2504.12285 Table 2): BitNet 2B4T avg 55.01 vs Qwen2.5-1.5B bf16 55.72, GPTQ-int4 52.15, AWQ-int4 51.17 (weight-only int4, bf16 acts),
  MMLU 53.2 vs 60.3. So "born ternary beats post-hoc int4" holds for weight-only LM benchmarks; knowledge (MMLU) is 7 points below the bf16 1.5B.
- bitnet.cpp GPU kernel = W2A8 GEMV via dp4a (decode only; A100 shapes 1.3-3.6x vs bf16). No prefill kernel exists -> wrote w2mm.

## 10:50 training at 50% (step 1658 / 3317)
- LC eval (half the lc rows): val_v5 .828 / adequacy .729 / gen_v16 .749 / gen_v18 .659 vs hobson .874 / .808 / .880 / .781 (gap -5 / -8 / -13 / -12).
  Train agree with hobson argmax ~0.86-0.88 on recent batches; CE ~0.40-0.52, KL ~0.07-0.11. Ternary code flips vs base: 0.30% of codes.
- chain.sh (eval -> check -> latency -> hobson latency -> GEMM micro -> A4 emulation evals) and chain2.sh (A4 QAT continuation 'mix') queued.

## 12:00 training done (8,494 s, 3,317 steps, 30.2M tokens; final code flips 0.33%) -- JB results (measured)
- Final LC: val_v5 .828 / adequacy .742 / gen_v16 .794 / gen_v18 .683 vs hobson .874 / .808 / .880 / .781.
- JB-hard .385 vs hobson .523 (McNemar 8 vs 26, p = .003); JB-all .619 vs .723 (9 vs 33, p = .0003); JB-long .312 vs .403.
  Brier JB-hard .705 vs .557; JB-all .464 vs .348.  Below hobson's own no-state baseline on JB-hard (.462).
- Per family (j10 / hobson): adequacy 7/10, adversarial 1/3, ambiguous 3/5, intent 20/23, judge_hard 7/9, long_policy 4/7, multi_hop 5/8,
  ordinal 8/12, policy 8/9, trap 5/7; surface families (extraction, fact, routing, tool_selection) tie.  => the ternary 2B reads worse everywhere.

## 12:20 suite results for the A8 ternary decider (tag a8; measured)
- REAL-agree agree .870, agree_sd .803 (F7 q08 .824, hob12 .884); REAL-label .733 vs .785 (McNemar 13 vs 34, p .003), Brier .392 vs .347, ECE .112 vs .116.
- CF acc .576 vs .594 (items 48 vs 62, p .21), flip .254 vs .268, fgh .596, dir .926. amount_insert .32 vs .10 (better), human_insert .21 vs .35, wrapup .84 vs .92.
- CF-probe acc .589 vs .653 (83 vs 124, p .005), flip .203 vs .328, fgh .40; amount_vs_limit 0 vs .33, id_match .76 vs .54, status .77 vs .89.
- SHUF change .323 / both_right .258 (hobson .339 / .245).
- Verdict on the accuracy bar: FAIL (JB-hard and JB-all significant drops, REAL-label .733 < .78, CF-probe below hobson).

## 12:30 latency (A10G, exclusive, CUDA graph, fresh ids H2D + replay + D2H, median of 30 after 5 warm; p95 within 0.3%) -- measured
- Same box, hobson-v19 bf16 with the doc's fused runtime (d1 lean2, all fusions, real 91-token question; 4q = state + 432 question tokens one pass):
  T=64 15.5 | 128 15.9 | 256 25.3 | 400 32.8 | 1000 57.1 | 4000 201.6 ms (1q);  4q: 32.9 | 37.1 | 43.7 | 48.4 | 83.6 | 215.8.
  (doc anchors: 9.4 @64, 15.7 @256, 52.7 @1000, 200.5 @4000 -- long lengths reproduce; short ones are ~1.6x slower here, question length differs.)
- Ternary BitNet decider, 1q (85-token question), ms:  T = 64 / 128 / 256 / 400 / 1000 / 4000
  bf16 (cuBLAS) 21.2 / 21.1 / 32.7 / 39.5 / 89.6 / 330.9
  i8  (int8 codes, Triton int8 MMA) 13.1 / 15.7 / 23.4 / 27.2 / 65.1 / 242.7
  w2  (2-bit packed weights -> int8 MMA) 12.7 / 16.7 / 23.0 / 26.6 / 60.7 / 231.5
  mix (qkv+gate/up int4 MMA, o/down int8) 9.2 / 10.1 / 15.5 / 17.8 / 43.9 / 174.6
  s4  (all int4 MMA; A4 everywhere, inaccurate) 6.6 / 7.5 / 11.0 / 13.1 / 30.1 / 137.4
- 4x fewer weight bytes (w2 vs i8) buys 3% at 149 rows and 7% at 1085 rows: prefill with a real question is already compute-bound.
- GEMM share at 1000: i8 54.1 of 65.1 ms; attention 4.5 ms at 1000, 45.5 ms at 4000 (30 full-attention layers vs hobson's 6).

## 12:50 A4 emulation on the trained A8 decider (no QAT; measured)
- mix (qkv + gate/up inputs int4 per token; o, down int8 -> 65% of GEMM FLOPs at int4): decision flips vs the A8 model: JB 9.1%, REAL 7.1% (77/1083),
  CF 9.4%, CF-probe 19.4%; mean TV .05-.06.  hobson for comparison (doc s5, post-hoc, rotation+GPTQ): 53% of GEMM work at 4 bits -> 2.2-2.6% REAL flips.
  => the natively ternary model is not more 4-bit-activation tolerant than hobson; its ternary weights cannot absorb a Hadamard rotation (W H is not ternary).
- GEMM micro (res_gemm_j10.json): CUTLASS s4 30-65 us at M=64 vs Triton int8 67-130 us; Triton int8 / w2 reach only ~40-50% of roofline at M<=256.

## 13:00 more A4 emulation (flips vs the A8 model) + QAT launched
- all GEMMs A4 with 16-element block scales (int grid; NVFP4-like, the Blackwell format): REAL 3.5% (38/1083), JB 5.6%, CF 4.5%, CF-probe 7.2%.
  (hobson NVFP4 W4A4 emulated: 5.2% REAL flips -- but that includes 4-bit weight error; BitNet's ternary weights are exact.)
- all GEMMs A4 per token (Ampere s4, my fastest kernel config): REAL 16.2%, JB 21.6%, CF 20%, CF-probe 32%. REAL-label .688. (hobson W4A4 RTN 13.5%, GPTQ 9%)
- LONG (A8): agree .879, agree_sd .818 (F7 q08 .867, hob12 .885).
- QAT continuation ck_mixa4: from ck_a8 final, qkv+gate/up trained at int4 per token, 500 steps, lr 5e-5, started 12:55.

## 13:45 QAT result + runtime fidelity (measured)
- A4 QAT continuation (qkv + gate/up int4 per token, 500 steps): JB-hard .454, JB-all .649, REAL-label .728, REAL agree_sd .731 (A8 .803), LONG sd .788,
  CF acc .559 / fgh .523, CF-probe acc .583 / fgh .40.  500 QAT steps do not recover the A8 model's agreement (no-QAT emulation: sd .734).
- Runtime fidelity with the training quantiser (fixed codes(): fp32 latent, no bf16 rounding): i8 / w2 / CUTLASS c8 vs emulated torso, 40 real
  questions: hidden cos >= 0.99994, argmax 38/40 (TV mean .006-.007, max .024 -> boundary cases; the emulation is bf16-rounded, the runtime exact int).
  CUTLASS mix (A4 qkv+gu): argmax 35/40, TV .041.  Packed 4-question vs single-question passes: TV .003-.007.
- (earlier check_full.log numbers, cos .96, were the bf16-rounded-latent bug in codes(); latency unaffected.)
- 13:40 a concurrent-launch slip (chain3 lat + chain3b lat) OOM'd both; no timing from that window was kept; c8/mixc latency re-run alone.

## 14:00 CUTLASS int8 latency + post-hoc ternary hobson (measured)
- CUTLASS s8 (c8), 1q: 64 10.6 | 128 12.0 | 256 18.1 | 400 21.1 | 1000 48.3 | 4000 206.1 ms; 4q: 22.8 | 28.9 | 33.4 | 40.1 | 67.5 | 246.5.
- mixc (CUTLASS s4 on qkv+gate/up, s8 on o/down), 1q: 8.0 | 8.9 | 13.7 | 16.0 | 37.2 | 160.1; 4q: 17.8 | 22.7 | 26.1 | 31.6 | 53.1 | 197.6.
- => best ACCURATE-format ternary (W1.58A8, c8) at 1000 tokens: 48.3 ms vs hobson W8A8 36.3 (doc) and hobson bf16 57.1 (same box).
- Post-hoc ternary hobson (absmean per tensor, int8 acts, no training): JB-all .299, JB-hard .362, REAL-label .400, REAL agree .358, CF acc .438,
  CF-probe .489 -> collapsed (the starting point BitDistill must repair with continued pretraining).
- BitDistill-lite run (ck_th): ternary hobson + LoRA-in-latent QAT, KL from hobson on F7 rows: ~11 s/step in the HF Qwen3.5 path -> ~240 steps in 45 min.

## 14:55 BitDistill-lite result (measured) -- all runs done
- ck_th: post-hoc ternary hobson (W1.58A8, 186 linears) + LoRA-in-latent QAT, F7 rows + hobson KL, 819 steps / ~8.1M tokens / 31 min (3.2 s/step).
  JB-hard .392, JB-all .429, REAL-label .655, REAL agree .703 / sd .451, LONG .828 / .636, CF acc .477 flip .057, CF-probe acc .505 flip .031.
  From the untrained collapse (JB-all .299, REAL-label .400) it recovers about a third of the gap at 0.08% of BitDistill's ~10B-token budget. Fails.
- Final score table: scores_table.txt (+ th row), scores_j10.json.

## 15:05 done
- DRAFT_REPORT.md final (about 1,490 words excluding table pipes). Box idle (all chains done); results, logs, scores and projections copied to ~/decider2/j10/.
- Files: results/*.jsonl (preds per suite and tag: a8, mixa4qat, a8_mixA4, a8_allA4b16, a8_allA4, hob_w158a8, th), scores_j10.json, scores_table.txt,
  res_lat_dedup.jsonl (all 84 latency configs incl. kernel split), res_hob_lat.jsonl, proj_j10.md, crest_j10.json, box_res_gemm_j10.json, box_check_full2.log.

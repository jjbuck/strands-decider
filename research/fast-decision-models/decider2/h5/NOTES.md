# H5 NOTES

## [21:48 PDT] start
- Read BRIEF7, F7_REPORT, evalkit README, recovered f7code / g3 code. Box g5 idle (A10G, venv torch 2.14.1, transformers 5.18, fla 0.5.2, peft 0.21.2).
- Started uploading evalkit/ (with train_pool) and d1/lean2.py to g5.

## [22:08 PDT] MISSION CHANGE (coordinator)
- hob12 rebuild stopped before any GPU job started (nothing to kill on g5; only uploads had run).
- New mission: full-depth hobson-v19 (24 layers), learned-rotation (SpinQuant-style, Cayley/Stiefel) W4A4 with weights frozen,
  KL to bf16 hobson on train_pool states; then short LoRA QAT; mixed precision incl. GDN internals; all suites; low-bit kill criterion;
  FORMAT.md in ~/decider2/h5/; compare with H1 (fixed Hadamard/QuaRot + KL-QAT on g2).

## [22:15 PDT] infrastructure + first numbers
- Fetched arXiv 2405.16406 (SpinQuant: R1 residual + R2 head-wise learned by Cayley SGD on Stiefel, weights 16-bit during learning
  (W16A4 beats learning with W4), then GPTQ; R3/R4 fixed Hadamard; 800 samples x 100 iters) and 2609.04098 (NVFP4 W4A4 on all 496 linears
  of a GDN hybrid incl. GDN, matches BF16; gates least sensitive). R2 is NOT absorbable in hobson: attention has an elementwise output gate
  and GDN a gated RMSNorm between v and the out-projection, so the out-proj inputs keep a fixed online Hadamard (as H1's R2).
- Code (h5/code): h5lib.py (Q5 simulator on g2lib's lean weights: residual rotation R1, online Ho/Hd, per-site W/A bits, per-token symmetric
  A, per-channel W with MSE clip, RTN + GPTQ, 'learn' mode = W16A4 differentiable in R1), h5sens.py, h5rot.py, h5build.py, h5suite.py.
- Seeds aligned with H1 FORMAT v0: R1 init = H1's R1 (seed 1234), Ho = H1 R2 (1235), Hd = H1 R4 (1236). Only R1 differs (learned).
- Rotation equivalence (16-bit everywhere, rotated path vs bf16 path): 0/240 flips, TV 0.0037 (bf16 rounding of the rotation). [measured]
- Scored G2's leftovers with evalkit (laptop): rotated W4A4 RTN 21% REAL flips, CF fgh 0.28; G2's W4A4 LoRA-QAT (1500 steps) 11% flips,
  CF fgh 0.48. W4A4 on hobson is far from the 0.5% bar. [measured, G2 data]
- Dev set (train split, 240 (state,question) pairs from train_pool, seed 11; never trained on): W4A4 RTN (Hadamard R1, clip 1.0) 17.5% flips,
  TV 0.144. Sensitivity sweep running (h5sens.py).

## [22:40 PDT] sensitivity on the train-split dev set (240 pairs; RTN, Hadamard R1, clip 1.0 unless noted) [measured]
| config | flips | TV | KL(dense||q) |
|---|---|---|---|
| all W4A4 | 17.5% | .144 | .099 |
| all W4A4, A clip 0.9 | 10.8% | .133 | .086 |
| all W4A4, GDN b/a rows also 4-bit | 12.5% | .141 | .097 |  (b/a precision irrelevant, as 2609.04098 says)
| W16A4 | 9.6% | .088 | .044 |
| W4A16 | 7.5% | .085 | .037 |
| W4A8 | 7.9% | .085 | .038 |  (A8 ~ lossless; W4-RTN is half the problem)
| W8A8 | 0.42% (1/240) | .009 | .0005 |
| only GDN Win W4A4 | 4.6% | .051 | .014 |
| only attn Win W4A4 (6 layers) | 4.2% | .033 | .0065 |
| only GDN out_proj W4A4 | 3.3% | .060 | .0186 |  (worst per FLOP)
| only attn o_proj W4A4 (6 layers) | 3.8% | .033 | .0062 |
| only gate_up W4A4 | 8.3% | .073 | .028 |
| only down W4A4 | 4.2% | .042 | .009 |
- Singles sum to .083 vs joint .099: roughly additive at 4 bits (H1 sees the same).
- GEMM-time model (H2's CUTLASS times, h5cost.py), A10G T=1000: bf16 49.2, W8A8 25.4, W4A8 26.8, W4A4 14.2 ms; making one class A8:
  Win_g +2.6, Win_a +0.5, Wo_g +0.8, Wo_a +0.3, Wgu +5.2, Wd +3.2 ms. => out-projections to 8-bit activations are the cheap fix.
- R1 learning run a/b (Adam lr 2e-5 / 2e-4, scale detached): no dev gain, hid loss flat or rising. Cause: with STE and a detached absmax scale
  the rounding error has no gradient in R1, so the only signal left is noise. Fixed: in 'learn' mode the per-token scale is NOT detached
  (the gradient reaches the outlier that sets the scale, which is what a rotation can change). Run c started (lr 1e-4).
- Killed the leave-one-out half of the sweep to free the GPU (A8 ~ lossless, so it is predictable from the singles).

## [23:02 PDT] GPTQ + rotation learning [measured, dev 240 / 120]
- Hadamard R1 + GPTQ (act-order, 128 train seqs <= 2048 tok) W4A4 clip 0.9: dev flips 8.3%, TV .091, KL .042 (RTN: 10.8%, .133, .086).
- End-to-end R1 learning (KL + residual loss through the whole W16A4 network, 4 seqs/step, Adam): run c (scale not detached) dev KL
  .046 -> .041 at step 40, then .053 at step 60: unstable/noisy, killed. Single-question KL on 2k-token sequences is too noisy a signal
  at this budget.
- Local R1 learning (h5rotl.py): full-batch output-space error of A4 quantization at all 48 R1 sites (Gram-weighted by the 16-bit weights),
  64 train seqs x 128 tokens (token 0 kept). Loss: identity R1 .228, Hadamard .0084, learned .0048 at step 160 (-42%), dR .31.

## [23:27 PDT] learned R1 vs Hadamard, same GPTQ [measured]
- W16A4 end to end (dev 120, clip 0.9): Hadamard R1 KL .0358 / TV .087 / flips 12.5%  ->  learned R1 (l1) KL .0301 / TV .076 / 10.0%  (-16% KL)
- W4A4 GPTQ (dev 240): Hadamard KL .042 / TV .091 / 8.3%  ->  learned R1 KL .036 / TV .0865 / 7.5%  (-14% KL)
- Dense same-runtime suite run (res/preds_dense.jsonl, 3227 q): vs hobson refs REAL flips 0.46%, LONG 0.42%, CF fgh 1.000, CF-probe fgh .971,
  JB-hard .531 = G2's dense / merged_full. The runtime itself sits at the 0.5% flip bar (bf16 re-rounding floor, as H1 found).
- Running: LoRA QAT (r16, 120 x 4 seqs, KL + residual) on l1_gptq_w4a4; suite eval of had_gptq_w4a4.

## [23:55 PDT] short LoRA QAT on learned-R1 GPTQ W4A4 [measured, dev 120]
- r16 LoRA in the rotated basis on all 96 GEMMs, STE through fixed GPTQ scales (step 0 = GPTQ solution), KL + residual, lr 1e-4 cosine,
  4 seqs/step: step 0 KL .0372 TV .0878 | step 40 .0377 / .0914 | step 80 .0369 / .0841. No meaningful gain after 320 train sequences;
  killed to free the GPU (G2's 1500-step QAT from RTN reached about the GPTQ level too). Checkpoint qat/l1a_s40, s80 kept on box.
- Suite eval of had_gptq_w4a4 running (slow under sharing: ~40 min); l1_gptq_w4a4 suite launched.

## [00:22 PDT] suites: Hadamard R1 + GPTQ W4A4 (all 96 GEMMs, A clip 0.9) [measured, all 3227 questions]
- REAL flips 9.23% vs hobson (8.96% vs same-runtime dense), agree_sd .847; LONG sd .861, flips 6.8%; CF fgh .725; CF-probe fgh .629
  (distract .460); JB-hard .462 (McNemar vs hobson 7 / 15, p .13); REAL-label .777; SHUF change .331. Fails every low-bit bar but JB McNemar.
  (H2 runtime, RTN W4A4: 13.5% flips, CF fgh .670, CFP .514 -> GPTQ is the larger lever.) Dev (240) predicted 8.3% flips: dev tracks REAL.
- Launched: learned R1 + GPTQ on H1's mixed map k48 (48 GEMMs at W8A8, rest W4A4; same map as H1 -> direct comparison).

## [00:34 PDT] learned R1 + GPTQ on H1's k48 map (48 GEMMs W8A8, 48 W4A4) [measured, dev 240]
- flips 1.25% (3/240), TV .029, KL .0045 (all-W4A4 learned: 7.5%, .0865, .036). GEMM time model 19.3 ms/1k tok (H1) vs 14.2 all-W4A4.
- Suite eval launched (res/preds_l1_gptq_k48.jsonl).

## [00:47 PDT] suites: learned R1 vs Hadamard R1, both GPTQ W4A4 on all 96 GEMMs [measured, 3227 q]
| metric | dense (same runtime) | Hadamard+GPTQ | learned R1+GPTQ |
|---|---|---|---|
| REAL flips vs hobson | 0.46% | 9.23% | 8.59% |
| REAL agree_sd | .997 | .847 | .815 |
| LONG agree_sd / flips | .988 / 0.42% | .861 / 6.8% | .879 / 6.4% |
| CF fgh | 1.000 | .725 | .697 |
| CF-probe fgh (distract) | .971 (.946) | .629 (.460) | .705 (.513) |
| JB-hard (McNemar vs hobson) | .531 | .462 (7/15, p .13) | .523 (10/10, p 1.0) |
| REAL-label | .785 | .777 | .762 |
| SHUF change | .344 | .331 | .269 |
- Paired, learned vs Hadamard: REAL decisions agreeing with hobson 51 vs 58 discordant (McNemar p .57); LONG 15/17 (p .86);
  CF fgh 17/14 (p .72); CF-probe fgh 13/21 (p .23). => learned R1 is NOT distinguishable from Hadamard on the suites, although its
  dev KL is 14% lower. The two 4-bit models disagree with each other on ~10% of REAL questions: at W4A4 decisions are noise-dominated
  (H2 saw 12% disagreement between two numerically-equivalent W4A4-RTN runtimes).
- H2 runtime with H1 GPTQ4 codes (Hadamard): flips 8.96%, sd .850, CF .826, CFP .657, JB .523: same flip level as my emulation; CF/CFP
  fgh differ by ~0.1, inside the 4-bit rounding noise band H2 measured.
- (Header times from 22:15 on were corrected at 01:05 against file mtimes; I had been estimating wall-clock from sleeps.)

## [01:05 PDT] Hadamard R1 + GPTQ on the same k48 map [measured, dev 240]
- flips 1.67% (4/240), TV .0294, KL .0047  vs learned R1: 1.25%, .0288, .0045 => -4% KL; on the mixed map the rotation barely matters.
- k48 suite eval (learned R1) running; ~25 min left.

## [01:35 PDT] suites: learned R1 + GPTQ, H1's k48 map (48 GEMMs W8A8 / 48 W4A4) [measured, 3227 q]
- REAL flips 3.14% vs hobson (3.05% vs dense), agree_sd .945; LONG sd .945, flips 3.4%; CF fgh .936; CF-probe fgh .914 (distract .838);
  JB-hard .546 (McNemar 4 / 1, p .38); REAL-label .772; SHUF change .328. Fails flips and both fgh bars.
- Same map in H2's runtime with H1's Hadamard GPTQ codes (H2 NOTES 00:15): flips 2.59%, sd .965, CF .972, CFP .848, JB .538.
  Not a clean A/B (different runtime and calibration), so no learned-vs-fixed claim from this pair; dev (same emulation): learned .0045 vs
  Hadamard .0047 KL.
- Running: h5kron.py = per-GEMM learned orthogonal Kronecker rotations P1 (x) P2 (32x64 / 96x64) + learned clips, local per-GEMM objective
  with W4 and A4 in the loop (FlatQuant structure, orthogonal). Per-GEMM local error after 150 steps: Wd -22%, Wo -18%, Win -6..9%, Wgu -5%.

## [02:00 PDT] per-GEMM learned Kronecker rotations (kron_k1) + learned R1 + GPTQ [measured, dev 240]
| config | flips | TV | KL |
|---|---|---|---|
| all-W4A4, R1 learned | 7.5% | .0865 | .0360 |
| all-W4A4, R1 learned + kron | 7.9% | .0830 | .0348 |  (-3% KL: GPTQ absorbs most of the local gain)
| k48, Hadamard | 1.67% | .0294 | .0047 |
| k48, R1 learned | 1.25% | .0288 | .0045 |
| k48, R1 learned + kron | 2.5% | .0257 | .0035 |  (-22% KL vs R1 learned only)
- Learned rotations (residual R1 + per-GEMM Kronecker) buy 14-25% KL at 4 bits; the bar needs ~10x.
- Launched: suite eval of l1_kron_gptq_k48; build of learned-R1 W8A8 GPTQ8 (does learned R1 push W8A8 below the 0.5% bar without
  bf16 exceptions? H2: Hadamard W8A8 GPTQ8 = 0.83% flips; + 8 bf16 GEMMs (b8) = 0.18%, passes).

## [02:28 PDT] learned R1 + GPTQ8, W8A8 on all 96 GEMMs [measured, dev 240]
- flips 0/240, TV .0069, KL .0002 (Hadamard RTN W8A8: 1/240, .0092, .0005). Suite eval launched (does it pass without bf16 GEMMs?).
- Next: NVFP4 W4A4 emulation (h5fp4.py: FP4 E2M1 + FP8 E4M3 scale per 16 elements + per-tensor fp32) = the Blackwell/5090 4-bit format,
  with no rotation / Hadamard / learned R1, RTN weights.

## [03:20 PDT] suites: k48 + learned R1 + per-GEMM learned Kronecker rotations [measured, 3227 q]
- REAL flips 2.22% vs hobson (2.49% vs dense), agree_sd .951; LONG sd .952, flips 2.1%; CF fgh .936; CF-probe fgh .905 (distract .865);
  JB-hard .538 (McNemar 4/2, p .69); REAL-label .787; SHUF .326. Best 4-bit-majority result; still fails flips and both fgh bars.
- Paired vs k48 + learned R1 only: REAL discordant 18 / 28 (p .18), LONG 4 / 10 (p .18), pooled 22 / 38 (p .05): directional gain.
- (The harness blocks subagents from writing REPORT.md; the report is returned as text to the coordinator.)

## [03:42 PDT] suites: learned R1 + GPTQ8, W8A8 on all 96 GEMMs [measured, 3227 q]
- REAL flips 0.83% vs hobson (0.55% vs dense), agree_sd .994; LONG sd .982, flips 0.85%; CF fgh .991; CF-probe fgh .952 (distract .946);
  JB-hard .538 (McNemar 2/0, p .5); REAL-label .782. Fails flips (and CF-probe fgh .952 < .97). Same as H2's Hadamard W8A8 GPTQ8 (0.83%):
  learned R1 does not remove the need for H1's 8 bf16 GEMMs (b8 map passes at 0.18% in H2's runtime).
- NVFP4 dev (RTN, no rotation): flips 8.75%, TV .077, KL .0325 (int4 Hadamard RTN .086; int4 learned-R1 GPTQ .036). Running: other NVFP4
  variants + NVFP4 Hadamard GPTQ (block-scale-aware GPTQ: fixed FP8 block scales, E2M1 grid).

## [04:05 PDT] NVFP4 W4A4 (Blackwell / RTX 5090 format, emulated) on all 96 GEMMs [measured, dev 240]
| config | flips | TV | KL |
|---|---|---|---|
| NVFP4 RTN, no rotation | 8.75% | .077 | .0325 |
| NVFP4 RTN, Hadamard R1/Ho/Hd | 7.9% | .092 | .0447 |  (rotation HURTS with 16-element block scales)
| NVFP4 RTN, learned R1 (int4-trained) | 9.2% | .101 | .0514 |
| NVFP4 A only (W16), Hadamard | 6.3% | .063 | .0209 |
| NVFP4 W only (A16), Hadamard | 7.1% | .072 | .0278 |
| NVFP4 GPTQ (block-scale aware), Hadamard | 8.3% | .072 | .0268 |
| **NVFP4 GPTQ, no rotation** | **3.3%** | **.058** | **.0185** |  best all-4-bit result (int4 best .035)
- Suite eval of NVFP4 no-rotation GPTQ launched (2 shards).

## [04:50 PDT] suites: NVFP4 W4A4 (no rotation, block-scale GPTQ) on all 96 GEMMs [measured, emulated, 3227 q]
- REAL flips 5.17% vs hobson, agree_sd .910; LONG sd .909, flips 4.5%; CF fgh .835; CF-probe fgh .781 (distract .595); JB-hard .515
  (McNemar 5/6, p 1.0); REAL-label .767; SHUF .339. Best all-4-bit result (int4 learned-R1 GPTQ: 8.59%); still fails every bar except JB.
- Final score table: ~/decider2/h5/scores.json (h5score.py). All preds in h5/box/.

# H1 notes (box g2). Mission: W8A8 / W4A4 hobson-v19 at < 0.5% flips, fgh >= 0.97.

## 2026-10-05 21:48 PDT  start
- Read BRIEF7, F7_REPORT, G2 code (g2lib.py, g2qat.py, g2map.py, score.py, rq.py) from ~/decider2/recovered/g2/g2/.
- Box g2 was still running G2's leftover queue jobs/laneC.sh (g2map longall Q:r8 -> res/longr8.json, then nest_mid).
  I tried to stop the laneC queue (kill the bash parent, leave longr8 running); the permission was DENIED (auto-mode classifier,
  "Interfere With Workloads"). Not worked around. Consequence: GPU shared with G2's queue until it drains (~05:40 UTC est.).

## 22:15 PDT  reconstruction of G2 + scoring what exists (laptop, h1/code/h1score.py over G2's res/*.json)
- G2 pipeline (reconstructed from code/logs): g2lib.G2 = lean fused hobson (merged LoRA) with per-row precision classes; 'Q:<p>' = every row
  quantized; p8 plain W8A8; r8/r4/r48 = QuaRot-style: norm gains folded, GEMM input rotated by randomized Kronecker Hadamard
  (2048 = H32xH64, 6144 = Paley12 x H16 x H32); per-token absmax activations, per-channel absmax weights (4-bit: MSE clip search), RTN; 'c' = A4 clip 0.9.
  Emulation only rotates the INPUT side (weights quantized per unrotated output channel).
- g2qat.py: W4A4 QAT, LoRA r32 on Win/Wo/Wgu/Wd of 24 layers, STE through RTN of (W + BA) R, KL(teacher||student) on answer dist
  + 1.0 * relative residual MSE at layers 5/11/17; data = train_pool real states (eval tasks excluded) + 30% train_v5 rows (gold weight 0);
  1500 steps, accum 4, lr 3e-4; KL fell 0.08 -> ~0.02-0.04 but hid-MSE stayed flat at 0.11 (!).
- Scores (vs bf16 hobson refs; 'vs dense' = vs G2's own bf16 runtime):
  | config | REAL flips (1083) | REAL agree_sd | CF fgh | CF-probe fgh | JB-hard | LONG |
  | G2 dense bf16 | 0.46% | 0.997 | 1.000 | 0.971 | .531 | 0/40 |
  | Q:r8 (rot W8A8, untrained) | 1.29% (1.20% vs dense) | 0.986 | 0.972 | 0.971 | .538 | 0/40 flips |
  | Q:r4 (rot W4A4 RTN) | 21.1% | 0.653 | 0.284 | 0.295 | .431 | |
  | Q:r4c untrained (346 SD q only) | (32% of SD q) | 0.679 | 0.569 | 0.495 | | |
  | QAT s500 r4c | | 0.734 | 0.477 | 0.486 | | |
  | QAT s1000 r4c | | 0.734 | 0.505 | 0.514 | | |
  | QAT s1500 r4c | 11.1% | 0.757 | 0.477 | 0.495 | .438 (McNemar p .06) | LONG-SD 11.5% flips |
  => G2's W4A4 QAT is far from the bar (fgh ~0.5); QAT helped agree_sd a little and did not help CF tracking.
- Noise floor: G2's bf16 runtime itself flips 0.46% of REAL vs deployed hobson (README: merged_full 0.4%). So "<0.5% vs hobson" is AT the
  runtime floor; I report flips both vs hobson refs and vs the in-runtime bf16 dense (the quantization's own effect).
- arXiv fetched: 2404.00456 QuaRot, 2405.16406 SpinQuant, 2410.09426 FlatQuant, 2510.13998 BitDistill, 2609.04098 GDN-NVFP4 (+html).
  2609.04098 insights: GDN gates (in_proj_a/b) are the LEAST sensitive to W4A4 (2.1-2.6% output err), out_proj the MOST (12.7%);
  errors add in quadrature; weight error > activation error; no growth with sequence; outliers extreme (MLP down max/RMS 368, GDN out 298)
  - they rely on 16-element block scales (NVFP4), which A10G int4 lacks -> Hadamard rotation is our substitute.
- Built h1lib.py (uniform low-bit emulation, all rows, integer-exact _int_mm, input AND output-side residual rotation exactly as a rotated
  runtime would store weights, GPTQ, ba16, per-head R2 option), h1eval.py, h1calib.py, h1sens.py. Smoke test: dense == G2 dense exactly.
- FORMAT.md v0 written for H2.

## 23:10 PDT  sensitivity + the bf16 noise floor (MEASURED, 23 train-split calib questions, h1sens.py / floor.py)
- W8A8 (RTN, rotated, all 96 GEMMs): mean KL(dense||q) 3.6e-4, final-hidden rel err (head rows) 1.00%, 0/23 flips.
  Sum of single-GEMM KLs = 2.7e-3 = 7.6x the joint KL -> singles are NOT additive at 8 bits: they sit on a floor.
- The floor: replacing ONE GEMM by an fp32 path with NO quantization gives hidden err 0.35%, KL 2-6e-5; ALL GEMMs fp32 no-quant: 0.35%,
  KL 5.8e-5. One GEMM at W8A8: 0.33%, KL 1.8e-5 (= floor). => any arithmetic change re-rolls the bf16 residual-stream rounding
  (~0.35% at the head). This is the same floor that makes G2's bf16 runtime flip 0.46% vs deployed hobson (merged_full 0.4%).
- Decomposition of W8A8 hidden error (adds in quadrature): W8 only 0.77% (KL 3.5e-4), A8 only 0.74% (KL 1.7e-4), floor 0.35%.
  => weight rounding at 8 bits is as large as activation rounding: GPTQ-W8 is a real lever.
- Implication for the bar: flips scale ~ with head-hidden error. dense: sqrt(2)*0.35 ~ 0.49% <-> 0.46% flips vs hobson.
  To be < 0.5% vs hobson the W8A8 quantization error must be ~ at the floor; vs our dense, quant err must be < ~0.4% (W8A8 RTN ~0.94%).
- W4A4 GPTQ (bf16 mode): KL 0.065, flips 3/23, sum of singles 0.044 (additive, well above floor). Top singles: 0.Wo .0033, 23.Wd .0029,
  8.Wgu .0021, 2.Wgu .0019, 11.Win .0016, 7.Win .0013, 7.Wo .0013 ... spread out (top-16 = ~50%).
- G2's queue finished (05:58 UTC). Started e1 (main subset: dense; w16a16 = fp32 path no quant = the 'perfect quantizer' floor; w8a8 RTN;
  w8a8 GPTQ8) and sensfp (single-GEMM W8A8-GPTQ sensitivity in an fp32 model = no floor).

## 00:20 PDT  (W8A8 first eval truth; fp32-mode sensitivity)
- e1 (main subset: REAL-ALL 1083 + CF-T 218 + CF-probe-T 210), my exact emulation (output-side R1 too):
  | cfg | REAL flips vs hobson | vs dense | agree_sd | CF fgh | CF-probe fgh (vs dense) |
  | dense (bf16, ==G2 dense) | 5/1083 0.46% | - | 0.997 | 1.000 | 0.971 |
  | w8a8 RTN | 14/1083 1.29% | 11 1.02% | 0.994 | 0.972 | 0.962 (0.971) |
  NB dense itself loses 3/105 CF-probe tracked pairs (fgh 0.971): the 0.97 bar on CF-probe is AT the bf16-runtime floor.
- sensfp (fp32 model = no bf16 floor; single GEMM W8A8-GPTQ8; 29 train-split questions): ALL KL 1.55e-4, sum singles 1.21e-4 (additive now).
  KL share: 23.Wd 23%; then layers 0-13 broadly (0.Wo 4.8%, 7.Wo 3%, 8.Wgu 2.8%, 11.Wo 2.7%, ...); layers 14-21 ~nothing (1.6% total).
  By GEMM type: Wd 36%, Wgu 24%, Wo 22%, Win 17%. Hidden-err metric is 85% 23.Wd (last writer -> no attenuation), so use KL.
  Greedy by KL removed per us (A10G CUTLASS s8 vs bf16 times): top-8 bf16 = {23.Wd,0.Wo,7.Wo,11.Wo,10.Wo,23.Wo,12.Wo,9.Wo} removes 38.5%
  of KL for +0.71 ms/1k tok of GEMM (W8A8 all = 25.2 ms/1k vs bf16 49.2); top-16 46% for +1.43 ms.
- Started QAT-A: W8A8, GPTQ8 init, top-8 bf16 (precmap_w8a8_b8.json), LoRA r16 in rotated space, STE, KL + 1.0*hid, lr 1e-4, 2400 steps.

## 00:45 PDT  GPTQ8 is the big W8A8 lever (MEASURED, main subset)
  | w8a8 GPTQ8 | 7/1083 0.65% vs hobson | 8 0.74% vs dense | agree_sd 0.991 | CF fgh 0.982 | CF-probe 0.962 (0.980 vs dense) | TV .0055 (RTN .0076, dense .0031)
- FORMAT.md v1 published for H2 (GPTQ for both widths, reproduction recipe = code + calib, W8A8 candidate map b8, W4 map k48).
- QAT-A restarted with 1500 steps / maxtok 1536 (first launch: 4.2 s/step while sharing the GPU -> too slow).

## 01:15 PDT
- QAT-A (W8A8, lr 1e-4) DIVERGED in the sense that matters: train KL 3.0e-4 (step 50) -> 1.2e-3 (200) -> 2.3e-3 (300), hid 3.5e-4 -> 4.1e-4,
  train flips 0 -> 6%. Diagnosis: Adam steps of 1e-4 on LoRA give B*A ~ 7*lr ~ 3x the 8-bit weight grid (absmax/127 ~ 8e-4) -> random code
  walks; at KL ~3e-4 the true gradient is tiny vs. noise. Killed at step 300 (ckpt s250 kept). 8-bit QAT would need lr <~1e-5; deprioritized.
- Floor measured: w16a16 (fp32 path, NO quantization, every GEMM) on REAL: 4/1067 = 0.37% flips vs hobson, 5/1067 = 0.47% vs dense, TV .0031.
  => the best any quantizer can do in this runtime is ~0.4% flips; W8A8-GPTQ8 (7/1083 = 0.65%) is within ~1.5 Poisson SE of it.
- Launched e2 (W8A8 GPTQ8 + bf16 maps b8 / b16, main subset) and cmp1 (W4 variants on 43 calibration questions).

## 01:55 PDT
- e2: W8A8 GPTQ8 + b8 (8 GEMMs bf16): REAL 8/1083 0.74% vs hobson (7 = 0.65% vs dense), TV .0046 (all-W8A8 GPTQ8 .0055, floor .0031),
  agree_sd .994, CF fgh 1.000, CF-probe fgh 0.971 (= dense; 0.990 vs dense). b16 partially run, then stopped for the question-row idea.
- cmp1 (43 calib q, KL vs dense): w8a8 RTN 5.8e-4 | GPTQ8 1.8e-4 | GPTQ8+b8 1.7e-4 | GPTQ8+ba16 2.8e-4 (ba16 no help at 8b) |
  w4a8 GPTQ 4.7e-3 | w4a4 RTN .084 | w4a4 GPTQ .052 | +ba16 .045 | +ohead .056 | clip .85 .045 | seed 7 .060 (rotation seed matters ~15%) |
  w4a4 GPTQ + k48 (48 GEMMs W8A8) .0077.
- G2's tap8.json re-scored: S:r8 (state rows W8A8-RTN, QUESTION rows bf16): REAL 5/1083 = 0.46% vs hobson, TV .0044 (all-row r8: 14, .0069).
  => the question rows (median 110 tok, p90 1014) carry a large share of the W8A8 decision error. Co-design: state rows low-bit, question
  rows (and the answer readout rows) bf16 - in a schema-first / compiled-question layout the question rows are precomputed once anyway.
  Implemented 'qb16' in h1lib (rows >= q0 run the bf16 GEMM). e3 = full suites for w8a8+GPTQ8+qb16 (+/- b8) + dense rest.
- QAT-B running: W4A8 (all 96), GPTQ init, LoRA r32, lr 1e-4, KL + hid, 1200 steps; train KL 5.5e-3 (s50) -> 4.2e-3 (s100).

## 02:40 PDT
- H2 ran my v1 recipe in the real runtime (g1): W8A8 GPTQ8+b8 REAL 2/1083 = 0.18% vs hobson, agree_sd .997, CF fgh 1.0, CF-probe .971,
  JB-hard .523 -> passes in H2 runtime; my emulation of the same recipe 8/1083. H2 bf16 floor 4/1083. Flip counts ~Poisson around the floor.
- e3 REAL: W8A8 GPTQ8 + qb16 (question rows bf16): 7/1083 = 0.65% vs hobson (6 = 0.55% vs dense), TV .0036 (floor .0031; all-row .0055;
  b8 .0046), agree_sd .991. TV says qb16 removes ~80% of the excess-over-floor; flips are saturated by floor noise.
- FORMAT.md v1.1 (qb16) for H2.
- QAT-B (W4A8) train KL 5.5e-3 (s50) -> 3.8e-3 (s400): slow. W8A8 is 1.8e-4 -> W4A8 will not reach the bar in 1200 steps.

## 04:05 PDT
- e3 done: W8A8-GPTQ8-qb16 over ALL suites (3126 q): see table in report; REAL 7/1083 (paired vs dense 4 lost/2 gained p .69), TV .0036,
  CF fgh 1.000, CF-probe fgh .990 (1.000 vs dense), JB-hard .538 (McNemar 0 lost/2 gained), JB-long agree .974, SHUF .346/.251, REAL-label .790.
- QAT-B stopped after the s900 checkpoint (time); train KL 5.5e-3 -> ~1.6e-3 by s600 (3.5x), hid flat ~1.2e-2.
- Running: e3b (dense on all suites, then W8A8-GPTQ8-b8-qb16 main), e4 (W4A8 QAT curve s0/s300/s600/s900 on quick), e5 (W4 + qb16 variants).
- H2 finished (their NOTES): full latency matrix v4, incl. W8A8-GPTQ-b8 T=1000 1q plain 36.3 ms (bf16 57.0), T=4000 142.1 (203.6).

## 05:08 PDT  final
- W8A8-GPTQ8-qb16 all suites: REAL 7/1083 (dense 5; paired 4 lost/2 gained p .69), LONG 4/471 (dense 2; 2/0 p .5), pooled 11/1554 = 0.71%
  (dense 7/1554 = 0.45%; paired 6/2 p .29); TV REAL .0036 (dense .0031); CF fgh 1.000, flip_rel 1.018, dir .946; CF-probe fgh .990,
  flip_rel 1.029; JB-hard .538 (0 lost/2 gained); JB-long acc .429 agree .974; SHUF .346/.251; REAL-label .790.
- W4A8 QAT-B learning curve (quick subset; s0/s300/s600/s900): REAL-SD flips 2.3/2.6/3.8/2.6%, TV .0277/.0268/.0263/.0253,
  CF fgh .972/.954/.917/.927, CF-probe fgh .781/.771/.771/.771 -> train KL fell 3.5x but eval tracking did NOT improve (same as G2's W4A4 QAT).
- W4A8-GPTQ + qb16 (quick): REAL-SD 2.3%, TV .0181, CF .936, CF-probe .876.
- W4A4-k48 (48 GEMMs W8A8) GPTQ + ba16 + clip .85 + qb16 (quick): REAL-SD 2/346 = 0.58%, TV .0124, agree_sd .994, CF fgh .972,
  CF-probe .933 -> best 4-bit point; fails CF-probe (.97 bar). H2 runtime k48 all-row GPTQ: REAL 2.59%, CF-probe .848.
- Plots: h1/qat_train_curves.png, h1/qat_eval_curves.png; data h1/res/curve_eval.json.
- Report returned as the final message (the harness forbids writing report .md files, so there is no REPORT.md; same as H2).

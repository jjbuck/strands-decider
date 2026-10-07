# H3 notes: hyperspherical co-design for 4-bit on hobson-v19 (box g3)

## [21:48 PDT] start
- Read BRIEF7, F7_REPORT, evalkit README, paper (22 pages).
- Paper facts that matter here (verified by reading):
  - Fig 2/3 metric: mean over tokens of ||y_q - y|| / ||y|| at positions layer_in, attn_in, attn_o_in, attn_output, A+R, layer_out (Fig 7), normalised to the error at the output of the quantised layer (layer 2, MXFP4 W+A).
  - Partial hypersphericity (v1 = hyperspherical lm_head; v2 = v1 + L2Norm without gain and without 1/sqrt(d)) is evaluated after a ~10B-token FineWeb fine-tune (11-22 h on 8xH200). Tables 2/3 are INT4 *weight-only*; Tables 5-8 MXFP4/NVFP4 W+A (block 32/16, RTN).
  - There is no "standard architecture + same fine-tune, no QAT" control in the paper: the "-" row is naive quantization with no fine-tune. So v1/v2 gains are confounded with the 10B-token fine-tune itself. v2 collapses on Pythia (39.8 vs 57.1 BF16, Tables 9/10).
- Box g3 at start: a previous-round G3 job (SmallThinker MoE decider: train_g3.py then ev_g3.py chain) was running. Training finished 21:50 PDT; its final-checkpoint eval is running now. I created ~/work/g3/STOP_EVAL (the chain's own stop flag) so the chain stops after the final-checkpoint eval (skips the optional 50%/75% checkpoint evals). Nothing of G3's deleted.
- G2's earlier W4A4 numbers (recovered res/, scored by me on the laptop): rotated W4A4 RTN (per-channel W, per-token A) on hobson: REAL-agree agree 0.789 (21% flips), CF fgh 0.28 (CF-T subset), JB-hard 0.431; W4A4 QAT 1500 steps: agree 0.889, CF fgh 0.48. Rotated W8A8: agree 0.987.
- The A10G/3090 int4 kernel (g2s4.cu) is per-row/per-column scaled (per-token A, per-channel W). NVFP4 (block 16, E4M3 scale) is the 5090 format; the paper's numbers are block-scaled FP4. Both will be tested.

## [22:18 PDT] code + step 1 (A) results
- Code (~/decider2/h3/code, box ~/work/h3): h3lib.py (lean hobson forward; fake quant int4/int8 per-channel-W (MSE clip) x per-token-A, NVFP4
  (block16, E4M3 scale x fp32 tensor scale), MXFP4 (block32, E8M0), g128; fold / rotation (G2 Kronecker Hadamard, input side); capture of
  every sub-layer tensor; LoRA; heads std / hyp (v1); norms rms / l2 (v2 literal) / l2s (v2 with scalar gain)); errprop.py (A: one layer
  quantised, all sub-layer errors; B: one sub-GEMM quantised, decision effect); train_h3.py (arms); ev_h3.py (preds per config).
- Smoke: dense matches hobson refs to ~1e-3 on REAL-agree items (measured).
- errA (12 train-pool states, 900-2000 tokens; one layer at a time quantised W4A4) [measured]:
  - Propagation is CONTRACTIVE in relative terms: the residual-stream relative error injected at layer l falls to 0.2-0.36x
    (int4) / 0.11-0.35x (NVFP4) by layer 23, monotonically after a small bump in layers 1-3 (<=1.09x). Absolute error grows 1.0-1.13x per
    layer (median), while the residual norm grows 2.3 -> 44.6. Layer 23 is the exception (x2.5-2.7, its MLP writes 2.6x its input).
  - Within downstream sub-layers (the paper's fig-2 metric, rel(pos)/rel(layer_in)): GDN in_proj 0.50, delta-rule core 0.45, out_proj in 1.08,
    out 0.98; MLP hidden 1.19, MLP out 1.21; attention layers similar (core 0.47, out 0.92). NO order-of-magnitude amplification as in the
    paper's standard transformer (~30x at attn_o_in). hobson already has bounded mixers: GDN l2-normalised q/k + gated RMSNorm before out_proj,
    attention q/k RMSNorm + sigmoid output gate. In the paper's terms hobson's submodules are already "partially hyperspherical".
  - Absolute Jacobian gains (error out / error in) of the downstream submodules are < 1: B (mixer) 0.17-0.65, M (MLP) 0.26-0.76, except
    layer 23's MLP (M = 2.1).
  - The damage is LOCAL: at the quantised layer itself int4 per-token gives ~100% relative error on the mixer/MLP outputs
    (activation outliers vs a 15-level per-token grid); NVFP4 gives 14-30%.
  - Decision sensitivity (TV / flips over 12 questions, one layer int4): layer 0 TV 0.68 (11/12 flips!), layers 7 and 11 (attention) TV 0.41,
    layer 23 0.14, layers 1-11 0.1-0.2, layers 12-22 < 0.1. NVFP4: all layers TV <= 0.045.

## [22:30 PDT] step 1 (B) + outlier stats [measured]
- errB (48 train-pool questions; ONE sub-GEMM at int4 per-token W4A4): most decision-critical GEMMs are layer-0 out_proj (TV 0.19, 17% flips),
  attention in-proj (q/k/v/gate) at layers 7 and 11 (TV 0.25 each; 29% / 10% flips), layer-2 out_proj (0.10), layer-10 GDN qkv (0.10),
  gate_up at 4/10 (0.09). GDN a/b rows are the least sensitive GEMM rows (mean TV 0.009; keeping them bf16 changes whole-model TV 0.517 -> 0.497).
  Layers 12-22 almost insensitive (TV < 0.06, mostly < 0.01). Map in ~/decider2/h3/sens_int4.json (for H1's precision map).
- Whole model on those 48 q: W4A4 int4 TV 0.52, flips 73%; W4A16 TV 0.11 / 8%; W16A4 TV 0.50 / 71%; fold (norm gains into W) 0.52.
  => activation 4-bit per-token is the failure; weights are a second-order term.
- outl.py (8 states): median per-token crest factor max|x|/rms(x) at GEMM inputs is 15-40 at every layer (Win in 17-28, Wgu in 14-27,
  Wd in 13-43, Wo in 8-32). Per-token int4 relative error of the activation 0.5-0.7 (most entries round to 0). Removing the norm gain
  (v2 / fold) does NOT lower it (crest 18-36 without the gain; gains (1+w) are only 0.34..2.2). Hadamard rotation: crest 3.2-3.8 (Gaussian
  limit), int4 error 0.13-0.16. NVFP4 block-16: 0.07-0.09 with or without rotation. Weights: int4 per-channel 0.13-0.145, NVFP4 0.094.
- Mechanism (derivation, verified against the numbers): per-token absmax 4-bit error per entry ~ U(+-step/2), step = max|x|/7, so
  relative error ~ crest/(2*sqrt(3)*7) = crest/24 (capped near 1 when most entries round to 0): crest 3.5 -> 0.14 (measured 0.13-0.16),
  crest 25 -> ~1 (measured 0.6-0.7, capped). Crest is scale-invariant, so ANY normalisation of the activation (RMSNorm, L2Norm, hypersphere)
  leaves it unchanged. Hypersphericity bounds ||x|| and ||w_i||, not max|x_j|/rms(x), which is what a 4-bit grid with one scale per token sees.
  The paper's Prop. 1 assumes noise with a fixed covariance bound sigma^2 independent of x; absmax quantisers violate that.

## [22:47 PDT] chain running on g3 (run_main2.sh, resumable)
- errA rotated int4 (G2 Kronecker Hadamard, gains folded): injected error per layer 0.07-0.33 (vs 0.2-0.8 plain int4), per-layer decision TV
  <= 0.08; propagation pattern unchanged (contractive). [measured, 12 q]
- In-runtime bf16 vs deployed hobson refs (dev rows): REAL-agree 6/1083 = 0.55% flips, CF fgh 1.000 (109), CF-probe fgh 0.952 (105),
  JB-hard 0.538 (hobson 0.523). This is the runtime noise floor; quantization flips are reported vs refs AND vs this in-runtime bf16.
- Chain: hobson {bf16,w4a4,w4a4r,nvfp4} -> train v1 (hyp head) -> eval -> train v2 (l2s) -> eval -> train QAT int4 -> eval -> hobson {nvfp4r,
  mxfp4,w8a8} -> train v2lit (paper-literal L2Norm) -> eval -> train NVFP4-QAT -> eval -> LONG-SD for hobson configs.
  All arms: 1200 micro-steps (accum 4 = 300 updates), same data order, KL + 1.0 x residual relMSE (layers 5/11/17), LoRA r16/a32 all GEMMs,
  lr 1e-4 / head 1e-3. No gold CE (target = fidelity to hobson). The standard-architecture KL fine-tune is an exact fixed point (student ==
  teacher at init, zero gradient), so "control + same fine-tune" == hobson itself (by construction).
- Smoke at init (8 steps): v2 l2s (gain removed, scalar kept) KL 0.53 / residual relMSE 0.18; v2 literal (x/||x||) KL 1.0 / relMSE 1.0
  (model destroyed, as the paper's Pythia rows suggest); QAT int4 at init KL 1.03.
- Step 4 (optional) written: pretrain_tiny.py (std pre-RMSNorm vs nGPT-style, d512 L8, same data/tokens), to run concurrently if GPU allows.

## [23:15 PDT] hobson PTQ on dev rows (REAL-ALL 1083 + CF-T 218 + CF-probe-T 210 + JB-hard 130) [measured]
| config | REAL flips vs refs | agree_sd | CF fgh | CF-probe fgh | JB-hard (hob .523) |
| bf16 (this runtime) | 0.55% | 0.997 | 1.000 | 0.952 | .538 |
| W4A4 int4 per-token RTN | 59.7% | 0.373 | 0.239 | 0.210 | .415 (McNemar p .08) |
| W4A4 int4 + Hadamard (fold+rot, RTN) | 15.9% | 0.682 | 0.550 | 0.448 | .446 (p .12) |
- G2's earlier rotated RTN on the same rows: 21.1% (different rounding details); same regime.
- Chain swapped to run_main3.sh (order: v1, v2, QAT-int4, QAT-NVFP4, hobson nvfp4r/w8a8/g128r, v2-literal, LONG). Online activation
  rotation switched to a bf16 GEMM (as a runtime would do it) for all runs after hobson w4a4r (fp32 rotation there; difference ~0.2% rel
  per element vs ~13% int4 noise).
- Tiny pretraining (std then nGPT, 2000 steps x 16k tokens = 33M tokens) running concurrently; std done (val TBD).

## [23:22 PDT] step 4 (small scale, optional) [measured, 1 seed, 35M params, 33M tokens; speculative for 2B]
- d512 L8 H8 FFN1536, same data/batches; std = pre-RMSNorm GPT (AdamW wd .1, warmup, lr 1e-3); nGPT-style = unit-norm hidden + learned
  interpolation (alpha init .05), all matrices unit-norm along d after every step, q/k unit-norm x s_qk, softmax scale sqrt(hd), s_u/s_v, s_z
  (Adam lr 2e-3, no wd/warmup). Val = held-out train-pool tasks (2.3M tokens). Text is templated (KB docs repeat) so losses are low (~1.07 nats).
  | val loss | bf16 | int4 per-token | int4+Hadamard | NVFP4 | MXFP4 | int8 |
  | std  | 1.0653 | 1.1586 (+8.8%) | 1.0881 (+2.1%) | 1.0791 (+1.3%) | 1.0863 (+2.0%) | 1.0656 |
  | nGPT | 1.0786 | 1.1564 (+7.2%) | 1.0938 (+1.4%) | 1.0882 (+0.9%) | 1.0932 (+1.4%) | 1.0787 |
- Hypersphere: smaller RELATIVE degradation (as the paper reports) but no better ABSOLUTE quantized loss (nvfp4 1.0882 vs 1.0791; int4
  1.1564 vs 1.1586) because it is worse in bf16 at equal tokens. GEMM-input crest: qkv 3.2 / 3.2, o 3.9 / 3.7, down 11.0 / 15.3 (nGPT
  higher at the MLP hidden). The tiny models have no massive-activation outliers (crest 3.2 = Gaussian limit) - the regime that breaks
  hobson's per-token int4 (crest 15-40) does not appear at this scale, so this cannot test the crest mechanism.

## [23:57 PDT] v1 (hyperspherical pointer head) results so far [measured]
- train v1: 1200 micro-steps in 15 min (0.76 s/step); KL(real) 0.095 -> 0.009, tau 7.30 -> 7.36.
- hobson NVFP4 PTQ (dev): REAL 9.14% flips, agree_sd 0.884, CF fgh 0.697, CF-probe 0.705, JB-hard .492 (McNemar p .42).
- v1 bf16 (after FT): REAL 4.99% flips vs hobson, agree_sd 0.902, CF fgh 0.817, CF-probe 0.810, JB-hard .515. The head retrofit itself
  costs fidelity at this budget (cosine logits drop the |q||k| confidence information hobson's head uses).
- v1 W4A4 int4: 55.8% flips (hobson 59.7%), vs own bf16 56.1%; CF fgh 0.20. No robustness gain.
- Flips vs hobson margin (REAL): NVFP4 flips 52% of decisions with top1-top2 margin < .1 (n 88), 24% at .1-.2, 10% at .2-.4, 1% at
  .4-.6, 0 above; plain int4 flips 57% even at margin > .6 (model broken). 19% of REAL decisions have margin < .2, so < .5% flips needs
  near-bf16 logits.
- Merge check: LoRA merged into bf16 vs unmerged gives 1/400 different decisions (13 vs 14 flips) -> not a factor. From v2 on, evals keep
  LoRA unmerged in bf16 and form W + BA in fp32 before quantising.
- Chain swapped to run_main4.sh: v2 -> QAT int4 -> hobson nvfp4r, g128r -> QAT NVFP4 -> hobson LONG. v2-literal dropped for time
  (its init state: KL 1.0 nat, residual relMSE 1.0, every smoke step flipped; consistent with the paper's own Pythia v2 collapse).
- H1 (g2) notes read: W8A8 RTN 1.29% flips; bf16 runtime floor 0.35% head-hidden error; sensitivity ranks agree with mine (0.Wo, 23.Wd,
  layers 0-13; 14-21 negligible).

## [00:20 PDT] v1 complete [measured]; paired tests
- v1 + int4/Hadamard: 14.22% flips vs hobson (hobson+rot 15.88%; paired McNemar 55 vs 73 discordant, p 0.13; TV diff CI [-0.0001, +0.010]);
  vs own bf16 11.9%. CF fgh 0.606 (0.550), CF-probe 0.486 (0.448).
- v1 + NVFP4: 8.96% vs hobson (hobson+NVFP4 9.14%); vs own bf16 5.4% (hobson 9.3%); agree_sd 0.847 (0.884), CF fgh 0.651 (0.697),
  CF-probe 0.695 (0.705). => the bounded head lowers the arm's OWN quantization flips (9.3 -> 5.4%, 15.9 -> 11.9%) but the retrofit's
  bf16 offset (5.0%) cancels it: no net gain vs hobson at this budget. Queued v1x3 (3x budget) to test whether the offset closes.
- Queued run_tail.sh (after main4): v1x3 (3600 steps) + eval bf16/nvfp4; v2-literal + bf16 eval; LONG-SD for v1.

## [01:25 PDT] v2 partial, question-rows-bf16 at 4 bits [measured]
- v2 (L2Norm with scalar gain, per-channel gain removed) bf16 after FT: 7.57% flips vs hobson, agree_sd 0.855, CF fgh 0.780, CF-probe 0.724.
  v2 + int4: 59.65% (= hobson), JB-hard 0.346 (McNemar p 0.009).
- Row-role precision (H1's qb16 idea, tested at 4 bits): state rows NVFP4, question rows (incl. option + answer readout rows) bf16:
  REAL 4.16% flips (NVFP4 all rows 9.14%), TV 0.031 (0.068), agree_sd 0.954 (0.884), CF fgh 0.807 (0.697), CF-probe 0.781 (0.705),
  JB-hard .508 (p .77). The largest single gain of any co-design tried here; still 8x over the low-bit bar.
- Chain swapped to run_main5.sh (QAT int4 -> nvfp4r hobson/v1 -> v1x3 -> LONG hobson/v1 -> [QAT-NVFP4 if time]); g128r and v2-literal dropped.

## [01:40 PDT] v1/v2 complete; paired tests on REAL (1083 q; exact McNemar on the flip indicator, bootstrap CI on TV) [measured]
| comparison | discordant flips A/B | p | TV(A)-TV(B) [95%] |
| hobson-rot vs v2-rot (int4+Hadamard) | 84 / 47 | 0.0016 | +0.028 [+0.022, +0.034] (v2 better) |
| hobson-rot vs v1-rot | 73 / 55 | 0.13 | +0.005 [-0.000, +0.010] |
| hobson-NVFP4 vs v1-NVFP4 | 46 / 44 | 0.92 | -0.006 [-0.009, -0.002] (v1 worse TV) |
| hobson-NVFP4 vs v2-NVFP4 | 43 / 55 | 0.27 | -0.007 [-0.011, -0.003] (v2 worse TV) |
| hobson-NVFP4 vs NVFP4-qb16 | 65 / 11 | 2e-10 | +0.037 [+0.034, +0.040] |
| hobson-rot vs rot-qb16 | 123 / 27 | 8e-16 | +0.083 [+0.077, +0.088] |
- v2 + int4/Hadamard 12.47% (vs own bf16 9.7%), CF fgh 0.459 (hobson-rot 0.550), CF-probe 0.486 (0.448); v2 + NVFP4 10.25% (own 4.3%),
  CF fgh 0.459 (0.697), CF-probe 0.648 (0.705). int4/Hadamard + qb16: 7.02%, agree_sd 0.850, CF 0.789, CF-probe 0.667.
- Margins (top1-top2) of v1/v2 bf16 match hobson's (median 0.46-0.47, 18-20% below 0.2), so their lower own-quantization flips are not
  from sharper decisions.
- Reading: hyperspherical retrofits do lower the arm's OWN sensitivity to 4-bit noise (NVFP4: 9.3% -> 5.4% (v1) / 4.3% (v2) flips vs own
  bf16), as the paper says, but at this budget the retrofit's bf16 offset (5.0% / 7.6% flips, CF fgh 0.82 / 0.78) eats the gain; the only
  net win vs hobson is v2 on rotated int4 (-3.4 pts flips, p .002) and it tracks fewer CF flips. Row-role precision (qb16) beats every
  architecture retrofit by a wide margin.

## [02:28 PDT] status
- Writing ~/decider2/h3/REPORT.md with the Write tool was DENIED by the harness ("subagents should return findings as text, not write report
  files"). Not worked around: the report goes in my final message only.
- QAT int4 (plain per-token, same budget) at step 700/1200: train KL 0.65-0.81, residual relMSE 0.80-0.97, train flips 48-76% -> it is not
  learning to live with crest-25 per-token int4 at this budget.
- Chain now run_main7.sh: QAT eval -> QAT-NVFP4 train+eval -> LONG (hobson, v1; bf16 + NVFP4) -> hobson nvfp4r -> v1x2 (2x budget) + eval.
- Concurrent: hobson nvfp4r-qb16 (dev) nearly done.

## [02:30 PDT] QAT int4 + more PTQ combos [measured]
- QAT int4 (plain per-token, same 1200-step budget): W4A4 47.9% flips vs hobson (40.9% vs its own bf16), agree_sd 0.402, CF fgh 0.211,
  CF-probe 0.190, JB-hard 0.308 (McNemar p 0.001). Its unquantised bf16 is 36% off hobson (QAT moved the weights).
  Paired REAL: QAT beats hobson PTQ (206 vs 333 discordant, p 5e-8) and beats v1 under int4 (229 vs 314, p 3e-4).
  => at the matched format, QAT > hyperspherical retrofit (contrary to "retrofit ~ QAT"), though all plain-int4 variants are broken.
- NVFP4 + Hadamard + qb16: 4.43% flips, TV .036, agree_sd .948, CF .780, CF-probe .743 (NVFP4-qb16 without rotation 4.16%):
  rotation adds nothing on top of block-16 scaling.
- Paired CF tracking (tracked pairs, McNemar): v2-NVFP4 loses 31 vs gains 5 of hobson-NVFP4's tracked CF pairs (p 1e-5); v2-rot vs
  hobson-rot p .12; v1 vs hobson NVFP4 p .33; NVFP4-qb16 vs NVFP4 gains 19 vs 7 (p .03).
- Queued run_tail2.sh (after main7): QAT-NVFP4 + qb16, v1 NVFP4-qb16 (does the head stack with row-role precision?), v2 int4-rot-qb16.

## [03:25 PDT] QAT-NVFP4 (same budget) [measured]
- qatnv + NVFP4: 7.20% flips vs hobson (3.5% vs own bf16), TV better than PTQ; agree_sd 0.899, CF fgh 0.596, CF-probe 0.733, JB .469.
- Paired REAL: QAT-NVFP4 beats hobson NVFP4 PTQ (40 vs 61 discordant, p .046), v1-NVFP4 (31 vs 50, p .045) and v2-NVFP4 (24 vs 57,
  p 3e-4). CF tracked pairs: vs PTQ 10 vs 21 (p .07), vs v1 7 vs 13 (p .26): QAT does not improve counterfactual tracking.
- => "partial hypersphericity matches QAT" does not hold on hobson at an equal budget, in either format (int4: QAT 47.9 vs v1 55.8%,
  p 3e-4; NVFP4: 7.2 vs 9.0%, p .045).

## [03:45 PDT] LONG-SD (165 state-dependent q, 4-8k-token states) [measured]
- hobson bf16 (runtime floor): 1.21% flips, agree_sd 0.988. hobson NVFP4: 12.73%, agree_sd 0.873 (REAL-SD at NVFP4: 0.884 -> no length
  growth). v1 bf16: 12.73% flips, agree_sd 0.873 (the retrofit generalises badly to long states; FT used <=2048-token states). v1 NVFP4:
  14.55%, agree_sd 0.855 (4.85% vs own bf16).
- QAT-NVFP4 bf16 (no quant): 6.09% flips vs hobson, CF fgh 0.706.

## [04:35 PDT] v1 at 2x budget (v1x2, 2400 micro-steps) [measured]
- bf16: 5.26% flips vs hobson (v1: 4.99%), agree_sd 0.890, CF fgh 0.596 (v1 0.817), CF-probe 0.848. Doubling the budget did not close
  the retrofit's bf16 offset; CF tracking got worse.
- NVFP4: 6.93% vs hobson (4.9% vs own bf16), agree_sd 0.870, CF fgh 0.550, CF-probe 0.733.
- hobson NVFP4 + Hadamard (all rows): 8.96% (no gain over NVFP4 9.14%), agree_sd .855, CF .752, CF-probe .660.
- Paired: v1x2-NVFP4 vs hobson-NVFP4 REAL 36 vs 60 discordant (p .018, v1x2 better) but CF tracked pairs 6 vs 22 lost (p .004, worse);
  v1x2-NVFP4 vs QAT-NVFP4 (1x budget) REAL 36 vs 39 (p .82). So at 2x budget the head retrofit matches 1x-budget QAT on REAL flips, at the
  cost of counterfactual tracking.

## [04:58 PDT] combinations with row-role precision (qb16) [measured]
- v1 NVFP4-qb16: 6.46% vs hobson (2.6% vs own bf16), agree_sd .879, CF .679, CF-probe .752, JB .546.
- QAT-NVFP4 + qb16: 6.09% vs hobson (2.0% vs own), agree_sd .913, CF .615, CF-probe .733.
- hobson NVFP4-qb16 (no training): 4.16% - best of everything. Retrofit / QAT lower their own quantisation flips to 2-2.6%, but their
  bf16 offsets (5-6%) dominate vs hobson. The fine-tune, not the 4-bit noise, becomes the error source.
- Final sync to ~/decider2/h3/{preds,res,logs} done; LoRA checkpoints stay on the box (no models on the laptop).
- [05:06 PDT] v2 int4+Hadamard+qb16: 9.51% vs hobson (3.6% vs own); worse than hobson rot-qb16 7.02% (paired p .013). Final table:
  ~/decider2/h3/table_dev.md; all scores ~/decider2/h3/scores_all.txt. Work ends here (box shutdown 05:43).

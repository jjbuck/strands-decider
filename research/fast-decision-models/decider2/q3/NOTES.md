# Q3 notes: decision-level rounding and quantization-aware distillation (B4), box q3

## 2026-10-06 20:02 PDT start
- Read BRIEF10 in full, FAST_DECISION_MODEL.md section 5, IDEAS_EXPLORED P1-P9 + floor, H1 REPORT/FORMAT/NOTES, H2 REPORT, H6 REPORT,
  J14 DRAFT_REPORT + j14train.py/j14lib.py/j14train_util.py, h3lib.py, h1lib.py, H2 qrt.py/evalrun.py, evalkit README.
- Box q3 (i-REDACTED, g5.8xlarge) launched 19:52, pending; setup script is connecting (q3.ssh written 20:01). No q3.ready yet.
- Q1/Q2 outputs: none yet (no FORMATS.md, no best_formats.json). Until FORMATS.md exists, the arithmetic is H1 FORMAT v1 as implemented
  by h1lib.H1.lin / H2 qrt.py: rotated residual (R1 seed 1234, R2 1235, R4 1236), int4 per-token absmax activations with clip 0.9,
  per-channel symmetric int4 weights, int32 accumulation, y = acc * s_tok * s_ch.
- Plan (compute arithmetic first): student fwd+bwd+recompute ~11 GFLOP/token + teacher fwd 2.75 => ~14 GFLOP/token; at an effective
  25-35 TFLOPS on the A10G that is 1.8-2.5k tokens/s, so 50M tokens = 5.5-7.7 h. The 10-h box lifetime makes 50M the realistic target;
  100M is out of reach without a timer reset and is not planned.
  1. B4.1a scales: GPTQ codes fixed, per-channel weight scales trained end to end on decision KL (EfficientQAT E2E-QP style).
  1. B4.1b rounding: AdaRound-style soft rounding of every weight, initialised from GPTQ's error-compensated weights (hard limit =
     GPTQ codes), end-to-end decision KL + hidden MSE, annealed regulariser, then hard rounding.
  2. B4.2 full-weight QAD: bf16 latent weights + stochastic-rounding Adam (factored v), STE through the exact W4A4 arithmetic,
     KL + rel. hidden MSE at layers 5/11/17/23 on answer+option rows + decoupled L2 pull, shared-prefix multi-question batches.
  3. Score best dev checkpoint on all 3,227 questions through H2's QRT kernels with exported codes (wcodes), plus emulation.

## 20:45 PDT box ready (20:13), init + smoke
- Code (laptop ~/decider2/q3/code, box ~/work/q3): q3lib.py (student/teacher, packed layout, QLinF STE, GPTQ with compensated
  weights, AdamSR), q3gptq.py, q3train.py (modes scale/round/qad), q3eval.py (emulated suites + code export for QRT), q3score.py (laptop).
- GPTQ init [M]: 64 train-split calibration pairs (H1's cal_items selection, 145,075 tokens), act-order GPTQ4 on the rotated folded
  weights; 130 s. Also stores GPTQ's error-compensated weights Wc (round(Wc/s) = GPTQ code exactly; 0.3% of entries needed a bf16 fix-up).
- Smoke [M] (6 train requests, 24 questions):
  - packed multi-question teacher vs one question per sequence: TV 0.0 (identical).
  - bf16 student in the rotated domain vs teacher: TV mean .0038 (max .012), residual rel. MSE 1e-4 at layers 5-23 = bf16 floor level.
  - W4A4-GPTQ student vs teacher: TV mean .089, 2/24 flips, KL .039; residual rel. MSE .07 (L5), .12 (L11), .037 (L17), .012 (L23).
  - all three modes backprop finite gradients.
- First speed [M]: student fwd+bwd 1.6 s per ~2,000 rows + teacher 0.22 s + optimizer 0.65 s/step => ~1,050 tokens/s => 50M tokens
  would take 13 h. Profile: GEMMs ~25% of CUDA time; the rest is elementwise (weight re-quantization on every call, activation quant,
  fp32 tensordot rotations, dtype copies). Fixes: int8 code cache per optimizer step, +-1 matmul Hadamards with exact products,
  torch.compile'd elementwise helpers in training only (dev/eval scoring stays eager = FORMATS.md arithmetic).
- Q2 FORMATS.md v0 (20:40) read: deployed F1 = int32 acc -> fp16(alpha*acc) -> (C16*s_t)*(sw/alpha); F2 (Q2's new kernels) = (acc*s_t)*sw
  -> bf16. My STE forward is F2 exactly (modulo the GEMM-input glue, which FORMATS.md says is not bit-exact in any emulation); q3eval has
  an F1 option (fp16 stage on Win/Wo/Wd; gate_up EVT has no fp16 stage).
- H2's CUTLASS libs built on q3 (g2s4, h2mix, h2evt rc 0) for deployed-kernel scoring.
- Scorer check [V]: q3score.py on H2's existing preds reproduces H2's table (bf16 0.37% flips; b8 0.18%; W4A4-GPTQ 8.96%, TV .0775,
  CF fgh .826, CF-probe fgh .657). The W4A4 TV must fall ~15x to reach b8's .0048.

## 20:55 PDT B4.1a (scales) running; dev baseline; coordinator note on q3b
- Dev set [M]: 100 requests / 366 questions from 15 train-split tasks held out of training by hash (J14's is_dev); teacher = my bf16
  lean runtime (packed layout). Train pool: 10,116 requests (states 32-5,000 tokens) + 12,557 train_v5 rows.
- GPTQ W4A4 init on dev (exact emulation, eager): flips 19/366 = 5.19% vs bf16 teacher, TV .0715, KL .0253, agree_sd .896 (96 sd q).
- Speed after the first fixes [M]: 1,550 tokens/s (student fwd+bwd + teacher, 2k-row requests); more glue compiled (norms, conv, GDN
  gates/gated norm, SwiGLU, epilogue, split instead of slicing), to be measured after the scale run.
- Coordinator: a bigger training GPU may appear as box q3b (H100 80 GB / RTX PRO 6000 96 GB / L40S 48 GB). Plan: keep working on q3;
  if ~/decider2/boxes/q3b.ready appears, re-run q3gptq.py there (no checkpoint copies via the laptop), move QAD training to q3b, and keep
  q3 for deployed-kernel scoring and latency.
- Box q3 lifetime: booted 19:52 PDT, auto-shutdown ~05:52 PDT (timer not yet reset).

## 21:00 PDT B4.1a result (decision-level per-channel weight scales, GPTQ codes fixed) [M, dev, exact emulation]
- Run ck_scale: 1.51M tokens, 152 Adam steps on log-scales (0.6M parameters), lr 1e-3 cosine, KL + 0.5 x hidden rel. MSE.
  | tokens | dev flips vs bf16 teacher (366 q) | dev TV | dev KL | agree_sd (96) |
  |---|---|---|---|---|
  | 0 (GPTQ) | 19 (5.19%) | .0715 | .0253 | .896 |
  | 0.76M | 21 (5.74%) | .0560 | .0183 | .906 |
  | 1.51M | 23 (6.28%) | .0535 | .0161 | .938 |
  TV -25%, KL -36% against GPTQ; flip count unchanged within noise (19 vs 23 of 366, paired counts not yet split). Speed 1,360 tok/s.
- B4.1b (AdaRound-style rounding, seeded from these scales; init hard rounding = GPTQ codes, mismatch 3e-8) started 21:00, 3M tokens.

## 21:20 PDT B4.1b restarted with a GPTQ-side init; Q1 best_formats.json read; plan for the w4q8 arm
- First rounding run (AdaRound init from the fractional part of GPTQ's compensated weight) killed after step 0: that soft model is
  much worse than its own hard rounding (soft TV .077 / 7.9% flips vs hard .053), because GPTQ's compensated weights are only good once
  rounded (their continuous output error equals RTN's: tr(E H E^T)/tr(W H W^T) 2.3e-3 vs GPTQ 2.5e-4 at 20.Win, from gptq4_stats.json).
  Also fixed an OOM (the regulariser was computed with autograd on). Restarted with h0 = .95 on GPTQ's side (soft TV .0509 ~ hard .0535),
  regulariser ramped from 0 with beta 20 -> 2, lr 1e-2 on V, 2M tokens.
- Noise floor of the dev flip count [M]: the same codes and scales scored 23/366 flips in one run and 20/366 in another that differed
  only in ~40 weight codes (3e-8 of all weights); TV .0535 vs .0534. Dev flips move by +-3 for negligible changes; TV is the metric.
- Q1 best_formats.json (21:12): best = null; candidates w4q8 (state rows W4A4, question rows W8A8 with a separate int8 weight copy,
  FORMATS.md section 8) and tc4q8. Q1's B1.1: question rows carry nearly all decision sensitivity at layers 13-23.
  => Added the w4q8 row-role format to the student (question rows: fixed GPTQ8 codes, W8A8; state rows: trainable int4 path) for a
  second QAD arm (on q3b if it appears, else after/instead of part of the plain arm, decided from the plain arm's curve).
- Queue on q3 (chain2.sh): GPTQ8 codes -> QAD plain W4A4, seeded from the better B4.1 result by dev TV, 50M tokens, dev + CF-probe +
  REAL-agree scoring every 10M tokens (scoring is reporting only; selection is dev TV).

## 21:32 PDT process mishaps, fixed; gradient check
- A `pkill -f chain2.sh` inside my own ssh command killed that command (pattern matched itself), so an lr-1e-2 rounding run kept going
  (its soft-model train KL rose .03 -> .15 and hidden MSE .07 -> .16 in 20 steps: Adam with a factored second moment gave outlier
  updates; now clipped per element at 3 and per tensor at RMS 1). Then chain2's process check raced with my relaunch and started QAD
  concurrently with a new rounding run, which OOM'd. All killed; one sequential chain (chain3.sh) now runs rounding (lr 3e-3) then QAD.
- Exactness fix: the cached int8 codes were computed by a torch.compile'd division in training, which differs from eager on a few
  boundary weights (QAD step-0 dev TV .0520 against .0535 for the same exported codes). Code cache is now always eager = exported codes.
- Gradient check [V] (q3gradcheck.py, random 64x256x128): QLinF lat dW, dx, soft dV, code d(log s) match autograd on the straight-through
  reference to 0.2-0.3% (bf16); +-1 matmul Hadamard equals Rot to 1e-6, its gradient equals g R^T to 0.17%.
- GPTQ8 codes (gptq8.pt, for w4q8 question rows) computed, 87 s.

## 21:48 PDT B4.1b result: end-to-end AdaRound changed no rounding in budget; QAD started 21:47
- Rounding run (seed = B4.1a scales; h0 .95 on GPTQ's side; lr 3e-3 on V, update clipping; regulariser ramp, beta 20 -> 2) [M]:
  after 1.0M tokens (104 Adam steps on 1.37B rounding variables) the hard rounding equals the seed exactly (0 of 1.37B codes changed;
  dev TV .0535, 23/366 flips, identical to the seed). Meanwhile the soft model's train KL rose (.04 -> .06-.07 per 10 steps against .02-.03
  for the scale run on the same batches) and hidden MSE .058 -> .075: V moved inside the cells without crossing h = .5.
  Arithmetic: flipping one rounding needs dV ~ 2 from the h=.95 start; at <= 9e-3 per step (lr x clip) that takes >= 220 consistent steps,
  more than this budget. The lr-1e-2 attempt (no clipping) degraded the soft model faster (KL .15 at step 20). Stopped at 1M tokens.
  => Decision-level rounding alone: no measured gain at ~1M tokens; the scale result (B4.1a) is the better B4.1 point and seeds QAD.
- QAD plain W4A4 (ck_qad) started 04:47 UTC, seed = ck_scale/best.pt (GPTQ codes + trained scales; latent = GPTQ's compensated weights
  rescaled), lr 2e-5 (warmup 30, cosine to 10%), L2 pull 100 toward the seed latent, w_hid .5, accum 4, 50M tokens, dev + CF-probe +
  REAL-agree scoring every 10M (reporting only).

## 22:05 PDT first QAD attempt diverged in 30 steps; restarted with latents on the grid points
- Curve scoring in "fast emulation" (torch.compile'd glue incl. activation quantizer; differs from eager only at rounding ties):
  1,723 questions (CF-probe + REAL-agree) in 288 s. B4.1a point (GPTQ codes + trained scales) [M, fast emulation, vs hobson refs]:
  REAL flips 6.56%, REAL TV .0621, agree_sd .902, McNemar vs H2 bf16 runtime 67 lost / 0 gained; CF-probe fgh .629.
  (H2's deployed W4A4-GPTQ for reference: 8.96%, TV .0775, CF-probe .657; different runtime and code draw, not paired.)
- QAD with the latent at GPTQ's compensated weights (lr 2e-5, warmup): train KL .12 -> .19 -> .21 and hidden MSE .077 -> .21 in 30 steps
  [M], although each weight moved <= ~1.5% of a grid step. Cause (arithmetic): Adam's early updates are ~sign(g) for every weight, so
  every weight within ~1% of a rounding boundary flips code on a noisy gradient; with compensated latents uniform in their cells that
  is ~2-3% of 1.37B weights per few steps, each flip a full grid step of error. Killed.
- Restarted (05:02 UTC) with latents on the seed codes' grid points (lat_init=center; anchor = same): a code flips only after the
  latent moves s/2 consistently (~75-125 full-lr steps at s = .003-.006), which filters gradient noise; the L2 pull (lambda 100) limits
  noise-driven drift to sqrt(lr/2 lambda) = 3e-4 = 5-10% of a grid step.
- Speed in QAD [M]: 1,650-1,730 tokens/s (A10G, accum 4, mem 18.2 GB). 50M tokens = ~8.2 h + ~30 min of curve scoring.
- 22:07 PDT: center-init QAD is healthy (train KL .032/.018 at steps 10/20, hidden MSE .058; 1,770 tokens/s).
- 22:07 PDT: BOX TIMER RESET (the one allowed reset): `sudo shutdown -c; sudo shutdown -h +600` on q3 at 05:07 UTC. New shutdown
  15:07 UTC = 08:07 PDT (was 12:52 UTC). Reason: 50M QAD tokens at 1,770 tok/s end ~13:25 UTC, plus ~1 h of deployed-kernel scoring.

## 22:22 PDT QAD progress
- 05:21 UTC: step 190, 1.92M tokens, 1,854 tok/s, mem 18.3 GB; train KL .02-.04 and hidden MSE .057-.060 per 10 steps (flat so far);
  relative latent drift 1.5% at step 100 (rms move ~1.8e-4 = ~5% of a grid step: the noise equilibrium sqrt(lr/2 lambda) = 3.2e-4).
  Codes can flip only after ~85 consistent full-lr steps (s/2 / lr), so changes should appear from step ~100-200 onward.
- Prepared final.sh (export codes; H2 QRT deployed-kernel scoring of bf16, GPTQ, B4.1a and every QAD checkpoint on all 3,227
  questions) and final_lat.sh (H2 bench.py, W4A4 vs W8A8-b8, T 64/256/1000/4000, 1q/15q). Staged prof_d1.py, qspecs.json, b8 precmap.
- q3b: not up yet (coordinator retrying). chain_q3b.sh ready: GPTQ re-run there, then w4q8 scales -> w4q8 QAD (100M tokens).

## 22:52 PDT QAD at 5.3M tokens (step 520), 1,866 tok/s
- Train loss per 100 steps [M]: KL .0216 / .0277 / .0233 / .0235 / ~.022; hidden MSE .0588 / .0581 / .0573 / .0576 / .0567: flat KL,
  hidden MSE -3.5%. First dev point at 10M tokens (~23:35 PDT) decides whether to keep lr 2e-5 or restart at a higher lr.
- Q2 (their NOTES 22:30, measured through deployed kernels): q2_ck64rr passes the fidelity bar but runs 0.916x b8; standalone GEMM sum
  vs b8 at M=1125: W4A4 .535, all-row-role (125 int8 rows) .801. So of the formats measured so far, only plain W4A4 meets the GEMM
  bar (<= .65); the w4q8 arm would be an accuracy experiment, not a speed-bar candidate.

## 23:45 PDT QAD first checkpoint (10M tokens, lr 2e-5) and an lr change
- Dev (exact emulation, 366 q) [M]: TV .0535 -> .0481 (-10%), KL .0161 -> .0129 (-20%), flips 23 -> 21 (within the +-3 noise),
  agree_sd .938 -> .906 (96 q, noise-level). Codes changed vs seed: 5.6e-5 of 1.37B (~76k weights); relative latent drift 2.6%.
- Eval suites, fast emulation [M, reporting only]: REAL flips 6.56% -> 7.02%, REAL TV .0621 -> .0589 (-5%), McNemar vs H2 bf16 runtime
  67/0 -> 74/2; CF-probe fgh .629 -> .667 (105 hobson-tracked pairs, SE ~.05).
- Reading: QAD at lr 2e-5 with grid-point latents changes almost no codes (the dead zone of s/2 needs ~85 consistent steps); the dev TV
  gain is real but slow. Arithmetic for lr 5e-5 with lambda 100: noise equilibrium 5e-4 = ~15% of a grid step (~0.03% of weights
  beyond s/2 at any time, ~0.4% of the weight-rounding noise power), consistent weights flip after ~35 steps.
- Resumed from the 10M checkpoint (state, optimizer, data pointers) with base lr 5e-5 (same token-based cosine), 06:42 UTC.
  The learning curve has an lr change at 10M tokens.
- 23:52 PDT: lr 5e-5 diverged [M]: train KL .026 -> .038 -> .086 and hidden MSE .056 -> .114 within 50 steps of the resume.
  Likely mechanism (arithmetic): with the factored second moment, weights whose v is underestimated take clipped steps of 3 x lr in
  noisy directions; at lr 5e-5 their L2-pull equilibrium spread is ~1.5e-3 ~ s/2, so they flip codes at random. Reverted: resumed
  again from the 10M checkpoint with lr 2e-5 (the setting that was stable), 06:51 UTC. ~6 h of box time remain for QAD + scoring.

## 00:18 PDT QAD 12.8M tokens (step 1260), 1,886 tok/s, stable at lr ~1.7e-5 (cosine); train KL ~.019, hidden MSE ~.056.
- q3b still not up. Next dev + curve point at 20M (~01:25 PDT).
- 00:48 PDT: 16.3M tokens (step 1600), drift 2.5% (flat since ~1M tokens = at the L2-pull equilibrium), train KL ~.024, hid ~.058.

## 01:32 PDT plain QAD stopped at 20M tokens; switching the remaining A10G time to Q1's recommended QAT base (w4q8)
- Plain W4A4 QAD learning curve so far [M]:
  | tokens | dev TV (exact emu) | dev KL | dev flips /366 | code changes | REAL TV (fast emu) | REAL flips | CF-probe fgh |
  |---|---|---|---|---|---|---|---|
  | 0 (B4.1a seed) | .0535 | .0161 | 23 | 0 | .0621 | 6.56% | .629 |
  | 10M | .0481 | .0129 | 21 | 5.6e-5 | .0589 | 7.02% | .667 |
  | 20M | .0514 | .0150 | 15 | 5.0e-5 | .0558 | 6.65% | .629 |
  (lr 2e-5 to 10M; a 50-step excursion at lr 5e-5 was discarded; then lr ~1.8e-5 -> 1.4e-5 by cosine.)
  REAL TV falls ~5% per 10M tokens; flips and CF-probe do not move beyond noise. Extrapolated to 50M: TV ~.048, about 10x above b8.
- Q1 best_formats.json (00:20): "none passes"; "Recommended base for QAT (Q3): w4q8 row role (deployed by Q2, 0.727x b8 e2e)".
  Q1 full kit (emulation): w4q8 REAL flips 3.69%, TV .0325, CF .798, CF-probe .676 (Q2 deployed: 3.88%, CF .835, CF-probe .762).
- Decision: stop the plain arm at 20M (best dev TV = t10M) and spend the remaining box time (shutdown 15:07 UTC) on: deployed-kernel
  scoring of the plain arm now (final.sh: H2 QRT bf16, GPTQ, B4.1a, t10M, t20M), then w4q8 scales (1.5M) + w4q8 QAD (20M tokens).
  Total QAD tokens across both arms: 40M (below the brief's 50-100M; the A10G does ~6.7M tokens/h).

## 02:05 PDT deployed-kernel scoring (H2 QRT on q3, FORMATS.md F1; all 3,227 questions) [M]
- bf16 runtime on this box (reference for McNemar): REAL flips 0.55% (H2's g1 run: 0.37%), TV .0032, CF 1.000, CF-probe .971.
- W4A4 with my GPTQ codes: REAL flips 9.23%, TV .0820, McNemar vs bf16 98 lost / 4 gained, CF .780, CF-probe .676, REAL-label .780,
  JB-hard .546 (9/12 vs hobson, p .66). H2's own GPTQ draw: 8.96% / .0775 / .826 / .657.
- Q2's q2gemm built on q3 (libq2gemm.so) for deployed scoring of the w4q8 arm later (QRT2C).

## 02:00 PDT (box 08:56 UTC) plain-arm deployed results [M, H2 QRT kernels, all 3,227 questions, vs hobson refs]
  | | GPTQ | + decision-level scales (B4.1a) | + QAD 10M | + QAD 20M |
  |---|---|---|---|---|
  | REAL flips (bar <= .70%) | 9.23% | 6.56% | 6.74% | 6.09% |
  | REAL TV (b8 .0048) | .0820 | .0603 | .0586 | .0588 |
  | McNemar vs bf16 runtime (lost/gained) | 98/4 | 67/2 | 72/5 | 65/5 |
  | CF fgh (bar .99) | .780 | .844 | .817 | .789 |
  | CF-probe fgh (bar .95) | .676 | .695 | .638 | .648 |
  | REAL-label (bar .78) | .780 | .790 | .785 | .7825 |
  | JB-hard (McNemar vs hobson p) | .546 (.66) | .508 (.84) | .462 (.12) | .500 (.66) |
- Paired agree-with-hobson on REAL: B4.1a vs GPTQ 64 gained / 35 lost (p .0046); QAD 10M vs B4.1a 45/47 (p .92).
- chain6 started 08:56 UTC: w4q8 scales (3M tokens) -> w4q8 QAD (20M tokens).
- QAD 20M vs B4.1a paired: 43 gained / 38 lost on REAL (p .66); REAL+LONG 54/55 (p 1.0). QAD adds nothing measurable over the scales.

## 02:15 PDT bug found and fixed in the w4q8 training path (training only; no reported number used it)
- The first w4q8 scale run showed train KL 1.2-1.8 and hidden MSE 1.01 from step 10, while its step-0 dev (eager) was sane
  (w4q8 GPTQ + GPTQ8 question rows, exact emulation: 9/366 flips = 2.46%, TV .0258, KL .0044).
- Cause [M, q3dbg8.py]: the torch.compile'd activation quantizer took (qmax, clip) as arguments of one dynamic graph; after compiling
  for int4 it produced wrong int8 rows (hidden MSE 1.03 vs .0065 eager). Fixed by compiling one closure per (qmax, clip).
  After the fix: compiled kl .0165 / hid .0077 vs eager .0158 / .0065 on the same request (the compiled division is not correctly
  rounded, so a few codes differ at ties; training only). The plain-W4A4 arm only ever used qmax 7 and was unaffected (its fast-emulation
  curve point equals the deployed kernels' REAL flips, 6.56% both).
- chain6 relaunched 09:13 UTC.
- 02:25 PDT: int8 rows now always use the eager quantizer/epilogue in training too (q3dbg8: FAST and eager both give hid .004-.007 on 4 requests incl. a short train_v5 row); chain6 relaunched.

## 02:45 PDT w4q8 arm: scales (B4.1a) result, QAD started
- w4q8 = state rows W4A4 (GPTQ4 codes, trainable), question rows W8A8 (GPTQ8, fixed). Dev, exact emulation [M]:
  | tokens | dev flips /366 | dev TV | dev KL | agree_sd |
  |---|---|---|---|---|
  | 0 (GPTQ) | 9 (2.46%) | .0258 | .0044 | .948 |
  | 1M scales | 6 | .0274 | .0051 | .969 |
  | 2M scales | 9 | .0252 | .0040 | .958 |
  Scales give -2.4% TV / -9% KL here (plain W4A4: -25% / -36%): with question rows at W8A8 the remaining error is state-row
  activation rounding, which per-channel weight scales cannot reach. Stopped at 2M (time).
- Speed: w4q8 training 1,400 tok/s in scale mode (int8 rows run the eager quantizer). Q2's runtime (QRT2C) needs ~13 GB at setup,
  so deployed w4q8 scoring waits for the end of training.
- w4q8 QAD (chain7) started ~09:42 UTC: seed = w4q8 scales t2M, grid-point latents, lr 2e-5, 12M tokens, dev + curve every 4M.

## 03:05 PDT w4q8 QAD running (1,640 tok/s, 19.6 GB); w4q8 seed on the suites
- w4q8 + scales (2M) seed, fast emulation, CF-probe + REAL-agree [M]: REAL flips 3.51%, REAL TV .0296, agree_sd .934, McNemar vs bf16
  runtime 36 lost / 4 gained, CF-probe fgh .686. (Q1 emulation of w4q8: 3.69%, TV .0325; Q2 deployed: 3.88%.)
- Expected end of w4q8 QAD (12M tokens + curves) ~12:10 UTC; then deployed w4q8 scoring via Q2's QRT2C and latency.
- 03:30 PDT: w4q8 QAD 3.8M tokens (step 380), train KL ~.005-.009, hidden MSE ~.004-.005, 1,668 tok/s. First dev point at 4M.
- 03:45 PDT w4q8 QAD 4M [M]: dev TV .0252 -> .0230, KL .0040 -> .0035, flips 9 -> 9, codes changed 3.9e-5, drift 2.4%.
  Fast-emulation suites: REAL flips 3.51% -> 3.69%, REAL TV .0296 -> .0282, McNemar vs bf16 36/4 -> 34/0, CF-probe fgh .686 -> .714.
- 04:16 PDT: w4q8 QAD 7.6M tokens; ETA end ~12:20 UTC, then final w4q8 deployed scoring (Q2 QRT2C) + latency.
- 04:55 PDT: w4q8 QAD 10.7M tokens; final_w4q8.sh queued (waits for QAD end): exports, Q2 QRT2C scoring of w4q8 QAD-best / GPTQ / scales, then latency.

## 05:35 PDT w4q8 deployed results (Q2 QRT2C on q3, my codes; all 3,227 questions) [M]
  | | w4q8 GPTQ | w4q8 + scales 2M + QAD 8M (dev best) |
  |---|---|---|
  | REAL flips | 3.05% | 3.69% |
  | REAL TV | .0312 | .0289 |
  | McNemar vs bf16 runtime (lost/gained) | 32/5 | 36/2 |
  | CF fgh | .826 | .872 |
  | CF-probe fgh | .667 | .714 |
  | JB-hard (McNemar vs hobson p) | .523 (1.0) | .500 (.65) |
  | REAL-label | .785 | .780 |
- Paired agree-with-hobson, QAD best vs GPTQ: 17 gained / 24 lost on REAL (p .35); REAL+LONG 23/29 (p .49). TV -7%.
- w4q8 QAD final dev (12M): TV .0238, flips 8; best by dev TV = t8M (.0229). Fast-emulation suites at 12M: REAL flips 3.32%, TV .0281,
  CF-probe .714.

## 05:45 PDT latency, projections, report
- Latency on q3 (Q2's q2bench, b8 / W4A4 / w4q8c side stream, 20 reps) [M] in res_e2e_q3.jsonl; T=1000/1q: b8 36.40 ms, W4A4 23.46
  (.645x), w4q8 26.18 (.719x); GEMM kernel sums b8 25.82 / W4A4 12.99 (.503x) / w4q8 20.17 (.781x). Projections (H2 proj.py model) in
  proj_q3.json: W4A4 T=1000/1q 3090 13.5 / 4090 8.9 / 5090 5.7 ms (5090 = NVFP4 assumption); b8 19.9 / 11.6 / 7.9.
- w4q8 scales-only deployed [M]: 3.60% flips, TV .0300, CF .908, CF-probe .762; vs w4q8 GPTQ 23 gained / 29 lost (p .49).
- All small results copied to ~/decider2/q3 (preds/, box_logs/, scores.json, scores_table.md, curve_scores.json). Weights and
  checkpoints stay on q3 (~/work/q3/ck_*). DRAFT_REPORT.md written. q3 left running until its timer (15:07 UTC); no GPU jobs left.

## 05:50 PDT done
- DRAFT_REPORT.md final (~1,330 words excluding table rules). Returned to the coordinator.

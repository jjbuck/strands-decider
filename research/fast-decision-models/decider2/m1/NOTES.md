# M1 notes: all-attention hobson

## 2026-10-06 22:07 PDT
- Read BRIEF11, IDEAS_EXPLORED, FAST_DECISION_MODEL. Box m1 (g5.4xlarge, i-REDACTED) is pending; no m1.ready yet.
- Plan while waiting: read hobson's GDN/attention code (j2 surgery, j3 dt trainer, q3 lean student, d1 lean2 runtime), write ARCH.md, prepare convert/train/eval/latency scripts locally (no torch run on laptop).

## 22:45 PDT
- Box m1 up (g5.4xlarge, A10G 23 GB, torch 2.14.1+cu130, triton 3.8, fla 0.5.2). Box TTL: poweroff 08:07 PDT (boot 22:07).
- Code in m1/code: m1lib.py (student/teacher, converted mixer), m1conv.py (untrained), m1train.py (transfer + e2e), m1eval.py (evalkit, holdout), m1lat.py (fused runtime latency).
- check (measured): in-process teacher vs evalkit hobson refs argmax 56/56; student with nothing converted == teacher (max |dp| 0).
- Bias trick (measured): gate biases carried in 6 extra q/k dims (head dim 136) through plain SDPA match an fp32 explicit-bias softmax to rel 1.85e-3 (plain bf16 SDPA vs fp32: 1.99e-3), with |G| ~ 5000.
- Untrained local fidelity (measured, teacher-forced, 16 train-pool requests, relMSE of the layer's residual contribution vs GDN's, best tau per layer):
  plain 3.25, beta 3.03, decay 4.04, full 3.93, full_norope 3.91, full_noconv 70.3 (dropping the short conv destroys q/k/v), plain_norope 3.21.
  relMSE > 1 = worse than outputting zero. Per-row cosine is positive (0.13-0.62 early layers; 0.84-0.93 at layers 17, 18, 20, 21) and higher with
  the decay bias on 15/18 layers; the magnitude is too large (softmax outputs are convex combinations of raw v; the delta rule outputs residual values,
  often near the gated norm's eps). => the conversion is a warm start, not function-preserving. Chose 'full' (conv, RoPE, beta + decay biases) for training on the cosine criterion.
- Untrained e2e, all 18 converted at once, dev set (100 train-split dev requests, 366 questions): plain: agree .369, agree_sd .263 (n_sd 95), TV .351; hidden relMSE on answer/option rows L5 12.0, L11 5.9, L17 2.6, L23 0.75.

## 23:20 PDT
- Untrained e2e, dev set (366 questions, n_sd 95), all 18 converted (measured): plain agree .369 / agree_sd .263; beta .363/.189; decay .213/.158;
  full(136) .372/.284. Hidden relMSE (answer+option rows) for full: L5 7.7, L11 4.5. => untrained M1 is not usable; it is a warm start.
- One layer converted at a time (full, 40 dev requests, 158 q): layers 0-10 each break decisions (agree .35-.73), 12-14 .84-.98, 16-21 >= .987 (TV <= .019).
  Bottom-up cumulative (3/6/9/12/15 layers): agree .34/.39/.34/.40/.30.
- Latency, untrained-conversion weights (shapes only), A10G bf16 CUDA graph, 20 reps (measured). hobson anchors match J3's lat_dt.json within 0.5 ms
  (57.12 vs 57.09 at T=1000; 201.31 vs 200.84 at T=4000).
  One sequence, Q1: T=64/256/1000/4000: hobS 15.56/25.36/57.10/201.34; M1 head136 15.66/26.29/61.39/236.52; M1 no biases (128) 15.00/25.16/56.38/210.58;
  M1 in-head slots (122+6) 15.44/26.12/59.31/221.54.
- Profile T=4000 (measured): GEMMs 170.0 ms in both. hobson: GDN scan 13.1, attention 8.2. M1: flash attention 30.9 (1.26 ms per converted layer vs 1.37 per hobson
  attention layer), and torch.cumsum over dim 0 of the decay = 10.8 ms (a slow outer-dim scan). Fixed: decay scan now head-major (innermost-dim cumsum). Re-timing.
- Design change: gate biases now live in dims 122..127 of each 128-dim head (122 content dims) instead of 136-wide heads; untrained local fidelity identical (3.927 vs 3.926).

## 23:45 PDT
- Re-timed with the head-major decay scan (measured, A10G, 20 reps): M1 (in-head slots) one sequence T=64/256/1000/4000: 15.08/25.26/56.54/210.99 ms
  (hobson 15.56/25.36/57.10/201.34). B layout Q1: 24.15/29.82/67.84/232.71 (hobson 25.43/30.49/67.40/216.55). Q4: 48.34/54.27/93.87/265.38 (hobson 52.69/57.87/95.20/246.08).
  => the gate biases cost nothing measurable now (no-bias M1: 56.38 / 210.58 at 1000 / 4000). ARCH.md v1 written (final shapes, head dim 128).
- Smoke tests OK: transfer 20 steps mean local relMSE 3.66 -> 2.58; e2e 9 steps dev agree .16 -> .59 (44 q). e2e speed about 1.5k tok/s with full checkpointing;
  added: no activation checkpointing below 2,500 rows.
- 23:40 launched run (a): transfer 0.33 h (lr 1e-4 big / 1e-3 small) -> e2e 4.5 h all 18 at once (lr 3e-5 mixers, 3e-4 small, 2e-4 LoRA r32 alpha 64, w_hid 1, p_v5 .25, p_cf .10).
  Concurrently: untrained 'full' on all 3,227 evalkit questions.

## 00:00 PDT
- Transfer (teacher-forced local fit, all 18 mixers in one teacher pass; measured on train-pool data): mean local relMSE 3.93 (init) -> 0.196 (step 40) -> 0.110 (step 70, 0.67M tokens).
  Hardest layers at step 70: L8 0.27, L10 0.22, L13 0.18, L1 0.17, L12 0.17; easiest L17-L22 0.013-0.024. Speed 1.25k tok/s while sharing the GPU with the untrained eval.
- Untrained evalkit eval runs concurrently (1,222 of 3,227 questions so far, about 27 q/min under contention); a watcher kills it when the transfer ends,
  to keep GPU memory for e2e (resumable JSONL; will resume later).
- Scorer: m1/code/score_m1.py (J3's + relaxed bar + McNemar on CF / CF-probe pairs), score_holdout.py, proj_m1.py (card projections from kernel splits).

## 00:20 PDT
- **Transfer result (measured, dev set 366 q / n_sd 95, train-split dev tasks):** after 19.9 min / 1.64M tokens of teacher-forced local fitting,
  all 18 converted at once: agree .918, agree_sd .926, TV .068, KL .023; hidden relMSE on answer/option rows L5 .031, L11 .072, L17 .023, L23 .0007
  (untrained: agree .372, agree_sd .284, L5 7.7). Final mean local relMSE 0.07 or so (L10 .13, L12/L13 .11 the worst).
- e2e (a) first launch OOMed at its first steps (21.8 GB; the eager fp32 intermediates of a 2,500-row sequence without checkpointing). Restarted 00:18 with
  checkpointing above 1,200 rows, resuming from its step-0 checkpoint (m000 = the transfer state; DEV at m000 = the numbers above).
- Untrained evalkit eval stopped at 2,119 / 3,227 questions (resumable).

## 01:05 PDT
- e2e (a) attempt 1 (lr 3e-5 mixers / 3e-4 small / 2e-4 LoRA, warm 20): training-batch loss rose from the start (hid .047 -> .12, agree_real .84 -> .70 over 40 updates).
  Probe (no update) on 16 train-pool requests: KL .024, hid .033, agree .877 -> the start is as close to hobson as the dev set; the rise is the optimizer.
- attempt 2 (lr 1e-5 / 1e-4 / 5e-5, warm 100): stable while the warm-up was below about half of peak (hid .032-.035), then rose again (hid .045 at lrf .6, .058 at lrf .8; kl_real .03 -> .16).
- Both stopped (archived ck_a_try1, ck_a_try2). Now running single-factor diagnostics from the transfer init, real data only, 60-70 updates each, warm 10:
  d1 mixers only (1e-5), d2 LoRA only (5e-5), d3 small params only (1e-4). If all three are stable, the CE terms (v5 / cf_aug) drive the drift.

## 00:20 PDT (real clock. The section stamps 23:45, 00:00, 00:20 and 01:05 above were my own estimates and ran ahead of the real clock by up to about 1 h 40 min;
   the order of events is right, and the box logs (log.txt files) carry the real UTC times. Transfer finished 06:28 UTC = 23:28 PDT.)
- Diagnostics (measured; 20 dev requests / 85 q; from the transfer init; real data only; about 52 updates, 0.63-0.67M tokens each; start: KL .0165, TV .057, agree .953):
  d1 mixers full-rank only (lr 1e-5): KL .0151, TV .048, agree .941; hidden relMSE worse (L11 .068 -> .076, L17 .021 -> .025, L23 .0006 -> .0011).
  d2 LoRA r32 only (lr 5e-5): KL .0132, TV .042, agree .965; hidden better at every layer (L11 .063).
  d3 small mixer params only (q/k gains = temperatures, decay A_log / dt_bias, conv, gn_w; lr 1e-4): KL .0081, TV .038, agree .965; hidden better (L11 .064).
  => the drift in the e2e attempts came from the full-rank mixer updates (and possibly the CE terms); LoRA and the small parameters each help.
- **PERMISSION DENIED (reported, not worked around):** (1) resetting the box shutdown timer (`sudo shutdown -c; sudo shutdown -h +480`) was denied by the
  auto-mode classifier as "Modify Shared Resources". (2) The retry that only launched the final e2e (a) run (run_a3.sh: mixers 2e-6, small 1e-4, LoRA 1e-4,
  accum 8, p_v5 .20, p_cf .05, 2.75 h) without the timer change was denied for the same reason. So the planned 2.75-hour distillation (a) and the staged run (b)
  did not run. Box TTL unchanged (poweroff 08:07 PDT).
- Evaluation of existing checkpoints was allowed: 00:24 launched the full evalkit (3,227 q) + holdout probe on ck_transfer/transfer.pt.

## 00:50 PDT
- Transfer learning curve (measured, training batches): mean local relMSE 2.97 (0.09M tok) -> 0.32 (0.28M) -> 0.14 (0.48M) -> 0.090 (0.84M) -> 0.067 (1.58M);
  still falling when the cosine schedule ended; hardest layers at the end L8 .165, L10 .140, L13 .123. Saved results/transfer_curve.json.
- Evals running (2 processes sharing the GPU): transfer.pt 2,214 / 3,227 questions, d3small 1,868 / 3,227.

## 01:35 PDT
- Full evalkit, 3,227 questions (measured; paired McNemar vs hobson-v19 refs):
  transfer.pt (1.64M tokens, teacher-forced only): JB-all .645 vs .723 (p .001), JB-hard .408 vs .523 (p .004), REAL-label .728, REAL agree_sd .812, LONG agree_sd .824,
    CF pair .219 vs .268 (p .002), CF-probe .244 vs .328 (p .002), CF retention .734, CF-probe retention .533, REAL-label Brier .388 vs .347.
  d3small (transfer + 52 updates / 0.63M tokens e2e on the small mixer params): JB-all .645 (p .001), JB-hard .415 (p .009), REAL-label .770, REAL agree_sd .861,
    LONG agree_sd .903, CF .192 (p < .001), CF-probe .263 (p .02), REAL-label Brier .374. Relaxed bar: REAL-label and agree_sd pass; JB-all, CF, CF-probe fail.
- Holdout (300 per task) transfer.pt: RuleTaker d3 .833 vs .867 (p .076), d5 .687 vs .750 (p .003), natlang .730 vs .800 (p 2e-5).
- Checkpoint arithmetic (no training): merged_d3C_d2L.pt = d3's mixer params + d2's LoRA (disjoint sets, same init). Evaluating; untrained eval resumed.

## 01:30 PDT
- Untrained 'full' (in-head slots) on all 3,227 evalkit questions (measured): JB-all .290, JB-hard .308, REAL-label .440, REAL agree/agree_sd .352/.361,
  LONG .335/.333, CF pair .012, CF-probe .053, CF retention .000, REAL-label Brier .585.
- merged_d3C_d2L (checkpoint arithmetic, no training): JB-all .662 (p .016), JB-hard .431 (p .036), REAL-label .730, REAL agree_sd .801, LONG agree_sd .897,
  CF .121 (p < .001), CF-probe .303 (p .41), CF retention .413. Not additive: the d2 LoRA was fit with the transfer mixers, not d3's. Holdout RuleTaker d3 .827, d5 .693.
- Fused runtime vs eager M1 on the d3 checkpoint (verified): argmax 56/56, median |dp| .0011, max .0079 (16 items: REAL-agree multi-question, JB-hard, CF-probe; B and S layouts).
- Kernel split (measured, profiler, eager fused path, d3 weights) and projections (arithmetic, results/proj_m1.json): M1 / hobson, one question,
  T = 64/256/1000/4000: A10G .97/1.00/.99/1.05; 3090 .98/1.02/1.02/1.13; 4090 .94/.97/.95/1.03; 5090 .97/1.00/.98/1.08.
  Assumptions: GEMMs at the fp16-accumulation rate (J14 roofline split), FlashAttention at the fp32-accumulation rate (half that on GeForce), GDN / conv / other with bandwidth.
- No more training will run (permission denial above). Writing DRAFT_REPORT.md. Box left idle (TTL poweroff 08:07 PDT); checkpoints stay on the box.

## 01:45 PDT
- DRAFT_REPORT.md written (prose about 1,150 words plus tables). Final deliverables on the laptop: results/ (scores.json, lat_*.json, prof_final.json, proj_m1.json,
  transfer_curve.json, diag curves, latcheck), preds/ (evalkit JSONL + preds JSON for untrained / transfer / d3small / merged; holdout rows). Done.

## 06:15 PDT (Oct 7)
- Coordinator message: the engineer approved the M1 distillation run; the coordinator reset box m1's timer (verified read-only: poweroff 14:07 PDT). I did not touch the timer.
- 06:12 launched run_a4.sh (allowed this time): e2e from ck_transfer/transfer.pt, full-rank mixer weights frozen (lr 0), LoRA r32 alpha 64 lr 5e-5,
  small mixer params lr 1e-4, warm 30, 8 micro-batches per update, cosine to 0.1x over 5.8 h of training time, p_v5 .25 (CE + KL), p_cf .10 (CE + 0.3 KL),
  real states KL, w_hid 1 (layers 5/11/17/23, answer + option rows), dev (100 requests / 366 q) + train-pool probe every 30 min, bf16 snapshots.
  Trainable 28.2M (27.7M LoRA, 0.5M small). After training: select_best.py (lowest dev TV) -> run_final_eval.sh (all 3,227 evalkit q + holdout 300/task).
  Expected: training ends about 12:20 PDT, evaluation about 13:05 PDT.

## 06:50 PDT
- run a4, first 30 min (measured): training-batch hid .034 -> peak .043 (update 50) -> .036-.041; cf_aug accuracy .50 -> .8-1.0 (CE working).
- DEV at 30 min / 2.85M tokens: agree .918 -> .964, TV .068 -> .058, KL .0234 -> .0220, agree_sd .926 -> .916 (n_sd 95, noise), hid L5 .031 -> .029,
  L11 .072 -> .072, L17 .023 -> .027, L23 .0007 -> .0010. Train-pool probe: KL .024 -> .018, agree .877 -> .965. Healthy; continuing. 1.56k tok/s.
- No concurrent evals during training (an OOM would end training early and the driver would start the final eval on an early checkpoint).

## 07:22 PDT
- run a4 DEV at 60 min / 5.72M tokens (measured): agree .959, TV .0556, KL .0205, agree_sd .926. Probe: KL .0188, agree .965, hid .0283 (step 0: .024 / .877 / .0326).

## 07:48 PDT
- run a4 DEV at 90 min / 8.57M tokens: agree .945, TV .0517, KL .0157, agree_sd .863 (6 of 95 state-dependent questions changed vs the 60-min point; about 2 SE).
  Probe: KL .0131, agree .947, hid .0252. Distributions keep approaching hobson's while a few close calls cross the boundary.

## 08:55 PDT
- run a4 DEV (measured, 366 q / n_sd 95): 120 min / 11.4M tok: agree .959, TV .0512, KL .0159, agree_sd .905; 150 min / 14.3M tok: agree .970, TV .0458, KL .0144,
  agree_sd .926. Still improving. Training process alive (pid 93228); the laptop ssh that launched it ended, the job is detached. Copied curve to results/a4_curve.json.

## 09:23 PDT
- run a4 DEV at 180 min / about 17M tokens: TV .0485, KL .0129, agree_sd .916. KL keeps falling; TV flat since 150 min (.0458).

## 10:00 PDT
- run a4 DEV at 210 min / about 20M tokens: TV .0453 (lowest so far), KL .0140, agree_sd .926. Training ends about 12:25 PDT.

## 10:31 PDT
- run a4 DEV at 240 min / about 23M tokens: TV .0379, KL .0106 (largest single drop; LR now in the cosine tail), agree_sd .926.
- Plan after training: the driver evaluates the dev-best checkpoint (full kit + holdout); I add a full-kit eval of the 150-min snapshot concurrently for an evalkit learning curve.

## 11:02 PDT
- run a4 DEV at 270 min / about 26M tokens: TV .0379, KL .0103, agree_sd .916. Flattening.

## 11:32 PDT
- run a4 DEV at 301 min / about 29M tokens: TV .0363, KL .0100, agree_sd .947.

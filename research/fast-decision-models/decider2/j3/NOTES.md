# J3 notes: decision-native architecture

## 2026-10-06 09:01 PDT
- Read BRIEF8, FAST_DECISION_MODEL.md, F7_REPORT, evalkit README. Box j3 not yet ready (no boxes/j3.txt). Doing design work on laptop.

## 09:20
- Read Brooker posts (encoder: e1b ties v19; hobson build). Read h7 (schema-first) code/report, h3lib/h7lib differentiable forward, lean2/TTL runtimes, kitrun/plib.
- Found prior untrained evidence in ~/decider2/tokens/stale_sweep.jsonl (35 long JevBench items, mean 1320 state tok): state rows STOP after layer k,
  later full-attention layers read K/V projected from the stale residual h_k, GDN layers see question rows only. Agreement with hobson (training-free):
  k3 .514, k7 .543, k11 .943, k15 .971 (tv .277/.240/.058/.006). => deep processing of STATE rows is largely redundant for decisions at k>=11,
  untrained. Not reported in FAST_DECISION_MODEL. This is the seed for the design.
- Box j3: not launched yet (retry_j.log has no j3 line). Designing.

## 09:35
- DESIGN.md written (C1 DT depth-split decision transformer = build; C2 slot-deep compiled schema, C3 exact value channel, C4 CF-contrastive, C5 static graph).
  Cost model code/costmodel.py -> costmodel.json: FLOPs/state token hobson 2.745 G; DT-A4 .500 (.18x), DT-A8 .949 (.35x), DT-A12 1.398 (.51x);
  DT-G (GDN memory too) .27/.42/.56x. A10G 1q@1000: hobson 60 (model; 52.7-57 measured), DT-A8 25.8, DT-A4 17.1 [arithmetic].
- Code (j3/code): dtlib.py (DT on h3lib/h7lib forward; bridges A/G; memory adapters), dt_eval.py (check / training-free sweep / eval),
  train_dt.py (F7 + CF aug + dense question-row distill; teacher in-process), dt_lat.py (fused runtime, hob1/hobB/dtA{Ls}), score_dt.py (laptop).
- score_dt.py verified on hobson refs (PASS, JB-all .723, REAL-label .785, CF flip .268, CF-probe flip .328).
- Box j3 launched 09:2x (i-REDACTED g5.2xlarge), waiting for ready.

## 09:50 box j3 ready 09:32; identity + training-free frontier (MEASURED)
- check: DT with Ls=24 == teacher path exactly (56/56 argmax, max|dp| 0); teacher vs evalkit hobson refs 56/56.
- Training-free sweep (hobson weights, hobson head; subset: REAL-agree 120 items, LONG 40, JB-hard 130, CF 120 pairs, CF-probe 120 pairs), tf/*.json,
  tf_scores.json. State rows stop after Ls layers; deep layers run on question rows reading the layer-Ls memory:
  | cfg | REAL sd | LONG sd | CF fgh | CFp fgh | REAL-label | JB-hard (hob .523) |
  | 24A (=hobson in this runtime) | .988 | 1.000 | 1.000 | .973 | .822 | .531 |
  | 16A | 1.000 | 1.000 | 1.000 | .973 | .811 | .546 |
  | 12G | 1.000 | 1.000 | 1.000 | .946 | .822 | .538 |
  | 12A | .952 | .963 | 1.000 | .865 | .789 | .477 (p .18) |
  | 8G  | .880 | .852 | .686 | .568 | .778 | .446 |
  | 8A  | .783 | .741 | .743 | .541 | .744 | .462 |
  | 4G  | .325 | .296 | .114 | .027 | .667 | .485 |
  | 4A  | .229 | .222 | 0    | 0    | .678 | .485 |
  (REAL-label subset n=90, hobson .811 on it.)
- => UNTRAINED, 16 of 24 layers of state-row processing are removable with no measurable change (0.68x state FLOPs... 16A = 0.69x),
  and at 12 layers (0.51x) agree_sd is already ~.95. Cliff between 12 and 8; 4 collapses.
- Coordinator addition (09:45): option-order invariance as design goal + metric (order-flip delta under 3 rotations; ECE + Brier). dt_order.py written.
- Plan: main run = elastic {8 (p .65), 12 (p .35)} bridge A; second run = option-set DT (invariant by construction) if time.

## 09:58
- Main run started 09:53: train_dt.py --Ls 8,8,12 (elastic, p(8)=2/3) --bridge A --updates 700 --accum 16 --max_hours 2.5, ck ~/work/j3/ck_a812.
  ~14 s/update (16 seqs), ~3.2k tok/s. Init: KL real .028, hid8 .058, hid12 .021.
- dtset.py (option-set DT): per question a tree (stem off the state; each option a branch off the stem at the SAME position, no numbers;
  tail/<answer> off the stem, attends to all options as a set). test_dtset.py (MEASURED, untrained hobson weights):
  invariance under rotation max|dp| .0031 (median .00017, bf16 noise); argmax agreement with hobson refs 27/30 at Ls=24, 23/30 at Ls=8;
  gradients flow to LoRA in all layers and to memory adapters.
- dt_order.py: order-flip metric (rotations 1,2,3 mod K distinct non-identity for noul/choice; reversal for score), mean |dp| over labels, TV.

## 10:05
- dt_order smoke (hobson, 8 REAL items): unchanged .875 under noul swap (H4: .884) - consistent.
- Fused runtime check (dt_lat.py check, hobson weights): DT-A8 runtime vs dtlib preds 40/40 argmax, max|dp| .0067; hobson branch layout
  (Ls=24) 40/40, max|dp| .011. dt_lat time smoke OK (timings shared with training, discarded).
- pipeline.sh launched on box (setsid nohup): waits for main run -> evals a812 L8/L12 (all suites) -> order (hobson, L8, L12) -> latcheck ->
  latency (exclusive) -> DT-set training (--layout set, Ls 8,8,12, 450 upd, 1.9 h) -> set evals + order -> training-free full evals 12A, 16A.
- main run: upd 40 at 528 s (13.2 s/upd) -> ~680 updates in 2.5 h; ETA end ~12:23.
- DESIGN.md: added C6 (DT-set) and the measured training-free frontier.

## 10:25
- Fused DT-set runtime validated against dtset (untrained, Ls 8): 32/32 argmax, max|dp| .0042. Fused DT-A8 vs dtlib 40/40 (.0067).
- Learning curve (subset, curve/): DT-A8 at s100 (1.6k seqs): REAL sd .783->.928, LONG sd .741->.852, CF flip .325->.717 (hob .292),
  CF-probe flip .283->.842 (hob .308), REAL-label .744->.767 (hob .811 on subset), JB-hard .462->.438 (hob .523, p .11; losses spread over
  adequacy/judge_hard/multi_hop/trap), JB-all ECE .228 (hob .135): overconfident on JevBench. Watching s300/s500 (curve.sh on box).
- pipeline.sh restarted (set-run args now read from set_args.txt at launch, so they can be changed after s300/s500).

## 10:46
- dt_lat.py: added bridge G memory pass (per deep GDN layer: folded [qkv|b|a] GEMM over the state rows + conv + GDN scan -> final state,
  tail) and DT-set q_pass (two-level varlen GDN). Validated: fused 12G vs dtlib tf_12G 40/40 (max|dp| .010); set (above) 32/32.
- dt_calib.py (per-kind temperature, Brooker's recipe, fitted on train_v5.holdout gold; never eval items), proj_lat.py (A10G measured ->
  3090/4090/5090 via costmodel ratios).
- pipeline.sh restarted with: ... + training-free FULL evals 12A, 16A, 12G + latency for dtG12 and dtA16 (function-preserving candidates).
- Training metrics windows (upd 1-60 / 180-240): kl_real .046 -> .037, hid8 .054 -> .042, hid12 .025 -> .020, acc_cf .79 -> .98.

## 11:05 learning curve DT-A8 s300 (4.8k seqs; subset, MEASURED)
- s100 -> s300: REAL-label .767 -> .822 (hob .811 same items), REAL sd .928 -> .892, LONG sd .852 -> .926, JB-hard .438 -> .469 (hob .523, p .35),
  CF flip .717 -> .725 (hob .292), CF-probe flip .842 -> .950 (hob .308), SHUF both_right .765, JB ECE .121 (hob .135), REAL-label ECE .103 (.115).
- => passes the new-architecture bar on the subset at 0.35x state FLOPs. Full suites after the run ends (~12:22).
- (box.sh get hung once via SSM rsync; using `run "cat file" > local` as fallback.)

## 11:08
- s300 at Ls 12 (subset): REAL-label .833, REAL sd .952, LONG sd .926, CF flip .742, CF-probe flip .983, JB-hard .462 (p .096), JB-long agree .870.
  JB-hard is consistently 5-6 points under hobson for both splits (n.s. on n=130); watch on full JB-all.
- pipeline v3 (restarted 11:08): main evals -> order (hob, L8, L12) -> calib fit -> latcheck -> latency (hob1, hobB, dtA/dtG 4/8/12/16, Q1/Q4)
  -> DT-set run (1.6 h) -> set evals/order/latency -> tf full 16A -> bold 4A run (1.0 h) + eval -> set L12 order -> tf full 12G, 12A.

## 11:49 curve s500 (subset)
- L8: s100/s300/s500 REAL sd .928/.892/.843, LONG sd .852/.926/.815, REAL-label .767/.822/.789 (hob .811), JB-hard .438/.469/.477,
  CF-probe flip .842/.950/.992, CF flip .717/.725/.733. L12 s300/s500: REAL sd .952/.916, LONG .926/.926, REAL-label .833/.800, JB-hard .462/.462.
- Fidelity to hobson (agree_sd) drifts DOWN after s300 while CF/CF-probe accuracy climbs: the CE terms (CF templates + v5 gold) pull away from
  hobson. Absolute REAL-label moves within noise (n=90, +-3 items). Final checkpoint (pre-registered) is what gets the full evals.

## 12:31 MAIN RESULT: DT-A8 trained (s700 = 11.2k seqs, 26.4M tok, 2.45 h), FULL evalkit (MEASURED, preds/a812_L8.json, scores.json)
- 0.35x state FLOPs. JB-all .684 vs .723 (McNemar p .20, model-only/hob-only see scores.json); JB-hard .477 vs .523 (p .41);
  REAL-label .782 vs .785; CF flip .719 vs .268, acc .812; CF-probe flip .988 vs .328 (in-template caveat), acc .994; SHUF both_right .703 (.245);
  REAL agree_sd .832, LONG agree_sd .879 (fidelity, not the new-arch bar); JB-all Brier .413 (hob .348), ECE .059 (.048);
  REAL-label Brier .347 (= hob), ECE .088 (.116). => PASSES the brief's new-architecture bar.
- CF by kind: human_insert 1.0, amount_insert .967, wrapup 1.0 (trained kinds); identity 0 and procedure 0 (untrained kinds, = hobson).
- Train log: kl_real (train-split states) .053 -> .016 over 700 updates, still falling (not plateaued).

## 12:37 DT-A12 (same elastic model at Ls=12; 1/3 of training samples), FULL evalkit (MEASURED)
- 0.51x state FLOPs. JB-all .693 (p .14; 5 model-only / 12 hob-only), JB-hard .477 (p .21), REAL-label .770 (hob .785; 7/13, n.s.) -> misses the
  .78 line by 4 items; CF flip .722, CF-probe flip .997; REAL sd .893, LONG sd .909; JB-all Brier .370 (hob .348), ECE .075; REAL-label ECE .087.
- DT-A8: JB-all 15 model-only / 24 hob-only; JB-hard 15/21; REAL-label 13/14.

## 12:47 option-order sensitivity (MEASURED, dt_order.py, REAL-agree + JB-all, 1286 noul/choice questions, 1969 distinct non-identity rotations)
| model | unchanged | mean abs dp | REAL-agree | JB-all | score reversal (n 28) |
| hobson | .919 | .037 | .928 | .891 | 1.000 |
| DT-A8 | .902 | .035 | .917 | .855 | .893 |
| DT-A12 | .928 | .035 | .937 | .899 | .929 |
- DT in hobson's layout inherits hobson's order sensitivity (~8-10% of decisions flip under rotation). DT-set (next run) is invariant by construction.

## 12:57 LATENCY (MEASURED, A10G exclusive, bf16 TTL fused kernels, CUDA graph, exact T, fresh banking states, 20 timed reps, p95 within .1 ms)
lat_dt.json, lat_table.json (proj_lat.py). Q1 = cc_asked_for_human (93 q tokens). hob1 = hobson single sequence (d1 layout);
hobB = hobson state pass + packed question pass (same two-pass harness as DT).
| Q1 ms | 64 | 128 | 256 | 400 | 1000 | 4000 |
| hob1 | 15.3 | 15.6 | 22.9 | 28.2 | 57.1 | 200.8 |
| hobB | 25.5 | 25.9 | 30.6 | 46.8 | 67.4 | 216.0 |
| DT-A4 | 16.4 | 16.4 | 17.3 | 20.2 | 24.2 | 52.5 |
| DT-A8 | 18.2 | 18.3 | 20.0 | 25.5 | 33.0 | 85.3 |
| DT-A12 | 20.0 | 20.2 | 22.7 | 30.9 | 41.5 | 117.9 |
| DT-A16 | 21.9 | 22.1 | 25.3 | 36.2 | 50.2 | 150.6 |
| DT-G12 | 21.5 | 21.8 | 24.8 | 34.2 | 47.6 | 140.9 |
Q4 (4 questions, 364 q tokens): hobB 52.8/57.9/74.2/95.1/245.5 at 64/256/400/1000/4000; DT-A8 45.2/47.1/52.7/60.4/114.5; DT-A4 43.3/.../51.7/81.8.
- DT-A8 vs hob1: 1.73x at 1000, 2.35x at 4000; DT-A4 2.36x / 3.82x. Below 256 tokens DT is 1-3 ms SLOWER than hob1: its question pass
  (gather conv + masked SDPA in torch, a second pass) is unfused glue (hobB, same harness, is slower than DT at every length).
- Runtime vs dtlib with the trained ckpt: L8 40/40 argmax (max|dp| .0056), L12 40/40 (.0085).
- Calibration fit on train_v5.holdout (dt_calib): even hobson's optimal alpha there is .7-2.0, i.e. that holdout does not transfer -> NOT applied;
  report raw ECE/Brier. BUT it exposed a capability loss: DT-A8 noul accuracy on that holdout .583 vs hobson .727 (mostly ruletaker rows);
  DT-A12 .677. dt_holdout.py (per task, + training-free 8/12/16) running alongside the set run.
- DT-set run started 12:53 (400 upd, 1.6 h max).

## 13:06 HELD-OUT CAPABILITY PROBE (MEASURED, dt_holdout.py on v19's train_v5.holdout, n=150 per task, holdout/a812.json)
| task | hobson | DT-A8 trained | DT-A12 trained | tf8 | tf12 | tf16 |
| emotion | .547 | .580 | .567 | .587 | .560 | .553 |
| hate_severity | .453 | .420 | .500 | .353 | .440 | .453 |
| massive_intent | .853 | .833 | .840 | .793 | .853 | .853 |
| ruletaker_d3 | .860 | .647 | .760 | .653 | .833 | .860 |
| ruletaker_d5 | .720 | .560 | .660 | .500 | .713 | .720 |
| ruletaker_natlang | .780 | .660 | .800 | .593 | .780 | .780 |
| sarcasm | .613 | .613 | .613 | .573 | .600 | .613 |
- train_v5 holds ruletaker d0-d2 only; d3/d5 test DEPTH GENERALIZATION of deduction over the state. DT-A8 loses 16-21 points there
  (training does not recover d3: .653 -> .647); DT-A12 trained loses 6-10 (tf12 only 0-3: my training mix eroded it); 16 = hobson exactly.
- => multi-hop composition inside the state needs depth on the state rows; the deep question rows do not substitute at 8 layers.
- pipeline v4 (13:05): replaces the 4A run with the decisive follow-up: continue DT-A8 from s700 on a ruletaker-heavy mix (d0-d2, 50% of
  rows, + real KL 35% + CF 15%, ~1 h); test on d3/d5 (data vs structure). Plus set evals/holdout/latency, tf full 16A/12G/12A.

## 13:14
- CF pair accuracy by state length, DT-A8 vs hobson: <2k .678 vs .269; 2-4k .805 vs .390; >=4k .632 vs .013 (76 pairs). By edit position: first
  half .692 vs .169, second half .739 vs .342. (Trained CF kinds dominate; the CF recipe, not the split, is what reads the details.)
- DRAFT_REPORT.md drafted with DT-A8/A12, latency, frontier, holdout; TBD = DT-set, RT continuation, tf full 16A.
- DT-set run at upd 70 (14.4 s/upd), ETA 14:30.

## 13:51
- DT-set run ~upd 250 (13.1 s/upd), ETA 14:21; tail.sh queued: full-suite eval of main s400 at L8 (matched-budget control for DT-set's 400 updates).

## 14:57 DT-set (400 updates, same recipe) + training-free 16A FULL suites (MEASURED; scores.json)
| | hobson | DT-A8 (700 upd) | DT-set8 | DT-set12 | untrained DT-A16 |
| JB-all (p) | .723 | .684 (.20) | .654 (.026) | .688 (.12) | .736 (.25) |
| JB-hard (p) | .523 | .477 (.41) | .431 (.09) | .462 (.12) | .546 (.25) |
| REAL-label | .785 | .782 | .760 | .787 | .782 |
| CF flip | .268 | .719 | .732 | .732 | .271 |
| CF-probe flip | .328 | .988 | .994 | .997 | .319 |
| REAL / LONG agree_sd | 1/1 | .832/.879 | .757/.873 | .855/.909 | 1.000/.982 |
| Brier JB / REAL-label | .348/.347 | .413/.347 | .419/.362 | .360/.328 | .347/.349 |
| ECE JB / REAL-label | .048/.116 | .059/.088 | .081/.053 | .070/.074 | .045/.112 |
| bar | PASS | PASS | FAIL (JB-all, REAL-label) | PASS | (fidelity) |
- DT-set8 order invariance (full REAL-agree + JB-all, 1969 rotations): unchanged .9985, mean |dp| .0009 (hobson .919 / .037; DT-A8 .902/.035).
- untrained DT-A16 as a function-preserving method: REAL agree .996 (bf16 floor merged_full .996), LONG .994 (.996), CF fgh 1.000,
  CF-probe fgh .943 (floor .971; 3 pairs). Speed 1.14x @1000, 1.33x @4000.
- Holdout ruletaker d3/d5/natlang: set8 .687/.580/.613, set12 .800/.673/.767 (hobson .860/.720/.780).
- Latency setA8 34.0 ms @1000 (dtA8 33.0), 86.3 @4000; setA12 42.7 / 119.1. Fused set runtime vs dtset 40/40 (max|dp| .0066).
- RT continuation (ruletaker-heavy, from DT-A8 s700) started 14:49, ~57 min.

## 15:31 DATA vs STRUCTURE test (MEASURED, holdout/rt8.json): DT-A8 continued 240 updates (3.8k seqs, 38 min) on a ruletaker-heavy mix
(ruletaker d0-d2 train rows 50%, real KL 35%, CF 15%), tested on held-out depths:
| | hobson | DT-A8 s700 | DT-A8 + RT | untrained 8 |
| ruletaker d3 | .860 | .647 | .687 | .653 |
| ruletaker d5 | .720 | .560 | .700 | .500 |
| ruletaker natlang | .780 | .660 | .773 | .593 |
- d5 and natlang recover to within 1-2 points of hobson with targeted data; d3 stays 17 points short. => the shallow-state reasoning loss is
  mostly a training-data/budget effect, with a residual on d3 that 38 min of data did not move. Full evalkit for the RT model running.

## 15:40 RT model (DT-A8 + ruletaker continuation) FULL evalkit (MEASURED): still passes the bar
- JB-all .675 (p .13), JB-hard .462 (p .27), REAL-label .780, CF flip .727, CF-probe flip .997, REAL/LONG sd .829/.855, JB Brier .417, ECE .073.

## 15:50 untrained DT-G12 FULL evalkit (MEASURED): REAL agree .993 / sd .997, LONG .989 / .970, CF fgh 1.000, CF-probe fgh .952, JB-all .732,
REAL-label .785, ECE .039/.110. Near function-preserving at 0.565x state FLOPs; 47.6 ms @1000 (1.20x), 140.9 @4000 (1.43x).
- DT-set12 order: unchanged .999, mean |dp| .00086 (1969 rotations).
- results.json written (suites, order, holdout, latency+projections, training-free subset). Logs copied: box_logs/j3_logs.tgz.

## 16:02 final evals (MEASURED)
- untrained DT-A12 full: JB-all .693, REAL-label .770, REAL/LONG sd .951/.927, CF fgh .982, CF-probe fgh .867 (trained DT-A12 had the same
  JB-all/REAL-label: training at 1/3 of samples did not move L12's absolute accuracy, only the CF kinds).
- matched-budget control: DT-A8 at s400 (of 700): JB-all .667 (p .06), JB-hard .469, REAL-label .777, CF .727, CF-probe .981 -> also misses
  the bar narrowly; DT-set8 (400 upd: .654 / .760) is within noise of it => set8's miss is budget, not the set layout.
- Pipeline v4 + tail done; box idle. DRAFT_REPORT.md final.

## 16:02 done
- DRAFT_REPORT.md final (~1430 words). Box j3 idle (all jobs finished); it auto-terminates at its TTL.

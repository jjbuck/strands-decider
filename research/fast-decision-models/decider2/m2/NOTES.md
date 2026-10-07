# M2 notes: decision-native layout (M2 segment isolation, M3 reader attention, M4 summary slots, M5 top-k reads)

Box `m2` (g5.8xlarge, A10G, launched 22:06:47 per boxes/retry_m12.log). Labels: [M] measured, [V] verified, [A] arithmetic, [S] speculation.

## Log

### 2026-10-06 22:11 PDT start
- Read BRIEF11 in full; IDEAS_EXPLORED (M rows, R1, R2, A5, A6, S2, floor on wall time); FAST_DECISION_MODEL sec 0-2; Q1 NOTES
  (state rows 0-1.7% of decision-gradient trace at layers 18-23; layer 23 state rows need only K/V; state-row sensitivity concentrated in
  rows: last tenth and sinks).
- Read J9 report + j9lib (pieces segmenter, U = '<state>\n' universal compile context, affine GDN composition S <- A_i(S - S_U) + E_i,
  K stored pre-RoPE and re-rotated), J3 dtlib/h7lib/h3lib/dt_eval/dt_holdout/score_dt/dt_lat, J3 NOTES (training-free depth frontier:
  12G ~ function-preserving, 8G REAL sd .880, 4 collapses; ruletaker d3 tf8 .653 vs .860), J2 report/j2lib head.
- Key arithmetic from the floor section: latency is GEMM row-layers. Segment isolation by itself does not cut row-layers; it pays off
  through (a) exact precompilation of constant documents (fewer live rows, A6 without its out-of-context error) and (b) shallow state
  rows with deep question readers (M3, fewer state row-layers), and (c) fixed block shapes for Inferentia.
- Box m2 not ready. Plan: write m2lib.py (segmenter + isolated forward on H7/H3), m2tf.py (training-free sweep), while waiting.

### 22:40 PDT box m2 ready (22:27; booted ~22:07 PDT -> auto-terminates ~08:07 PDT). Checks [V] (res/test.json)
- Code (laptop copy ~/decider2/m2/code, box ~/work/m2): m2seg.py (segmenter), m2lib.py (M2 forward on H7/H3: per-layer segment
  isolation for attention and GDN, local/global RoPE, composition exact/sum/last/zero, frozen state rows = depth split, top-k reads),
  m2tf.py (training-free sweep), m2test.py, m2score.py (J3's scorer + relaxed bar + flips), m2train.py / m2train_util.py (distillation).
- Layout detail: the 4 END tokens '</state>\n' move to the front of every question branch (reader rows); U = the 4 tokens of '<state>\n'
  is the shared sink every segment sees (J9's universal context). With every flag off this is hobson's layout.
- 1. native config vs H7 teacher path: 40/40 argmax, max|dp| .0063 (bf16 noise); 40/40 vs evalkit hobson refs.
- 2. segmentation: 0 token mismatches in 750 states (offsets mapping); U = 4, END = 4 tokens in all 750. 'nat' segments per state:
  REAL-agree 10.9 (mean 180 tokens), LONG 13.5... (see test.json), CF-probe 25.7 (117 tokens); 'const' isolates 60% of REAL-agree rows
  (docs + hook notes), 31% of CF-probe.
- 3. composition (random activations, 5 segments, fla kernels): fold S <- A_j(S - S_U) + E_j vs one native-order scan: rel err 1.2e-3
  (bf16 inputs); the order-free sum S_U + sum_j (E_j - S_U) is off by 390% (rel err 3.9).
- 4. isolation is exact: changing a token in segment 0 changes the layer-23 rows of the last segment by exactly 0.0 (the perturbed
  segment's rows change by up to 16.5).
- 22:33 sweep pass 1 launched (19 configs x 966 items, 2 shards, res/tf1): REAL-label items (316), 40 LONG, 130 JB-hard, 120 CF pairs,
  120 CF-probe pairs.
- Segmenter v2 (after the sweep started): a KB document entry's volatile parts (rank 'N.' and 'Score:' line) become their own 2-6
  token 'docmeta' segments, so a document segment has identical tokens in every request (needed for exact precompilation). The sweep
  pass 1 uses v1 (docs include rank/score).

### 23:00 PDT sweep pass 1 at ~30% (256 of 966 items; partial, MEASURED, untrained hobson weights; REAL-label n = 101, REAL-agree subset)
| config | REAL agree / agree_sd | REAL-label | LONG agree |
|---|---|---|---|
| native (this runtime) | .995 / .983 | .792 | 1.000 |
| iso all layers, natural segments | .797 / .712 | .663 | .861 |
| iso sections | .822 / .678 | .743 | .861 |
| iso fixed 256-token blocks | .858 / .780 | .693 | .778 |
| const (only docs + hook notes isolated, one-sided) | .929 / .797 | .772 | .889 |
| iso attention only / GDN only | .858 / .729 ; .888 / .847 | .782 ; .733 | |
| iso layers >= 8 / >= 16 / < 8 | .939 / .915 ; 1.000 / 1.000 ; .838 / .729 | .802 ; .782 ; .693 | |
| iso + sum composition / + no GDN read | .619 / .424 ; .832 / .729 | .505 ; .782 | |
| freeze 8 / freeze 12 (J3 8G / 12G) | .858 / .661 ; 1.000 / 1.000 | .713 ; .782 | |
| iso + freeze 8 / 12 | .843 / .593 ; .807 / .729 | .752 ; .673 | |
| const + freeze 12 | .929 / .797 | .772 | |
| top-64 keys per head for question rows (M5) | .964 / .881 | .772 | |
- Reading: untrained hobson depends on cross-segment mixing of state rows in its EARLY layers (isolating layers 0-7 costs ~.25 agree_sd)
  and hardly at all in late layers (isolating 16-23 costs 0; 8-23 costs .07). State depth 12 is free; 8 costs. The order-free sum
  composition is destructive (.42). CF/CF-probe pair counts still tiny (9/10).
- ARCH.md v1 written for N1 (block-local state attention + per-block GDN from S_U, frozen after k, global question readers).
- Added: m2lib R order (gran=const;ro=1: reader rows after U + all documents, positions too = J9 layout R with exact doc isolation),
  docfirst composition, memory adapters (J3) for frozen rows, elastic-depth training, m2pre.py (precompile splice check), m2hold.py
  (RuleTaker probe; 'sent' granularity = every sentence its own segment as a stress test), m2lat.py (fused runtime).
- Plan: two 2.5 h distillation runs, elastic k in {8, 12}: N = gran=nat;iso=all;comp=docfirst (all segments isolated), C = gran=const;
  iso=all;ro=1 (only constants isolated, read first). Then full evalkit at k = 8 and 12, RuleTaker, precompile check, latency.

### 23:25 PDT sweep pass 1 at 60% (MEASURED, untrained; REAL-label n = 269, CF 59 pairs, CF-probe 67 pairs; fgh = retention)
| config | REAL agree / sd | REAL-label | LONG agree | CF fgh | CF-probe fgh |
|---|---|---|---|---|---|
| native | .994 / .986 | .781 | 1.000 | 1.000 | 1.000 |
| iso all layers, nat | .845 / .797 | .684 | .843 | .800 | .450 |
| iso sections | .866 / .784 | .732 | .871 | .733 | .700 |
| iso 256-blocks | .890 / .824 | .743 | .829 | .733 | .524 |
| const (docs + notes only) | .942 / .865 | .777 | .914 | .933 | .850 |
| iso attention only / GDN only | .866 / .770 ; .907 / .858 | .747 ; .740 | | .750 ; .800 | .650 ; .500 |
| iso layers >= 8 / >= 16 / < 8 | .948 / .939 ; .996 / .993 ; .870 / .797 | .777 ; .777 ; .714 | | .933 ; 1.0 ; .867 | .800 ; 1.0 ; .500 |
| iso + sum / + no GDN read | .638 / .446 ; .826 / .750 | .535 ; .747 | | .150 ; .611 | 0 ; .450 |
| freeze 8 / 12 | .897 / .784 ; .996 / 1.000 | .743 ; .773 | | .800 ; 1.0 | .619 ; 1.0 |
| iso + freeze 8 / 12 | .864 / .696 ; .847 / .797 | .770 ; .695 | | .800 ; .800 | .400 ; .450 |
| const + freeze 12 | .944 / .872 | .773 | | .933 | .850 |
| top-64 keys (M5) | .965 / .892 | .777 | .957 | .933 | .800 |
- Local (precomputable) vs native RoPE positions for isolated rows: max|dp| .027, 1 of 1026 argmax differ -> positions cost nothing.

### 23:55 PDT sweep pass 1 COMPLETE (MEASURED, untrained hobson weights, res/tf1, res/tf1_scores.json; 966 items: REAL-label 400 q,
REAL-agree (those 316 requests), LONG 40, JB-hard 130, CF 120 pairs, CF-probe 120 pairs). sd = state-dependent agreement; fgh = retention.
| config | REAL agree / sd | LONG agree / sd | REAL-label | JB-hard (p) | CF fgh | CF-probe fgh |
|---|---|---|---|---|---|---|
| native (this runtime) | .995 / .990 | 1.000 / 1.000 | .787 | .538 | 1.000 | 1.000 |
| iso all layers, nat segments | .855 / .817 | .837 / .889 | .723 | .485 (.42) | .771 | .405 |
| iso sections | .872 / .798 | .880 / .852 | .755 | .538 | .771 | .676 |
| iso 256-token blocks | .896 / .837 | .804 / .741 | .757 | .531 | .686 | .486 |
| const (docs + notes only, one-sided) | .944 / .889 | .913 / .815 | .785 | .538 | .971 | .838 |
| iso attention only | .885 / .812 | .870 / .852 | .755 | .562 | .743 | .622 |
| iso GDN only | .905 / .861 | .859 / .852 | .760 | .454 (.08) | .714 | .486 |
| iso layers >= 8 | .955 / .938 | .935 / .852 | .787 | .523 | .971 | .838 |
| iso layers >= 16 | .996 / .995 | 1.000 / 1.000 | .785 | .538 | 1.000 | 1.000 |
| iso layers < 8 | .879 / .817 | .859 / .889 | .743 | .462 (.12) | .771 | .432 |
| iso, sum composition | .663 / .452 | .467 / .481 | .568 | .485 | .143 | .000 |
| iso, no GDN read (zero) | .844 / .779 | .891 / .889 | .760 | .508 | .686 | .378 |
| iso, native positions | .855 / .817 | .848 / .889 | .723 | .477 | .771 | .405 |
| freeze 8 (J3 8G) | .904 / .803 | .946 / .852 | .755 | .446 (.10) | .714 | .568 |
| freeze 12 (J3 12G) | .995 / 1.000 | .989 / 1.000 | .782 | .538 | 1.000 | .946 |
| iso + freeze 8 | .877 / .726 | .859 / .704 | .785 | .469 (.30) | .686 | .378 |
| iso + freeze 12 | .858 / .817 | .837 / .889 | .730 | .485 | .771 | .405 |
| const + freeze 12 | .945 / .894 | .913 / .815 | .782 | .538 | .971 | .811 |
| top-64 keys per head (M5) | .972 / .909 | .967 / .926 | .787 | .531 | .971 | .865 |
- hobson on these items: REAL-label .785, JB-hard .523, CF pair acc .292, CF-probe .308.
- 23:49 pipeline launched: train N (gran=nat;iso=all;comp=docfirst, elastic 8/12, 2.6 h cap), then C (gran=const;iso=all;ro=1)
  in parallel with N's evals. Smoke test [V]: fused runtime with compiled documents vs m2lib preds 16/16 argmax, max|dp| .0046
  (nat, k = 8) and 16/16, .0059 (R order, k = 8): the runtime computes the same function and compiled = live documents.

### 00:05 PDT RuleTaker probe, untrained (MEASURED, res/hold_untrained.json; train_v5.holdout, n = 150 per task, accuracy)
| | d3 | d5 | natlang |
|---|---|---|---|
| hobson | .860 | .720 | .780 |
| every sentence its own segment ('sent'), all layers | .700 | .593 | .700 |
| 'sent' + freeze 8 / 12 | .680 / .700 | .560 / .593 | .680 / .707 |
| freeze 8 / 12, no isolation (= J3 tf8 / tf12) | .687 / .847 | .527 / .727 | .607 / .793 |
| 32-token blocks isolated | .653 | .587 | .600 |
- RuleTaker states are one paragraph, so 'nat' segmentation leaves them as one segment (only depth matters there); 'sent' is the stress
  test (15-20 segments per state). Isolating every fact costs 13-16 points untrained, like cutting the state to 8 layers.
- Training N: 18.6 s/update (16 sequences, ~2.8k tokens each; ~1,800 tokens/s, J3 had ~3,000) -> ~500 updates in 2.6 h. Peak 7.5 GB.
  Dev (J14 is_dev split, 348 questions, 105 state-dependent) at update 0: k=8 agree .891 / sd .752; k=12 .891 / .810.

### 00:30 PDT M5 top-k (MEASURED, untrained, sweep subset; question rows keep the top-k state keys per head at the 6 attention layers)
| k | REAL agree / sd | REAL-label | CF fgh | CF-probe fgh | REAL tv |
|---|---|---|---|---|---|
| 16 | .939 / .875 | .777 | .971 | .838 | .049 |
| 64 | .972 / .909 | .787 | .971 | .865 | .029 |
| 256 | .991 / .971 | .787 | 1.000 | .946 | .013 |
- Run N dev (is_dev, 348 q / 105 state-dependent): upd 0 k8 .891/.752, k12 .891/.810 -> upd 74 (0.5 h, 2.5M tokens) k8 .931/.781,
  k12 .940/.829 (agree / agree_sd). tv .105 -> .095 (k8).

### 01:30 PDT run N learning curve (MEASURED, dev = J14 is_dev split, 348 questions, 105 state-dependent; agree / agree_sd / tv)
| update (hours, tokens) | k = 8 | k = 12 |
|---|---|---|
| 0 (untrained) | .891 / .752 / .105 | .891 / .810 / .090 |
| 74 (0.5 h, 2.5M) | .931 / .781 / .095 | .940 / .829 / .089 |
| 178 (1.0 h, 6.1M) | .931 / .790 / .083 | .931 / .790 / .080 |
| 284 (1.5 h, 9.7M) | .888 / .838 / .084 | .917 / .857 / .075 |
- ~17.5 s/update (16 sequences); letting N run to its 2.6 h cap, then C 2.3 h in parallel with N's evals.

### 01:45 PDT precompile splice check, untrained weights (MEASURED, res/pre_native.json, res/pre_iso_untrained.json)
60 banking REAL-agree requests (228 questions); every KB-document and hook-note segment (61% of state rows) computed standalone in
[U + segment] and its hidden rows spliced into the request at all 24 layers:
- hobson's own layout (= A6's out-of-context compile): 221/228 decisions unchanged (7 changed, 3.1%), max|dp| .261, median .0104;
  the spliced rows differ from the in-context rows by up to 22% (relative norm, layer 23).
- isolated layout (nat, iso all, docfirst, k = 8): 226/228 unchanged, max|dp| .0056, median .0011; spliced rows differ by <= 1.05%
  (bf16 GEMM-shape rounding). The 2 changed decisions moved by .003 and .0029 (ties).
| 387 (2.0 h, 13.1M) | .943 / .848 / .068 | .943 / .848 / .063 |
| 512 (2.6 h, 17.3M, final s512) | .957 / .914 / .058 | .954 / .905 / .052 |
- 02:34 PDT N done (still improving steeply between 2.0 and 2.6 h). C (gran=const;iso=all;ro=1, elastic 8/12, 2.3 h) started;
  N evals (full evalkit k=8 and k=12, RuleTaker, precompile splice) running alongside.

### 03:00 PDT trained N (s512) at k = 8, FULL evalkit (MEASURED, res/full/N_k8.json; McNemar vs hobson)
- JB-all .649 vs .723 (p .016, significant drop); JB-hard .431 vs .523 (p .081); REAL-label .792 (hobson .785); REAL agree .924 / sd .844;
  LONG agree .947 / sd .909; CF pair acc .704 (hobson .268), CF-probe .553 (.328); CF fgh .954, CF-probe fgh .648; SHUF both_right .695
  (hobson .245); Brier REAL-label .348 (.347), JB-all .446 (.348).
- Strict bar: fails JB-all only. Relaxed: fails JB-all (.649 < .693) and REAL sd (.844 < .85).
- CF-probe by kind: the DISTRACTOR variants fail: id_match_distract 0.0 (hobson .381), status_equal_distract 0.0 (.25),
  amount_vs_limit_distract .21 (.24); the plain kinds 1.0 (amount, date, id). With every message / tool call / tool result its own
  segment, binding a record to the id it was looked up for (call <-> result, two records) is left to the question rows, and they do
  not learn it in 2.6 h.

### 03:20 PDT trained N (s512) at k = 12, FULL evalkit (MEASURED, res/full/N_k12.json)
- JB-all .675 (p .117), JB-hard .454 (p .188), REAL-label .780, REAL agree .925 / sd .864, LONG .945 / .885, CF .697 (fgh .917),
  CF-probe .541 (fgh .648), Brier REAL-label .352 (.347), JB-all .395 (.348). STRICT BAR: PASS (REAL-label exactly at .78).
  Relaxed: fails JB-all only (.675 < .693).
- JevBench per family at k = 8 (n small): ambiguous .14 vs .71 (n 7), adequacy .58 vs .83, policy .58 vs .75, ordinal .83 vs 1.0,
  judge_hard .41 vs .53, multi_hop .33 vs .44; temporal_numeric up (.33 vs .20).

### 03:20 PDT N evals done (MEASURED)
- RuleTaker (res/hold_N.json; n = 150 each; RuleTaker states are one segment under 'nat', so k8/k12 test depth; 'sent' isolates
  every sentence): d3 hobson .860 / k8 .713 / k12 .740 / k8 sent .640; d5 .720 / .580 / .607 / .560; natlang .780 / .673 / .680 / .633.
  Training eroded k12 (untrained freeze-12 d3 .847), as J3 saw.
- Precompile splice with the TRAINED N at k = 8 (res/pre_N.json): 228/228 decisions unchanged, max|dp| .0057, median .00065, with
  61% of state rows (documents + hook notes) computed standalone.
- **Box timer reset (the one allowed reset): 03:20 PDT, `sudo shutdown -c; sudo shutdown -h +390` -> box m2 now powers off at
  16:50 UTC (09:50 PDT) instead of ~15:07 UTC.** Reason: latency must run on an exclusive GPU after C, and N's curve was still rising
  steeply (dev sd .848 -> .914 in its last 0.6 h), so a continuation of N is worth ~1.5 h.
- Plan: C ends ~04:52 PDT -> latency (exclusive, ~40 min) -> N continuation (1.5 h) alongside C's evals -> N-continued evals.

### 04:20 PDT run C learning curve (dev, agree / sd / tv)
| update (hours) | k = 8 | k = 12 |
|---|---|---|
| 0 (untrained) | .920 / .848 / .072 | .966 / .952 / .040 |
| 65 (0.5 h) | .925 / .848 / .106 | .937 / .848 / .096 |
| 169 (1.0 h) | .940 / .848 / .087 | .948 / .876 / .075 |
| 285 (1.5 h) | .937 / .838 / .084 | .945 / .829 / .075 |
- C starts much closer to hobson at k = 12 and training first pulls it away (as J9 saw with all-weights LoRA on a close model).
- phase3.sh launched (waits for C; then latency on an exclusive GPU; then N continuation + C evals).
| 404 (2.0 h) | .960 / .895 / .065 | .968 / .914 / .055 |
| 470 (2.3 h, 15.9M, final s470) | .943 / .914 / .056 | .968 / .952 / .049 |
- 05:02 PDT C done; phase 3 latency (exclusive GPU) started: grid with N's checkpoint, grid with C (R-order runtime), then the 84 real.

### 05:10 PDT latency grid (MEASURED, res/lat_grid.json; A10G, bf16 fused runtime, CUDA graph per exact shape, 15 timed reps, fresh
inputs; median ms, p95 within 0.1 ms; T = state tokens incl. U; question 125 tokens + 4 END; c = share compiled in 256-token blocks)
| T, 1 q | hob1 | hobB | dtG8 | dtG12 | m2 k8 c0 | m2 k8 c.55 | m2 k12 c0 | m2 k12 c.55 | m2 k24 c0 | m2 k24 c.55 |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 15.3 | 25.6 | 20.3 | 21.6 | 20.9 | - | 22.5 | - | 27.2 | - |
| 256 | 22.9 | 30.7 | 22.9 | 24.9 | 24.2 | - | 26.6 | - | 33.4 | - |
| 1000 | 57.0 | 67.5 | 41.1 | 47.7 | 46.4 | 34.6 | 54.2 | 39.6 | 77.4 | 54.4 |
| 4000 | 204.7 | 216.4 | 116.1 | 141.2 | 134.2 | 81.1 | 162.4 | 95.9 | 246.7 | 140.7 |
4 questions (380 q tokens): hobB 53.6 / 58.8 / 96.1 / 246.8; m2 k8 c0 48.9 / 52.3 / 74.9 / 164.6; k8 c.55 - / - / 63.1 / 111.4.
- Isolation overhead of my runtime: m2 k24 c0 vs hobB +10 ms (+15%) at 1000, +30 ms (+14%) at 4000; m2 k8 vs dtG8 +5 / +18 ms. Sources:
  the exact-composition scan (a second GDN scan per layer), gathers for per-segment conv history and segment order, U-key concatenation.
  All glue (unfused torch), not arithmetic. The question pass (J3's q_pass glue) costs ~10 ms against hob1 at 1000 (hobB vs hob1).
- C runtime grid (R order, live rows one reader stream; res/lat_grid_C.json), 1 q: T=1000 k8 c0 44.3, k8 c.55 33.5, k12 c0 51.1,
  k12 c.55 37.9, k24 c0 71.3; T=4000 k8 c0 128.9, k8 c.55 79.1, k12 c.55 93.0, k24 c0 230.9 (hob1 57.1 / 204.7).
- 05:10 PDT real-request latency crashed (hob1 graph built inside the capture: CPU->GPU copy during capture). Fixed; phase3b relaunched
  with an exclusive GPU (N continuation and C evals were stopped by PID first; both resume).
- J9's A6 runtime (j9lat.py real, copied read-only) re-run on box m2: 84 requests native mean 160.2 ms, A6 compiled 98.4 (1.63x);
  J9's own box: 161.5 / 98.8. res/lat_real_j9_on_m2box.json.

### 05:40 PDT real-request latency, N layout (MEASURED, res/lat_real.json; J9's 84 eval-split requests, first question, exact shapes,
15 reps, fresh inputs; docs + hook notes compiled; natural segments; J3's question-pass glue)
| | n | state / live tokens | hob1 mean (median, p95) | A6 (J9 runtime, same box) | N k8 mean (median, p95) | N k12 | speedup of means k8 / k12 / A6 |
|---|---|---|---|---|---|---|---|
| all | 84 | 2961 / 1900 | 160.1 (143.2, 309.8) | 98.4 | 90.2 (81.1, 161.1) | 105.1 | 1.77 / 1.52 / 1.63 |
| banking | 59 | 3286 / 1792 | 178.8 (169.8, 324.9) | 97.0 | 93.6 (76.0, 185.4) | 107.8 | 1.91 / 1.66 / 1.84 |
| retail | 25 | 2194 / 2154 | 116.0 | 101.8 | 82.4 | 98.8 | 1.41 / 1.17 / 1.14 |
- Per-request median speedup k8: banking 2.26x. My runtime pays ~10 ms of J3's question-pass glue (hobB vs hob1) that J9's runtime and
  hob1 avoid; phase4.sh re-times one-question requests with TTL's fused cached pass for the question (QFUSED), after the N2 evals.
- C layout (R order, one live stream; res/lat_real_C.json): all 84 k8 86.2 (median 76.0, p95 156.1) / k12 99.1 -> 1.86x / 1.61x;
  banking k8 89.9 / k12 102.4 -> 1.99x / 1.75x (per-request median 2.35x); retail 77.5 / 91.5 -> 1.50x / 1.27x.
- 05:41 PDT phase3c: N continuation (resume s512, cosine over 820 updates, cap 4.1 h cumulative) + C evals running together.

### 06:05 PDT trained C (s470) at k = 8, FULL evalkit (MEASURED, res/full/C_k8.json)
- JB-all .654 (p .011), JB-hard .423 (p .029) [JevBench has no documents: C there = depth split k8 + training]; REAL-label .780;
  REAL agree .923 / sd .853; LONG .936 / .873; CF .714 (fgh .954); CF-probe .981 (fgh 1.000); SHUF .713; Brier REAL-label .344.
- CF-probe distractor kinds: C .875-1.0 vs N 0.0-.21 -> isolating the DYNAMIC segments (N) is what loses record binding
  (tool call <-> tool result, two records); isolating only the constants (C) keeps it.
- Strict: fails JB-all, JB-hard. Relaxed: fails JB-all only.

### 06:20 PDT trained C (s470) at k = 12, FULL evalkit (MEASURED, res/full/C_k12.json)
- JB-all .697 (p .263), JB-hard .485 (p .332), REAL-label .770, REAL agree .924 / sd .870, LONG .938 / .885, CF .702 (fgh .927),
  CF-probe .981 (fgh .990), Brier REAL-label .349 (.347), JB-all .371 (.348). RELAXED BAR: PASS. Strict: fails REAL-label (.770 < .78).
- Summary: N k12 passes strict (not relaxed: JB-all .675 < .693); C k12 passes relaxed (not strict: REAL-label .770); both k = 8 fail
  JevBench (significant drops).
- C RuleTaker (res/hold_C.json; documents absent, so = depth): d3 k8 .747 / k12 .800, d5 .633 / .687, natlang .660 / .633
  (hobson .860 / .720 / .780). C precompile splice at k8 (res/pre_C.json): 228/228 unchanged, max|dp| .0076, median .0007.
- REAL-label McNemar vs hobson: N k8 p .70 (15/12), N k12 .86 (16/18), C k8 .85 (12/14), C k12 .35 (11/17) -> none significant.

### 06:50 PDT N continuation (N2; resumed s512, LR re-warmed to 7e-5 on a cosine over 820 updates; dev agree / sd / tv)
| update (cum. hours, tokens) | k = 8 | k = 12 |
|---|---|---|
| 570 (3.1 h, 19.3M) | .945 / .886 / .069 | .957 / .895 / .062 |
| 668 (3.6 h, 22.6M) | .960 / .895 / .051 | .966 / .905 / .049 |
- ARCH.md v2 (status + which k to time).
| 773 (4.1 h, 26.1M, final s773) | .954 / .886 / .053 | .966 / .905 / .051 |

### 07:30 PDT N2 (s773) at k = 8, FULL evalkit (MEASURED, res/full/N2_k8.json)
- JB-all .636 (p .005), JB-hard .408 (p .028), REAL-label .782, REAL agree .936 / sd .861, LONG .945 / .891, CF .712 (fgh .982),
  CF-probe .559 (fgh .667; distractor kinds still 0-.25). More training raised real-traffic agreement (.844 -> .861) and lowered
  JevBench (.649 -> .636): the k = 8 JevBench gap is not closing with tokens on this mix.

### 07:40 PDT N2 (s773) at k = 12, FULL evalkit (MEASURED, res/full/N2_k12.json)
- JB-all .6926 (160/231; p .349), JB-hard .477 (p .418), REAL-label .787 (p 1.0), REAL agree .940 / sd .870, LONG .941 / .867,
  CF .719 (fgh .982), CF-probe .537 (fgh .638), Brier REAL-label .355 (.347), JB-all .385 (.348).
- STRICT: PASS. Relaxed: JB-all 160/231 = .6926 against >= .693 -> one JevBench item short; everything else passes.
- N2 RuleTaker (res/hold_N2.json): d3 k8 .653 / k12 .807, d5 .560 / .620, natlang .607 / .693 (hobson .860 / .720 / .780).
- 07:38 PDT phase 4 (QFUSED: one question as TTL's fused cached pass): check vs m2lib N k8 preds 15/16 argmax, dp median .0011,
  max .018; the one changed decision was a tie (m2lib p .512 / .488).

### 07:50 PDT latency grid with the fused one-question pass (MEASURED, res/lat_grid_qf.json, res/lat_grid_C_qf.json; 1 q, median ms)
| T | hob1 | dtG8 | dtG12 | N k8 c0 | N k8 c.55 | N k12 c0 | N k12 c.55 | C k8 c.55 | C k12 c0 | C k12 c.55 |
|---|---|---|---|---|---|---|---|---|---|---|
| 64 | 15.3 | 15.8 | 17.1 | 16.4 | - | 18.0 | - | - | 17.3 | - |
| 256 | 22.9 | 18.4 | 20.4 | 19.7 | - | 22.1 | - | - | 21.1 | - |
| 1000 | 57.1 | 36.3 | 42.9 | 41.6 | 29.8 | 49.4 | 34.8 | 28.7 | 46.3 | 33.1 |
| 4000 | 204.7 | 110.1 | 135.2 | 128.2 | 75.0 | 156.3 | 89.9 | 73.0 | 148.4 | 87.0 |
- Projections [A] (J9's method), N k12 c.55: 1000 -> 18.4 / 9.8 / 7.0 ms (3090 / 4090 / 5090; hobson 29.6 / 14.6 / 10.7);
  4000 -> 46.1 / 21.6 / 16.3 (hobson 103.8 / 46.2 / 35.8). C k12 c.55 at 1000: 33.1 -> 17.5 / 9.4 / 6.7 [A].

### 08:05 PDT real-request latency with the fused one-question pass (MEASURED, res/lat_real_qf.json, res/lat_real_C_qf.json;
84 requests, mean ms (median, p95); A6 = J9's runtime on this box)
| | hobson | A6 | N k8 | N k12 | C k8 | C k12 |
|---|---|---|---|---|---|---|
| all 84 | 160.1 (143.2, 309.8) | 98.4 | 79.1 (74.4, 138.9) | 94.0 (89.3, 160.3) | 75.1 (70.0, 132.4) | 88.0 (83.1, 150.9) |
| banking 59 | 178.8 | 97.0 | 80.0 | 94.3 | 76.3 | 88.8 (76.6, 159.5) |
| retail 25 | 116.0 | 101.8 | 77.1 | 93.5 | 72.2 | 86.2 |
- Speedup of means vs hobson: all 84: N k8 2.02x, N k12 1.70x, C k8 2.13x, C k12 1.82x, A6 1.63x; banking: C k12 2.01x (per-request
  median 2.05x), N k12 1.90x, A6 1.84x.
- Phase 4 done 08:03 PDT. All results copied to ~/decider2/m2/res and box_logs/. Box idle; it powers off at 09:50 PDT (timer reset).

### 08:20 PDT done
- DRAFT_REPORT.md final (~1,240 words). ARCH.md v2 for N1. No permission denials. Box m2 is idle and powers off at 09:50 PDT
  (one timer reset, noted above). Checkpoints on the box only: ~/work/m2/ck_N (s512 = 2.6 h, s773 = 4.1 h), ~/work/m2/ck_C (s470).

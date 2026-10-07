# J14 notes: state-first compiled question bundle (box j14)

## 13:07 start
- Read BRIEF8, FAST_DECISION_MODEL.md, J9 NOTES/DRAFT, H7 DRAFT, J5 DRAFT, evalkit README, h3lib/h7lib/j9lib.
- Box j14 pending (launched 13:03).
- Design (J14 layout, "in-place partial compile", STATE FIRST):
  [state: hobson exact, question-independent] [question in its natural position: most rows COMPILED once per deployment
  in a universal context U = render_state('') (= hobson's own no-state rendering), only a few rows LIVE (option-end rows + <answer>)].
  Compiled rows contribute their cached pre-conv qkv / beta / g (GDN, replayed from the state's final GDN state = exact affine
  composition) and K (pre-RoPE, re-rotated to natural positions) / V (attention). Live rows run hobson weights at natural positions.
  => state rows are exactly hobson; live rows are at hobson's positions with hobson's weights; ONLY the compiled question rows'
  hidden states are context-free. Contrast H7: question-first (state rows see the bundle; slots out of place).
- Consequence for training: the state prefix is frozen hobson -> computed once under no_grad, cached (K/V + GDN state + conv tail);
  backprop only through question rows (compile pass over U+Q, live rows). Training cost per request ~ one hobson prefix + tiny suffixes.
- Arms: untrained (live-set variants: opt-end+ans, +suffix, +option tails, +options block; GDN replay vs skip);
  compile adapter LoRA (J9); FULL-FT compile path with L2-SP (compile weights are compile-time only -> free at runtime);
  + live-row LoRA (minimal live-path adaptation, state cache untouched). Dev = held-out train-split tasks.
- 37 distinct question specs in train_pool; JevBench questions are unseen -> JB tests compile-path generalization.

## 13:58 box ready (13:23); lib verified; UNTRAINED layout fails badly
- j14lib verified on box [V]: all-live suffix (state cached) = plain hobson forward within runtime noise (|dp| <= .003);
  compiled-cache path = fresh compile path exactly; EMPTY-STATE identity (compile context == state) -> compiled == hobson (|dp| <= 3e-4).
- Untrained, REAL-agree (777 of 1083 questions scored before I stopped it), agree / agree_sd / tv:
  all-live (floor) .996 / 1.000 / .003
  oa (opt-ends + answer live, 10 rows/q) .716 / .068 / .244     oa+sfx (21 rows/q) .716 / .068 / .244
  oa+sfx GDN-skip .613 / .388 / .330    oa+sfx+ok4 (49 rows) .743 / .156     ob (whole options block live, 271 rows/q) .861 / .641
  ob+q16 .873 / .671.  agree_sd ~0.07 = the live rows decide like NO-STATE hobson.
- Reading: hobson gathers state evidence AT THE QUESTION ROWS (instruction tokens attend to the state); the answer/option rows read the
  question rows. Context-free question rows carry no evidence, and with GDN replay they also wash out the state's GDN memory
  (skip > replay untrained). => the compile adapter alone cannot inject evidence; the live rows must learn to read the state.
- 14:00 launched arm A: compile LoRA r16 (lr 1e-4) + LIVE-ROW LoRA r64 (lr 2e-4; question live rows only, state cache untouched),
  lv oa+sfx, GDN replay, KL + dense rows (5/11/17/23), 600 updates x 4 requests, dev = 120 held-out train-split requests.

## 14:50 arm A learning fast; runtime built and verified
- Arm A dev (120 held-out train-split requests, 378 q, n_sd 107): agree_sd step 0 .028 -> 50 .551 -> 100 .710 (procedure .52, other .86);
  agree .725 -> .910; tv .226 -> .118. Already above H7's plateau (.685 after 900 updates).
- Lib: added 'frozen' conv semantics (compiled rows' post-conv q/k/v constant; only live rows see natural conv neighbours) so the
  compiled cache is fully constant and the leading compiled run of each question is ONE exact affine GDN transfer (A_q, B_q).
  Arm A was trained with 'natural' conv (differs only at <=3 compiled rows after the state / after each live row).
- Runtime j14rt.py = H2 QRT (bf16 fold / rotated W8A8-b8) on J5 short-M kernels + J14 layout (live rows at natural positions,
  persistent compiled span buffers, varlen GDN replay or affine leading-run transfer, masked SDPA over state + compiled K/V).
  [V] runtime vs j14lib (frozen, untrained compile) on 11 real REAL-agree states: argmax 11/11, max|dp| median .0014 / max .0076
  (replay), .0012 / .0053 (affine).
- Latency sets (real question ids): bk15 = 15 banking questions, 1733 question tokens, 14 live rows each (oa+sfx);
  bk4, bk1; jb1 (median JevBench question, 103 tok, 16 live), jb4.

## 14:05 (clock check: the two previous headers were ~40-60 min fast; laptop PDT = box UTC-7)
- Arm A stopped at step 200 (cosine over 600): dev agree_sd 50 .551 / 100 .710 / 150 .813 (best.pt) / 200 .682; tv .144/.118/.115/.111.
  Churn on the 23-way procedure questions (n_proc 48: .52/.69/.46); other questions ~.86-.92. LoRA arm looks like H7's plateau pattern.
- Added the FULL-FT arm 'fullall': ONE full-FT weight set (bf16 + stochastic-rounding Adam, factored 2nd moment, decoupled L2-SP pull
  to hobson, lambda 300) used by state rows, live rows AND compile rows -> no runtime cost (same single weight set as hobson).
  Teacher = frozen hobson (separate hobson prefix pass). State prefix now differentiable (checkpointed per layer).
- 14:01 launched arm B = fullall, lr 2e-5, conv 'frozen' (runtime-exact), oa+sfx live set, 500 updates x 4 requests, states <= 5000 tok.
  19.5 GB on the GPU -> nothing else can run beside it.
- Coordinator relayed J9's final: compile adapter good; residual concentrated in the 23-way procedure questions; reorder (not compile)
  dominated J9's loss. For J14 the order is hobson's (state first) so the reorder loss is absent by construction; the loss is
  'question rows read no state'. Plan: gate on per-question agreement (procedure vs other is in every dev line), then a procedure-
  focused arm with option-line tails live for many-way questions (option-definition text as candidate rows).

## 14:58 arm B (full FT, one weight set, L2-SP) no better than LoRA; stopped at 290
- B dev (464 q, n_sd 117; different 120-request dev because maxtok 5000): step / agree / agree_sd / tv / sd_proc / sd_other
  0 .752/.026/.210/.00/.05 | 50 .847/.496/.149/.12/.80 | 100 .847/.701/.154/.60/.78 | 150 .869/.744/.108/.71/.77 (best.pt) |
  200 .884/.726/.111/.67/.77 | 250 .888/.701/.176/.62/.77.  drift ||W-W0||/||W0|| .0087 (L2-SP holds); tv unstable.
  Full FT moves the state rows too; 'other' questions stall at ~.77 (A: ~.86-.92). Same budget, no gain over LoRA -> stopped.
- 14:58 launched: full eval of A best (s150) [natural conv = as trained, and frozen conv = runtime semantics];
  arm C = A's recipe at half the LR, conv frozen, procedure questions weighted 2x, BIGGER LIVE SET 'ob@8+ok4@8+sfx'
  (whole options block live for <= 8 options; last 4 rows of every option line for the 23-way questions = option-definition text
  as live candidate rows).

## 15:45 arm A (s150) FULL EVAL (3,227 q): shortcut model; detail reading collapses
- A s150 (compile LoRA r16 + live-row LoRA r64, oa+sfx = 17.3 live of 205 question rows on average), frozen conv (runtime semantics):
  REAL agree .821 / agree_sd .720 / tv .138; LONG agree .885 / agree_sd .764; JB-all .693, JB-hard .508 (hobson .523, McNemar p .84);
  REAL-label .703 (hobson .785, p < .001); CF pair acc .170 (hobson .268), CF fgh .193; CF-probe pair acc .019 (hobson .328), fgh .029.
  natural conv (as trained) is the same within noise (agree_sd .717, LONG .770, CF fgh .174).
- per question (REAL agree_sd): cc_ends/states_amount/UserAskedForHuman/cc_procedure_found 1.00, details_match .93, but needed_procedure .47,
  procedure .50, rule_bound_values .61, WrapsUp .56, wants_change .14; LONG needed_procedure .43, step_fit .10.
- The model answers the SAME on both items of most CF / CF-probe pairs (CF-probe: true/true on 195 of 320 pairs; CF asked_for_human:
  'no' on both sides in 91 of 100 pairs). Its REAL agreement comes from coarse correlates (hook, stage), not from reading the edited
  detail. 14-17 trained live rows do not replace ~100-200 question rows as readers.
- 15:45 B s150 full eval running; C (bigger live set) dev: step 50 .598 (tv .109), step 100 .785 (proc .63, other .92, tv .108).

## 16:30 arm B (full FT, s150) FULL EVAL: full FT forgets
- B compiled layout (oa+sfx): REAL agree .820 / agree_sd .613; LONG .875 / .727; CF pair .089 fgh .303; CF-probe pair .006 fgh .010;
  JB-all .567 (hobson .723), JB-hard .415 (p .044, significant drop); REAL-label .728 (p .005).
- B's own weights in HOBSON's layout ('all' live): REAL agree_sd .673, JB-all .619, JB-hard .438 -> the full fine-tune itself has moved
  away from hobson (drift .87% of weight norm under L2-SP lambda 300) and lost generality, while the compiled layout is no better than A.
- C (bigger live set) dev: 150 .822 / 200 .794 / 250 .832 (proc .69, other .95), tv .078 / .076 / .070. Full eval of C s250 running,
  with and without the hybrid policy H8 (questions with > 8 options run uncompiled, exact hobson).

## 17:05 arm C done (400 upd); C s250 FULL EVAL; latency matrix running (exclusive GPU)
- C dev: 300 .869 (proc .73, other .983, tv .063) | 350 .832 | 400 .869 (tv .059 best) -> final = s400 (tie on agree_sd, lower tv).
- C s250 full eval (3,227 q), live set ob@8+ok4@8+sfx (64 live of 205 question rows on average; 79 of 103 for the median JB question):
  REAL agree .917 / agree_sd .824 / tv .081; LONG .917 / .842; CF pair .113 (hobson .268), fgh .394; CF-probe pair .156 (.328), fgh .352;
  JB-all .719, JB-hard .523 (p 1.0); REAL-label .770 (p .38).
  + hybrid H8 (questions with > 8 options uncompiled, exact hobson): REAL agree .958 / agree_sd .916 / tv .047; LONG .962 / .921;
  REAL-label .782 (p 1.0); CF/CF-probe/JB unchanged (no many-way questions there that hobson tracks).
- Per question REAL agree_sd (C): details_match .93, states_amount .92, rule_bound_values .94, WrapsUp .92, UserAskedForHuman .84,
  cc_procedure_found .82; LONG step_fit .10. Off-procedure 'other' .887 on eval-split (dev said .95-.98: generalisation gap across tasks).
- Verdict forming: bigger live sets buy real-traffic agreement and JB (because most of a JB question becomes live), but CF / CF-probe
  detail tracking stays far below the .90 bar -> the evidence is read by the INSTRUCTION rows, which are the ones compiled.

## 17:45 LATENCY (A10G, exclusive GPU, CUDA graph per exact shape, fresh state ids, 15 reps, median; p95 within ~0.1-2 ms) lat_oa/lat_ob.jsonl
- W8A8-b8 (H1 b8 map, RTN codes for speed; J5 short-M kernels), T = 32 / 64 / 128 / 256 / 400 / 1000:
  jb1 plain 7.98 / 9.36 / 10.79 / 15.34 / 19.70 / 41.83 | J14-oa 7.52 / 8.61 / 10.12 / 14.74 / 19.35 / 40.50 | J14-ob 8.76 .. 42.94
  jb4 plain 18.96 / 20.73 / 23.59 / 28.20 / 33.24 / 55.34 | J14-oa 9.68 / 10.24 / 12.82 / 17.00 / 22.83 / 43.92 | J14-ob 15.71 .. 49.22
  bk15 plain 68.19 / 70.07 / 72.11 / 77.70 / 83.33 / 106.74 | J14-oa 20.27 / 22.02 / 24.75 / 28.39 / 32.66 / 53.70 | J14-ob 41.24 .. 77.54
- bf16: jb1 plain 10.57 -> oa 9.74 (T32) ; jb4 25.07 -> 11.84 ; bk15 92.92 -> 23.90 ; T1000 bk15 146.1 -> 71.5.
- affine leading-run GDN = replay within 0.5 ms: GDN kernels drop (bk15 T32 7.45 -> 3.63 ms) but the fp32 A_q @ S bmm adds ~2 ms.
- Where J14's time goes (W8A8 jb1 T32, 48 rows): GEMM 3.95 (plain 5.63 at 135 rows), GDN 1.41 (plain .88: extra span call), other 1.68
  (plain 1.20: index_copy / rope / masks), 846 kernels vs 431. 1 question is at the int8 knee already -> only 6% gain.
  bk15: GEMM 48.4 -> 8.0 ms, but span GDN replay 7.45 ms = plain's GDN (the compiled rows are still replayed through the recurrence).

## 18:05 MECHANISM LADDER (untrained hobson, CF-probe, 320 pairs; live rows / question rows)
- all 126/126: pair .328, fgh .981 | oa+sfx 14: .006 / .010 | ob 70 (options block incl. the probe's own criteria): .062 / .105 |
  ob+q32 102: .378 / .848 | ob+q64 124: .328 / .981.
- => the detail is read by the question's CONTENT rows (the last 32-64 instruction rows: "In the get_account_details result for
  acct_750650, is the balance greater than the daily_transfer_limit?"). Only the boilerplate head of a question compiles losslessly.
- 17:53 C400 eval OOM-crashed (3 processes); killed the compile-only arm E (dev step 0 = .280, as C's untrained start); C400 eval restarted.

## 18:55 C s400 FULL EVAL + ladder on REAL/CF
- C s400 (ob@8+ok4@8+sfx, 64 live of 205 rows): REAL agree .928 / agree_sd .827 / tv .065; LONG .926 / .855; CF pair .140 fgh .495;
  CF-probe pair .150 fgh .333; JB-all .714 (McNemar 5/7, p .77), JB-hard .515 (4/5, p 1.0); REAL-label .785 (p 1.0); Brier REAL-label .362
  (hobson .347), JB-all .365 (.348); SHUF both_right .124 (hobson .245).
  + H8 (many-way questions uncompiled, 157 live of 205 rows): REAL .965 / .922 / tv .037; LONG .966 / .927; REAL-label .785. CF/CF-probe/JB same.
- Untrained ladder (frozen; REAL-agree 1,083 q; CF 406 pairs; CF-probe 320 pairs), live rows/q of 204:
  oa+sfx 16.9: REAL agree_sd .055, CF fgh .000, CF-probe fgh .010, REAL-label .650
  ob 147.5: .555 / .349 / .105 / .703
  ob+q32 179.5: .832 / .982 / .848 / .790
  ob+q64 200.1: .974 / 1.000 / .981 / .790
  all 203.7: .997 / 1.000 / .981 / .787
  => lossless compilation covers only the question header (~2% of question rows at ob+q64).
- 18:57 compile-adapter-only arm E (LoRA r16 on compile rows, live path exactly hobson, C's live set, lr 1e-4) running, 100 updates.

## CLOCK CORRECTION (laptop `date` = 16:59 PDT when E's eval launched)
- Section headers from "14:58" to "18:55" above were written from a drifting estimate; true times were ~14:30 to ~16:50.
- Arm E (compile adapter ONLY; live path exactly hobson; C's live set) dev: 0 .280 | 50 .654 (proc .31, other .93, tv .123) |
  100 .729 (proc .54, other .88, tv .096). Full eval of E s100 running; untrained ladder-2 (instruction tail live + option ends + suffix,
  option-line text compiled: q16/q32/q64+oa+sfx) running on CF-probe, CF, REAL.

## 17:25 E s100 FULL EVAL; ladder-2; arm F launched
- E (compile adapter only, live path = hobson exactly, C's live set ob@8+ok4@8+sfx): REAL .877 / agree_sd .717 / tv .100; LONG .898 / .794;
  CF pair .118 fgh .413; CF-probe pair .134 fgh .295; JB-all .727, JB-hard .538 (p .69); REAL-label .752 (p .06).
  + H8: REAL .933 / .864; LONG .968 / .958 (passes .95); REAL-label .750 (p .016). CF/CF-probe/JB unchanged.
- Ladder-2 (untrained; instruction tail live, option-line text compiled), REAL agree_sd / CF fgh / CF-probe fgh / REAL-label:
  q16+oa+sfx (33 live/204) .159 / .183 / .229 / .677; q32+oa+sfx (49) .254 / .257 / .467 / .715; q64+oa+sfx (70) .329 / .211 / .343 / .735.
  => both the instruction content rows AND the option-criteria rows read the state; only header boilerplate compiles losslessly.
- 17:23 arm F = compile adapter only (live path hobson) with live set ob+q32 (untrained: REAL .832, CF .982, CF-probe .848, REAL-label .790):
  can the adapter close the REAL gap while the exact live content rows keep detail reading? 200 updates.

## 17:58 arm F (compile adapter only, live set ob+q32, live path exactly hobson) PASSES ON DEV
- dev (378 q, n_sd 107): step 0 .794 (tv .061) | 50 .925 (tv .029) | 100 .953 (proc .92, other .98, tv .026) | 150 .963 (proc .96, other .97, tv .024).
- But ob+q32 computes most question rows: banking bk15 1,268 of 1,733 live (27% compiled); the median JevBench question is 100% live.
- Queued (exclusive GPU): latency for the ob+q32 live set (bk1/bk4/bk15, plain + replay) and oa on bk1/bk4; then F s150 full eval.

## 18:05 latency for F's live set (ob+q32) and oa on banking 1/4 q (W8A8-b8, T = 32 ... 1000)
- bk1 plain 7.07 / 8.14 / 10.72 / 15.29 / 19.60 / 42.40 | oa 7.51 / 8.07 / 10.07 / 14.89 / 19.39 / 40.39 | ob+q32 8.71 / 9.56 / 12.49 / 17.18 / 21.12 / 42.73
- bk4 plain 18.45 / 18.90 / 21.22 / 25.07 / 31.02 / 53.73 | oa 9.58 / 10.18 / 12.68 / 16.07 / 22.10 / 43.59 | ob+q32 15.59 / 17.80 / 19.68 / 24.65 / 29.35 / 51.78
- bk15 plain 68.98 / 71.12 / 72.63 / 78.39 / 84.04 / 107.65 | oa 20.27 .. 53.70 | ob+q32 58.42 / 59.08 / 60.99 / 66.89 / 73.30 / 95.50
  => the fidelity-preserving live set buys 1.13-1.18x for 4-15 questions and is slower for 1 question (runtime overhead > 21 saved rows).
- 18:02 arm G launched: compile adapter only, live set q32+ok4+sfx (instruction tail + last 4 rows of every option line + suffix;
  option-definition text and question head compiled), to find a better point on the trade-off. F s150 full eval running.

## 18:28 F s150 FULL EVAL: PASSES EVERY BAR, but compiles only 12% of question rows
- F (compile adapter only, live path exactly hobson; live = option block + last 32 instruction rows + suffix):
  REAL agree .975 / agree_sd .951 / tv .023; LONG .977 / .952; CF pair .276 (hobson .268; 3/0, p .25) fgh 1.000;
  CF-probe pair .319 (.328; 6/9, p .61) fgh .914; JB-all .719 (0/1), JB-hard .515 (0/1, p 1.0); REAL-label .782 (p 1.0);
  Brier REAL-label .345 (hobson .347), JB-all .353 (.348); SHUF both_right .256 (.245).
  + H8: REAL .982 / .968; LONG .989 / .970; REAL-label .787.
- Compiled share of question rows (from preds nl/Lq): A 92% (REAL 93, JB 85) | C,E 69% (REAL 74, JB 38) | C+H8 24% |
  F 12% (REAL 11, CF-probe 18, JB 8) | F+H8 10%.
- Speed of F's live set (above): 1.13-1.18x at 4-15 banking questions, slower at 1 question in this runtime.
- G (q32+ok4+sfx, compile adapter only) dev: 0 .327 | 50 .682 | 100 .766 (other .93, proc .56).

## 18:45 G s100 FULL EVAL (compile adapter only, live = last 32 instruction rows + last 4 rows of each option line + suffix)
- 68% of question rows compiled (REAL 73, CF-probe 54, JB 45): REAL .882 / agree_sd .766 / tv .092; LONG .900 / .782;
  CF pair .200 fgh .697; CF-probe pair .263 fgh .629; JB-all .710, JB-hard .515 (9/10, p 1.0); REAL-label .752 (p .06).
  + H8: REAL .938 / .902; LONG .966 / .933; REAL-label .755 (p .043).
- Trade-off (compiled share -> CF-probe fgh / REAL agree_sd): A 92% .03/.72 | C 69% .33/.83 | E 69% .30/.72 | G 68% .63/.77 | F 12% .91/.95.

## 18:50 wrap-up
- DRAFT_REPORT.md final (about 1,410 words excluding table delimiters). Scores: score_*.txt and preds/*_scores.json; latency: lat_*.jsonl and lat_table.txt
  (projections: lat_proj.json); runtime fidelity: fid_bf16.json; box logs: box_logs/.
- Checkpoints stay on the box (~/work/j14/ck{A,B,C,E,F,G}); none were copied to the laptop.
- Box j14 left idle (no jobs running); it auto-terminates at its TTL.

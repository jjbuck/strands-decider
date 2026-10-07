# J6 notes: question in the weights (per-question adapters + hypernetwork)

## 09:08 start
- Read BRIEF8, FAST_DECISION_MODEL.md, F7, evalkit README, Brooker's two posts, h7 report/code (h7lib branch forward = my base).
- Fetched: Text-to-LoRA 2506.06105, S-LoRA 2311.03285, Punica 2310.18547, Context distillation 2209.15189, HyperTuning 2211.12485, Drag-and-Drop LLMs 2506.16406.
- Inventory: 38 deployed question specs in train_pool, each exactly 1 spec text. REAL-agree/LONG/CF use only these 38 (0 unseen). CF-probe 'probe' differs per item (unseen -> hypernet only). JevBench: per-task questions (hypernet only).
- train_pool: 12,747 requests, 34M state tokens (median 2049), ~47k (state, question) pairs; question sets of 1/4/8/15.
- Box j6 not ready yet.

### First-principles cost model (A10G, bf16, ~64 TFLOPS achieved, 600 GB/s)
- hobson in-context: rows = T + sum|q_i| (|q| = 78..1200 tokens; 15-q banking bundle = 3720 tokens).
- Variant B "late-bound adapter" (state read question-blind = exactly hobson's state rows; question only in weights of K+1 slot rows): rows = T + sum(K_i+1). 15 q at T=1000: 1000+~80 rows vs 4720 -> ~4.4x fewer FLOPs.
- Variant A "question from layer 0" (adapter on state rows too): rows = n_q * T. Wins only for 1 question or short states (n_q*T < T + sum|q|): T < ~266 for 15q, T < ~120 for 4q. Loses 3x at 15q/1000.
- So B is the cost-class change for multi-question; A is the Brooker test (question-aware reading), and its cost is n_q x.

## 09:45 box j6 ready (09:17); code up; (a) training launched 09:29 (correction: not 09:42)
- j6lib.py: J6 = H3 differentiable hobson (merged) + `state_cache` (question-blind state rows) + `branch` (segments continuing cached states,
  multi-request) + `full` (variant A, adapter on every row). QAdapters: shared LoRA r16 + per-question LoRA r8 on Win/Wo/Wd of all 24 layers
  (Wgu skipped so the fused runtime can serve it: deltas enter via GEMM-output add or residual pre-add), + per-question slot-input vectors
  [K option slots + answer]. Batched over segments with padded bmm (Punica/S-LoRA style).
- [V] teacher layout (state_cache + one branch per question) vs hobson refs: argmax 49/49, max|dp| 0.0056 (REAL-agree + LONG items).
- [V] fused TTL state pass vs h3lib state pass: teacher |dp| <= 0.004. TTL 54 ms/1000 tok vs 78.
- [V] multi-request merged branch vs per-request: max|dp| 0.009 with random adapters (numerical).
- train_a.py launched 09:29: 38 questions, KL(teacher||student) + 1.0 * rel-MSE at layers 5/11/17/23 (h7's dense term), 4 requests/update,
  lr 2e-4 cosine over 6000 (deadline 120 min ~ 5200 updates ~ 1.65 epochs of train_pool). 1.37 s/update, 7.7k state tok/s, 16 GB.
  First 100 updates: kl .46 -> .43, hid .78 -> .26.

## 09:50
- (a) curve on fresh (first-epoch) data: upd 90 kl .51 agree .63 | 290 .20/.81 | 490 .12/.89 | 590 .11/.87 (agree = plain argmax agreement with hobson on not-yet-seen train-split requests).
- Written while (a) trains: lat_j6.py (fused runtime arms ctx_packed / ctx_plain / w_late / w_early), train_h.py + HyperAdapters (hypernet), train_e.py + eval_e.py (Brooker test: late vs early from the (a) checkpoint, same samples), score_j6.py (laptop scorer: every suite, Brier/ECE, McNemar). Scorer validated on merged_full / nostate baselines (reproduces README numbers).
- Fetched more: Prompt Injection 2206.11349 (fixed prompt into parameters, up to 280x FLOPs), Gisting 2304.08467, LoRA 2106.09685, HyperNetworks 1609.09106.
- Cost model on deployed sets (rows through the network): hobson packed T+sum|q| vs late-bound weights T+sum(K+1):
  Q1 states_amount 84 vs 3; Q1 procedure 1014 vs 24; Q4 (details_match, failure_cause, needed_procedure(48-way), rule_bound_values) 1613 vs 59; Q15 code_* 1480 vs 45.
  train_pool mean: state 2664 + 889 question tokens per request -> 1.32x fewer rows on this (long-state) traffic; at T=64..400 the question term dominates (4-14x rows).

## 10:15
- (a) curve (fresh first-epoch requests, 25-update windows): upd 240 kl .42/agree .70 | 740 .094/.883 | 1240 .056/.905 | 1740 .050/.915 | 1790 .039/.941. Still improving, ~1.5 s/update.
- pipeline.sh running on box: waits for train_a, then smoke tests train_h/train_e, eval_a (+teacher), lat check, lat timing (exclusive GPU), train_h 95 min, train_e 45 min (maxtok 2048, 800 upd), eval_h, eval_e, eval_late.
- Rows ratio on real train_pool traffic (in-context vs late-bound weights): mean 1.32x, median 1.41x; by state length 0-400: 2.67x, 400-1000: 2.22x, 1-2k: 1.61x, 2-4k: 1.22x, 4k+: 1.18x. The win is the question term, so it is large on short/multi-question requests and modest on long states.

## 10:57
- (a) at upd 3460 (1.09 epochs), kl .023, agree .925 (now partly on repeated requests), hid .024. Ends at the 120-min deadline (~11:29, ~4750 updates).

## 11:55 (a) results [M] (s4723, 120 min, 1.48 epochs of train_pool; eval through FastState state rows + j6lib slot rows)
- [V] in-runtime teacher (hobson in context through the same code): REAL agree .995 / sd .997, LONG .996/.988 -> noise floor ok.
- [V] fused runtime vs j6lib: ctx 60/60 argmax (max dp .0064); w_late 60/60 (max dp .0038).
- (a) per-question adapters, late-bound: REAL agree .934, agree_sd .864, tv .050 | LONG .951 / .891 | REAL-label .785 (hobson .785; McNemar 13/13 p 1.0)
  CF acc .514 (hobson .594), flip .126 (.268), fgh .431, dir .978 (.948); McNemar CF 7 vs 72, p 1e-14 -> significant drop. SHUF both_right .114 (.245).
  h7 (schema-first prefix) for comparison: REAL sd .685, LONG .836.
- Where it fails: CF human_insert .019 vs hobson .346 (the asked-for-human questions have ~0 state-dependent examples in real traffic: REAL n_sd 0-9 -> adapters learned the prior); procedure / needed_procedure agree_sd .74 / .79 (LONG needed_procedure .72); wrapup .755 vs .918. amount_insert .122 vs .100 (>= hobson).
- Response: train_e (Brooker test) now distils hobson on 50% counterfactually edited TRAIN states (h7 gen_cf human/amount/wrapup; probe items excluded; hobson's answers as targets, construction labels unused). Its late arm = continuation of (a) on that mix.
- Latency [M] Q4 (deployed 4-q set incl. 48-way needed_procedure): T=128 ctx_packed 163 ms vs w_late 23.0 (7.1x); 256: 168 vs 32.6; 400: 175 vs 40.3; 1000: 208 vs 64.8 (3.2x); 2000: 254 vs 117. w_early (n copies of the state) 54 / 77 / 119 / 251 / 487.
  BUT ctx_packed has ~70 ms of fixed overhead from the gather-form conv on 1.6k question rows (h7's small-R code). Fixed (d1 conv kernel per segment) + added ctx_seq (hobson's shared-prefix layout, state once then each question continues the cache). Re-measure the ctx arms after the pipeline.

## 12:10 latency run 1 [M] (A10G, bf16 fused runtime, CUDA graph, n=20, fresh real states; median ms; p95 within 0.1 ms except w_early Q15 T4000 +50)
- Q1p (procedure, 1014-token 23-way question): ctx_plain 56.8/57.3/61.6/83.2/103.9/147.7/246.8 at T=64/128/256/400/1000/2000/4000; w_early(n=1, merged = plain path) 9.8/15.3/22.4/27.9/52.8/103.9/200.1; w_late(sets path) 14.8/19.1/28.7/36.4/57.2/109.3/207.2.
- Q1s (states_amount, 84 tokens): ctx_plain 15.3/15.6/23.0/28.1/56.8/107.7/200.6; w_early 9.6/14.7/22.3/27.3/52.7/103.9/199.9; w_late 16.1/20.4/29.9/37.6/58.4/110.6/208.5.
- Q4: ctx_packed* 162.8/163.2/168.2/175.1/208.1/253.6/353.8; w_late 18.8/23.0/32.6/40.3/64.8/116.6/211.1; w_early 39.7/54.2/77.2/118.7/251.4/487.2/971.8.
- Q15: ctx_packed* 154.7/155.1/159.6/164.6/190.0/244.0/347.1; w_late 19.8/24.0/33.5/41.2/65.7/114.1/212.1; w_early 67/128/240/360/876/1772/3632.
  (* ctx_packed carries ~70 ms gather-conv overhead; being re-measured with the fix + ctx_seq.)
- Sets-path fixed overhead (w_late vs w_early at the same rows, n=1): ~6 ms (2nd GDN chunk call x18, masked SDPA x6, LoRA kernels). For n=1 the late model can run as a plain continuation (no branch machinery): cost ~ w_early + LoRA.
- w_early = question from layer 0: n x the state; 15 questions at T=1000 = 876 ms (13x w_late). Cost-prohibitive for multi-question.
- train_h started 11:49 (1.07 s/update; 3200 updates ~ 57 min).

## 12:50
- train_h (hypernet) at upd 2310/3200: pool agree .89 kl .07 (31 seen deployed questions), v5 (each row an unseen-before question): acc .63 vs teacher .90, agree .63, hid .21. Ends ~13:07, then train_e (45 min), eval_h, eval_e, eval_late.

## 13:20
- train_h done 12:51 (3200 upd, 61 min): last 400 upd pool agree .87 kl .07; v5 acc .73 vs teacher .90, agree .79.
- train_e (Brooker test, both arms from (a) s4723, same samples, 50% CF-edited train states) at upd 350: late kl .015-.019 vs early .020-.025 (early never below late so far); hid equal.

## 13:50 hypernet (b) results [M] (s3200; question spec -> generated delta + slot inputs; question-blind state)
- All suites: JB-all .524 (hobson .723; McNemar 19 vs 65, p 5e-7); JB-hard .423 (.523; 18 vs 31, p .085; below no-state .462); REAL agree_sd .514; LONG .618;
  REAL-label .728 (.785; p .009); CF acc .475 flip .049; CF-probe acc .503 flip .006 (chance).
- Held-out deployed questions (7 never trained: states_amount, WrapsUp, cc_insists, details_match, failure_cause, cc_refuses, UserAskedForHuman):
  REAL agree_sd .262, LONG .171, REAL-label .672 vs hobson .844 on the same items (8 vs 30, p .0005). (a)'s own per-question adapters on the same subset: .976 / 1.000 / .844.
- Seen deployed questions: REAL agree_sd .659 (vs (a) .800 same subset).
- Verdict (b): does not generalize to unseen question specs at this budget (61 min, ~9.6k train_v5 questions + 31 deployed). Clean negative: a 41k-number code + slot inputs from hobson's own no-state reading of the question is not enough to carry an arbitrary question.
- train_e done (704 updates each arm). Training KL on identical samples: early - late = +.0037 +- .0003, early worse in 66/70 windows (second half late .0162 vs early .0198).

## 14:12 Brooker test + CF-state distillation results [M]
- late = (a) + 704 updates distilling hobson on 50% CF-edited TRAIN states (targets = hobson's answers): REAL agree .930 / sd .858; LONG .924 / .873;
  REAL-label .777 (hobson .785; McNemar 11 vs 14, p .69); CF acc .624 / flip .352 / fgh .862 / dir .983 -> better than hobson on CF (52 vs 27, p .0066); SHUF .331 (.245).
  vs (a): REAL sd 19/17 p .87, LONG 9/6 p .61 (no change), CF acc 101 vs 11 p 2e-19.
- early = late + per-question adapter on the STATE rows (question-aware reading from layer 0), same samples:
  REAL .934 / .870; LONG .913 / .848; REAL-label .790; CF acc .633 / flip .367 / fgh .899 / dir .990; SHUF .349.
  early vs late (paired): REAL sd 8 vs 4 p .39; LONG sd 0 vs 4 p .125 (early worse); CF acc 8 vs 1 p .039 (early better); REAL-label 6 vs 1 p .125.
  Training KL early worse in 66/70 windows (+.0037).
- CF gains are in the augmented edit kinds: human_insert .596 (late) / .615 (early) vs hobson .346; wrapup .959 / .980 vs .918; amount_insert .033 / .056 vs .100 (below hobson); identity, procedure_intent, insists stay 0 (hobson 0 too). State >= 4k: 0 (hobson .013).
- Brier/ECE: REAL-label hobson .347/.116, a .360/.101, late .365/.084, early .366/.104; CF hobson .521/.111, late .541/.116, early .538/.126; hyper REAL-label .394, JB-all .601 (hobson .348).
- ctx latency re-run (fixed conv) in progress: Q1p T64 ctx_packed 61.3 (was 100.6), ctx_seq 62.1; Q1s T64 16.9 / 19.1.

## 15:00 final additions [M]
- Fixed in-context baselines (lat_ctx.json) and 1-question late on the plain causal path (w_late1, lat_w1.json); merged in lat_final.json, projections proj_final.json.
  [V] w_late1 runtime vs j6lib: 60/60, max dp .0063; fixed ctx_packed 59/60 (one near-boundary flip, max dp .009).
- Procedure capacity probe (train_p.py, 30 min on procedure requests only, from (a) s4723, same samples): r8 control REAL sd .760 / LONG .776; r8+r32 extra rank REAL .771 / LONG .755 (a: .771 / .735). Extra rank worse on training KL in 32/32 windows. -> not capacity-limited; 87-93% of misses are hobson near-ties (top-2 margin < .2).
- DRAFT_REPORT.md written (<= 1500 words).

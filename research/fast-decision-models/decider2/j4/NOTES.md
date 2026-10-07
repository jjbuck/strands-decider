# J4 NOTES (decision pretraining)
09:00 start. Read BRIEF8, FAST_DECISION_MODEL, F7_REPORT, evalkit README. Box j4 launched 08:56:57, waiting for ready.
09:15 Read Brooker posts + ELECTRA (arXiv 2003.10555, PDF pages 5-9 read). Verified ELECTRA numbers:
  Table 1: ELECTRA-Small 14M, 1.4e18 FLOPs, GLUE 79.9 vs BERT-Small 75.1 (same FLOPs) vs GPT 78.8 (4.0e19, ~29x FLOPs).
  ELECTRA-Base 85.1 vs BERT-Base 82.2 (same 6.4e19). Table 2: ELECTRA-400K 89.0 @7.1e20 vs RoBERTa-500K 88.9 @3.2e21 (4.5x), XLNet 89.1 @3.9e21 (5.4x).
  Table 5 (the key ablation): ELECTRA 85.0, All-Tokens MLM 84.3, Replace MLM 82.4, ELECTRA-15% 82.4, BERT 82.2.
  => most of ELECTRA's gain is from a loss over ALL tokens (density), only ~0.7 GLUE from being discriminative. Gains shrink with size (6 pts @ hidden 128 -> ~3 @ 768).
  First-principles consequence: NTP is already an all-tokens dense objective, so ELECTRA's density argument does NOT separate RTD from NTP.
  What is sparse for a decision model is the DECISION supervision: hobson's FT = ~1 label (~1-2 bits) per ~370 tokens.
  Hypothesis to test: make the decision-interface supervision dense and self-supervised: many verifiable decisions per state,
  generated from the state's own text (replaced-span detection asked AS a decision at the pointer head), packed as branches off one state pass.
  Controls: (a) FT only; (c) same compute as more FT; (d) NTP on the same states at equal compute (isolates objective from data/domain exposure).
09:55 Built on box j4: j4lib.py (forest forward: packed varlen roots + child branches from root GDN state/conv tail; verified forest == plain
  per-sequence forward and == hobson refs within bf16 noise on 6 CF items), gen_dp.py (DP corpus: 5000 train-split states (agent 80%, generated
  docs 20%), 197,879 questions: verify 72.8k, lookup 53.6k, order 21.5k, count 20.9k, numcmp 17.7k, who 10.7k, datecmp 0.8k), mk_ft.py (40k FT rows,
  F7 mix, 5.5% real KL rows), teach.py (hobson calibrated targets, 26k rows, ~10k tok/s), train.py (ft/dp/ntp), evalj4.py, evq.py, drive.sh.
  Throughput (A10G, LoRA r16, ckpt): 3.2k tok/s fwd+bwd (no-ckpt 4.3k at 4k tok, 16.9 GB); fwd-only 14k tok/s.
  Plan (equal compute in tokens through the model): DP 4.4M tok; FT 12k rows (~3.5M tok); (c) = (a) continued to a+DP tokens; (d) NTP 4.4M tok on
  the same DP states then same FT. Constant LR after 30-update warmup in every FT so checkpoints r1k..r12k are sample-efficiency points.
  Density arithmetic: FT gives 1 decision label / ~292 tokens (measured on rows); DP gives 40 labels / ~4.1k tokens = 1 / ~100 tokens (2.9x);
  ceiling ~1 / 55 tokens (the question length), so decision-label density can rise at most ~5x per FLOP in this format.
09:50 drive.sh launched (DP phase first; teacher still finishing concurrently).
09:47 (clock fix: earlier 09:50/09:55 stamps were early estimates; laptop clock is truth) teacher labels done: 26,000 FT rows in 1032 s.
  Harness check: hobson weights through my forest eval (evalj4 --hobson): JB-all .727 / JB-hard .531, agree .996 / .992, tv .003 / .004
  (refs merged_full: .731/.531, agree .992 on JB-hard) => eval path at the bf16 noise floor.
  score_j4.py on hobson refs reproduces README numbers (CF .594/.268, CF-probe .653/.328, REAL-label .785, kinds table).
  DP learning fast: at update 50 (0.67M tok): lookup .96, verify .93, who .93, order .76, count .60, numcmp .58 (train-batch acc).
  evq.py eval queue running (waits for checkpoints).
10:30 DP phase done (16:40->17:13 box UTC, 33 min): 4.40M tokens, 323 updates, 995 states, ~37k questions. Final train-batch acc:
  verify .99, lookup 1.00, who 1.00, numcmp .97, order .93, datecmp ~.9, count .71-.79 (count hardest).
  Arm (a) FT started 17:13 UTC. FT runs at ~1.4k tok/s while an eval shares the GPU (dp_only zero-shot eval running).
  Eval queue OOM'd once when a third GPU process started (train 9.7 GB + eval 10.9 GB); evq now waits while any eval runs. Training unaffected.
  Eval now merges every LoRA into bf16 weights before scoring (as merged_full).
10:58 dp_only (DP phase alone, its verification head, no task FT): JB-all .446, JB-hard .385, REAL-label .573, CF acc .504 / flip .113,
  CF-probe acc .658 / flip .316 (hobson .653/.328): zero-shot near transfer only on id_match 1.00 (hobson .54) and status_equal .94; amount/date ~0.
  a_r1000 (FT only, 1k rows): JB-all .550, JB-hard .377, REAL-label .725, CF .589/.276, CF-probe .586/.188, SHUF .277.
  Bug fixes in the queue launcher (pgrep self-match; pkill -f killed my own ssh shell twice; now launched via start_evq.sh).
11:30 a_r4000: JB-all .628, REAL-label .735, CF .613/.325, CF-probe .606/.219.
  a_r12000 (FT only, 12k rows = 4.15M tok): JB-all .688 (McNemar vs hobson 13/21 p .23), JB-hard .492 (13/17 p .58), REAL-label .740 (10/28 p .005),
  REAL agree_sd .844, LONG agree_sd .897, CF .670/.434 (flip > hobson, 75/8), CF-probe .645/.300 (32/41 p .35), SHUF both_right .411.
  CF kinds: amount .50, human .545, wrapup .94, identity 0, insists 0, procedure 0. CF-probe: amount ~0, date .14/.06, id .76/.33, status .89/.27.
  b (DP + FT) at 7.7k rows, train-batch acc .815 vs a's .81 at 11.5k.
11:50 b_r1000 vs a_r1000 (paired McNemar b-only/a-only): JB-all .636 vs .550 (32/12 p .004); JB-hard .415 vs .377 (16/11 p .44);
  REAL-label .680 vs .725 (21/39 p .027, b WORSE); CF acc .669 vs .589, flip .438 vs .276 (pairs 100/34 p<1e-6);
  CF-probe acc .745 vs .586, flip .494 vs .188 (105/7). Far transfer on CF: human_insert .686 vs .288, wrapup .49 vs .29; identity/procedure/insists still 0.
  Near transfer on CF-probe: amount_vs_limit .77 vs .07, status_distract .66 vs .36, id_match 1.0 vs .16; date_order 0 (DP had only 793 datecmp q).
12:15 b_r4000 vs a_r4000: JB-all .671 vs .628 (17/7 p .064), JB-hard .469 vs .400 (14/5 p .064), REAL-label .748 vs .735 (35/30 p .62),
  CF flip .409 vs .325 (58/24 p .0002), CF-probe flip .550 vs .219 (107/1). b@4k ~= a@12k on JB-all/JB-hard/REAL-label.
  b ended 18:48 UTC, c (equal compute: a continued to 8.56M tok) running; then ntp -> d. drive2.sh queued after: bf (DP torso + FRESH head, 4k rows),
  a2/b2 (seed 1 repeats at 4k rows) to size run-to-run noise on JB.
12:25 b_r12000 vs a_r12000 (paired): JB-all .671 vs .688 (11/15 p .56), JB-hard .454 vs .492 (8/13 p .38), REAL-label .753 vs .740 (25/20 p .55),
  CF flip .463 vs .434 (43/31 p .20), CF-probe flip .391 vs .300 (46/17 p .0003). LONG agree_sd .873 vs .897.
  => DP's head start CONVERGES AWAY by 12k rows under the F7 recipe; only CF-probe keeps a (smaller) gain. b's CF-probe flip fell .494 (1k) -> .550 (4k)
  -> .391 (12k); amount_vs_limit .77/.79 -> .05: the fine-tune (CE gold + KL to hobson, who scores .33 there) un-teaches the DP skill.
  Re-planned: d (NTP control) at 4k rows only (the regime where DP differs); then bf (fresh head), a2/b2 (seed repeats) at 4k. drive3.sh.
12:30 c done: a continued to 24,416 rows = 8.56M tokens (= a 4.15M + DP 4.40M). c_final eval + ntp (19:28 UTC) running.
  Added arm (e): DP-MIXED fine-tune (b's start + one DP state of <=40 questions per FT update, DP states 995+ unseen in the DP phase), 12k rows,
  to test whether keeping the decision-pretraining objective alive stops the erosion. Smoke-tested ntp and mix modes. drive4.sh order:
  ntp -> d(4k) -> e(12k) -> bf(4k, fresh head) -> a2, b2 (seed 1, 4k).
13:05 c_final (a continued to 24.4k rows = b's total compute): JB-all .727 (= hobson; 12/11), JB-hard .546 (12/9), REAL-label .753 (11/24 p .041),
  REAL agree_sd .896, LONG agree_sd .873, CF .636/.367, CF-probe .681/.363, SHUF .354.
  b vs c (equal compute, paired): JB-all 10/23 p .035 (b worse), JB-hard 9/21 p .043 (b worse), REAL-label 22/22, CF pairs 59/20 p<1e-5 (b better),
  CF-probe pairs 28/19 p .24. More ordinary FT LOWERS CF flip (a .434 -> c .367): the KL-to-hobson recipe pulls toward hobson's conservatism.
  => at equal compute DP buys CF detail reading and costs ~5 JB points; it is a trade, not a dominance.
13:10 compute-matched pair: b@4k (5.78M tok incl. DP) vs c@16k (5.50M tok): JB-all .671 vs .710 (18/27 p .23), JB-hard .469 vs .523 (17/24 p .35),
  REAL-label .748 vs .773 (12/22 p .12), CF flip .409 vs .365 (56/38 p .079), CF-probe flip .550 vs .363 (82/22 p<1e-8).
  c@16k REAL-label .7725 (hobson .785, 11/16 p .44) but c@24.4k .7525: REAL-label noise ~+-2 pts (n 400).
13:38 NTP control. ntp phase: 4.40M tok on the same DP states, nll 1.64 -> 0.68. d_r1000 (NTP-merged torso, fresh head, same FT):
  JB-all .619, JB-hard .431, REAL-label .543 (!), CF .548/.192, CF-probe .545/.094, SHUF .184.
  b_r1000 vs d_r1000 (same states, same tokens, objective differs): JB-all 27/23 p .67 (tie), REAL-label 93/38 p<1e-6, CF pairs 130/30, CF-probe 129/1.
  d vs a @1k: JB-all better (34/18 p .037), REAL-label much worse (20/93), CF worse (36/70), CF-probe worse (0/30).
  => the JB-all gain at small FT is generic to any intermediate LoRA phase; the detail-reading gain is specific to the decision objective,
  and generative adaptation to agent states HURTS real-traffic decisions at small fine-tune budgets.
13:55 d_r4000: JB-all .654, JB-hard .438, REAL-label .693 (vs a 19/36 p .03 worse; vs b 11/33 p .001 worse), CF .668/.431 (vs a 57/14 better; vs b 41/32 tie),
  CF-probe .661/.322 (vs b 8/81 worse). => at 4k rows the CF (far-transfer) gain is shared with NTP on the same agent states (domain exposure);
  the CF-probe gain is specific to the decision objective; NTP costs real-traffic accuracy (REAL-label), DP does not.
14:20 e (DP mixed into every FT update): e_r1000 ~= b_r1000 (all paired p > .04). e_r4000: JB-all .645, JB-hard .415, REAL-label .710 (vs b@4k 13/28 p .028 worse),
  CF flip .347 (vs b 25/50 p .005 worse; wrapup_replace .18), CF-probe flip .559 (= b .550). Keeping DP alive during FT keeps the probe skill but
  costs real-traffic accuracy and CF at 4k. Seed repeats a2/b2 dropped for time (drive5: e -> bf only).
14:50 e_r12000 (DP mixed, 12k rows): CF acc .730 / flip .559 (best arm; vs b 52/13, vs c 85/7), SHUF .545 (best), CF-probe .691/.384 (= b),
  JB-all .684, JB-hard .485 (n.s. vs hobson), REAL-label .715 (vs hobson 13/41 p .0002; vs b 13/28 p .028; vs c 18/33 p .049), LONG agree_sd .788 (others .87-.90).
  => keeping DP alive buys state sensitivity (CF 2.1x hobson) at a cost on real traffic; still erodes the probe skills (amount_vs_limit .77 -> .30).
  bf (fresh head) trained 21:27-21:43 UTC; b2 (seed 1) training; evals queued. results.json + score_all.txt written.
15:00 bf_r4000 (DP-merged torso, FRESH head, same FT): JB-all .680, JB-hard .469, REAL-label .745, CF .682/.463, CF-probe .791/.584, SHUF .439.
  vs b_r4000 (carried head): JB 5/3, REAL-label 12/13, CF pairs 31/9 p .0007 (bf better), CF-probe 24/13 p .10. => the gain is in the torso, not the head.
  vs a_r4000: JB-all 17/5 p .017, JB-hard 13/4 p .049, REAL-label 34/30, CF 71/15, CF-probe 119/2.
  bf@4k ~= a@12k on JB-all (.680/.688), JB-hard (.469/.492), REAL-label (.745/.740); CF .463 vs .434; CF-probe .584 vs .300.
15:10 DRAFT_REPORT.md written (seeds + latency placeholders). b2 (seed 1, DP->FT 4k) done 22:00 UTC; a2 running; evals bf_r1000, b2_r4000, a2_r4000 queued.
  Latency plan: j4lat.py (copy of H7's harness, plain hobson layout, CUDA graph, n=20, fresh real states) on the b model once the GPU is idle.
15:25 bf_r1000: JB-all .589, JB-hard .431, REAL-label .635, CF flip .352, CF-probe .481 (fresh head lags b at 1k, catches up by 4k).
15:35 b2_r4000 (seed 1 of b@4k): JB-all .649 (b .671), JB-hard .431 (.469), REAL-label .703 (.748; seed-vs-seed McNemar 16/34 p .015!), CF flip .370 (.409), CF-probe .516 (.550), SHUF .357 (.401). Seed noise is large on REAL-label.
15:58 Latency [M] (b model = base + DP + b LoRA merged; H7/H4 TTL fused runtime, plain hobson layout: state pass then each question as a cached
  branch; CUDA graph, exclusive GPU, fresh real banking states, n=20): Q1 64/128/256/400/1000/4000 = 19.0/19.4/25.8/37.4/63.1/212.1 ms;
  Q4 = 48.6/48.9/55.6/67.3/93.4/243.9 ms; p95 within 0.13 ms. Same harness measured hobson plain 64.0 ms (T1000 Q1) in H7 -> same cost class.
  Seeds: a2@4k CF flip .458 vs a@4k .325 (seed-vs-seed 71/17!) -> CF flip is seed-noisy; CF dmean (threshold-free) at 4k: a .233/.305, b .250/.243
  -> no robust CF gain at 4k. CF-probe dmean a .183/.205 vs b .376/.315 -> robust. Third seeds (a3/b3 @4k) + a2/b2 @1k evals running.
16:05 seeds @1k: a {CF flip .276/.261, CF-probe .188/.250, JB-all .550/.597, REAL-label .725/.720, SHUF .276/.227}; b {CF .438/.448, CF-probe .494/.475, JB-all .636/.615, REAL-label .680/.700, SHUF .475/.468}. => at 1k the CF far-transfer gain is robust across seeds; at 4k it is within seed noise (a .325/.458, b .409/.370). a3/b3 (seed 2) @4k evals pending.
16:27 3-seed @4k: JB-all a .632 [.619-.649] b .657 [.649-.671]; JB-hard .410 vs .444; REAL-label .725 vs .720; CF flip .347 [.259-.458] vs .365 [.315-.409]; CF-probe flip .264 vs .483 [.384-.550]; CF dmean .235 vs .251; CF-probe dmean .176 vs .345. All evals done; seeds.json, results.json, curves.png written.
16:35 DRAFT_REPORT.md final (<1500 words). Verdict: DP is a robust near-transfer detail-reading lever (CF-probe ~2x at <=4k rows, 3 seeds) and a
  strong head start at 1k rows (CF, SHUF, JB-all); not robust on CF / REAL-label / JB beyond 1k; loses to equal-compute ordinary FT on JB;
  eroded by the KL-to-hobson fine-tune. NTP control: same JB gain, no probe gain, hurts REAL-label. No arm meets the bar (REAL-label >= .78).

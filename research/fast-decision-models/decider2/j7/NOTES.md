# J7 notes: reader-native input representation for hobson (box j7, g5.8xlarge)

## 09:08 PDT start
- Read BRIEF8, FAST_DECISION_MODEL.md (full), F7_REPORT, evalkit README, Brooker's two posts (fetched), h7 report/notes/code (train_h7.py, gen_cf.py, h3lib.py).
- Box j7 launched 08:57, not ready yet (waiting for boxes/j7.ready).
- Key facts that shape the plan:
  - States are rendered TEXT (markdown docs, "key: value" tool results, JSON tool results, "user:/assistant:" turns), not raw JSON.
    Example REAL state: 7000 chars / 1883 Qwen tokens = 3.7 chars/token. REAL-agree median state 2054 tokens.
  - CF identity pairs: customer says "my phone number is 617-555-0934", record says "phone_number: 617-555-0834". hobson 0/51 identity pairs;
    h7's CF-augmented fine-tune also 0 on identity and procedure (its augmentation had no identity kinds). Equality of two distant digit
    strings is the capability hobson lacks.
  - h7: augmentation with the CF-probe generator itself lifts CF-probe to .977 (in-distribution). So CF-probe gains from training on that
    generator say nothing about representation. My augmentation must use other templates/formats, and the arms must share it.
- First-principles points:
  - A reader has no LM head, so vocabulary size costs nothing in FLOPs (embedding = gather). A generator pays V*d per output token.
    So the reader can carry a huge domain vocabulary and arbitrary deterministic per-token side channels (value hashes, field paths) for free.
  - Cost is linear in tokens (GEMMs 46 of 52.7 ms at 1000 tokens), so a k-fold token cut is ~k-fold latency at these lengths.
  - 18 of 24 layers are GDN (no positional encoding beyond order); only 6 attention layers use RoPE. "Structure-aware positions" can act
    only through RoPE in 6 layers or through additive coordinates; I use additive coordinates and keep RoPE on original token positions.
- Plan (arms share data/recipe; deltas are attributable to representation):
  1. CPU: token counts of every suite with Qwen; train a token-level super-BPE over Qwen ids on train_pool (train split only), digits atomic;
     measure reduction on REAL/LONG/CF/CF-probe/JB states and question texts.
  2. C  = hobson + LoRA r16 + head, KL(hobson) + gold CE + own detail augmentation (different templates from CF-probe; no identity kinds).
  3. V  = C + zero-initialised side channels: value-identity hash, field key, record id, role/turn (b + c).
  4. TV = V on super-tokens; new-token embeddings by token distillation (dense hidden alignment at aligned positions) inside the same run.
  5. Every suite, absolute accuracy, McNemar vs hobson; latency at exact lengths in the fused runtime (token cut = latency cut).

## 09:25 box j7 ready (09:19); (a) token counts MEASURED
- Super-BPE over Qwen ids (code/superbpe.py): trained on 12,784 train-split texts (train_pool states + the 37 distinct deployment question
  texts), 34.0M Qwen tokens, 114k distinct mergeable segments, digits and newline tokens never merged; 65,536 merges in 36 s (CPU).
  Truncations to the first 4,096 / 16,384 merges are prefix subsets (same ids), so one embedding table serves all three levels.
- State-token reduction (Qwen tokens / super tokens, whole suite; median per item in brackets) and question-token reduction:

  | vocab added | REAL-agree state | LONG state | CF state | CF-probe state | JB-all state | REAL questions | REAL state+q |
  |---|---|---|---|---|---|---|---|
  | 4k  | 1.60 (1.62) | 1.59 | 1.52 | 1.50 | 1.02 | 1.23 | 1.51 |
  | 16k | 1.97 (1.95) | 2.06 | 1.85 | 1.71 | 1.03 | 1.31 | 1.80 |
  | 64k | 2.41 (2.41) | 2.67 | 2.14 | 1.91 | 1.04 | 1.59 | 2.19 |
- So the deployment's own vocabulary cuts real gate states 1.6-2.7x; out-of-domain JevBench is untouched (1.02-1.04x), as expected.
- Embedding table cost: 64k x 2048 bf16 = 268 MB, a gather; no FLOPs (no LM head).
- Plan for T: one model trained with the vocabulary level drawn per example from {4k, 16k, 64k} (prefix-nested), so one run gives the
  accuracy-vs-compression curve.

## 09:55 code built and verified (box j7)
- code/chan.py (b+c side channels: canonical value identity hash, Abacus-style digit place, field key, record instance, role x section, turn
  recency); host cost 2.97 ms per 1000 Qwen tokens in Python regex (a Rust port would be far less; not measured).
- code/superbpe.py (a); code/j7lib.py (J7 = H7 forward + extended vocab + channels + original-position RoPE); code/gen_aug.py (own detail
  augmentation, 4000 pairs: stated_match / lookup_value / above_amount / before_date, half with a distractor record; JSON, key:value or
  markdown-table records; no evalkit template, field, tool name or question wording; no CF identity kind).
- VERIFIED (test_j7.py, 40 REAL items, 160 q): J7 forward with Qwen ids vs hobson refs: argmax 160/160, max|dp| median .0015 (max .012);
  zero-init channels: identical. Untrained transplant (mean-of-constituents init) agreement: 4k .919, 16k .900, 64k .350.
- Host costs per 1000 Qwen tokens (Python): HF tokenize 2.15 ms, super-merge 1.21 ms, channel annotate 2.97 ms.
- CPU check (cputest.py): super-token option rows map to the same original tokens; constituents reproduce the Qwen ids exactly.
- Training C (train_j7.py) try 1 (lr 1.5e-4, aug 46% of sequences, w_aug 1): kl_real rose to .18, train agree .71 by update 40 -> killed.
  try 2 (lr 1e-4, head lr 2e-4, p_real .6 / p_aug .2 / p_v5 .2, w_aug .5): kl_real .02-.11, agree .76-.94 (noisy) by update 40; kept.
  ~7 s/update (8 sequences, ~2.1k tokens each, teacher in-process), 400 updates ~ 47 min.
- pipe.sh queued: lat -> train T -> eval C -> train V -> eval T64k -> train TV (init T) -> eval V -> eval TV64k -> eval T16k.

## 10:30 C trained; latency curve MEASURED (res/lat_j7.json)
- C: 400 updates, 43 min, final train-window kl_real .01-.03, agree_real ~.95+, aug acc ~.9-1.0 (hobson ~.55-.75 on the same aug items).
- Latency (A10G, fused TTL runtime, hobson weights, plain layout = state pass with cache + each question as a branch, CUDA graph, fresh
  real banking states, 20 timed reps after 3 warm-ups; p95 within 0.1 ms of the median everywhere). Super-token rows = Qwen rows / r with
  r = REAL median compression (16k: 1.95, 64k: 2.41); questions merged with the same vocabulary.

  | Qwen state tokens | 64 | 128 | 256 | 400 | 1000 | 4000 |
  |---|---|---|---|---|---|---|
  | hobson Q1 | 19.05 | 19.37 | 25.86 | 37.40 | 62.99 | 211.52 |
  | + channels (V) | 19.12 | 19.47 | 25.94 | 37.51 | 63.31 | 212.83 |
  | super 16k Q1 | 18.65 | 19.60 | 24.65 | 25.57 | 47.08 | 118.45 |
  | super 64k Q1 | 17.68 | 18.10 | 18.41 | 24.46 | 36.99 | 98.41 |
  | super 64k + channels Q1 | 17.75 | 18.16 | 18.47 | 24.54 | 37.10 | 99.01 |
  | hobson Q4 | 48.60 | 48.94 | 55.64 | 67.31 | 93.25 | 243.30 |
  | super 64k Q4 | 44.80 | 45.22 | 45.55 | 51.63 | 64.24 | 126.37 |
  | super 64k + channels Q4 | 44.96 | 45.37 | 45.70 | 51.80 | 64.44 | 127.06 |
- Speedups at equal content: 1000 tokens 1.70x (Q1) / 1.45x (Q4); 4000 tokens 2.15x / 1.93x. Channels cost 0.1-0.3 ms at 1000 (<=0.6%).
- Below ~256 tokens the layout's floor dominates (two weight streams: state pass + question branch, ~9 ms each); there tokens buy little.
  Single-pass Q1 layout (state+question in one sequence) queued in pipe2 (SINGLE=1).
- T (super-token transplant) training started 10:27: 4.3 s/update, student rows 1064 vs teacher 2341 (2.2x); hid relMSE .36 -> .28 by update 20.

## 11:16 C scored (MEASURED, all 3227 questions; preds/C.json, results/score_C.json)
- C (hobson + LoRA, F7-style KL + gold + my out-of-template detail augmentation, Qwen tokens, no channels), vs hobson:
  JB-all .719 (h .723, McNemar p 1.0) | JB-hard .523 (= h) | REAL-label .782 (h .785) | REAL agree .946 / sd .908 | LONG .964 / .939 |
  CF acc .607 (h .594) flip .310 (h .268) fgh .973 | CF-probe acc .952 (h .653) flip .903 (h .328) fgh .981 | SHUF both .287 (h .245).
- CF-probe by kind (C vs h): amount .88/.33, amount_d .84/.24, date .78/.06, date_d .75/.03, id 1.0/.54, id_d 1.0/.38, status 1.0/.89, status_d .98/.25.
  => augmentation with OTHER templates, field names, tools, record formats and question wording transfers to the CF-probe suite almost fully.
  JSON-record detail reading is a training-data problem, not a representation problem.
- CF identity kinds: C 1/51 pairs (h 0/51). Both move P(true) the right way (dir .90 / .84) but stay above .5 on mismatches
  (mean P(true) on mismatch items .74 C, .77 h). stated_match training (single field, my wording) did not transfer to details_match /
  identity_established (2+ details). This is the open test for the value channel (V).
- Eval speed: 2 shards in parallel, 12 min for all suites.

## 12:13 T (super-token transplant, 64k level) scored (MEASURED; preds/T64k.json, results/score_T64k.json)
- T: LoRA + head + super-token embeddings (mean + shared per-position gains + per-token delta), 400 updates, KL + dense token distillation
  (relMSE vs hobson rows at aligned original positions, layers 3/7/11/15/19/23), vocab level per example 4k .2 / 16k .3 / 64k .5,
  real 80% / v5 20%. Train-window hid relMSE .36 -> .12-.14, agree ~.9 at the end (plateauing).
- At 64k (2.2-2.4x fewer tokens): JB-all .710 (h .723, p .65) | JB-hard .538 | REAL-label .730 (h .785, McNemar p .0007: FAIL) |
  REAL agree .896 / sd .801 | LONG .932 / .836 | CF acc .551 flip .200 fgh .514 | CF-probe acc .620 flip .244 fgh .505 | SHUF .176.
  CF human_insert .026 (h .346); CF-probe distractor kinds collapse (id_d .02, status_d .05).
  agree_sd .80 sits between the rand25@7 (.73) and rand50@7 (.87) baselines: as if half the state were dropped after layer 7.
- Verdict for (a) at 64k with this budget: the 2.4x token cut does NOT come at equal accuracy. Queued: T at 16k and 4k (same model),
  T learning curve (s100/s200/s300 on REAL), zero-shot transplant at 16k.

## 12:50 V scored (MEASURED; preds/V.json, results/score_CV.json); paired V vs C (code/pair_cmp.py)
- V = C + zero-init channels (value identity, digit place, key, record, role/section, turn), same data/recipe/seed.
  JB-all .723 | JB-hard .531 | REAL-label .795 (h .785) | REAL agree .944 / sd .908 | LONG .949 / .915 | CF acc .622 flip .335 fgh .991 |
  CF-probe acc .933 flip .866 fgh .962 | SHUF .310.
- Paired V vs C (discordant pairs V-only / C-only): CF pairs 14/4 (p .031; identity 3/0, amount 5/1, human 6/3);
  CF-probe 6/18 (p .023; ALL from date_order 3/15, p .0075); REAL-label 6/1 (p .125); JB-all 2/1; REAL agree_sd 5/5; LONG agree_sd 0/4 (p .125).
- CF identity: V 4/51 pairs (C 1/51, h 0/51). The identity channel does not turn identity edits into a solved primitive at this budget.
- Reading: the channel helps a little where equality matters (CF, p .03) and hurts date ORDER (identity hashes of dates carry no order;
  place coordinates apply to numbers only). Not a capability change.
- Queued held-out aug pairs (SEED 99, 400 pairs) for C, V, hobson: does C already solve in-distribution digit matching without channels?

## 12:56 TV (T s400 + channels + aug, 400 more updates, levels 16k .3 / 64k .7) scored at 64k (MEASURED; preds/TV64k.json)
- JB-all .697 (p .31) | JB-hard .500 (p .66) | REAL-label .750 (h .785, p .020 FAIL) | REAL agree .903 / sd .806 | LONG .928 / .842 |
  CF acc .575 flip .249 fgh .486 (human_insert .03 vs h .35) | CF-probe acc .817 flip .638 fgh .914 | SHUF .217.
- Paired TV64k vs V (same channels+aug, Qwen tokens): CF 40/75 (p .001; human_insert 0/64, amount 39/2), CF-probe 7/80 (amount 1/34,
  date 5/44), REAL-label 7/25 (p .002), REAL agree_sd 9/44, LONG agree_sd 4/16.
- So at 64k the super-token input costs real accuracy even after 800 updates of distillation + aug: numeric comparisons (digits are not
  merged, but their context is), human-request detection in user turns, and real-traffic labels.

## 13:03 T at 16k scored; single-pass latency MEASURED (results/lat_single.json)
- T at 16k (1.97x on REAL states): JB-all .693 (p .09) | JB-hard .492 (p .34) | REAL-label .772 (p .50) | REAL agree .910 / sd .809 |
  LONG .928 / .861 | CF acc .548 fgh .560 (human_insert .06) | CF-probe acc .639 fgh .581 (distract kinds collapse). Barely better than
  64k: the loss is the transplant at this budget, not the compression level.
- Single-pass layout (state + 1 question in one sequence; the engine's 1-question path), A10G, median/p95 ms:

  | Qwen state tokens | 64 | 128 | 256 | 400 | 1000 | 4000 |
  |---|---|---|---|---|---|---|
  | hobson | 15.24/15.25 | 15.56/15.57 | 22.84/22.85 | 28.14/28.15 | 57.06/57.07 | 201.17/201.21 |
  | + channels | 15.29 | 15.60 | 22.90 | 28.22 | 57.35 | 202.47 |
  | super 16k | 9.33 | 14.70 | 15.51 | 22.33 | 37.36 | 108.46 |
  | super 64k | 9.53 | 9.74 | 15.24 | 15.53 | 27.82 | 92.50 |
  | super 64k + channels | 9.58 | 9.79 | 15.28 | 15.58 | 27.89 | 93.06 |
  64k: 1.60x at 64, 1.81x at 400, 2.05x at 1000, 2.17x at 4000. hobson at 1000 = 57.06 matches the FAST_DECISION_MODEL anchor (57.0).

## 13:12 held-out detail pairs (MEASURED; results/score_aug.json): exactness is not a representation bottleneck
- 400 held-out pairs from gen_aug (SEED 99, train-split states, values never seen), both items right:

  | kind | hobson | C (Qwen tokens, no channels) | V (+ channels) |
  |---|---|---|---|
  | above_amount / +distractor | .40 / .08 | .98 / 1.00 | 1.00 / .98 |
  | before_date (question date in another format) / +d | .18 / .08 | .98 / .98 | .98 / 1.00 |
  | lookup_value / +d | .62 / .34 | 1.00 / 1.00 | 1.00 / .98 |
  | stated_match / +d | .58 / .30 | .98 / .76 | .98 / .80 |
  | all | .323 | .960 | .965 |
- 400 LoRA updates on digit-level Qwen tokens take hobson from .32 to .96 on exact-value reading, including cross-format date comparison
  and binding under distractors. Canonical value codes + digit place + field/record coordinates add .005. The 0% identity / procedure CF
  kinds are not value-encoding failures (C and V both move P(true) the right way but keep a strong 'true' prior on details_match).

## 13:25 T at 4k (1.60x) scored; fused-runtime check of TV is OFF (investigating)
- T at 4k: JB-all .710 (p .45) | JB-hard .508 (p .69) | REAL-label .777 (p .70) | REAL agree .922 / sd .844 | LONG .938 / .897 |
  CF acc .583 flip .264 fgh .706 | CF-probe acc .642 flip .306 fgh .676. Absolute accuracy within ~1-3 points of hobson everywhere, but
  not the same function (agree_sd .84) and marginally below the bar (REAL-label .777 < .78, CF .583 < .594, CF-probe .642 < .653).
- Same model, compression vs agree_sd: 1.60x .844 | 1.97x .809 | 2.41x .801 (REAL); LONG .897 / .861 / .836.
- lat_j7 check (TV ckpt merged into the fused runtime vs j7lib eval preds, 24 CF questions): argmax 23/24 but max|dp| median .34 -> the
  runtime path is not reproducing the trained TV function exactly. Queued checks on C (no super, no channels) and T (super only) to
  localise. Latency numbers do not depend on this (same kernels, same shapes); the accuracy numbers come from j7lib.

## 13:35 runtime check localised; value-channel ablation (MEASURED)
- lat_j7 check: C (no super, no channels) fused runtime vs j7lib: 24/24 argmax, max|dp| median .0009 (VERIFIED). T (super only): dp .35.
  Cause: super ids start at len(tokenizer) = 248,077 but the fused table was cat(embed[248,320 padded rows], super) -> wrong rows.
  Fixed (embed[:V]); re-check of T and TV queued (pipe4). Timing is unaffected (same gather, same shapes). j7lib (train/eval) was right.
- V with the value-identity channel zeroed at inference (CF, CF-probe): CF acc .622 -> .619, identity 4/51 -> 4/51, CF-probe .933 -> .938.
  The trained V model does not use the value-identity code at all. Clean negative for the identity channel at this budget.

## 14:08 T2 (structure-aware transplant: user/assistant message lines stay Qwen, 64k elsewhere; 1.90x on REAL) scored (MEASURED)
- JB-all .688 (p .12) | JB-hard .515 | REAL-label .733 (h .785, p .001 FAIL) | REAL agree .889 / sd .778 | LONG .930 / .861 |
  CF acc .544 fgh .468 (human_insert .00) | CF-probe acc .619 fgh .562. Paired vs T64k: no significant difference anywhere.
- human_insert diagnosis (mean P(true) on the no-request / request item; dir): hobson .265/.425 (.99), C .272/.455 (.99),
  T4k .258/.374 (1.0), T64k .262/.318 (.97), T2 .269/.287 (.83). The transplanted models still move the right way, but the signal is
  attenuated, and MORE so when the message lines themselves are left at Qwen granularity. The damage is to how the merged context is
  read (GDN state / attention over imperfect super-token rows), not to the merged words themselves.
- Queued pipe5: continue T for 700 more updates at 64k only (budget test), then full eval.

## 14:20 runtime path VERIFIED; zero-shot transplant; learning curve (MEASURED)
- After the table-offset fix, fused runtime vs j7lib: T 24/24, max|dp| median .0005 (max .0032); TV 24/24, median .0004 (max .0037);
  C 24/24, median .0009. The latency path computes the trained functions.
- Zero-shot transplant (16k, mean-of-constituents init, no training): JB-all .727 | JB-hard .561 | REAL-label .705 (p .0001) |
  REAL agree .802 / sd .679 | LONG .756 / .430 | CF fgh .596 | CF-probe fgh .486.
- T at 64k, REAL agree_sd by update (cosine lr, 400 updates): s100 .668 | s200 .772 | s300 .795 | s400 .801 (tv .123 -> .084).
  Flattening at the decayed lr. pipe5 continues T for 700 updates at a fresh lr, 64k only (started 14:17).

## 14:55 status
- TL (T s400 + 700 updates at 64k only, fresh cosine) at update 510: train hid relMSE .112 (T s400: .12-.14). Eval ~15:15.
- DRAFT_REPORT.md written with all measured results so far; TL row pending.
- Results/preds/logs copied to ~/decider2/j7/{results,preds,box_logs}.

## 15:17 TL (T + 700 updates at 64k; ~1100 updates, ~20M teacher tokens total) scored (MEASURED; results/score_TL.json)
- JB-all .706 (p .45) | JB-hard .515 (p 1.0) | REAL-label .767 (p .25) | REAL agree .922 / sd .841 | LONG .932 / .855 |
  CF acc .568 flip .234 fgh .679 (human_insert .167) | CF-probe acc .619 flip .263 fgh .591 | Brier REAL-label .372.
- Paired TL vs T64k: REAL-label 23/8 (p .011), REAL agree_sd 26/12 (p .034), CF 26/12 (p .034; human_insert 22/0).
- So budget helps significantly: agree_sd .668 / .772 / .795 / .801 / .841 at 100 / 200 / 300 / 400 / 1100 updates. Still below the bar
  (REAL-label .767 < .78, CF .568 < .594, CF-probe .619 < .653).
- pipe6: one more budget point (TL + 700 updates = ~1800), started 15:17, eval ~16:15.

## 16:18 TL2 (1,800 updates, ~33M teacher tokens) scored; final
- TL2 at 64k: JB-all .688 (p .10) | JB-hard .492 (p .42) | REAL-label .765 (p .20) | REAL agree .921 / sd .832 | LONG .936 / .885 |
  CF acc .573 flip .244 fgh .752 (human_insert .244) | CF-probe acc .628 flip .266 fgh .591 | Brier .359.
- Paired TL2 vs TL: REAL agree_sd 13/16 (p .71), REAL-label 10/11, CF 20/16; LONG agree_sd 7/2. REAL agreement has plateaued at ~.84
  with LoRA r16 + per-token embeddings; LONG and CF fgh still creep up.
- Final verdict in DRAFT_REPORT.md. All GPU work finished 16:12; box j7 idle (auto-terminates ~18:58).
- Artifacts: code/ (superbpe, chan, gen_aug, j7lib, train_j7, eval_j7, lat_j7, score_j7, pair_cmp, score_aug, traffic_lat, pipe*.sh),
  results/ (token stats, lat_j7.json, lat_single.json, traffic_lat_64k.json, score_*.json, check*.json, protstat.json, score_aug.json),
  preds/ (all model predictions), box_logs/ (training logs). Checkpoints stay on the box (ck_*), per the no-models-on-laptop rule.

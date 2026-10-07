# H7 notes (box g4): compiled question schema made accuracy-preserving for the full 24-layer hobson-v19 (no shrinking)

## 00:46 PDT start
- Read BRIEF7, H4 REPORT/NOTES, H2 REPORT, H6 NOTES/code (h6eval.py: layout [question minus final '<answer>'][state][one '<answer>' slot], options read from the compiled prefix rows).
- g4 idle; ~/venv has fla 0.5.2 / torch 2.14 / transformers 5.18. Pushed train_pool.jsonl, train_v5.jsonl, H3's h3lib.py (differentiable lean hobson + LoRA) to g4.

## 01:00 layout (matches H6 exactly when the bundle holds one question)
- main sequence = [Q1' Q2' .. Qn'][state] (Qk' = question k's rendered tokens minus its final '<answer>' token; bundle in spec order, causal, compiled once),
  then n one-token branches = the '<answer>' slot of each question, all at position P+Ls.
- each slot starts from the main sequence's final GDN state + conv tail (slots never see each other); at attention layers slot k attends ONLY to its own
  question's bundle span + all state rows + itself (per-question mask over the bundle). Pointer-head options = bundle rows (state-independent, compiled).
- code: ~/decider2/h7/code/h7lib.py (H7 = H3 + branched forward), test_h7.py.
- VERIFIED (test_h7.py, 3 REAL items x 4 q): teacher branch layout vs plain state-first forward: logit maxdiff <= 0.017, argmax 12/12;
  schema n=1 slot-branch vs plain [Q'][state][<answer>] sequence: maxdiff <= 0.008; LoRA grads through the branch path vs plain: cos >= 0.9997 (192 tensors).
- untrained 4-question bundle vs teacher argmax: 1/12 on the cc_* questions (near-50/50 logits) -> must train.

## 01:05 data + training
- CF-style augmentation (code/gen_cf.py, train-split states only, 0 eval tasks): 2440 pairs = F0 CF-probe generator (8 kinds x 180, copied from the g4 replica
  of f0/build_cf.py) + human_insert 450 + amount_insert 300 + wrapup_replace 250 (own templates).
- train_h7.py: LoRA r16/a32 on Win/Wo/Wgu/Wd of all 24 layers + trainable pointer head (copy of hobson's). Teacher = frozen bf16 hobson state-first in-process.
  mix: real 0.70 (bundle = full qset .45 / random 2-4 subset in random order .30 / single question .25; KL), cf 0.20 (1.0 CE gold + 0.3 KL),
  v5 0.10 (0.3 CE gold + 1.0 KL). Option-order permutation p .5 (choice, noul), teacher canonical, matched by label name. maxtok 6144 (state head 1/4 + tail 3/4).
  AdamW(.9,.95) lr 1.5e-4 / head 5e-4, warmup 30, cosine to 1500 updates (floor 0.1), accum 4, ckpt every 300 updates.
- 01:08 launched (~7.5 s/update, avg T ~3000). 01:10 untrained baseline eval (bundle layout) launched concurrently.

## 01:20
- 01:16 restarted training from the update-50 resume point with the cosine horizon cut to 1050 updates (~8.5-10 s/update with an eval sharing
  the GPU; 1500 would not finish before the 05:43 shutdown). Checkpoints s300/s600/s900/s1050 (LoRA + head only, ~45 MB... see ck/).
- evq.sh on g4 scores each checkpoint (bundle layout, all 2421 items / 3227 questions) as it appears.
- code/h7lat.py: H4's TTL runtime + H7 slot rows (GDN one-step branch per slot from the state's final state, masked slot attention, head option keys
  precomputed at compile), and hobson's plain layout (state cached once, each question a branch) in the same harness.

## 01:35
- MEASURED untrained hobson in the schema layout (bundle = item's questions, h7lib, preds/base_bundle.json): REAL agree .400 / agree_sd .350,
  LONG agree .261 / sd .206, JB-hard .362 (hobson .523, McNemar p .01), CF fgh .009, CF-probe fgh .000, REAL-label .412. Worse than no-state
  (REAL agree .68): state-first-trained hobson does not work at all with the question in front. The layout must be trained (H4 measured .76 for this-that,
  which was trained on both layouts).
- VERIFIED runtime (h7lat.py check, TTL fused + slot rows) vs h7lib on 12 multi-question items (44 questions): argmax 43/44 (one near-tie),
  max|dp| median .0018, max .0082 = bf16 runtime noise.

## 01:50 layout change: per-question slot SETS (state-aware option rows)
- Single-slot run (layout v1 = H6's): train KL/agreement plateaued from update ~130 to 330 (train agree vs teacher ~.76, CF CE stuck at ln2 = .69,
  i.e. CF pairs answered identically). H6 independently saw no learning in the same layout (their 01:17 diagnosis: option rows computed BEFORE the
  state give the pointer head fixed option keys; all state evidence must squeeze into one decision row). Kept ck/s300.pt (single-slot) and scoring it.
- Layout v2 'sets' (h7lib.sets_inputs): bundle unchanged [Q1' .. Qn'] (compiled once); after the state, each question gets a slot SET =
  [its option-end tokens (the rows hobson's pointer head reads), '<answer>'] re-emitted at positions Tm.., causal within the set, attending to its own
  question's bundle span + all state rows (per-question mask); GDN = branch from the state's final GDN state (conv history = last 3 state rows), sets
  never see each other (varlen chunk, one initial state per set). Pointer head reads options from the set rows (state-aware, as hobson was trained).
  Cost per request = state + sum_k (K_k + 1) rows (noul: 3 rows/question; 15 noul questions = 45 rows) instead of state + full question text.
- VERIFIED (test2_h7.py): sets n=1 vs plain sequence [Q'][state][option-end tokens][<answer>]: logit maxdiff <= .020; LoRA grads cos >= .9994.
  Untrained sets layout vs teacher argmax 7/10 on 4 items (single-slot: 1/12).
- 01:49 launched sets training: fresh LoRA r16 + hobson head, lr 2e-4, cosine 900 updates, ckpt every 300 -> ck_sets/. evq2.sh scores each ckpt (layout sets).

## 02:12 single-slot s300 scored (MEASURED, preds/s300_bundle.json, all suites)
| | hobson | untrained schema (bundle) | single-slot s300 |
|---|---|---|---|
| REAL agree / agree_sd | 1 / 1 | .400 / .350 | .746 / .393 |
| LONG agree / agree_sd | 1 / 1 | .261 / .206 | .762 / .424 |
| CF fgh / CF-probe fgh | 1 / 1 | .009 / .000 | .413 / .000 |
| JB-hard (McNemar p) | .523 | .362 (.01) | .346 (.001) |
| REAL-label | .785 | .412 | .675 |
- The single-slot model learned the no-state prior (plain agreement up, agree_sd barely moved; CF-probe flip 0, wrapup_replace 1.0 = proposed
  message is right next to the slot). It does not read the state. Confirms H6's diagnosis.
- 02:13 sets run resumed from update 150 with a dense distillation term: relative MSE between the slot-set rows (option-end rows + '<answer>')
  and hobson's own state-first rows for the same tokens at layer outputs 5/11/17/23 (unpermuted questions only), weight 1.0 (H3's --hid recipe).
  Initial hid ~0.6-0.86.

## 02:55 sets s300 scored (MEASURED; 150 updates KL-only + 150 updates KL + hidden distillation)
| | hobson | untrained | single-slot s300 | sets s300 |
|---|---|---|---|---|
| REAL agree / agree_sd | 1 / 1 | .400 / .350 | .746 / .393 | .743 / .616 |
| LONG agree / agree_sd | 1 / 1 | .261 / .206 | .762 / .424 | .777 / .642 |
| CF acc / flip / fgh | .594 / .268 / 1 | .440 / .015 / .009 | .512 / .123 / .413 | .754 / .606 / .835 |
| CF-probe acc / flip / fgh | .653 / .328 / 1 | .502 / .003 / 0 | .500 / 0 / 0 | .505 / .009 / .010 |
| JB-hard (McNemar p) | .523 | .362 (.01) | .346 (.001) | .431 (.16) |
| SHUF both_right | .245 | .031 | .116 | .589 |
| REAL-label | .785 | .412 | .675 | .647 |
- The sets layout + dense distillation reads the state: CF flip .606 (hobson .268; amount_insert .87 vs .10, human_insert .76 vs .35), but CF-probe
  is at chance (P(true) .51 +- .04 on both items of every pair: the deep JSON record is not read yet).
- VERIFIED runtime (h7lat.py checksets: TTL fused, LoRA merged into bf16 weights, slot sets via varlen GDN + masked SDPA) vs h7lib (unmerged
  LoRA) on 18 items / 50 questions: argmax 50/50, max|dp| median .0013, max .0050.
- copied ck_sets/s300.pt -> ~/decider2/h7/ckpt/h7_sets_s300.pt (fallback for H6).

## 03:32 sets s600 scored (MEASURED, all suites, preds/sets_s600.json)
- REAL agree .821 / agree_sd .647; LONG .860 / .770; CF acc .784, flip .667 (hobson .268), fgh .972; CF-probe acc .944, flip .887 (hobson .328),
  fgh .914; JB-hard .423 (hobson .523, McNemar p .09); SHUF both_right .664 (.245); REAL-label .720 (.785).
- CF-probe by kind (model / hobson): amount_vs_limit .84/.33, +distract .74/.24, date_order .86/.06, +distract .72/.03, id_match 1.0/.54,
  +distract .98/.38, status_equal .94/.89, +distract 1.0/.25.  CF: amount_insert 1.0/.10, human_insert .84/.35, wrapup .96/.92, identity 0/0, procedure 0/0.
- CAVEAT: the CF-probe augmentation uses the same generator code as the eval suite (different, train-split states and fresh random records), and the
  CF augmentation covers human/amount/wrapup with my own templates. CF / CF-probe gains are therefore partly in-distribution for the templates;
  the clean real-traffic metric is REAL/LONG agree_sd, which is far from the .95 bar.
- copied ck_sets/s600.pt -> ~/decider2/h7/ckpt/h7_sets_s600.pt.

## 04:00 sets s900 (final, best) scored (MEASURED, all suites, preds/sets_s900.json)
- REAL agree .881 / agree_sd .685 (REAL decision flips vs hobson 11.9%); LONG .917 / .836; CF acc .817, flip .732 (hobson .268), fgh .991;
  CF-probe acc .977, flip .953 (hobson .328), fgh .981; JB-hard .408 (hobson .523, McNemar p .04: FLAG, -11.5 points); JB-long agree .403;
  SHUF both_right .726 (.245); REAL-label .785 (= hobson .785; of the 46 REAL-label questions where it disagrees with hobson, each is right on 22).
- Co-design bar: CF fgh PASS, CF-probe fgh PASS, REAL agree_sd FAIL (.685), LONG agree_sd FAIL (.836), JB-hard flagged.
- Where the state-dependent disagreement is: the two 23-way procedure choices (needed_procedure / procedure, ~1000-1200-token questions).
  agree_sd excluding them: REAL .796 (n 250), LONG .948 (n 116); on them: REAL .396 (n 96), LONG .571 (n 49). Trend s300->s600->s900 still rising
  (REAL sd .616/.647/.685, LONG .642/.770/.836).
- JB-hard by family (model/hobson correct): policy 5/9, long_policy 4/7, adversarial 1/3, trap 5/7, ambiguous 3/5, tradeoff 2/4, temporal_numeric 6/3.

## 04:00 latency (MEASURED, A10G g4 exclusive, bf16, TTL fused runtime + LoRA merged, CUDA graph, fresh states, n=20, median / p95 ms) lat_h7.json
| T | Q | sets (H7 layout) | single slot | plain (hobson layout, state cached once, each question a branch) | bundle tokens, compile once |
|---|---|---|---|---|---|
| 1000 | 1 | 56.71 / 56.72 | 56.14 | 64.01 / 64.02 | 92, 22.5 ms |
| 1000 | 4 | 57.81 / 57.82 | 57.13 | 94.49 / 94.51 | 360, 27.3 ms |
| 1000 | 15 | 66.06 / 66.07 | 61.16 | 226.33 / 226.36 | 1718, 97.2 ms |
| 4000 | 1 | 210.02 / 210.05 | 209.63 | 215.15 / 215.33 | 92 |
| 4000 | 4 | 211.93 / 211.98 | 211.24 | 247.14 / 247.19 | 360 |
| 4000 | 15 | 219.85 / 219.92 | 218.54 | 385.59 / 385.78 | 1718 |
- sets rows per request: state + 3 (Q1) / 12 (Q4) / 46 (Q15).

## H6 HANDOFF (best checkpoint)
- ckpt: ~/decider2/h7/ckpt/h7_sets_s900.pt (LoRA + head only, 60 MB; also s300/s600). Keys: lora.{i}.{Win,Wo,Wgu,Wd}.{A,B} (r16, scale
  _meta.lora_scale = 2.0, delta W = scale * B @ A in the lean fused row layout: GDN Win rows = [in_proj_qkv 6144 | in_proj_z 2048 | in_proj_b 16 |
  in_proj_a 16], attention Win rows = [q_proj 4096 | k_proj 512 | v_proj 512], Wgu rows = [gate_proj | up_proj]); head.{norm,q,k}.* = pointer head
  (hobson's PointerHead shape). Base = hobson-v19 with its own LoRA merged (plib.P merge_and_unload). Loader: h3lib.H3.load_trainable, or
  h7lat.merge_lora (merges into the HF torso) - both in ~/decider2/h7/code/.
- LAYOUT (NOT H6's single-slot layout; that one did not train, see 01:50 / 02:12): per request ids = [state tokens] + for each question k
  [q_k[o] for o in opt_k] + [q_k[-1]]  (option-end tokens of the rendered question, then '<answer>'); bundle (compiled once) = concat of q_k[:-1].
  Positions: bundle 0..P-1, state P..P+Ls-1, each slot set restarts at P+Ls (P+Ls+t for its t-th row). Slot set k: causal within itself,
  attends to bundle span of q_k + all state rows; GDN: varlen branch from the state's final GDN state (conv history = last 3 state rows).
  Head: decision = set k's last row, options = set k's first K rows (all final-normed). Temperature by kind as hobson.
  Reference implementations: h7lib.H7.sets_inputs / sets_logits (training, unmerged LoRA) and h7lat.RTS.fwd_sets + sets_meta (fused runtime).

## 04:06
- checksets with s900 (runtime vs h7lib): argmax 50/50, max|dp| median .0009, max .0059 (logs/latcheck900.log).
- 04:00 continued training from update 900 (resume, cosine horizon 1400, ckpt every 100) while sets1_s900 (one question per sequence) is scored.
- evq4.sh on g4: at 04:48 PDT stops training and scores the newest checkpoint (sets layout, all suites) -> preds/sets_sNNNN.json.
- DRAFT_REPORT.md written (s900 numbers).

## 04:18 sets1 (one question per sequence = per-question compile) vs bundle, s900 (MEASURED)
- sets1 scored on all multi-question items (REAL-agree 174, LONG 142; 1122 questions); single-question items are the identical computation
  (verified: 684/684 same decisions), filled from sets_s900 -> preds/sets1_s900.json.
- Decisions bundle vs one-question-per-sequence on multi-question items: same on .965 (H4: this-that shared schema prefix .786).
  sets1: REAL agree_sd .682 (bundle .685), LONG .812 (.836), REAL-label .787. Bundling the questions does not cost accuracy.
- sets1 eval was stopped after the LONG items to free the GPU for the continued training (CF/CF-probe/JB are single-question items).

## 04:47
- Continued training 900 -> 1300 (cosine horizon 1400, lr 7e-5 -> 2.2e-5), stopped at 04:45 (s1000..s1300 saved on g4 ck_sets/), scoring s1300
  (sets layout, all suites, exclusive GPU). ckpt copied: ~/decider2/h7/ckpt/h7_sets_s1300.pt.
- H6 (g1) adopted the slot-set layout in their deployed kernels and measured latency (their NOTES 02:55 / 04:18): W8A8-GPTQ b8 sets 33.6 ms (1q)
  / 40.6 ms (15q) at T=1000; untrained sets layout there: REAL flips 56.8%, agree_sd .197.

## 04:59 s1300 scored (MEASURED, preds/sets_s1300.json): plateau
- REAL agree .883 / agree_sd .691; LONG .921 / .818; CF fgh .991, flip .741; CF-probe fgh 1.000, acc .995, flip .991; JB-hard .423 (p .09);
  SHUF .729; REAL-label .757; REAL flips 11.7%. Excl. procedure: REAL .716, LONG .948; procedure: REAL .625, LONG .510.
- 900 -> 1300 at decayed lr: real-traffic agreement flat (+.006 REAL, -.018 LONG) with churn between questions => plateau for this recipe.
- DELIVERED / best for H6: ~/decider2/h7/ckpt/h7_sets_s900.pt (best LONG sd + REAL-label; s1300 kept as h7_sets_s1300.pt). Layout: see H6 HANDOFF above.
- Final report: ~/decider2/h7/DRAFT_REPORT.md.

# Hobson: generation-by-generation history

Source material for a report on how the model developed. Each entry records what
changed, why, what was measured, and what was decided, and the files it comes from. The
entries are dated; the development history they were written from is private. Numbers
are copied from saved results (`research/data/`, built by `research/scripts/collect.py`)
or, where no raw result survives, from the README or a pre-registration; the source is
named in each case. Figures are in `research/figures/`
(`research/scripts/make_figures.py`).

Project span: 20-28 September 2026. Hardware throughout: one RTX 3090
(24 GiB); Windows until v12, WSL2 from v13.

## At a glance

JevBench is the public set of 231 tasks (48 easy, 72 standard, 111 hard). "Default"
means the run was adopted as the recipe going forward.

| gen | date | parent | torso | change | JevBench | hard | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1 | 09-20 | - | Qwen3-1.7B-Base | slot head, 7 classification tasks | not run | | default |
| v2 | 09-20 | v1 | Qwen3-4B-Base | 14 tasks, 3 primitives | not run | | default |
| v3 | 09-20 | v2 | Qwen3-1.7B-Base | register rubric moved into training | not run | | default |
| v4 | 09-21 | v3 | Qwen3-1.7B-Base | instruction variants, noul 4 -> 8 types | 0.628 (145)* | | default |
| v5 | 09-21 | v4 | Qwen3-1.7B-Base | RuleTaker; 3,072-token window | 0.636 (147) | 44 | default |
| v6 | 09-22 | v5 | Qwen3-1.7B-Base | head seeded from LM + KL anchor | 0.649 (150) | 44 | default |
| v7 | 09-22 | v6 | Qwen3-1.7B-Base | pointer readout | 0.667 (154) | 46 | default |
| v8 | 09-23 | v7 | Qwen3-1.7B (instruct) | instruct torso | 0.658 (152) | 48 | experiment |
| v9 | 09-23 | v7 | Qwen3-1.7B-Base | +20k generated rule-execution docs | 0.658 (152) | 42 | experiment |
| v10 | 09-23 | v7 | Qwen3-1.7B-Base | +20k generated minimal pairs | 0.628 (145) | 38 | experiment |
| v11a | 09-24 | v7 | Qwen3-1.7B-Base | question transforms (30% of corpus) | 0.619 (143) | 41 | experiment |
| v11b | 09-24 | v7 | Qwen3-1.7B-Base | KL on changed questions | 0.623 (144) | 40 | experiment |
| v12 | 09-24 | v7 | Qwen3-1.7B-Base | distil frozen Qwen3.5-4B | 0.632 (146) | 43 | experiment |
| v13 | 09-25 | v7 | Qwen3.5-2B-Base | new torso | 0.680 (157) | 49 | default |
| v14 | 09-25 | v13 | Qwen3.5-2B-Base | +12,909 multi-step rows, 4B teacher | 0.697 (161) | 52 | default |
| v15 | 09-25 | v14 | Qwen3.5-2B-Base | +4,566 ShARC/ConditionalQA rows | 0.693 (160) | 50 | experiment |
| v16 | 09-26 | v14 | Qwen3.5-2B-Base | +2,148 generated document questions | 0.706 (163) | 55 | default |
| v17 | 09-27 | v16 | Qwen3.5-2B-Base | multi-step rows replayed toward v14's own answers | 0.710 (164) | 55 | default |
| v18 | 09-27 | v17 | Qwen3.5-2B-Base | +1,667 generated questions on five weak skills | 0.710 (164) | 56 | default |
| v19 | 09-28 | v18 | Qwen3.5-2B-Base | +6,166 answer-adequacy rows (HelpSteer2, generated) | 0.723 (167) | 56 | default |

\* v4 was measured at the old 1,024-token window; no raw results survive (the README in
the private development history, "0.628, 36th of 53").

Reference systems on the same 231 tasks (`runs.csv`, kind `reference`): decider-2b v10
0.701 (162), decider-2b v11 0.758 (175); frozen models read through the chat template
(SemIf method): Qwen3-1.7B 0.558, Qwen3.5-2B 0.658, Qwen3-8B 0.684, Qwen3.5-4B 0.805.

---

## Phase 1 (v1-v4): building the interface, judged on an internal held-out set

The model: a pretrained LM torso plus LoRA, read at an `<answer>` position by a
"slot head" (a linear map from that hidden state to 24 option slots). Three answer
types — `noul` (yes/no), `choice` (one of N), `score` (rubric level). No external
benchmark until v4; progress was judged on whole tasks held out of training.

### v1 — 2026-09-20
- **What:** Qwen3-1.7B-Base, slot head, 7 training tasks, 2 held out (choice only);
  40,620 training rows, 66 min.
- **Also:** calibration leakage fixed in the eval sampling.

### v2 — 2026-09-20
- **What:** nine new recipes doubled the corpus (72,620 rows, 14 tasks, 4 held out, all
  three primitives), on a Qwen3-4B torso; 3.8 h. Per-primitive temperature calibration
  fitted against ECE.
- **Finding:** held-out `score` failed (`formality`, 0.271, +0.021 over chance; fitted
  temperature saturated at 6.00).
- **Also concluded, later reversed:** that torso size is not the lever (see
  *Scaling test* below).

### v3 — 2026-09-20
- **What:** back to 1.7B; swapped `formality` (a register rubric) into training and
  held out `hate_severity`. One hypothesis: v2's score failure came from every training
  rubric being a valence judgement.
- **Result:** `formality` 0.613 in-distribution, so the task is learnable; held-out
  score 0.379 (+0.129 over chance), score ECE 0.117 -> 0.004, temperature 6.00 -> 4.34.
  The accuracy gain is confounded (different held-out task); the calibration gain is not.
- **Also:** head-capacity probe — a linear readout is enough; score failure
  located as mostly readout, partly adaptation.

### v4 — 2026-09-21
- **What:** 5-6 instruction phrasings for 13 tasks (sampled per epoch); `noul` from
  four judgement types to eight (vitaminc, paws, wnli, pubmed_qa), 18,620 -> 35,449 rows.
- **Result:** no held-out change (probe set 0.733 -> 0.725); `subjectivity` down. In-
  distribution ECE 0.099 -> 0.068; score temperature 4.34 -> 2.16.
- **First external benchmark:** JevBench 0.628, 36th of 53 (1,024-token window). Showed
  1.000 on fact/routing/tool_selection and 0.167 `tradeoff`, 0.278 `multi_hop`, 0.316
  `long_policy`: the model had never been asked to compose anything.
- **Finding recorded then:** five interventions (torso capacity, head capacity, data
  volume, noul breadth, instruction diversity) failed to move held-out accuracy. Two
  were later reinterpreted (below).

## Phase 2 (v5-v7): external benchmark, readout changes

### v5 — 2026-09-21/22
- **What:** RuleTaker (apply a stated ruleset; graded depth 0/1/2 trained, 3/5 and
  NatLang held out).
- **Result on RuleTaker:** depth-3 (never trained) +0.173, NatLang +0.195.
- **Result on JevBench:** 0.6277 at 1,024 tokens, identical to v4 to four decimals;
  `long_policy` unchanged. ECE doubled (0.068 -> 0.135) because RuleTaker dominated the
  calibration set and pulled `noul`'s temperature below 1.
- **Two fixes on the way:** (1) the state was tokenised before the question and could
  leave the question 8 tokens — all 19 `long_policy` items sat at exactly 1,032 tokens;
  reserving the question first recovered ECE 0.115 -> 0.079, +1 task (`jb_fix`). (2)
  Window sweep 1,024 / 3,072 / 4,096: 0.632 / 0.636 / 0.636, `long_policy` 0.263 ->
  0.368, 14 other families bit-identical; 3,072 made the default.
- **Lesson:** "RuleTaker teaches the task, not the capability" — in-family transfer.
  First of three generated-data negatives (v5, v9, v10).

### Scaling test — 2026-09-22
- **What:** two head-ablation checkpoints on the same corpus and steps, torso only
  different: Qwen3-1.7B 0.641 (148) vs Qwen3-4B 0.684 (158), +0.043, hard 0.396 -> 0.450
  (`jb_head17`, `jb_head4b`). McNemar p = 0.174.
- **Why it matters:** the internal held-out set had ranked the 4B *below* the 1.7B
  (0.845 vs 0.850). It is dominated by saturated classification tasks and does not rank
  models. Consequences adopted: benchmark externally before concluding; treat any
  difference under +0.043 (10 tasks) on 231 tasks as unresolved.

### v6 — 2026-09-22
- **What:** three arms on the v5 corpus, then two combined:
  instruct torso 0.619 (143); `head_init: lm_head` (seed slots from the LM's output rows
  for digits 1-9) 0.641 (148); `kl_frozen_weight: 0.3` (KL toward the frozen torso's
  option-number readout) 0.641 (148). v6 = lm_head + KL: **0.649 (150)**.
- **Findings:** both anchored arms reached ECE 0.085-0.086 with no temperature fitted
  (v5 0.092 after fitting); the KL arm improved exactly the families its frozen teacher
  was better at (`adequacy`, `long_policy`) — the first prediction in the project made
  before a run and confirmed. The instruct torso lost `policy` (0.500, the frozen base's
  level). Combined, the accuracy effects added and the calibration effects partly
  cancelled (uncalibrated ECE 0.095; 0.090 after fitting).

### v7 — 2026-09-22
- **What:** pointer readout. Option k's logit is a single attention score between the
  `<answer>` state and the hidden state at the last token of option k's line; no
  per-slot parameters, no cap on option count.
- **Why:** a frozen-feature probe on v6's hidden states: pointer head at 70k parameters
  0.720 held-out vs linear slot head 0.662 and a 2.1M-parameter MLP slot head 0.694.
- **Result:** **0.667 (154)**, ECE 0.068, Brier 0.442, paraphrase consistency 0.833 —
  every JevBench metric better than v6 (+16 / -12 tasks, p = 0.57). By family: `tradeoff`
  0.333 -> 0.667, `policy` 0.667 -> 0.833; `ambiguous` 0.429 -> 0.143, `temporal_numeric`
  0.333 -> 0.133.
- **Status:** the default for v8-v12, and again the explicitly recommended recipe
  after v8-v10 failed.

## Phase 3 (v8-v12): experiments on v7, none adopted

Pre-registration began with v9: predictions and failure modes committed
before data generation or training, outcomes appended without editing.

### v8 — 2026-09-23
- **What:** `Qwen3-1.7B` (instruct) in place of the base torso, under the pointer readout.
- **Result:** 0.658 (152), best ECE recorded to that point (0.053). The v6 instruct
  arm's losses on `policy` (-0.083), `tradeoff` (-0.167) and `routing_hard` (-0.400)
  reproduced to the task through a different readout: they are costs of instruction
  tuning. A twice-measured negative.

### v9 — 2026-09-23
- **What:** +20,000 generated rule-execution documents (thresholds, unit conversion,
  date windows, banded lookups), labels computed and re-verified at 0% disagreement.
- **Result:** 0.658 (152); targeted families 16/50 -> 16/50 (48 of 50 verdicts
  identical); synthetic eval 0.436 -> 0.972. Internal validation rose (0.835 -> 0.895)
  while JevBench fell.

### v10 — 2026-09-23
- **What:** +20,000 generated minimal pairs for cross-referencing and requirement
  checking — the same document with one decisive fact changed and the answer flipped.
- **Result:** 0.628 (145); targeted families 14/35 -> 11/35, `multi_hop` 7/18 -> 4/18.
  The model read the decisive fact and carried it to unseen vocabulary (0.901 on an
  unseen domain) but not to real documents: generated documents share one structure.
- **Lesson:** third templated-data negative (v5, v9, v10) with the same shape.
- **Also:** length-grouped batching added.

### v11a / v11b — 2026-09-24
- **Why:** the new question-sensitivity probe (`evaluation/question_sensitivity.py`) showed
  v7 gives the same answer to a changed question 94-99% of the time: each state in the
  corpus is asked one fixed question, so state and options determine the label.
- **What:** 30,135 question-transformed rows (is "X" right/wrong, the one that does NOT
  apply, thresholds, flipped yes/no wrappers). v11a replaced 30% of the corpus with
  them; v11b added them as KL-only rows toward the frozen torso's reading.
- **Result:** both lost — v11a 0.619 (143, p = 0.007 vs v7), v11b 0.623 (144,
  p = 0.013), the first losses clearly outside task-sampling noise. v11a learned
  negation lexically (7 of 13 losses have a negation word in the question; Fisher
  p = 0.007); v11b answered "yes" on 60 of 74 yes/no tasks.

### v12 — 2026-09-24
- **Teacher selection:** frozen Qwen3-8B 0.684 (failed the 0.75 bar), frozen
  Qwen3.5-4B 0.805 (passed; reproduces SemIf's published 187/231).
- **What:** v7's recipe with the KL anchor replaced by the 4B teacher's option
  distributions on 91,408 training rows.
- **Result:** held-out classification 0.653 -> **0.674**, the best of any version; JevBench
  0.632 (146, p = 0.096). The teacher is no better than v7 on short classification;
  its advantage is on long documents, which the corpus did not contain.
- **Also:** Qwen3.5 (hybrid Gated DeltaNet + attention) torso support added.

## Phase 4 (v13-v17): Qwen3.5 torso and document data

### v13 — 2026-09-25
- **What:** v7's recipe on Qwen3.5-2B-Base (1.90B; 18 of 24 layers Gated DeltaNet),
  LoRA extended to the DeltaNet projections, trained under WSL2 for the fused kernels.
  The shared-prefix cache was disabled for hybrid torsos.
- **Result:** 0.680 (157), 17 gained / 14 lost vs v7 (p = 0.72); pre-registered bar 162
  missed. Training added +5 tasks over the frozen torso's reading (152), against +25 on
  Qwen3. Probe unchanged.
- **Decision:** made the default anyway, pending a seed replicate.

### v14 — 2026-09-25
- **What:** +12,909 multi-step rows — ContractNLI (balanced per claim; the per-claim
  prior alone scored 0.679), MuSiQue (answerable/unanswerable pairs), BoardgameQA —
  each with its gold label and the 4B teacher's distribution. HotpotQA held out.
- **Result:** HotpotQA (never trained on) 0.705 -> **0.750** (+100 / -57, p = 0.0007):
  the first data change to transfer outside its own source. JevBench 0.697 (161, bar
  163), hard 49 -> 52, `multi_hop` 6 -> 9 of 18; best Brier (0.438) and paraphrase
  consistency (0.861) to that point. Top confidence band overconfident on JevBench
  (right 0.789 at a claim of 0.952).
- **Decision:** made the default; recipe became `configs/train.yaml` and
  `recipe.sh`.
- **Also:** decider-2b (Mapika, same torso) measured on our harness: v10 162, v11 175.
  Its lead is one stage of document questions, not released.

### v15 — 2026-09-25
- **What:** +4,566 rows of public rule-application data — ShARC (balanced so dialogue
  shape and last follow-up answer predict nothing) and ConditionalQA. Gold labels only.
- **Result:** ShARC 0.576 -> 0.752, ConditionalQA "depends" 0.056 -> 0.111; JevBench
  0.693 (160; 7 gained, 8 lost); MuSiQue 0.852 -> 0.828 (p = 0.0017), mostly on its
  answerable questions (0.843 -> 0.806).
- **Decision:** not adopted. Short rule data teaches short rule application.

### Data generation pilots — 2026-09-26
- **Nova Premier** (Bedrock) writing and verifying its own questions: 7 of 30
  hand-checked kept labels wrong or ill-posed; documents a median 536 words.
- **Qwen3.6-27B writer + Qwen3.5-397B-A17B verifier** (OpenRouter, Apache-2.0 models
  only): 1 of 30 wrong, median 1,066 words, 81% kept, about $7 for 50 documents.

### v16 — 2026-09-26
- **What:** +2,148 generated questions over 1,000 realistic workplace documents
  (34 domains, 22 kinds), ten skills, each kept only when both verifier answers matched
  the writer's (2,804 of 3,486). Yes/no balanced per skill; choice rows where the answer
  is also the longest option (34% vs 23% chance) thinned to chance. About $130 in API
  fees at OpenRouter list prices. Raw export committed in `data/generators/gen_v16/`.
- **Result:** generated questions from four held-out domains 0.646 -> **0.837** (+83 /
  -16, p < 0.0001; "no" answers 0.435 -> 0.823, "yes" unchanged). JevBench 0.706 (163,
  bar 167; +10 / -8), hard 52 -> 55, `trap` 8/8, `tradeoff` 4/6; best Brier (0.396) and
  macro (0.715); top confidence band now right as often as claimed. HotpotQA 0.750 ->
  0.712 (p = 0.002), MuSiQue 0.852 -> 0.827 (p = 0.0002) — the pre-registered failure.
- **Decision:** by the pre-registered rule v14 stays the leader; v16 was then made the
  default on the strength of the other results.
- **Also:** shared-prefix cache ported to the hybrid torso: 8 questions over
  a ~2,000-token state 369 ms vs 1,964 ms batched.

### v17 — 2026-09-26/27
- **What:** v16's recipe and rows, with one change: the teacher distributions on v14's
  12,909 multi-step rows are v14's own (softmax of v14's raw logits, temperature 1;
  `src/hobson/data/replay.py`) in place of the frozen Qwen3.5-4B's — decider-2b's replay
  toward the parent, KL(p_parent || p_model). One setting, fixed before training; the
  pre-registration records HotpotQA as no longer an untouched transfer test.
- **Result:** the replayed sources came back past v14 — MuSiQue 0.880 (v16 0.827, v14
  0.852; +80 / -16 against v16, p < 0.0001), ContractNLI 0.865, BoardgameQA 0.817 — and
  the generated skill stayed (0.840). HotpotQA did not recover: 0.723 (v16 0.712, v14
  0.750; p = 0.015 against v14). JevBench 0.710 (164), the highest here (+6 / -5
  against v16, p = 1.0); best Brier (0.392), macro (0.720) and ordinal MAE (0.571) of
  the Qwen3.5 runs; ECE 0.083 (v16 0.069). Held-out classification 0.641; probe 0.943.
- **Decision:** by the pre-registered rule v16 stayed the default: prediction 1 failed on
  HotpotQA, the first named failure mode ("replay protects the rows replayed and not
  the transfer they supported"). v17 was then made the default on the strength of the
  other results (`configs/train.yaml`, `configs/train-parent.yaml`, `recipe.sh`), as v13,
  v14 and v16 had been before it.

## Phase 5 (v18-v19): data aimed at named weaknesses

### v18 — 2026-09-27
- **What:** v17's recipe and rows plus 1,667 generated rows from three exports, each
  written by Qwen3.6-27B and kept when two Qwen3.5-397B-A17B answers agreed with the
  writer's: a run weighting five weak skills four to one (692 documents), the completed
  mixed-skill pilot (100) and the OpenRouter pilot, not trained on before (50). Two new
  writer rules: every fact a rule needs is given or the answer is "cannot be determined",
  and no unstated facts are assumed. The weighted run cost about $96.
- **Result:** the new held-out-domain generated eval 0.660 -> **0.761** (+32 / -7,
  p = 0.0001); the five targeted skills 0.586 -> 0.744 (+26 / -5, p = 0.0002). JevBench
  0.710 (164), level with v17 (+9 / -9, p = 1.0): hard 55 -> 56, `long_policy` 7 -> 10,
  but `temporal_numeric`, which two of the skills were aimed at, 3 -> 2. Brier 0.392 ->
  0.388, ECE 0.083 -> 0.079; macro 0.720 -> 0.711, ordinal MAE 0.571 -> 0.644. MuSiQue
  0.880 -> 0.892, BoardgameQA 0.817 -> 0.801, HotpotQA 0.723 -> 0.705. Held-out
  classification 0.641; probe 0.944.
- **Decision:** by the pre-registered rule v17 stayed the default: HotpotQA missed its
  floor by one question (676 of 959 = 0.7049 against 0.705). v18 was then made the
  default. The recorded reading: the model learned the generator's versions
  of the five skills, and the gain did not reach `temporal_numeric` or `judge_hard`.
- **Also:** in-distribution fit measured afterwards for v16-v18: 0.883, 0.886, 0.886,
  level with v13 and v14 (0.886, 0.888). A re-run of v14 reproduced its saved report
  exactly.

### v19 — 2026-09-27/28
- **What:** v18's recipe and rows plus 6,166 answer-adequacy rows: a request, a response,
  and "does the response adequately answer the request?" (yes/no; gold labels, no
  teacher). 4,866 from HelpSteer2 (CC-BY-4.0): adequate when helpfulness and correctness
  are both at least 3, inadequate when either is at most 1, the middle dropped, balanced
  and paired by prompt. 1,300 generated by the v16-v18 pipeline over twelve request
  categories, with eight named defects and five kinds of adequate answer that look
  flawed; two categories held out for the eval. Both written from the benchmark family's
  one-line description, not its items. Generation cost about $32.
- **Result:** HelpSteer2 held-out responses 0.483 -> **0.739** (+90 / -30, p < 0.0001;
  inadequate recall 0.068 -> 0.761), ahead of the frozen Qwen3.5-4B (0.645); generated
  adequacy from held-out categories 0.577 -> 0.788 (balanced). Every other evaluation
  within about 0.02 of v18, none significantly (MuSiQue 0.879, HotpotQA 0.726,
  BoardgameQA 0.810, ContractNLI 0.862, held-out classification 0.647; probe 0.939).
  JevBench **0.723 (167)**, the highest here (+14 / -11, p = 0.69): standard 60 -> 63,
  hard 56; `adequacy` 7 -> 9 of 12, `judge_hard` +3, `ordinal` +2, `temporal_numeric` +2;
  `long_policy` -4, `multi_hop` -3. Calibration moved clearly: Brier 0.388 -> **0.342**
  (lower on 147 tasks, higher on 84; sign test p < 0.0001), ECE 0.079 -> 0.052, ordinal
  MAE 0.644 -> 0.427, paraphrase consistency 0.833 -> 0.861, each the best recorded. The
  yes-bias shrank across all 74 yes/no tasks: "yes" said on 68% -> 55%, "no" recall
  0.487 -> 0.641.
- **Decision:** all four predictions held, so v19 replaces v18 as the default by its own
  rule, the first run to clear its pre-registered bar.

---

## Cross-cutting threads

- **The benchmark's resolution.** 231 tasks cannot resolve differences below ~+0.043
  (10 tasks). No generation-to-generation JevBench accuracy change in the lineage is
  significant on its own; the significant results are on larger held-out sets (HotpotQA
  for v14, generated questions for v16 and v18, adequacy for v19), the v11 losses, and
  v19's per-task Brier (lower on 147 of 231 tasks).
- **Seed noise was never measured.** Every comparison treats a run's per-task answers as
  fixed. A seed replicate is the outstanding measurement.
- **Internal validation misleads.** It ranked the 4B below the 1.7B and rose in v9 while
  JevBench fell; used only for training diagnostics and calibration.
- **Reading the question.** Every model from v7 to v19 gives the same answer to a changed
  question 0.92-0.95 of the time on held-out choice tasks (`question_sensitivity.svg`),
  except v11a (0.72), whose drop is its lexical negation shortcut, not better reading.
- **Where the gains came from.** Readout (v6, v7), torso generation (v13), document
  data with a teacher or a verifier (v14, v16), and a missing skill taught directly
  (answer adequacy, v19), which also moved calibration across the whole benchmark.
  Templated generated data (v5, v9, v10), question transforms (v11), distillation on
  short text (v12) and short public rule data (v15) did not move JevBench; generated data
  aimed at weak skills (v18) moved its own held-out eval but not JevBench.
- **Multi-step regressions from new document data.** v15 and v16, with entirely different
  data, both cost MuSiQue's answerable questions ~0.04. Replaying the multi-step rows
  toward v14's own answers (v17) restored MuSiQue, ContractNLI and BoardgameQA past v14,
  but not HotpotQA, the source none of them trained on.

## Comparability caveats for charts

- v5 is shown at the 3,072-token window (`jb_3072o`), v6 after calibration (`jb_v6c`).
- ECE is after per-primitive temperature fitting on held-out classification tasks; the
  calibration set changed with the corpus (v5's RuleTaker pulled it off, for example).
- Latency: v5-v12 served on Windows, v13 onwards under WSL2. JevBench asks one question
  per task, so its latencies never use the shared-prefix cache; `latency_vs_questions.svg`
  is v14's benchmark of that cache, not re-measured on later runs of the same
  architecture.
- Held-out short tasks: the same four tasks (emotion, hate_severity, massive_intent,
  sarcasm) for v4 and v7 onwards; v6 and v8-v10 were not measured; v1-v3 used different
  held-out sets. v5's held-out set added three RuleTaker splits: its four common tasks are
  recombined from per-task results (exact for accuracy, NLL and mean confidence; ECE not
  recoverable), and its calibration is not comparable because its temperatures were
  fitted on the RuleTaker-heavy set. No Brier score was saved for the held-out runs.
- In-distribution: a fixed 6,000-row sample of the short-task training file, mostly rows
  the model trained on, whose task set changed (v4 18 tasks, v5 onwards 21, scaling runs
  14); measured for v4, v5, v7, v13, v14 and v16-v18. v19 is not yet measured. It covers
  none of the multi-step, generated or adequacy rows.
- Multi-step sets: built for v14; v13's figures come from PREREGISTRATION-v14.md (no
  raw per-row results), v14-v19 from per-row outputs. HotpotQA stopped being an
  untouched transfer test at v17, which was designed in response to its drop.

## Data provenance (`research/data/`)

| file | contents | source |
| --- | --- | --- |
| `runs.csv` | one row per JevBench run: metrics, tier counts | `jb_*/summary.json`, `results.jsonl` |
| `jevbench_results.csv` | per run, per task: correct 0/1 | `jb_*/results.jsonl` |
| `jevbench_families.csv` | per run, per family: correct / n | derived |
| `jevbench_tasks.csv` | task id, family, tier (no task text) | JevBench public task file |
| `heldout_short.csv` | held-out classification: accuracy, ECE, NLL, mean confidence, by primitive (v4, v5 on its four common tasks, v7-v19, scaling runs) | `reports/*heldout*.json` |
| `indist.csv` | in-distribution accuracy, ECE, NLL (v4, v5, v7, v13, v14, v16-v18, scaling runs) | `reports/*indist*.json` |
| `probe_heldout_choice.csv` | question-sensitivity probe, held-out choice tasks | `reports/qsens_*.json` |
| `multistep_and_generated.csv` | multi-step sets and generated eval, per source | per-row eval outputs; v13 from PREREGISTRATION-v14.md |
| `training_history.csv` | v17's training run: loss by term every 20 steps, validation loss and accuracy | `checkpoints/hobson-2b-v17/history.json` |
| `jevbench_latency.csv` | per run, per task: latency and input tokens; warm-up request flagged | `jb_*/results.jsonl` |
| `latency_questions.csv` | v14, 1-16 questions over one state: shared prefix vs batched | `bench_prefix.py` output, entered by hand in `collect.py` |

Not saved anywhere: JevBench raw results for v1-v4; held-out results for v8-v10;
any seed replicate.

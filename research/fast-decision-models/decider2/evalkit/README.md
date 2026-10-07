# evalkit (F0, round 5): capacity-sensitive evaluation for decider compressions

Purpose: stop calling a compression "lossless" because it agrees with hobson on easy or forgiving inputs. Every suite is printed next to
hobson's own numbers and next to baselines that read *less* of the state: no state at all, all state dropped after layer k, or random k% of
state positions kept after layer k. A design is interesting only where it clearly beats those baselines. It preserves capacity only if it
also tracks the counterfactual flips (CF, CF-probe) that hobson tracks.

Status:
- `READY_v0` (12:30): split, train pool, JB-hard, JB-long, REAL-agree.
- `READY_v1`: adds LONG, CF, CF-probe, SHUF and REAL-label. All suites and **all hobson references** are final.
- All baselines are complete, on the full grid, for every suite.
- Regenerate the table at the bottom with `python3 make_table.py`.

## Quick start

```python
import sys; sys.path.insert(0, '~/decider2/evalkit')          # on a box: box.sh BOX put ~/decider2/evalkit/suites evalkit/ ; box.sh BOX put ~/decider2/evalkit/kitrun.py evalkit/ ; ...evalkit.py (train_pool.jsonl is 185 MB: only if you train)
import evalkit as EK
for suite, iid, q, state, spec in EK.all_question_items():         # every (item, question) to answer, deduplicated: 3227
    ...                                                             # run your model on (state, spec)
preds = {iid: {q: {label: prob}}}                                   # one dict for all suites is fine
EK.report(preds)                                                    # every suite: your model vs hobson vs baselines (+ CF breakdown by edit kind)
EK.score('CF', preds)                                               # metrics for one suite: {'model': {...}, 'hobson': {...}, 'nostate': {...}, ...}
EK.kill_check(preds, budget=0.10, layer=7)                          # the brief's accuracy criteria + random selection at your budget
EK.breakdown('CF-probe', preds, by='kind')                          # flip rate per edit kind / 'domain' / 'pos' (edit in first or second half of the state)
```

**Prediction format.** `preds[item_id][question_name] = {label: prob}`. Labels:
- noul: `'true'` / `'false'` (a bare float is read as P(true));
- choice: the option names;
- score: `'0'..'n-1'`.

Decisions are the argmax. Missing items count as not covered (`coverage`).

**Render inputs exactly as the references did.**
- Render as strands-decider does: `render_state(state)` + `render_question(spec)`, one question per sequence. Questions are independent in SystemOneEngine.
- Do not truncate. The references set `model.config.max_length = 16384`. The deployed window is 4096 tokens and cuts the end of a longer state; LONG states are 4-8k tokens, and 28 of the 561 REAL-agree requests also exceed 4096 with their longest question.
- On a box, `kitrun.py` gives the exact token ids:
  ```python
  from kitrun import load_P, prep_question
  P = load_P()
  pr = prep_question(P, item, q)   # pr['s'] + pr['q'], option rows pr['opt'], state length pr['q0'], pr['rq']
  ```
- Write a preds JSON on the box, `box.sh get` it, and score on the laptop. evalkit.py is pure Python plus numpy.

## Suites

| suite | n | states | ground truth | what it catches |
|---|---|---|---|---|
| JB-hard | 130 tasks | JevBench public, the 11 reading and policy families (every family except extraction, fact, intent, ordinal, routing, routing_hard, tool_selection) | JevBench labels | reasoning over the state. hobson gets 0.523 here, against 0.462 with no state |
| JB-long | 77 tasks | JevBench tasks with at least 256 state tokens (the set e1 and b1 used) | JevBench labels | long-ish reading. **Accuracy says nothing here: hobson 0.403, no-state 0.416.** Use agreement |
| JB-all | 231 | all of JevBench public | labels | reference only (hobson 0.723) |
| REAL-agree | 561 requests, 1083 questions | held-out real gate requests (eval-split tau tasks; banking 361, retail 200), state under 4000 tokens, stratified over domain, spec, hook and question set; at most 4 questions per request | agreement with hobson | real traffic. `agree_sd` removes the questions a model gets right without reading |
| LONG | 187 requests, 471 questions | held-out real requests with 4000-8000 state tokens (banking only; retail states are short) | agreement | long-context reading |
| CF | 406 pairs (802 items) | real eval-split states in 3 domains (banking 161, retail 125, airline 120) | by construction: one edit flips the registered question's answer | lost details, identities, counts, intents |
| CF-probe | 320 pairs (640 items) | real eval-split states (banking 130, retail 110, airline 80) with one synthetic tool-call-plus-JSON-result inserted in the first 60% of the conversation. Half also carry a distractor record for another id with the opposite answer | by construction | numbers, dates, statuses and ids read from deep in the context |
| SHUF | 387 pairs | CF items whose state is swapped for another eval task's state with the opposite ground truth; same question | by construction | models that ignore the state |
| REAL-label | 400 (item, question) | REAL-agree questions, balanced toward minority answers per question | Claude Opus 5 label; Sonnet 5 gives a second opinion and agrees on 368 of 400 (`label_meta`) | accuracy on real traffic, not anchored to hobson |

CF edit kinds. Each pair's `kind` field in `suites/CF.pairs.jsonl` names the kind; `detail` says exactly what changed, and `pos` gives where in the state.
- `human_insert` (156): an explicit request for a human, inserted as a new customer message at a random user turn in the first 75% of a conversation that had none. Questions: `cc_asked_for_human`, `asked_for_human`, `UserAskedForHuman`.
- `insists_3v4` (20): 3 against 4 separate inserted requests for a human. Question: `cc_insists` ("at least four separate times").
- Identity (51). Questions: `details_match`, `identity_established`.
  - `id_digit_existing`: one digit of a phone number or date of birth the customer stated is changed, so it no longer matches the record.
  - `id_digit_inserted`, `id_remove_one`, `id_add_second`: an inserted customer statement of two matching details, against one wrong digit, against only one detail; or one detail, then two.
- `procedure_intent` (40): the customer's newest message is replaced by explicit intent X against intent Y. The assistant holds neither procedure's documents. Question: `procedure`, a 23-way choice.
- `amount_insert` (90): a sentence stating a dollar amount is inserted after the first sentence of a proposed message that had none. Question: `states_amount`.
- `wrapup_replace` (49): the proposed message is replaced by a mid-work question against a closing message. Question: `WrapsUp`.
- CF-probe kinds: `amount_vs_limit`, `date_order`, `status_equal` and `id_match` (one digit), each with and without `_distract`. Question name: `probe`.

**Validation.**
- **By hand:** 48 random pairs (32 CF, 16 CF-probe), 0 errors.
- **By Opus 5, independently:** both items of 100 random CF pairs and 40 CF-probe pairs (280 items) agreed with the construction labels on 276. All 4 misses were `procedure_intent`. They exposed a bug in the held-document check, which made 33 of the 40 procedure pairs invalid. Those 33 pairs were replaced, and all 80 procedure items (the 7 survivors and the 33 replacements) now agree with Opus.
- **After the fix:** 0 known errors in 326 checked items (95% upper bound about 1%).

## Metrics

- `acc`: argmax against ground truth. `acc_hobson_same`: hobson's accuracy on the same covered questions, shown when coverage is below 1.
- `agree`: argmax against hobson.
- `tv`: mean total-variation distance to hobson's distribution. `kl`: mean KL(hobson ‖ model).
- `agree_sd` (state-dependent agreement): agreement restricted to questions where hobson with no state decides differently from hobson. Plain agreement on real traffic is inflated by questions that can be answered without reading: no state already agrees 0.68 on REAL-agree.
- CF and CF-probe:
  - `flip`: the fraction of pairs where the model gets BOTH items right, that is, it tracks the flip.
  - `flip_rel`: flip divided by hobson's flip. Kill criterion: at least 0.95.
  - `flip_given_hobson`: of the pairs hobson tracks, the fraction the model also tracks.
  - `dir`: how often P(b's answer) rises from item a to item b. This is threshold-free.
  - `dmean`: the mean of that rise. `dmean_rel`: the model's mean rise divided by hobson's.
- SHUF: `change` is how often the decision changes when the state is swapped; `both_right` is how often both decisions are right.
- Noise floor: `merged_full` is the same function in another runtime. It agrees with the deployed engine on 0.996 of REAL-agree decisions. Agreement above about 0.995 cannot be told apart from exact.

## How the references were produced (box f0, A10G)

- **hobson**: `StrandsAgents/strands-decider-2B-hobson-v19`, exactly as deployed. Stored in `refs/<suite>.hobson.jsonl`.
  - Loaded with `strands_decider.infer.load_engine`; the PEFT LoRA is not merged.
  - Same code paths as SystemOneEngine: the shared-prefix cache for multi-question requests, the batched path for a single question.
  - Probabilities are taken before the engine rounds them to 4 decimals.
  - `max_length` is raised to 16384. The result is identical to the deployed 4096 whenever the state plus the longest question fits.
- **Fidelity against the deployed service** (`refs/fidelity.jsonl`):
  - Compared 419 hobson-local gate records (1535 questions), with the engine at the deployed max_length of 4096, against the answers stored in the gate logs.
  - max |dP| = 0.0163; median 0.0018.
  - 5 argmax flips. In each, both probabilities are within 0.005 of the decision boundary.
  - The stored answers came from the laptop's MPS server (`agents/decider_server.py`, device mps), so bit-exact agreement across devices is not expected.
- **Baselines** (`refs/<suite>.base.jsonl`, per question) run hobson-v19 with the LoRA merged, through `tokens/plib.py` (Hugging Face layers). Code: `kitrun.py`.
  - `merged_full`: the full model in this runtime.
  - `nostate`: an empty state.
  - `drop@k`: every state row removed after layer k; 4 sink tokens kept.
  - `randX@k`: X% of state rows kept at random after layer k, with the sink kept and original positions.
  - `qattnX@k`: X% of state rows chosen by hobson's own question-to-state attention at layer k. This was the earlier rounds' best untrained selection.
  - Every suite has the full grid: drop@{0,3,7}, rand{10,25,50}@{0,3,7}, qattn10@{3,7}.
  - LONG and CF-probe were run in two passes (reduced grid, then the rest), merged per question in the same base file.

## What the references say about the suites (hobson-v19 itself)

- **hobson tracks few counterfactual flips.** CF: flip 0.268, dir 0.948. CF-probe: flip 0.328, dir 0.884. hobson almost always moves its probability the right way but often stays on the same side of 0.5.
- **CF flips by kind:**

  | kind | hobson flip |
  |---|---|
  | wrapup_replace | 0.92 |
  | human_insert | 0.35 |
  | amount_insert | 0.10 |
  | insists_3v4 | 0.05 |
  | identity (all four kinds) | 0.00 |
  | procedure_intent | 0.00 |

  On the identity pairs hobson says "true" on both sides, missing the wrong digit. On `procedure_intent` it mostly answers `no_procedure_needed` when the assistant already holds many documents.
- **Position matters.** hobson tracks 0.17 of the CF edits in the first half of the state and 0.34 of those in the second half.
- **CF-probe by kind:**

  | kind | hobson flip |
  |---|---|
  | status_equal | 0.89 |
  | status_equal_distract | 0.25 |
  | id_match | 0.54 |
  | id_match_distract | 0.38 |
  | amount_vs_limit (with or without distractor) | 0.24-0.33 |
  | date_order (with or without distractor) | 0.03-0.06 |

  hobson cannot compare dates or bind a field to an id when a distractor record is present.
- **The baselines on CF and CF-probe:**
  - **CF, flip_rel:**

    | baseline | flip_rel |
    |---|---|
    | drop@7 | 0.33 |
    | rand10@7 | 0.29 |
    | rand25@7 | 0.54 |
    | rand50@7 | 0.81 |
    | rand50@3 | 0.47 |
    | rand50@0 | 0.38 |

    REAL-agree agreement for the same baselines is 0.79 to 0.94, so CF exposes a loss that agreement hides.
  - **qattn10@7 scores 1.28 on CF but retains only 0.88 of hobson's own flips (`flip_given_hobson`).** Attention-guided selection keeps the salient inserted sentence and drops the context that keeps hobson conservative. A flip_rel above 1 is therefore not proof of more capacity.
  - **CF-probe's flip_rel does not track how much of the state is read.** drop@7 scores 1.09 and rand10@7 1.02, because early layers match strings lexically. The question carries the id or status, and hobson's later layers override that match. Use `flip_given_hobson` on CF-probe (drop@7 0.64, rand10@7 0.61, rand50@7 0.72, qattn10@7 0.82) and the `_distract` kinds, which defeat the lexical shortcut.
- **What this means.**
  - `flip_rel` measures how much of hobson's limited capacity a compression keeps. `flip` and `acc` measure distance from the truth.
  - Report both, plus `dir` and `dmean_rel`, which still have power where hobson does not flip.
  - Precision: hobson tracks 109 CF pairs and 105 CF-probe pairs, so `flip_given_hobson` near 0.95 has an SE of about 2 points.

## Split and train pool

- `split.json`: per domain, task ids are sorted by request count and 3 of every 10 are drawn for eval (seed 20261005).
  - banking_knowledge: 29 of 97 tasks are eval.
  - retail: 34 of 114.
  - airline: 15 of 50. Airline has no gate records; its ids are reserved, and its CF states come from eval-split airline baseline transcripts.
- `train_pool.jsonl` holds the 12,747 deduplicated real requests of train-split tasks: state, question specs and token counts.
- **Self-distillation and any training data must exclude `split.json["eval_tasks"]`.** Every suite except JevBench uses eval-split tasks only.

Request reconstruction:
- 31,013 gate records became 18,153 unique (state, question set) requests.
  - 12,159 records had no answers (the oracle arm).
  - 686 were duplicates.
  - 15 had stored answers that did not fit the current spec.
- Questions come from the trial's own spec (a `gate_spec` Python file or a wire rubric) at the record's lifecycle point.
- Every stored answer was checked to have the options of the reconstructed question.

## Caveats

- No airline gate traffic exists. The airline CF items use real airline trajectories (65 baseline transcripts), rendered with the deployed default renderers and asked the retail guard and wrap-up questions.
- LONG is banking only. REAL-agree caps states at 4000 tokens, so it does not overlap LONG.
- LONG agreement is no harder than REAL-agree: rand25@7 scores 0.915 on LONG and 0.893 on REAL-agree. The hard long-context test is CF on states of at least 4000 tokens (`EK.breakdown('CF', preds, by='len')`). There hobson tracks only 1 of 76 pairs, so long-state CF measures hobson's limits more than a compression's.
- CF edits are templated sentences inserted into real states. Some read unnaturally, such as an amount sentence in an unrelated message. The ground truth is right by construction, not by fluency. hobson reacts less to inserted amounts than to natural ones: P(true) is 0.65 on natural messages with "$" against about 0.28 after an insertion.
- JB-hard has a binomial SE of about 4.4 points, so "within 1.5 points" cannot be resolved on JB-hard alone. Use CF and agree_sd for statistical power.
- REAL-label is not a sample of the natural answer distribution. For each question, up to half of its items were drawn where the decision stored in the gate log (from hobson or clef-flash) was that question's minority answer.
- REAL-label labels come from a model (Opus 5), not from humans. On these labels, qattn10@7 (0.787) scores the same as hobson (0.785), so accuracy at n = 400 does not separate designs near the top; agreement metrics do.

## Baseline table

<!--TABLE-->
| config | JB-hard acc | JB-hard agree | JB-long acc | JB-long agree | REAL-agree agree | REAL-agree agree_sd | REAL-agree tv | LONG agree | LONG agree_sd | CF acc | CF flip | CF flip_rel | CF dir | CF dmean_rel | CF-probe acc | CF-probe flip | CF-probe flip_rel | CF-probe dir | CF-probe dmean_rel | SHUF change | SHUF both_right | REAL-label acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n | 130 | 130 | 77 | 77 | 1083 | 1083 | 1083 | 471 | 471 | 406 | 406 | 406 | 406 | 406 | 320 | 320 | 320 | 320 | 320 | 387 | 387 | 400 |
| hobson | 0.523 | 1.000 | 0.403 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 | 0.594 | 0.268 | 1.000 | 0.948 | 1.000 | 0.653 | 0.328 | 1.000 | 0.884 | 1.000 | 0.339 | 0.245 | 0.785 |
| merged_full | 0.531 | 0.992 | 0.416 | 0.987 | 0.996 | 0.997 | 0.003 | 0.996 | 0.988 | 0.599 | 0.276 | 1.028 | 0.951 | 1.000 | 0.652 | 0.328 | 1.000 | 0.878 | 1.000 | 0.351 | 0.256 | 0.785 |
| nostate | 0.462 | 0.469 | 0.416 | 0.481 | 0.681 | 0.000 | 0.244 | 0.650 | 0.000 | 0.451 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.635 |
| drop@7 | 0.477 | 0.608 | 0.455 | 0.597 | 0.789 | 0.408 | 0.148 | 0.792 | 0.418 | 0.495 | 0.089 | 0.330 | 0.842 | 0.357 | 0.678 | 0.356 | 1.086 | 0.869 | 0.929 | 0.127 | 0.093 | 0.715 |
| drop@3 | 0.454 | 0.531 | 0.416 | 0.519 | 0.681 | 0.020 | 0.242 | 0.645 | 0.000 | 0.448 | 0.000 | 0.000 | 0.401 | -0.044 | 0.502 | 0.006 | 0.019 | 0.575 | 0.024 | 0.000 | 0.000 | 0.647 |
| rand10@7 | 0.423 | 0.615 | 0.403 | 0.623 | 0.849 | 0.610 | 0.113 | 0.854 | 0.612 | 0.486 | 0.079 | 0.294 | 0.764 | 0.308 | 0.655 | 0.334 | 1.019 | 0.759 | 0.733 | 0.098 | 0.044 | 0.720 |
| rand25@7 | 0.423 | 0.692 | 0.351 | 0.701 | 0.893 | 0.731 | 0.082 | 0.915 | 0.788 | 0.525 | 0.145 | 0.541 | 0.840 | 0.552 | 0.630 | 0.281 | 0.857 | 0.747 | 0.667 | 0.168 | 0.116 | 0.743 |
| rand50@7 | 0.485 | 0.769 | 0.429 | 0.766 | 0.940 | 0.870 | 0.048 | 0.949 | 0.867 | 0.563 | 0.217 | 0.807 | 0.892 | 0.806 | 0.647 | 0.322 | 0.981 | 0.800 | 0.840 | 0.248 | 0.186 | 0.775 |
| rand10@3 | 0.454 | 0.477 | 0.416 | 0.481 | 0.749 | 0.338 | 0.187 | 0.713 | 0.248 | 0.462 | 0.044 | 0.165 | 0.574 | 0.090 | 0.530 | 0.109 | 0.333 | 0.572 | 0.097 | 0.101 | 0.036 | 0.665 |
| rand25@3 | 0.423 | 0.546 | 0.364 | 0.545 | 0.810 | 0.503 | 0.148 | 0.786 | 0.473 | 0.477 | 0.074 | 0.275 | 0.638 | 0.221 | 0.527 | 0.141 | 0.429 | 0.556 | 0.141 | 0.137 | 0.049 | 0.672 |
| rand50@3 | 0.423 | 0.623 | 0.351 | 0.636 | 0.882 | 0.714 | 0.090 | 0.909 | 0.770 | 0.512 | 0.126 | 0.468 | 0.796 | 0.472 | 0.577 | 0.216 | 0.657 | 0.684 | 0.389 | 0.173 | 0.103 | 0.762 |
| rand10@0 | 0.392 | 0.485 | 0.325 | 0.455 | 0.705 | 0.321 | 0.218 | 0.673 | 0.200 | 0.464 | 0.034 | 0.128 | 0.554 | 0.071 | 0.497 | 0.016 | 0.048 | 0.512 | 0.012 | 0.114 | 0.028 | 0.657 |
| rand25@0 | 0.431 | 0.523 | 0.364 | 0.519 | 0.761 | 0.434 | 0.179 | 0.747 | 0.388 | 0.477 | 0.071 | 0.266 | 0.599 | 0.183 | 0.520 | 0.069 | 0.210 | 0.519 | 0.043 | 0.163 | 0.067 | 0.670 |
| rand50@0 | 0.446 | 0.477 | 0.351 | 0.468 | 0.861 | 0.725 | 0.112 | 0.834 | 0.624 | 0.494 | 0.101 | 0.376 | 0.781 | 0.426 | 0.519 | 0.128 | 0.390 | 0.553 | 0.163 | 0.191 | 0.093 | 0.728 |
| qattn10@7 | 0.438 | 0.692 | 0.351 | 0.766 | 0.935 | 0.853 | 0.057 | 0.932 | 0.848 | 0.628 | 0.345 | 1.284 | 0.894 | 1.090 | 0.695 | 0.403 | 1.229 | 0.894 | 1.165 | 0.398 | 0.318 | 0.787 |
| qattn10@3 | 0.438 | 0.538 | 0.390 | 0.571 | 0.802 | 0.517 | 0.139 | 0.777 | 0.424 | 0.570 | 0.239 | 0.890 | 0.897 | 0.836 | 0.650 | 0.325 | 0.990 | 0.728 | 1.025 | 0.323 | 0.238 | 0.738 |

Rows are hobson-v19 itself and baselines run on hobson-v19. 'n' = questions (agreement suites) or pairs (CF, CF-probe, SHUF).
<!--/TABLE-->

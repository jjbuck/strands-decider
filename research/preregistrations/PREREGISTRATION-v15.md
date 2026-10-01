# Pre-registration: v15, rules applied to a user's situation

Committed before training. Same discipline as v9-v14.

## Why

decider-2b (Mapika) runs on the same torso as v14. Measured on our harness, its v10 scores
162/231, level with v14's 161 (14 tasks against 13, p = 1.0). Its v11 scores 175/231,
ahead of both (v11 against v10: 24 tasks against 11, p = 0.041; against v14: 27 against
13, p = 0.039). The one thing v11 adds over v10 is a stage of harder decisions,
including 11,356 questions over teacher-written business documents and policies. That
data is not released.

This run tests the nearest public substitute: questions whose answer turns on applying a
rule to a user's situation, including rules whose outcome depends on a condition the
user has not established. On the JevBench families this targets (`policy`, `trap`,
`long_policy`, `ambiguous`), v14 scores 24/46 and decider v11 29/46.

## What is being changed

v14 (the leader) plus `data/policy_v15.jsonl`, built by `python -m hobson.data.policy`
(`src/hobson/data/policy.py`, tested in `tests/test_policy.py`):

| source | training rows | what it is |
| --- | --- | --- |
| ShARC (CC BY-SA 3.0) | 3,414 | short rules from government websites, a user's question, situation and the follow-up questions answered so far: yes, no, or depends on an unstated fact |
| ConditionalQA | 1,152 | long gov.uk guidance pages, windowed to keep every evidence and condition sentence: yes, no, depends on a condition the scenario does not establish, or not stated in the page |

Every label comes from the dataset. The balancing is set by the shortcuts found before
training:

- **ShARC.** Its "irrelevant" rows (a question paired with an unrelated rule) all have no
  scenario and no dialogue, so they are dropped. The remaining labels follow three
  things that need no reading: whether a scenario is present, the dialogue length, and
  the last follow-up answer. The last answer alone predicts 0.534 on dev, against 0.377
  for the commonest label. Rows are capped at 12 per rule first, then balanced to equal
  label counts within each cell of those three. After balancing, no cell predicts the
  label.
- **ConditionalQA.** 68% of its yes/no answers are "yes", so yes and no are balanced.
  "Depends" (310) and "not stated" (91) are kept whole. Span answers are skipped.

The policy rows carry **gold labels only**. No teacher distribution was computed for
them. v14's teacher distributions stay on its multi-step rows, at unchanged indices.
Everything else is v14's recipe. Config: `configs/experiments/v15.yaml`.

## Baselines, measured before training

Evaluation sets (`data/policy_v15_eval.jsonl`), from splits whose rules and pages never
appear in training, at their natural label distributions:

| evaluation | n | commonest answer | v14 |
| --- | --- | --- | --- |
| ShARC (dev) | 2,132 | 0.377 | 0.576 |
| — yes / no / depends | 804 / 766 / 562 | | 0.821 / 0.492 / 0.338 |
| ConditionalQA (dev) | 155 | 0.400 | 0.477 |
| — yes / no / depends / not stated | 62 / 43 / 36 / 14 | | 0.871 / 0.302 / 0.056 / 0.357 |

v14 almost never answers that a ConditionalQA answer depends on a condition (0.056), and
leans towards "yes" on both sets.

JevBench, v14: 161/231, hard tier 52/111, ECE 0.086; target families 24/46 (`policy`
9/12, `trap` 6/8, `long_policy` 7/19, `ambiguous` 2/7). HotpotQA (held out since v14):
0.750. Held-out short tasks: 0.637.

## Predictions

1. **JevBench at least 167/231 (0.723)**, six tasks over v14, the same margin v14 was
   held to.
2. **Target families at least 28/46** (v14 24/46): `policy`, `trap`, `long_policy`,
   `ambiguous`.
3. **In-distribution**, the data teaches what it should:
   - ShARC at least 0.70, with "depends" at least 0.50;
   - ConditionalQA at least 0.60, with "depends" at least 0.30.
4. **No displacement:** held-out short tasks at least 0.617 (v14 − 0.02), and HotpotQA
   at least 0.730 (v14 − 0.02).
5. **JevBench ECE at most 0.09.**

Exploratory, no numeric prediction: the other hard families; question sensitivity on
`tests/question_sensitivity.py` (v14: same answer to a changed question 0.94 of the time).

## What would count as failure

- **3 holds, 1 and 2 do not:** the model learned these datasets but not JevBench's
  authored policies. That is the v9/v10 failure, on real rules.
- **`policy` falls below 9/12:** a conflict of conventions. JevBench's policy questions
  say to treat unproved conditions as not satisfied (answer no), whereas these sources
  offer "depends". If the model carries "depends" into questions that have no such
  option, it will lean away from the stated convention.
- **4 fails:** the new data displaced what the old corpus or the multi-step rows taught.
- **ShARC "depends" at least 0.50 while ConditionalQA "depends" stays under 0.15:** it
  learned ShARC's dialogue form of "depends", not conditions in long text.

## Which model leads afterwards

v15 becomes the leader if 1 and 4 hold; otherwise v14 stays. A seed replicate remains
outstanding (PREREGISTRATION-v11.md).

## Power

231 JevBench tasks cannot resolve differences below about +0.043 (10 tasks), so
prediction 1 is suggestive. Prediction 2 rests on 46 tasks and is weaker still. ShARC dev
(2,132 rows, paired against v14 on identical rows) carries the statistical weight.
ConditionalQA dev is small (155 rows): +0.12 there is about 2.5 standard errors.

## Outcome (added after the run)

Nothing above this section was edited after training. **By the rule fixed above, v14
stays the leader:** prediction 1 failed. v15 learned ShARC well, learned a little of
ConditionalQA, and moved JevBench not at all.

| | prediction | v14 | v15 | |
| --- | --- | --- | --- | --- |
| 1 | JevBench >= 167/231 | 161 | 160 (0.693) | FAIL |
| 2 | target families >= 28/46 | 24 | 26 | FAIL |
| 3a | ShARC >= 0.70; "depends" >= 0.50 | 0.576; 0.338 | **0.752; 0.651** | pass |
| 3b | ConditionalQA >= 0.60; "depends" >= 0.30 | 0.477; 0.056 | 0.581; 0.111 | FAIL, both |
| 4 | held-out short tasks >= 0.617; HotpotQA >= 0.730 | 0.637; 0.750 | 0.633; 0.754 | pass |
| 5 | JevBench ECE <= 0.09 | 0.086 | 0.063 | pass |

**JevBench did not move.** 7 tasks gained and 8 lost against v14 (p = 1.0); hard tier
52 -> 50. Of the target families only `ambiguous` changed (2 -> 4 of 7); `policy`,
`trap` and `long_policy` have the same counts. Elsewhere
`temporal_numeric` fell 4 -> 2 and `adequacy` rose 7 -> 8. `policy` did not fall below
9/12, so there is no sign of the feared conflict between "depends" and JevBench's
"unproved conditions are not satisfied".

**One named failure mode occurred: ShARC's "depends" was learned and ConditionalQA's
was not** (0.651 against 0.111). On ShARC dev, v15 gains 540 rows and loses 164 against
v14 (p < 0.0001), mostly on "depends" (0.338 -> 0.651) and "no" (0.492 -> 0.745).
On ConditionalQA it gains 21 and loses 5 (p = 0.0025), but almost all of that is "no"
(0.302 -> 0.628): the gain is the removal of v14's lean towards "yes", not reading
conditions in long pages. With 310 "depends" rows against ShARC's 1,112, the dialogue
form of the idea — a follow-up question not yet answered — is what was taught.

**A regression outside the predictions: MuSiQue fell 0.852 -> 0.828** (26 gained, 55
lost, p = 0.0017), mostly on answerable questions (0.843 -> 0.806). The new rows offer
"depends" and "not stated" beside MuSiQue's "cannot be determined"; some of MuSiQue's
answerable questions now draw that kind of answer. HotpotQA, BoardgameQA (0.752 ->
0.773, p = 0.056) and ContractNLI (0.842 -> 0.837) are within noise.

**The question-sensitivity probe is unchanged:** same answer to a changed question on
held-out choice tasks 0.923 of the time (v14 0.937).

**Reading.** Short rule-application data teaches short rule application. It transfers
neither to long pages (ConditionalQA "depends") nor to JevBench's authored policies.
Measured on the same harness, decider v11's stage-2 data moved JevBench by 13 tasks; the
public substitute tried here moved it by −1. What decider's data has that this does not
is length and variety of documents: 34 business domains and 22 document kinds, written
for the purpose, each question checked by two further answers.

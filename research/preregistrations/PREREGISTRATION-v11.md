# Pre-registration: v11, teaching the model to read the question

Committed before training either arm. Same discipline as v9 and v10.

## What we found, and why this is different from v9 and v10

`tests/question_sensitivity.py` keeps the state and the options fixed and changes only
the question. v7 gives the same answer to a changed question 94-99% of the time. Asked
which option a text does **not** fit, it picks the one it does fit (0.730 on held-out
tasks) where the frozen base picks it 0.028 of the time; asked which option is listed
first, it scores 0.080 against the frozen base's 0.700.

The corpus taught this. Each state is asked one fixed question per task, so the state
and the option set determine the label on their own, and the cheapest solution is to
ignore the question. Where the question does vary per row (boolq, MNLI) v7 reads it;
where it never varies, it doesn't.

v9 and v10 added generated *documents* and failed because the model learned their one
structure. v11 keeps v5's real states unchanged and varies only the *question*
(`src/hobson/data/question_transforms.py`):

| transform | from | the question | rows |
| --- | --- | --- | --- |
| `is_answer` | choice | `Is "X" the correct answer?` / `...the wrong answer?`, X gold or not | 5,786 |
| `complement` | choice | two options; the right one, or the one that does NOT apply | 3,896 |
| `threshold` | score | `Is the rating k or higher?` / `...lower than k?`, with the scale | 6,319 |
| `polarity` | noul | the row's question inside a wrapper that keeps or flips it | 14,134 |

Every label is derived from the source row's gold label. `tests/test_question_transforms.py`
re-derives each one from the rendered text independently: 0 disagreements on all
30,135 rows. Yes/no labels are balanced (share true 0.491-0.496).

## The two arms

Each changes one thing against v7 (same torso, readout, KL anchor 0.3, window, one
epoch; length grouping **off**, as v7 trained):

- **Arm A** (`configs/experiments/v11a.yaml`): 30% of v5's rows (30,135) are replaced by
  their transformed version. Same row count and steps as v7.
- **Arm B** (`configs/experiments/v11b.yaml`): all of v5, plus the same 30,135 transformed
  prompts as **KL-only** rows: no label, trained only toward the frozen torso's reading
  of the changed question, at KL weight 1.0 (a labelled row's loss also has weight 1).
  This tests whether preserving the base model's own question-following is enough.

## Held out of training

The probe's question forms never appear in training, so the probe measures reading
questions rather than recall of templates. `PROBE_FORMS` lists them and a test enforces
it.

- **Never trained at all**: "which option is listed first / last", and another task's
  question over the wrong options (`irrelevant`).
- **Concept trained, form held out**: "which option does this text clearly NOT fit" over
  all 3-9 options (training has only a two-option complement), and the wrapper
  `Is the answer to the following question "no"?` (training has other wrappers).

## Baselines (v7)

Probe, choice, `same_as_real` = share of changed questions answered exactly as the real
one (lower is better). Held-out tasks: first 0.948, last 0.943, NOT 0.948 (mean
**0.946**); tasks in training: 0.990, 0.988, 0.988 (mean **0.989**). Held-out choice:
first accuracy 0.080 and last 0.033 (chance 0.184); NOT picks the original class 0.730.
Frozen base on held-out: first 0.700, last 0.388, NOT 0.028.

Probe, yes/no, `negated` on the per-row-question tasks (boolq, MNLI): accuracy
**0.072** (chance 0.5).

Held-out tasks, real questions (`hobson eval` on `holdout_v5_norule.jsonl`, 6,000 rows):
overall **0.653**, choice 0.728, noul 0.629, score 0.492.

JevBench: 154/231 = 0.6667. The families where the answer turns on reading the
question: `policy` 10/12, `trap` 5/8, `adversarial` 4/6, `ambiguous` 1/7 — together
**20/33**.

## Predictions for arm A

1. **Follows changed questions**: mean `same_as_real` over first / last / NOT below
   **0.60** on held-out choice tasks *and* on trained choice tasks.
2. **Reads questions it was never trained on**: "listed first" and "listed last"
   accuracy each above **0.284** (chance + 0.10) on held-out choice tasks.
3. **Held-out form of a trained concept**: NOT picks the original class below **0.40**
   on held-out choice tasks; `negated` accuracy above **0.60** on the per-row-question
   yes/no tasks.
4. **Costs nothing on real questions**: held-out overall accuracy at least **0.633**
   (v7 − 0.02) and choice at least **0.708**.
5. **JevBench** at least **0.6667**.

## Predictions for arm B

The frozen base's own mean `same_as_real` on held-out choice is about 0.28, but a KL pull
competes with 100k labelled rows that reward ignoring the question, so a weaker effect:

1. Mean `same_as_real` below **0.80** on held-out choice tasks.
2. NOT picks the original class below **0.60** on held-out choice tasks.
3. Held-out overall accuracy at least **0.633**.
4. JevBench at least **0.6667**.

## What would count as failure

- **1 holds but 2 does not** (arm A): answers now vary with the question without
  following it — the model learned that questions matter, not how to read them.
- **1-3 hold on trained concepts but fail on held-out forms**: learned the templates. The
  v9/v10 failure, one level up.
- **4 fails**: reading questions cost real accuracy — a net loss however good the probe.
- **Arm A passes 1-4 and JevBench does not move.** Stated in advance as a live
  possibility: the frozen base follows questions and still scores 0.550, so
  question-following may be necessary without being the binding constraint on
  JevBench. That would be a result about JevBench, not a failure of the corpus.

The four question-reading JevBench families (20/33) carry no numeric prediction: n=33 is
too small. Expected direction: up. Any movement there is exploratory.

## Which arm, if any, becomes v11

An arm is a candidate only if its probe predictions (1-3 for A, 1-2 for B) hold, its
held-out accuracy holds, and JevBench is at least 0.6667. If both qualify, the higher
JevBench wins; a tie goes to arm A, which needs no training change. Otherwise v7 stays.

## Power

JevBench cannot resolve an overall difference below about +0.043, so prediction 5 is a
guard. The probe (400-600 examples per condition) and the held-out eval (6,000 rows)
carry the statistical weight: at these sizes a change of 0.05 is several standard errors.

## Outcome (added after the runs)

Nothing above this section was edited after training. **Neither arm qualifies; v7
stays.** Both lost significantly on JevBench, by different routes.

### Arm A

| | prediction | v7 | arm A | |
| --- | --- | --- | --- | --- |
| 1 | mean `same_as_real` < 0.60, held-out / trained choice | 0.946 / 0.989 | 0.652 / 0.680 | FAIL |
| 2 | "listed first" / "last" > 0.284 | 0.080 / 0.033 | 0.090 / 0.040 | FAIL |
| 3 | NOT < 0.40; `negated` > 0.60 | 0.730; 0.072 | **0.115**; 0.340 | half |
| 4 | held-out >= 0.633, choice >= 0.708 | 0.653 / 0.728 | 0.646 / 0.731 | pass |
| 5 | JevBench >= 0.6667 | 154/231 | **143/231 = 0.6190** | FAIL |

JevBench: 2 gained, 13 lost, McNemar p = 0.007. Every family that moved went down:
`policy` 10 -> 7, `long_policy` 8 -> 6, `extraction` 24 -> 22, and one each in
`ordinal`, `probability`, `tradeoff`, `adversarial`. The question-reading families fell
from 20/33 to 16/33; ordinal MAE worsened from 0.592 to 0.733, matching a held-out
`score` drop (0.492 -> 0.451, all `hate_severity`).

The model learned the trained *concept* of negation — NOT over 3-9 options, a form it
never saw, fell from 0.730 to 0.115 — but nothing about questions it had never seen
("listed first/last" unchanged). And the negation it learned looks lexical. Stated as a
hypothesis before checking: 7 of the 13 JevBench losses have a negation word in the
question, against 41 of the other 218 (one-sided Fisher p = 0.007), and all 6 lost
yes/no tasks flipped yes -> no. JevBench questions carry incidental negation that does
not reverse anything ("Treat unproved required conditions as not satisfied"); a model
trained so that a negating wrapper flips the answer gets those wrong. The negation
regex is a post-hoc definition, so this is suggestive, not established.

### Arm B

| | prediction | v7 | arm B | |
| --- | --- | --- | --- | --- |
| 1 | mean `same_as_real` < 0.80, held-out choice | 0.946 | 0.941 | FAIL |
| 2 | NOT < 0.60, held-out choice | 0.730 | 0.745 | FAIL |
| 3 | held-out >= 0.633 | 0.653 | 0.650 | pass |
| 4 | JevBench >= 0.6667 | 154/231 | **144/231 = 0.6234** | FAIL |

The KL pull did not move choice at all. It moved yes/no, toward the frozen torso's
reading, which is near chance: on negated fixed-question prompts arm B now answers
0.500 with `same_as_real` 0.36 — it stopped following the original question without
starting to follow the new one. On JevBench it says yes on 60 of 74 yes/no tasks
(v7 53, expected 35).

JevBench: 2 gained, 12 lost, McNemar p = 0.013. Losses are *not* concentrated in
negation questions (3 of 12; Fisher p = 0.47), and only 2 tasks were lost by both
arms — a different failure from arm A's, which supports reading arm A's as specific to
its labels.

### What this says

1. **Labelled question transforms teach the transforms' concepts, keyed on their
   words.** The probe's NOT result looked like understanding; JevBench says it was
   partly a keyword.
2. **The frozen torso is a poor teacher for yes/no.** Its option-number readout is near
   chance there, and anchoring to it made the model worse, not more question-aware.
3. **Question-following as the probe measures it has not yet been shown to matter for
   JevBench.** Neither arm moved first/last; both lost JevBench for other reasons.

### A caveat that applies to every comparison in this project

McNemar treats v7's per-task answers as fixed. It accounts for which tasks were
sampled, not for training noise: v7 retrained with another seed would also differ from
v7 by some number of tasks, and that has never been measured. Every run since v7 —
v8 152, v9 152, v10 145, v11a 143, v11b 144 — has scored below v7's 154, which is also
what regression toward the mean would produce if v7's run was a favourable draw. The
losses here are large enough that seed noise is unlikely to explain all of them, but
until a seed replicate of v7 exists the p-values above overstate the certainty.

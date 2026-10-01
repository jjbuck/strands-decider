# Pre-registration: v20, reading the question and confidence

Committed before training. Same discipline as v9-v19.

## Why

v19 is 63 of 72 on JevBench's standard tier (decider-2b v11 64, the frozen Qwen3.5-4B
71). Its nine misses, read from outcomes and probabilities only, fall into three
patterns: six near coin-flips (`adequacy`, `policy`); items answered one way when asked
one way and the other way when asked another (the tier asks each of 36 items twice);
and two confident misses on questions with a catch-all option (`other`, `general`). The
question-sensitivity probe says the same from our own data: with the state and options
fixed, v19 gives the same answer to a changed question 93-95% of the time on held-out
tasks, and the same answer to a *negated* yes/no question 92.5% of the time.

Separately, v19 is less confident than the 4B where both are right (mean top probability
0.78 against 0.94 on the standard tier), which is most of decider's Brier lead there.
Recalibration showed temperature cannot fix this
([PREREGISTRATION-v19-calmix.md](PREREGISTRATION-v19-calmix.md)): v19's right and wrong
answers need opposite corrections.

## What is being changed

v19's recipe and rows plus four changes, each aimed at one of those and each with its
own held-out measure. Config: `configs/experiments/v20.yaml`.

1. **Paraphrases.** Each generated document question (3,815 training rows) carries up
   to three paraphrases as instruction variants, sampled per epoch at collate time as
   the short tasks' already are. Written by Qwen3.6-27B and kept only if Qwen3.5-397B-A17B
   judges each to ask exactly the same thing (`scripts/gen_paraphrases_openrouter.py`;
   14,464 of 14,876 kept; 3,757 of the 3,815 rows have at least one). On six deliberately broken rewrites (flipped polarity, changed
   number, dropped or added condition) the checker rejected all six. The rows and their
   order are unchanged (`data/generated_v16p.jsonl`, `generated_v18p.jsonl`).
2. **Instruction flips.** Short policies, each case asked twice with an instruction
   appended to the question that reverses the correct answer — how unproven conditions
   are treated, whether uncovered requests are permitted, which conflicting clause
   prevails, which version applies, whether a list of examples is complete, business or
   calendar days (`scripts/gen_flips_openrouter.py`; the state is identical across a
   pair, so the answer cannot be read off it). Same writer, same two-judgement keep rule
   per variant; a pair is kept only if all four judgements agree. 801 pairs written
   under a cost cap, 419 kept (52%; 74 more were left unverified at the cap): 359
   training pairs (718 rows) and 60 held-out pairs (120 rows). Six hand-read, one per
   flip type, all right, one borderline (a policy saying its examples "include" items
   half-settles the list question).
3. **Catch-all options.** 5,000 rows from the corpus's own choice tasks with named
   categories (ag_news, banking77, clinc150, dbpedia, yahoo_topics), each with an
   "other" / "none of these" option: in half the true category is removed and the
   catch-all is right, in half it is a distractor (`python -m hobson.data.catchall`). A
   row already at 24 options, the model's width, drops one wrong option at random to
   make room.
4. **Distillation where the teacher is right.** The frozen 4B's distributions on the
   59,525 yes/no and choice short-task rows where its answer matches the gold label
   (of 91,408 it labelled for v12; 75% agree), at the same weight as the replay targets
   on the multi-step rows (`data/teacher_v20.jsonl`). v12 distilled the 4B on every
   short-task row and lost `policy` and `routing`, because the teacher disagrees with
   the annotators on a quarter of them; here it is only ever pulled toward answers
   that are right, and only its confidence is new.

Generation cost about $72 on OpenRouter, pilots included.

*Amended before training (2026-09-28): the first launch failed at step 0 because 46
catch-all training rows (and 9 eval rows) had 25 options, one more than the model's
24-wide training reference. The transform now drops a wrong option from rows already
at 24; that changes the random draws, so the catch-all eval is a different sample of
the same two tasks, and its v19 and 4B baselines below were re-measured on it (v19
was 0.455 / 0.665 on the first sample). Prediction 3's thresholds were re-derived by
the same rule: 0.600 for the catch-all when right, and v19 less 0.02 as a distractor.
Nothing else changed.*

## Baselines

| evaluation | v19 (default) | frozen 4B |
| --- | --- | --- |
| **paraphrase consistency** (581 held-out generated eval rows, each also asked with a paraphrase): same answer / both right | 0.983 / 0.809 | — |
| **instruction flips** (60 held-out pairs): both answers right / same answer to both instructions | 0.233 / 0.767 | 0.583 / 0.417 |
| **catch-all** (held-out `massive_intent`, `emotion`, 800): catch-all right / distractor | 0.438 / 0.618 | 0.685 / 0.578 (the 634 rows it labelled) |
| probe, held-out yes/no: same answer to the negated question | 0.925 | — |
| probe, held-out choice: same answer to a changed question (first / last / not) | 0.950 / 0.935 / 0.948 | — |
| held-out short tasks: accuracy / ECE / NLL | 0.647 / 0.054 / 0.883 | — |
| JevBench at the 4096 default: tasks / standard / Brier / ECE | 168 / 63 / 0.342 / 0.051 | 186 / 71 / 0.252 / 0.056 |
| JevBench paraphrase consistency | 0.861 | — |
| MuSiQue / ContractNLI / BoardgameQA / HotpotQA | 0.879 / 0.862 / 0.810 / 0.726 | — |
| generated v16 / v18 eval | 0.846 / 0.757 | — |
| adequacy: HelpSteer2 / generated (balanced) | 0.739 / 0.788 | 0.645 / 0.810 |

## Predictions

1. **Paraphrases (a guard, not a gain):** paraphrase consistency at least 0.975. v19 is
   already at 0.983 on these questions, with the same accuracy under either wording
   (0.847 / 0.856 on v16's set, 0.769 / 0.773 on v18's), so there is no headroom for a
   gain here; this checks that adding wordings does no harm. Where paraphrases could
   help — JevBench's paired wordings — is too small to test (36 pairs).
2. **Instruction flips:** both answers of a pair right on at least 0.400 of the held-out
   pairs (24 of 60; v19 0.233 — the chance rate is 0.25 — and the frozen 4B 0.583).
3. **Catch-all:** catch-all right at least 0.600 (v19 0.438), with the catch-all as a
   distractor at least 0.598 (v19 0.618, less 0.02).
4. **Distillation:** held-out short tasks NLL at most 0.883 and ECE at most 0.064 (v19
   0.883 / 0.054, ECE allowed 0.01 of noise), and JevBench Brier at most 0.342.
5. **Nothing else is lost:** MuSiQue at least 0.865, ContractNLI at least 0.845,
   BoardgameQA at least 0.795, HotpotQA at least 0.710; generated v16 eval at least
   0.830 and v18 at least 0.740; adequacy at least 0.720 (HelpSteer2) and 0.770
   (generated, balanced); held-out short-task accuracy at least 0.627.
6. **JevBench** (4096 window) at least 165 of 231, standard tier at least 63, ECE at
   most 0.07.

Exploratory, no numeric prediction: the question-sensitivity probe; JevBench's own
paraphrase consistency; the standard tier by family (`policy`, `intent`, `routing`,
`adequacy`); how often each flip type is answered right; mean top probability on
right answers on the standard tier (v19 0.78, 4B 0.94).

## Which model is the default afterwards

v20 replaces v19 if predictions 1, 5 and 6 hold and at least two of predictions 2-4
hold. Otherwise v19 stays.

## What would count as failure

- **1-3 hold on our evals and the standard tier does not move:** the generated forms of
  question-reading are learned and do not carry to JevBench's wording.
- **4 fails, or Brier rises:** confidence learned from a teacher that is right on a
  task does not carry to decisions it was not distilled on; or the teacher's
  confidence on short tasks leaks into overconfidence on hard ones.
- **5 fails on the multi-step or adequacy sets:** the new rows (5,718 of 129,057)
  crowd out the skills v17-v19 added.

## What this test cannot show

- **Four changes at once.** Each has its own held-out measure, so each prediction can
  be read on its own, but a JevBench change cannot be attributed among them.
- **The generated evaluations share their generators' style and noise**; JevBench does
  not. 72 standard-tier tasks cannot resolve fewer than about 5, nor 231 fewer than
  about 10, and the seed replicate that would measure training noise is backlogged.
- **The catch-all eval is built from two held-out tasks**, one of them (`emotion`) with
  labels far from any trained task.

## Outcome (added after the run)

Nothing above this section was edited after training. **By the rule fixed above, v19
stays the default:** predictions 1 and 2 held, but 3, 4, 5 and 6 each failed on one
count, and 5 and 6 were required.

| | prediction | v19 | v20 | |
| --- | --- | --- | --- | --- |
| 1 | paraphrase consistency >= 0.975 | 0.983 | 0.986 | pass |
| 2 | flip pairs both right >= 0.400 | 0.233 | **0.900** | pass |
| 3 | catch-all right >= 0.600; as a distractor >= 0.598 | 0.438; 0.618 | **0.835**; 0.538 | FAIL (distractor) |
| 4 | held-out NLL <= 0.883, ECE <= 0.064; JevBench Brier <= 0.342 | 0.883, 0.054; 0.342 | **0.851**, 0.053; 0.355 | FAIL (Brier) |
| 5 | nothing lost (floors on eight sets) | | HelpSteer2 adequacy 0.714 against 0.720; the rest held | FAIL (adequacy, by 2 rows) |
| 6 | JevBench >= 165, standard >= 63, ECE <= 0.07 (4096 window) | 168, 63, 0.051 | 169, **61**, 0.059 | FAIL (standard) |

**Each piece learned its own task.** Instruction-flip pairs went from chance (0.233;
0.25 by guessing) to 0.900, past the frozen 4B's 0.583 — in-distribution for the
generator, so the size says more about the generator's style than about the skill. The
catch-all is now picked when the right category is missing (0.438 -> 0.835, +166 / -7),
but too often when it is present (0.618 -> 0.538, +2 / -34, p < 0.0001; on `emotion`
0.395 -> 0.295): part of what was learned is that a catch-all is often right. On the
held-out short tasks the distillation did what it was for: NLL 0.883 -> 0.851 (yes/no
0.706 -> 0.689, choice 0.840 -> 0.800), accuracy 0.647 -> 0.654, ECE level. Paraphrase
consistency held (0.986), with no gain available there.

**Nothing else moved significantly.** HotpotQA 0.726 -> 0.745 (+54 / -36, p = 0.07),
MuSiQue 0.879 -> 0.875, ContractNLI and BoardgameQA level, both generated sets level,
generated adequacy 0.788 -> 0.778 balanced. HelpSteer2 adequacy fell 0.739 -> 0.714
(+12 / -18, p = 0.36), below its floor by two rows; its inadequate recall fell 0.761 ->
0.701.

**JevBench: 169/231 at the 4096 default** (v19 168; +11 / -10, p = 1.0), the most any
model here has scored, but not where it was aimed. The hard tier rose 57 -> 60
(`long_policy` +2 net, `tradeoff` +2, `multi_hop` +1, `judge_hard` +1) and the standard
tier fell 63 -> 61: `policy` 9 -> 7, everything else level. JevBench's own paraphrase
consistency rose 0.833-0.861 -> **0.917**, the best recorded here (decider-2b v11 0.944),
but partly by agreeing with itself on wrong answers: v19 answered the two wordings of
`policy-02` and `policy-03` differently, getting one of each right; v20 gets all four
wrong. Brier 0.342 -> 0.355 (standard 0.193 -> 0.243), from a few confident wrong
answers — task by task it is lower on 123 and higher on 108 (p = 0.36). On the standard
tier the teacher's confidence did not carry: mean top probability on right answers
0.776 -> 0.772 (4B 0.94). "Yes" answers on JevBench's yes/no tasks fell 55% -> 51%
(gold 47%).

**Reading.** The four changes taught four generated or transformed forms, each
measurably, and none of them reached the standard tier they were chosen for; by the
AWS retrains' spread (SD 3.2 tasks per run), 168 -> 169 and 63 -> 61 are both inside
training noise. What the evidence supports: reading an instruction appended to a
question is learnable from pairs (0.233 -> 0.900) but the form learned does not
transfer to JevBench's `policy` items; the catch-all rows need the distractor case
weighted more heavily than the absent case, not equally; and distilling a teacher on
short tasks lowers held-out NLL without raising confidence on benchmark decisions.

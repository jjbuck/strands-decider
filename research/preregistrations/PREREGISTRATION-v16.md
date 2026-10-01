# Pre-registration: v16, generated document questions

Committed before training. Same discipline as v9-v15.

## Why

decider-2b's v11 adds one stage to its v10: harder decisions, most of them 11,356
questions over teacher-written business documents, each kept only when two further
answers agreed with the writer's. Measured on our harness, that stage is worth 13
JevBench tasks (v10 162 -> v11 175, p = 0.041), and v10 is level with v14 (162 against
161). The data is not released.

v15 tried the nearest public substitute (ShARC and ConditionalQA): the model learned
ShARC and JevBench did not move (160). This run tests the thing itself: documents and
questions written for the purpose, the way decider built them.

## The data

Written by `scripts/gen_documents_openrouter.py` through OpenRouter, with open-weight
Apache-2.0 models only:

- **Writer: Qwen3.6-27B** (decider's writer), thinking on. Each call writes one realistic
  workplace document (a domain from 34, a kind from 22, 900-1,600 words in numbered
  sections) and one question per assigned skill, 3 or 4 per document: all conditions,
  exception, precedence, dates, numbers, lookup chain, tradeoff, underdetermined,
  surface trap, rubric. Each question is applied to a case, cites at least two
  sections, and names its tempting wrong answer.
- **Verifier: Qwen3.5-397B-A17B**, a different model, thinking on. Every question is
  answered twice in fresh contexts, options shuffled, without the writer's answer. A
  question is kept only if both answers match the writer's.
- **Types and answers are assigned per question:** ~27% yes/no with a 50/50 required
  answer, the rest choice; "underdetermined" is always choice. The first 71 documents
  were written before this fix (a yes/no target shown on every question made the
  writer use yes/no for 86% of them); they are kept, since their labels passed the same
  filter, and the yes/no balancing below covers them.
- **No JevBench content.** The skills were written from our own reading of which
  decisions fail; no benchmark item was shown to either model.

The run: 1,000 documents (median 1,056 words), 3,486 questions, 2,804 kept (80%;
"underdetermined" 44%, "dates" 71%, every other skill 81-90%); kept yes/no answers 482
"yes" and 464 "no". Cost about $130 in API fees at OpenRouter list prices. The raw export — documents, questions, rationales
and every verifier answer — is committed in `scripts/gen_v16/`.

Two pilots decided the setup. Nova Premier (Bedrock) as writer and its own verifier:
7 of 30 hand-checked kept labels wrong or ill-posed, documents a median 536 words.
Qwen3.6-27B with the 397B verifier: 1 of 30 wrong, median 1,066 words. On the first 223
documents of this run, a hand check of 30 kept questions (13 of them "yes", the case
most likely to go wrong) found 1 clearly wrong label (an arithmetic slip both verifiers
repeated) and 5 debatable ones. On the 627 questions kept from those documents the
frozen Qwen3.5-4B scores 0.770 (first 200 rows) and v14 0.630, with v14's lean towards
"yes" intact (0.87 on "yes", 0.54 on "no").

`python -m hobson.data.generated` (`src/hobson/data/generated.py`, tested in
`tests/test_generated.py`) builds:

- **train** `data/generated_v16.jsonl`, **2,148 rows** (754 yes/no, 1,394 choice) from
  the 2,454 kept outside the four held-out domains. Two shortcuts found before training
  are removed: yes/no answers are balanced within each skill (−67 rows), and choice
  rows where the answer is also the longest option — 34% of them, against a 23%
  chance rate, because the writer describes the right answer most fully — are thinned
  at random until that rate equals chance (−239 rows).
- **eval** `data/generated_v16_eval.jsonl`, **350 rows** (125 yes/no, 225 choice): the
  documents in the held-out domains (insurance claims, municipal permits, laboratory
  safety, payroll), natural label distribution, options in a fixed shuffled order (the
  writer put the answer second in 42% of choice questions).

## What is being changed

v14 (the leader) plus `data/generated_v16.jsonl`, appended last, gold labels only.
v15's policy rows are not included: v16 differs from v14 only in this data. v14's
teacher distributions stay on its multi-step rows at unchanged indices. Config:
`configs/experiments/v16.yaml`.

## Baselines, measured before training

| evaluation | n | chance | cue heuristic | v14 | frozen Qwen3.5-4B |
| --- | --- | --- | --- | --- | --- |
| generated, held-out domains | 350 | 0.328 | 0.383 | **0.646** | 0.794 |
| — yes / no / choice | 63 / 62 / 225 | | | 0.87 / 0.44 / 0.64 | 0.86 / 0.82 / 0.77 |

The cue heuristic answers yes/no questions with the commonest answer and choice
questions with the longest option (the cue thinned out of training, left in the eval
set at its natural 0.316). v14 is weakest on "underdetermined" (0.29, n=17) and
"all conditions" (0.53, n=34), and its lean towards "yes" is the largest single gap.

JevBench, v14: 161/231, hard tier 52/111, ECE 0.086. HotpotQA (held out since v14):
0.750. MuSiQue (dev pairs): 0.852. Held-out short tasks: 0.637.

## Predictions

1. **JevBench at least 167/231 (0.723)**, six tasks over v14 — the bar v14 and v15
   were held to.
2. **Hard tier at least 58/111**, six over v14. decider's stage moved the hard tier
   (0.459 -> 0.577 on our harness) and little else.
3. **Generated eval (held-out domains) at least 0.746** (v14 + 0.10). Documents from
   domains never trained on.
4. **No displacement:** held-out short tasks at least 0.617, HotpotQA at least 0.730,
   MuSiQue at least 0.832 (each v14 − 0.02). v15's policy rows cost MuSiQue 0.024; that
   is the regression to watch.
5. **JevBench ECE at most 0.09.**

Exploratory, no numeric prediction: per-family changes on the hard tier; generated
eval by skill and by answer ("yes" / "no"); question sensitivity on
`tests/question_sensitivity.py` (v14: same answer to a changed question 0.937 of the
time on held-out choice tasks).

## What would count as failure

- **3 holds, 1 and 2 do not:** generated documents teach generated documents — v15's
  result again, with better data. That would argue against the full generation run,
  not for it.
- **1 or 2 holds, 3 does not:** the JevBench gain comes from something other than
  reading these documents (for example a shift in answer priors); read the per-family
  and per-answer results before believing it.
- **4 fails:** the new data displaced what the old corpus or the multi-step rows
  taught.

## Which model leads afterwards

v16 becomes the leader if 1 and 4 hold; otherwise v14 stays. A seed replicate remains
outstanding (PREREGISTRATION-v11.md).

## Scale and power

This is about a fifth of decider's stage (2,148 training questions against its 11,356
document questions, which came with 8,000 generated-family and 3,293 human-labelled
rows). A null result here is weaker evidence against the approach than a positive one
is for it. 231 JevBench tasks cannot resolve differences below about +0.043 (10
tasks), so predictions 1 and 2 are suggestive; prediction 3 (350 rows, paired against
v14: +0.10 is about 35 rows, roughly four standard errors) carries the statistical weight.

## Outcome (added after the run)

Nothing above this section was edited after training. **By the rule fixed above, v14
stays the leader:** predictions 1, 2 and 4 failed. v16 learned the generated documents
thoroughly, and on domains it never saw; JevBench moved two tasks, within noise, and
the multi-step sets regressed.

| | prediction | v14 | v16 | |
| --- | --- | --- | --- | --- |
| 1 | JevBench >= 167/231 | 161 | 163 (0.706) | FAIL |
| 2 | hard tier >= 58/111 | 52 | 55 | FAIL |
| 3 | generated eval (held-out domains) >= 0.746 | 0.646 | **0.837** | pass |
| 4a | held-out short tasks >= 0.617 | 0.637 | 0.629 | pass |
| 4b | HotpotQA (held out) >= 0.730 | 0.750 | 0.712 | FAIL |
| 4c | MuSiQue >= 0.832 | 0.852 | 0.827 | FAIL |
| 5 | JevBench ECE <= 0.09 | 0.086 | 0.069 | pass |

**The generated data was learned, and it transfers across domains.** On the 350
held-out-domain questions v16 gains 83 and loses 16 against v14 (p < 0.0001): choice
0.640 -> 0.831, "no" answers 0.435 -> 0.823, "yes" answers 0.873 -> 0.873. The gain is
not a shift in answer prior — "yes" is unchanged while "no" nearly doubles — and it is
spread over every skill (lowest: "underdetermined" 0.706, n=17).

**JevBench moved in the right places, too little to count.** 163/231: 10 gained, 8 lost
(p = 0.82); hard tier 52 -> 55. `trap` 6 -> 8 of 8 and `tradeoff` 2 -> 4 of 6 moved to
or past decider v11 (8 and 3); `long_policy` 7 -> 8 and `ambiguous` 2 -> 3 matched it.
`temporal_numeric` fell 4 -> 2 (v15 lost the same two), `probability` 5 -> 4, `ordinal`
12 -> 11. ECE improved (0.069, the best of any model here) and latency is unchanged.

**The multi-step sets regressed — named failure mode 4.** HotpotQA, held out since v14,
fell 0.750 -> 0.712 (49 gained, 85 lost, p = 0.0024), and MuSiQue 0.852 -> 0.827 (18
gained, 49 lost, p = 0.0002), almost all on its answerable questions (0.843 -> 0.803).
v15 lost MuSiQue's answerable questions by the same amount (0.843 -> 0.806) with
entirely different data. What the two share is new rows whose options include "cannot
be determined" / "not stated" (v16: 123 training rows answered "cannot be determined");
that explains MuSiQue's answerable losses better than HotpotQA's, whose options never
include it. Not yet tested.

**The question-sensitivity probe is unchanged** (same answer to a changed question on
held-out choice tasks 0.929 of the time; v14 0.937).

**Reading.** At this scale — a fifth of decider's stage — generated documents teach a
transferable skill on documents like them (+0.19 on unseen domains) but not yet one
that moves JevBench outside noise, and they cost the multi-step sources some of what
v14 gained. decider's stage also carried 8,000 generated-family rows, 3,293
human-labelled rows, and replay trained toward the parent's own answers, which
protects exactly what regressed here. Before a full-size generation run, the cheaper
test is this data again with replay of v14's multi-step rows toward v14's own
distributions.

## Decision after the outcome

v16 was made the default model and recipe on 2026-09-26 by decision, overriding the
rule above: its JevBench score (163), Brier (0.396), macro (0.715) and calibrated top
band, and its gain on unseen-domain generated questions, were judged to outweigh its
losses on HotpotQA and MuSiQue. v14 was promoted the same way over v13. v17
(PREREGISTRATION-v17.md) tests replay toward v14 to recover what v16 lost.

# Pre-registration: v12, distilling a frozen Qwen3.5-4B

Committed before labelling the corpus or training. Same discipline as v9-v11.

## Why a teacher

Every corpus change since v5 has taught a surface: RuleTaker, generated documents,
templated question transforms. Every label came from a template or a classification
dataset. v11b showed the mechanism of a KL target works but its target (our own frozen
1.7B) was near chance on yes/no.

The teacher has to be better than v7 at exactly this interface. Measured locally, each
read the way SemIf reads a frozen model (chat template, JSON decision, one forward pass,
softmax over option letters), bar fixed before each run at 0.75:

| | JevBench | ECE | |
| --- | --- | --- | --- |
| v7 (ours, trained) | 0.6667 (154) | 0.068 | |
| Qwen3-8B, frozen | 0.6840 (158) | 0.305 | below the bar |
| **Qwen3.5-4B, frozen** | **0.8052 (186)** | **0.056** | passes |

The Qwen3.5-4B run reproduces SemIf's published result (187/231; the same verdict on
230 of 231 tasks). Against v7 it gains 48 tasks and loses 16, concentrated where no
corpus change ever moved v7: `judge_hard` 7 -> 13, `multi_hop` 7 -> 12, `ambiguous`
1 -> 5, `adequacy` 8 -> 12, `trap` 5 -> 8. It is also better calibrated than v7, so its
probabilities can be used as targets directly. The size limit applies to our model,
not to what it learns from.

JevBench was used only to choose the teacher. No JevBench task is labelled or trained on.

## What is being changed

One thing against v7: the KL reference. v7 anchors to its own frozen torso (weight 0.3,
rows with <= 9 options). v12 replaces that with the teacher (`kl_frozen_weight: 0`,
`teacher_weight: 1.0`, rows with <= 16 options) — the standard distillation mix of
gold label and teacher distribution at equal weight. Everything else is v7's: same
corpus (`train_v5.jsonl`), torso, pointer readout, window, one epoch, length grouping
off. Config: `configs/experiments/v12.yaml`.

Teacher labels come from `src/hobson/data/teacher.py`, whose prompts are checked
byte-identical to SemIf's own (`tests/test_teacher.py`). Batched labelling agrees with
SemIf's batch-of-one path on the argmax of all 120 rows compared, probabilities within
0.047 (bf16 batching). On a 300-row sample of the training corpus the teacher's argmax
matches the gold label on 230 (0.767): it disagrees with annotator conventions on
fine-grained tasks, which is why the gold label loss stays in.

Known mismatch: the teacher labels each row's canonical question; during training the
collator sometimes shows a paraphrase of it. The meaning is the same.

## Baselines (v7)

- JevBench 154/231; ECE 0.068; hard tier 46/111.
- v7's answer equals the teacher's on **157/231** JevBench tasks.
- Held-out tasks, real questions (`holdout_v5_norule.jsonl`, 6,000 rows): **0.653**.
- Held-out agreement with the teacher's argmax: measured on the same 6,000 rows once the
  held-out file is labelled, before v12 is evaluated; recorded in the outcome.

## Predictions

1. **Distillation took** (mechanism): v12's answers equal the teacher's on at least
   **165/231** JevBench tasks, and its held-out agreement with the teacher's argmax
   exceeds v7's by at least **0.03**.
2. **JevBench at least 162/231 (0.701)**, eight tasks over v7.
3. **Held-out real-question accuracy at least 0.633** (v7 − 0.02).
4. **JevBench ECE at most 0.08.**

The families where the teacher gains most carry no numeric prediction; expected
direction up.

## What would count as failure

- **1 fails**: the student did not move toward the teacher — a weight or mechanism
  problem, and says nothing about whether a teacher helps.
- **1 holds, 2 does not**: the student learned the teacher's answers on the training
  corpus's easy classification prompts, and that does not carry to JevBench's hard
  families. The teacher's advantage is on long, multi-step documents the corpus does
  not contain; distilling on the wrong prompts is the most likely failure.
- **3 fails**: pulling toward the teacher cost real accuracy on unseen tasks.

## If it passes

v12 would replace v7 only after a second seed confirms it: every run since v7 has scored
below v7's 154, and training noise has never been measured (PREREGISTRATION-v11.md).

## Power

As before, 231 tasks cannot resolve an overall difference below about +0.043, so
prediction 2 is suggestive rather than conclusive. Prediction 1 is where the power
is: agreement is paired per task and per row.

## Outcome (added after the run)

Nothing above this section was edited after training. **v12 does not qualify; v7
stays.**

| | prediction | v7 | v12 | |
| --- | --- | --- | --- | --- |
| 1a | same answer as teacher on JevBench >= 165/231 | 157 | **156** | FAIL |
| 1b | held-out agreement with teacher's argmax >= v7 + 0.03 | 0.6958 | **0.7270** (+0.031) | pass |
| 2 | JevBench >= 162/231 | 154 | **146** (0.6320) | FAIL |
| 3 | held-out real-question accuracy >= 0.633 | 0.653 | **0.674** | pass |
| 4 | JevBench ECE <= 0.08 | 0.068 | 0.093 | FAIL |

The failure named as most likely — *1 holds, 2 does not* — in its clearest form yet. The
student moved toward the teacher on the distribution it was distilled on (held-out
agreement +0.031, on 5,384 rows) and not at all on JevBench (156 vs 157). Of the 74
JevBench tasks where v7 and the teacher disagree, v12 switched to the teacher's answer
on 9, and the teacher was right on 4 of those.

JevBench: 5 gained, 13 lost, McNemar p = 0.096. Losses in `policy` 10 -> 7, `routing`
11 -> 9, `tradeoff` 4 -> 2, one each in `long_policy`, `probability`, `multi_hop`; gains
in `adversarial` and `trap`. Ordinal MAE improved (0.592 -> 0.552).

**What did improve is real, and on the corpus's own ground.** Held-out real-question
accuracy is the best any model here has recorded: 0.674 against v7's 0.653, with
`hate_severity` 0.492 -> 0.563 and `sarcasm` 0.629 -> 0.656 — the unseen rubric and the
unseen yes/no task, v7's two weakest.

**Why, most likely.** Labelling the held-out file showed the teacher is no better than v7
on short classification (0.640 against gold vs v7's 0.653 on the same file); its 0.805
on JevBench comes from long documents and multi-step judgement, which the training
corpus does not contain. Distillation transfers the teacher's behaviour on the prompts
it is given. These prompts carried its calibration of short-text judgements, which
helped the held-out tasks, and none of its advantage on JevBench.

Every run since v7 has now scored below it: 152, 152, 145, 143, 144, 146. The caveat in
PREREGISTRATION-v11.md applies with more force: v7's 154 may be a favourable draw, and
a seed replicate is the measurement that would settle it.

# Pre-registration: calibrating v19 on a broader held-out mix

Committed before fitting. No training: only v19's three temperatures change.

## Why

v19's temperatures (`noul` 0.91, `choice` 0.77, `score` 0.96) were fitted, like every
model's since v5, on the calibration half of four held-out *short classification*
tasks (`holdout_v5_norule.jsonl`). The decisions the model is judged on are mostly
not short classification: long documents, generated workplace questions, answer
adequacy. On JevBench's standard tier v19 is right 63 times at a mean top probability of
0.78, where the frozen Qwen3.5-4B is right 71 times at 0.94, and a temperature fitted
on one kind of data need not suit the others. This tests how much of the calibration
gap is temperature alone.

## What is being changed

The temperatures are refitted, with the same code and objective (ECE per primitive,
`evaluate.fit_temperature_by_kind`), on a mix of the **calibration halves** of all six
held-out sets the project has, at most 500 rows from each so that no one source
dominates:

| source | rows | kinds |
| --- | --- | --- |
| `holdout_v5_norule` (four unseen classification tasks) | 21,000 | noul, choice, score |
| `multistep_v14_eval` (MuSiQue, ContractNLI, BoardgameQA, HotpotQA) | 4,084 | noul, choice |
| `generated_v16_eval`, `generated_v18_eval` (held-out domains) | 350, 247 | noul, choice |
| `adequacy_hs2_eval`, `adequacy_gen_eval` | 234, 302 | noul |

Halves are the existing deterministic split (`evaluate.partition_examples`, seed 0),
applied to each file separately. Only the `noul` and `choice` temperatures are refitted:
`score` rows exist only in the classification set, which the mix does not change, and
500 rows of it hold too few `score` rows for the fitter's 200-row minimum, so `score`
keeps v19's 0.96. Applied to a copy of the checkpoint; the shipped one is untouched
until the decision below.

## The decision rule

Decided on the **test halves** of the same six sets, never on JevBench. The new
temperatures are adopted if, averaged over the six sets with equal weight:

1. ECE is lower than with v19's temperatures, and
2. NLL is not higher (by more than 0.005).

Temperature cannot change a `choice` or `noul` answer, and `score` keeps its
temperature, so no answer changes: accuracy is identical by construction.

JevBench (at the 4096 default, against v19's 168/231, Brier 0.3416, ECE 0.0507) is run
afterwards and reported as an external check. It does not enter the decision, and the
mix, the 500-row cap and the rule are not revised after seeing it.

## What this cannot show

- Whether the gap to the 4B is temperature at all: one scalar per primitive cannot make
  a model more confident on the answers it gets right *and* less on those it gets wrong.
- Anything about accuracy.

## Outcome (added after the fit)

Nothing above this section was edited after fitting. **By the rule, v19's temperatures
stay.** The refit sharpened both primitives a long way (`noul` 0.91 -> 0.53, `choice`
0.77 -> 0.56; fitted on 1,557 calibration rows, since the smaller sets' halves are under
500). Mean ECE over the six test halves fell 0.202 -> 0.108, but mean NLL rose 0.533 ->
0.545 (+0.012, against an allowance of 0.005), so rule 2 fails.

| test half | n | accuracy | ECE v19 -> mix | NLL v19 -> mix | mean confidence |
| --- | --- | --- | --- | --- | --- |
| short classification | 3,000 | 0.643 | **0.054** -> 0.082 | **0.906** -> 1.005 | 0.615 -> 0.691 |
| multi-step documents | 2,005 | 0.827 | 0.191 -> **0.085** | 0.452 -> **0.435** | 0.636 -> 0.743 |
| generated, v16's set | 178 | 0.888 | 0.165 -> **0.068** | 0.329 -> **0.305** | 0.722 -> 0.821 |
| generated, v18's set | 126 | 0.770 | 0.134 -> **0.093** | **0.569** -> 0.610 | 0.678 -> 0.780 |
| adequacy, HelpSteer2 | 122 | 0.721 | 0.321 -> **0.162** | **0.552** -> 0.568 | 0.400 -> 0.577 |
| adequacy, generated | 150 | 0.860 | 0.346 -> **0.157** | 0.391 -> **0.347** | 0.514 -> 0.703 |

**The finding is the table, not the verdict.** With the shipped temperatures, fitted on
short classification, v19 is well calibrated there and **under-confident everywhere
else** — by 0.19 on multi-step documents and by more than 0.3 on adequacy, where it is
right 72-86% of the time at a stated 40-51%. No single temperature per primitive
serves both: sharpening fixes the documents and the adequacy sets and breaks short
classification.

**JevBench, reported and not used** (4096 window): accuracy unchanged at 168, as it must
be; Brier 0.3416 -> 0.3593 and ECE 0.051 -> 0.089, worse. The same split shows up there:
the standard tier's Brier improves (0.193 -> 0.177, level with decider-2b v11's 0.174)
and the hard tier's worsens (0.581 -> 0.632); the top confidence band doubles (53 -> 106
answers) and goes from all right at a claimed 0.957 to 0.915 right at a claimed 0.966.

**Reading.** Temperature cannot close the gap to the 4B's confidence: the answers v19
gets right and the answers it gets wrong need opposite corrections, and one scalar
moves both. The under-confidence is on the safe side for routing (fewer answers clear
an automation threshold, not more wrong ones do), but it means the documented
confidence bands hold for short classification only. Closing it needs a change in
training, where right answers become sharper without wrong ones following.

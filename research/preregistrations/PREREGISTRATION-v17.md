# Pre-registration: v17, replay toward v14

Committed before training. Same discipline as v9-v16.

## Why

v16 (the default) learned the generated document questions — 0.646 -> 0.837 on four
domains it never saw — but lost ground on the multi-step sets v14 was built on: HotpotQA
0.750 -> 0.712 (p = 0.002) and MuSiQue 0.852 -> 0.827 (p = 0.0002), almost all of that on
MuSiQue's answerable questions. v15 lost the same MuSiQue questions with different data.
v14's multi-step rows were in both runs, on their labels, with the frozen Qwen3.5-4B
teacher's distributions; nothing asked the new model to behave like v14 on them.

decider-2b protects each parent that way: replayed rows are trained toward the parent's
own answer distribution, KL(p_parent || p_model). Its 4B ablation (v2 against v2.1)
found replay on hard labels sharpened every answer and cost old skills; replay toward
the parent kept them.

## What is being changed

v16's recipe and rows, unchanged, except the teacher distributions on v14's 12,909
multi-step rows: **v14's own** distributions (softmax of v14's raw logits, temperature
1, over each row's options; `python -m hobson.data.replay`, `src/hobson/data/replay.py`)
in place of the frozen Qwen3.5-4B's. Same weight (1.0), same loss (gold-label
cross-entropy plus KL toward the teacher); generated and short-task rows as in v16.
Config: `configs/experiments/v17.yaml`.

v14 gives the gold answer on 0.882 of these rows, and its distributions are soft
(median top probability 0.74; lower quartile 0.56), so the targets carry v14's
uncertainty and, on about one row in eight, its mistakes — against which the
gold-label term still pulls.

One setting, fixed here. If v17 fails, the weight and the choice of rows are not tuned
against the results below and rerun.

## Baselines

| evaluation | v14 | v16 (default) |
| --- | --- | --- |
| HotpotQA (held-out source, 959) | **0.750** | 0.712 |
| MuSiQue (1,199) | **0.852** | 0.827 |
| — answerable / unanswerable | 0.843 / 0.862 | 0.803 / 0.850 |
| ContractNLI / BoardgameQA | 0.842 / 0.752 | 0.840 / 0.757 |
| generated, four held-out domains (350) | 0.646 | **0.837** |
| JevBench (231) | 161 | **163** |
| hard tier | 52 | 55 |
| held-out short tasks | 0.637 | 0.629 |
| JevBench ECE | 0.086 | 0.069 |

## Predictions

1. **The multi-step sets come back:** HotpotQA at least 0.740 and MuSiQue at least 0.845
   — within 0.01 of v14 (v16: 0.712 and 0.827).
2. **The generated skill stays:** generated eval (held-out domains) at least 0.800 (v16
   0.837, v14 0.646).
3. **JevBench at least 161** (v14's), and ECE at most 0.09.
4. **Held-out short tasks at least 0.617.**

Exploratory, no numeric prediction: JevBench against v16 per family, especially `trap`,
`tradeoff` and `temporal_numeric`; how closely v17 follows v14 on the multi-step
training rows.

## What would count as failure

- **MuSiQue recovers, HotpotQA does not:** replay protects the rows replayed and not the
  transfer they supported. That would say v14's HotpotQA result is not held by its
  answers on its own rows.
- **1 holds, 2 does not:** the two kinds of training pull against each other, and replay
  toward v14 simply rebuilds v14.
- **Neither moves:** the regression in v16 is not about the multi-step rows' targets.

## Which model is the default afterwards

v17 replaces v16 as the default if 1, 2 and 4 hold and prediction 3's JevBench floor
does. Otherwise v16 stays.

## What this test cannot show

- **HotpotQA is no longer an untouched transfer test.** It was held out to measure
  transfer, and this run is designed in response to its drop, so its recovery is weaker
  evidence than its original result. No fresh transfer set is added in this run.
- **231 JevBench tasks cannot resolve differences below about 10 tasks**, and the seed
  replicate that would measure training noise is still outstanding. Prediction 3 is a
  floor, not a claim of improvement.

## Outcome (added after the run)

Nothing above this section was edited after training. **By the rule fixed above, v16
stays the default:** prediction 1 failed on HotpotQA. The first named failure mode
occurred — replay restored the rows it replayed and not the held-out transfer — but
v17 is ahead of v16 on almost every other measure taken.

| | prediction | v14 | v16 | v17 | |
| --- | --- | --- | --- | --- | --- |
| 1 | HotpotQA >= 0.740 and MuSiQue >= 0.845 | 0.750 / 0.852 | 0.712 / 0.827 | 0.723 / **0.880** | FAIL (HotpotQA) |
| 2 | generated eval >= 0.800 | 0.646 | 0.837 | **0.840** | pass |
| 3 | JevBench >= 161; ECE <= 0.09 | 161; 0.086 | 163; 0.069 | **164**; 0.083 | pass |
| 4 | held-out short tasks >= 0.617 | 0.637 | 0.629 | **0.641** | pass |

**The replayed sources came back and passed v14.** Against v16: MuSiQue 0.827 -> 0.880
(+80 / -16, p < 0.0001; answerable 0.803 -> 0.865, unanswerable 0.850 -> 0.895),
ContractNLI 0.840 -> 0.865 (p = 0.005), BoardgameQA 0.757 -> 0.817 (+71 / -17, p <
0.0001). All three are also above v14 itself (MuSiQue +0.028, ContractNLI +0.023,
BoardgameQA +0.065): v14's own distributions made better targets for these rows than
the frozen Qwen3.5-4B's did.

**HotpotQA did not come back.** 0.723 against v16's 0.712 (+50 / -40, p = 0.34) and
v14's 0.750 (+40 / -66, p = 0.015). Whatever v16 lost in transfer to an unseen
multi-hop source, pinning its answers on the sources it did see did not restore it.

**The generated skill stayed** (0.840; +8 / -7 against v16), including v16's "no"
answers (0.839).

**JevBench: 164/231**, the highest of any model here (+6 / -5 against v16, p = 1.0;
+11 / -8 against v14, p = 0.65), hard tier 55, standard 61. Brier 0.392, macro 0.720,
ordinal MAE 0.571 and the question-sensitivity probe (same answer to a changed question
0.943 on held-out choice tasks; v16 0.929) are its best or near-best; ECE 0.083 is
worse than v16's 0.069, with a top band still right as often as it claims (0.940 at a
claimed 0.948, 67 tasks) and a middle band claiming more than it delivers (0.667 at
0.746). None of the JevBench differences is resolvable at 231 tasks.

**Reading.** Replay toward the parent worked as decider describes for the sources it
covers, and cost nothing on the new data. The HotpotQA loss is something else: v16's
generated documents changed how the model reads paragraph sets it never trained on,
and neither v14's rows nor v14's answers on them hold that in place.

## Decision after the outcome

v17 was made the default model and recipe on 2026-09-27 by decision, overriding the rule
above: it is ahead of v16 on JevBench (164), Brier, macro, the multi-step sources it
replays and the held-out short tasks, level on generated questions, and slightly ahead
on HotpotQA, the one test it missed. v14, v16 and now v17 were each promoted this way.

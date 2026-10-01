# Pre-registration: v19, answer adequacy

Committed before training. Same discipline as v9-v18.

## Why

JevBench's standard tier is where the default, v18, is closest to decider-2b (60 of 72
against 64), but one family in it is a systematic failure rather than noise:
`adequacy`, where v17 and v18 each answer "adequate" on 11 of 12 public tasks and get
7. Every model trained here scores 7-9 on that family; the frozen Qwen3.5-4B scores 12.
decider's Brier on the standard tier (0.174) is also far ahead of ours (v18 0.269, v17
0.242), mostly from sharper confidence where it is right.

The failure is not specific to JevBench. On a balanced held-out set of HelpSteer2
responses (below), every model trained here answers "adequate" about nine times in
ten and is at chance:

| model | accuracy | adequate recall | inadequate recall |
| --- | --- | --- | --- |
| v14 | 0.479 | 0.915 | 0.043 |
| v16 | 0.483 | 0.932 | 0.034 |
| v17 | 0.487 | 0.906 | 0.068 |
| v18 (default) | 0.483 | 0.897 | 0.068 |
| frozen Qwen3.5-4B | **0.645** | 0.675 | 0.615 |

The frozen 4B separates the two (mean P(adequate) 0.70 on adequate responses, 0.39 on
inadequate), so the labels carry a learnable signal. Nothing in the training corpus
asks the question: the nearest sources are entailment (MNLI), BoolQ and fact
verification (VitaminC), none of which judges a response against a request.

## What is being changed

v18's recipe and rows, unchanged, plus answer-adequacy rows: a request, a response,
and "does the response adequately answer the request?" (yes/no; four phrasings of the
question, sampled at collate time). Gold labels only; the new rows carry no teacher
distribution. Config: `configs/experiments/v19.yaml`.

| source | rows | adequate / not | how labelled |
| --- | --- | --- | --- |
| HelpSteer2 train split (`data/adequacy_hs2.jsonl`) | 4,866 | 2,433 / 2,433 | human ratings |
| generated (`scripts/gen_adequacy/` -> `data/adequacy_gen.jsonl`) | 1,300 | 650 / 650 | writer + two verifier judgements agree |

**HelpSteer2** (nvidia/HelpSteer2, CC-BY-4.0): prompts mostly from ShareGPT; responses
from NVIDIA's in-house models, none from OpenAI or Anthropic; each response rated 0-4
by 3-5 Scale AI annotators. Adequate = helpfulness >= 3 and correctness >= 3;
inadequate = helpfulness <= 1 or correctness <= 1; the middle band is dropped.
Single-turn prompts only (multi-turn prompts carry earlier assistant turns of unknown
origin). Balanced to equal "yes" and "no", adequate responses drawn first from the
same prompts as inadequate ones, so the request alone does not predict the label.
Built by `python -m hobson.data.adequacy` (`src/hobson/data/adequacy.py`).

**Generated** (`scripts/gen_adequacy_openrouter.py`): the v16-v18 pipeline (writer
Qwen3.6-27B, two judgements by Qwen3.5-397B-A17B, kept only if both agree with the
writer), six items per batch in one of twelve request categories, each item assigned
its verdict (half and half) and a kind: for inadequate items one of eight specific
defects (wrong fact, answers a different question, missed constraint, skipped part,
calculation error, needless refusal, ignores supplied context, accepts a false
premise); for adequate items a plain good answer or one that looks flawed and is not
(terse, informal, corrects a false premise, states an assumption, caveated). The
categories and kinds were written from the family's one-line description ("answer
adequacy judging"), not from benchmark items. Two request categories (transforming
supplied data; planning with constraints) are held out for the eval. 380 batches, 2,280
items, 1,852 kept (81%): 1,550 training items, balanced to 650 each way by the v16
builder (`python -m hobson.data.generated --src scripts/gen_adequacy`), and 302 eval
items (121 adequate, 181 not). The verifiers disagreed most with the rough-looking
adequate kinds (56-75% kept, against 74-100% for the defects), which is why the kept
items lean to "no" before balancing. Cost about $32 on OpenRouter, pilot included.

**Label check.** In the 120-item pilot, 12 of 14 hand-read kept labels were right; one
was borderline (a comparison marked adequate that calls one option "the clear winner"
on an invented claim) and one probably wrong (a timeline marked adequate whose dates
contradict each other). Response length does not predict the label (median 95 words
adequate, 100 inadequate). In 30 kept items read from the full run, all 30 verdicts are
right (one writer's explanation garbles its figures around a correct verdict). Many
defects are blatant once noticed, needless refusals especially; others are subtle (a
weighted sum off by one, a sort order reversed, a skipped third criterion).

One setting, fixed here. If v19 fails, the mix, the weight and the thresholds are not
tuned against the results below and rerun.

## Baselines

| evaluation | v18 (default) | v17 | frozen 4B |
| --- | --- | --- | --- |
| HelpSteer2 adequacy eval (234, validation split, balanced) | 0.483 | 0.487 | 0.645 |
| generated adequacy eval (held-out categories, 302), balanced accuracy | 0.577 | 0.575 | 0.810 |
| — adequate / inadequate recall | 1.000 / 0.155 | 1.000 / 0.149 | 0.884 / 0.735 |
| JevBench `adequacy` (12) | 7 | 7 | 12 |
| JevBench standard tier (72) | 60 | 61 | 71 |
| JevBench standard-tier Brier | 0.269 | 0.242 | — |
| JevBench (231) | 164 | 164 | 186 |
| JevBench ECE / Brier | 0.079 / 0.388 | 0.083 / 0.392 | — |
| MuSiQue / ContractNLI / BoardgameQA / HotpotQA | 0.892 / 0.866 / 0.801 / 0.705 | 0.880 / 0.865 / 0.817 / 0.723 | — |
| generated v16 eval (350) / v18 eval (247) | 0.863 / 0.761 | 0.840 / 0.660 | — |
| held-out short tasks | 0.641 | 0.641 | — |

## Predictions

1. **Adequacy is learned, both ways:** HelpSteer2 adequacy eval at least 0.620 (v18
   0.483, frozen 4B 0.645), with inadequate recall at least 0.40 and adequate recall
   at least 0.60 -- a gain from judging, not from moving the bias to "no".
2. **It transfers to other requests:** balanced accuracy (the mean of the two recalls)
   on the generated adequacy eval, whose request categories were never trained on, at
   least 0.727 (v18 0.577 + 0.150; frozen 4B 0.810).
3. **Nothing else is lost:** MuSiQue at least 0.875, ContractNLI at least 0.850,
   BoardgameQA at least 0.785, HotpotQA at least 0.690 (each within about 0.015 of v18);
   generated v16 eval at least 0.840 and v18 eval at least 0.730; held-out short tasks
   at least 0.620.
4. **JevBench at least 162**, ECE at most 0.09, standard tier at least 60.

Exploratory, no numeric prediction: JevBench `adequacy` (v18 7 of 12; the public
family is 6 items in two phrasings, so it can move only in steps of one or two items);
standard-tier Brier; `policy` and the other yes/no families, and the share of "yes"
answers across JevBench's yes/no tasks, in case "no" spreads beyond adequacy.

## What would count as failure

- **1 fails:** 4,866 human-rated rows plus the generated ones do not teach a 2B
  single-pass model to judge adequacy that the frozen 4B judges at 0.645 untrained.
- **1 holds on accuracy but not on recall:** the model has moved its bias, not learned
  the judgement.
- **1 holds, 2 does not:** the model learned HelpSteer2's rating conventions, not
  adequacy.
- **1 and 2 hold, JevBench `adequacy` does not move:** the benchmark's family asks
  something different from both sources (its imported items are graded by a
  deterministic grader, not by raters). 12 tasks can only show this weakly.

## Which model is the default afterwards

v19 replaces v18 as the default if predictions 1 to 4 all hold. Otherwise v18 stays.

## What this test cannot show

- **The HelpSteer2 eval shares its source with the HelpSteer2 training rows** (same
  raters, same models writing responses; no prompt is in both). Prediction 2 is the
  out-of-source test, and it shares the generator's style and label noise with the
  generated training rows.
- **JevBench's `adequacy` family is 12 tasks** and cannot resolve the change; the
  standard tier (72) cannot resolve differences below about 5 tasks, nor JevBench
  (231) below about 10. The seed replicate is still outstanding.

## Outcome (added after the run)

Nothing above this section was edited after training. **All four predictions held, so
by the rule fixed above v19 replaces v18 as the default** — the first run in this
project to clear its own pre-registered bar.

| | prediction | v18 | v19 | |
| --- | --- | --- | --- | --- |
| 1 | HelpSteer2 >= 0.620; inadequate recall >= 0.40; adequate recall >= 0.60 | 0.483; 0.068; 0.897 | **0.739; 0.761; 0.718** | pass |
| 2 | generated adequacy, balanced accuracy >= 0.727 | 0.577 | **0.788** | pass |
| 3 | MuSiQue >= 0.875, ContractNLI >= 0.850, BoardgameQA >= 0.785, HotpotQA >= 0.690; generated v16 >= 0.840, v18 >= 0.730; held-out short >= 0.620 | 0.892 / 0.866 / 0.801 / 0.705; 0.863 / 0.761; 0.641 | 0.879 / 0.862 / 0.810 / 0.726; 0.846 / 0.757; 0.647 | pass |
| 4 | JevBench >= 162; ECE <= 0.09; standard tier >= 60 | 164; 0.079; 60 | **167; 0.052; 63** | pass |

**Adequacy was learned, both ways, and past the teacher.** On HelpSteer2's held-out
responses v19 gained 90 and lost 30 against v18 (p < 0.0001), catching 76% of
inadequate responses where v18 caught 7%, while still accepting 72% of adequate ones.
It is ahead of the frozen Qwen3.5-4B there (0.739 against 0.645), and close to it on
the generated requests from categories never trained on (0.788 balanced against
0.810; +133 / -38 against v18).

**Nothing else was lost.** Every other evaluation is within about 0.02 of v18, none
significantly: MuSiQue 0.892 -> 0.879 (+19 / -34, p = 0.05), HotpotQA 0.705 -> 0.726
(+60 / -40, p = 0.06), BoardgameQA 0.801 -> 0.810, ContractNLI 0.866 -> 0.862, the two
generated sets 0.863 -> 0.846 and 0.761 -> 0.757, held-out short tasks 0.641 -> 0.647.

**JevBench: 167/231 (0.723), the best of any model here** (+14 / -11 against v18, p =
0.69 — not resolvable on its own). Standard tier 60 -> 63, one short of decider-2b
v11's 64; hard 56, easy 48. `adequacy` 7 -> 9 of 12 ("adequate" said on 9 of 12, down
from 11). The gains were also in `judge_hard` +3, `ordinal` +2 and `temporal_numeric`
+2; the losses in `long_policy` -4 (net -3: back from v18's 10 to 7 of 19) and `multi_hop` -3.

**The clearest JevBench change is calibration.** Brier 0.388 -> 0.342 (lower on 147
tasks, higher on 84; sign test p < 0.0001), the best recorded here; standard-tier Brier
0.269 -> 0.193 (decider-2b v11: 0.174), hard 0.629 -> 0.582. ECE 0.079 -> 0.052; the top
band's 53 answers were all right at a claimed 0.957. Ordinal MAE 0.644 -> 0.427 and
paraphrase consistency 0.833 -> 0.861 are also the best recorded.

**"No" spread beyond adequacy, in the right direction.** Across JevBench's 74 yes/no
tasks (47% "yes"), v18 said "yes" to 68% and v19 to 55%: "no" recall 0.487 -> 0.641,
"yes" recall 0.857 -> 0.771, net +3 tasks. `policy`, which the pre-registration flagged
as the family where a spreading "no" could cost, went 10 -> 9.

**Reading.** Judging a response against a request was a missing skill, not a capacity
limit: 6,166 rows taught it to a 2B single-pass model beyond what the frozen 4B does
untrained. The skill seems to carry into the model's yes/no answers generally — the
yes-bias measured on every earlier model shrank across JevBench, not just in
`adequacy` — and with it the calibration. The `long_policy` gain that v18 made did not
survive; v19 is level with v17 there.

*Correction (2026-09-28, before the promotion commit): the `long_policy` figure above
first read "back from v18's 10 to 6"; v19 scores 7 of 19 (4 lost, 1 gained). No other
number was changed.*

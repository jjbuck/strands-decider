# Pre-registration: v14, multi-step documents with a teacher

Committed before labelling the training rows or training. Same discipline as v9-v13.

## Why

JevBench's hard tier fails our models in three ways, found by reading the failing items:
sequential lookups through long procedures (`multi_hop`, `long_policy`), conditions that
must all hold (`policy`), and rules that override or withdraw others (`policy`, `trap`).
The training corpus contains none of them — it is short classification.

Two earlier lessons shape this run:

- **Templated generators teach their surface** (v5 RuleTaker, v9, v10, v11a). These rows
  come from real documents (ContractNLI, MuSiQue) or, capped at a quarter, from one
  generated set (BoardgameQA).
- **Distillation transfers the teacher's behaviour on the prompts it is given** (v12). The
  frozen Qwen3.5-4B teacher was no better than v7 on short classification, so distilling
  it on the old corpus moved nothing on JevBench. On these sources it is clearly better
  than v13 (below), so here its targets carry something the labels alone do not.

## What is being changed

v13 (the leader: Qwen3.5-2B-Base, v7's recipe) plus `data/multistep_v14.jsonl`, built by
`python -m hobson.data.multistep` (`src/hobson/data/multistep.py`, tested in
`tests/test_multistep.py`):

| source | training rows | what it is |
| --- | --- | --- |
| ContractNLI (CC BY 4.0) | 3,930 | full NDAs; is each of 17 claims entailed, contradicted or not mentioned. **Balanced per claim**: unbalanced, answering each claim with its usual label scores 0.679 without reading the contract |
| MuSiQue full (CC BY 4.0) | 5,979 | 2-4 hop questions with distractor paragraphs; each question with both its answerable and unanswerable version. Options are the answer, the earlier hops' answers, distractor entities, and "cannot be determined" |
| BoardgameQA (CC BY 4.0) | 3,000 | conflicting rules resolved by stated preferences: proved, disproved, unknown |

Every label comes from the dataset. Each multi-step row also carries the teacher's
distribution (`teacher_weight: 1.0`, as in v12). Everything else is v13's recipe, except
`group_by_length` is on: the new rows run to ~2,000 tokens among ~250-token rows, and
batching them unsorted would mostly train on padding. It changes which rows share a
batch, not which rows are seen. Config: `configs/experiments/v14.yaml`.

**Held out entirely: HotpotQA** (CC BY-SA 4.0) — two-hop comparisons over ten Wikipedia
paragraphs, eight of them distractors, only questions whose answer is yes, no, or one of
the two compared entities. It is the transfer test: multi-step reading learned from the
other sources, applied to a source never seen.

## Baselines, measured before training

Evaluation sets (`data/multistep_v14_eval.jsonl`), all from splits never trained on:

| evaluation | n | trivial | v13 | teacher |
| --- | --- | --- | --- | --- |
| BoardgameQA (valid) | 900 | 0.379 commonest answer | 0.519 | 0.596 |
| ContractNLI (dev, natural distribution) | 1,026 | 0.679 per-claim guess | 0.564 | 0.772 |
| **HotpotQA (validation, held out)** | 959 | 0.50 chance | **0.705** | 0.846 |
| MuSiQue (dev pairs) | 1,199 | 0.500 always "cannot" | 0.202 | 0.557 |
| — answerable | | | 0.391 | 0.417 |
| — unanswerable | | | 0.013 | 0.697 |

v13 never says a MuSiQue question cannot be answered (0.013).

JevBench, v13: 157/231, hard tier 49/111, ECE 0.085. Held-out short tasks (real
questions): 0.646.

## Predictions

1. **JevBench at least 163/231 (0.706)**, six tasks over v13.
2. **Transfer: HotpotQA at least 0.745** (v13 + 0.04), a source never trained on.
3. **In-distribution** — the data teaches what it should:
   - MuSiQue at least 0.55, with unanswerable accuracy at least 0.50 and answerable
     accuracy not below 0.39;
   - ContractNLI at least 0.70 — above the 0.679 a per-claim guess gets, so the model
     must be reading the contract;
   - BoardgameQA at least 0.60.
4. **Held-out short tasks at least 0.626** (v13 − 0.02).
5. **JevBench ECE at most 0.09.**

Exploratory, no numeric prediction: the families the sources target (`multi_hop`,
`long_policy`, `policy`, `trap`, `ambiguous`; v13 30/64), and question sensitivity on
`tests/question_sensitivity.py`.

## What would count as failure

- **3 holds, 1 and 2 do not**: the model learned these sources and nothing that
  transfers — the v9/v10 failure, on real documents.
- **2 and 3 hold, 1 does not**: multi-step reading transfers across datasets but not to
  JevBench's authored scenarios. A result about the benchmark as much as the data.
- **ContractNLI at or below 0.679**: it learned claim priors despite the balancing, or
  learned nothing from the contracts.
- **4 fails**: the new data displaced what the old corpus taught.

## Which model leads afterwards

v14 becomes the leader if 1, 2 and 4 hold; otherwise v13 stays. A seed replicate
remains outstanding for both (PREREGISTRATION-v11.md).

## Power

231 JevBench tasks cannot resolve differences below about +0.043 (10 tasks), so
prediction 1 is suggestive. The evaluation sets (900-1,199 rows each, paired against
v13 on identical rows) carry the statistical weight: +0.04 on 959 rows is roughly three
standard errors.

## Outcome (added after the run)

Nothing above this section was edited after training. **By the rule fixed above, v13
stays the leader** — prediction 1 missed by two tasks — though v14 is the best-scoring
model this project has produced.

| | prediction | v13 | v14 | |
| --- | --- | --- | --- | --- |
| 1 | JevBench >= 163/231 | 157 | **161** (0.697) | FAIL, by 2 |
| 2 | HotpotQA (held out) >= 0.745 | 0.705 | **0.750** | pass |
| 3a | MuSiQue >= 0.55; unanswerable >= 0.50; answerable >= 0.39 | 0.202; 0.013; 0.391 | **0.852; 0.862; 0.843** | pass |
| 3b | ContractNLI >= 0.70 (per-claim guess 0.679) | 0.564 | **0.842** | pass |
| 3c | BoardgameQA >= 0.60 | 0.519 | **0.752** | pass |
| 4 | held-out short tasks >= 0.626 | 0.646 | 0.637 | pass |
| 5 | JevBench ECE <= 0.09 | 0.085 | 0.086 | pass |

**The transfer is real.** On HotpotQA, never trained on, v14 gains 100 questions and loses
57 against v13 (McNemar p = 0.0007), on both its yes/no (0.699 -> 0.742) and its choice
questions (0.711 -> 0.756). Multi-step reading learned from ContractNLI, MuSiQue and
BoardgameQA carries to a source it never saw.

**JevBench moved in the same direction, below the bar.** 161/231: 8 gained, 4 lost against
v13 (p = 0.39); hard tier 49 -> 52; `multi_hop` — the family MuSiQue targets — 6 -> 9 of 18;
`ambiguous` fell 4 -> 2. Brier 0.438 and paraphrase consistency 0.861 are the best any
model here has recorded; macro 0.690 likewise. v14 gives the teacher's answer on 172
tasks against v13's 166.

**A correction.** The exploratory baseline above — "v13 30/64" on `multi_hop`,
`long_policy`, `policy`, `trap`, `ambiguous` — was written without being computed. It is
32/64; v14 scores 33/64.

**The in-distribution numbers were checked for artefacts of the conversion, because the
student beat its teacher by a wide margin** (MuSiQue 0.852 against 0.557):

- *Option position.* Evaluation lists the dataset's answer first; training shuffles.
  Rescored with shuffled options, v14's MuSiQue is 0.862 — no position effect.
- *Answer string missing* (unanswerable versions replace the supporting paragraphs, so
  the answer often does not appear). On unanswerable questions whose answer string *is*
  still in the text, v14 scores 0.705 against 0.906 where it is absent: part of the gain
  is this cue, but 0.705 is still above the teacher's 0.553 on the same rows.
- *Answer type* (distractors are paragraph titles and earlier hops' answers, so sometimes
  only one option has the type the question asks for). Where several options share the
  expected type, v14 scores 0.820 against 0.960 where one does — against the teacher's
  0.414 on the same rows.

Both cues explain part of the in-distribution gain and neither explains most of it. The
held-out HotpotQA result, where the conversion is different, is the one to rely on.

**The question-sensitivity probe is unchanged** (same answer to a changed question 0.94
of the time on held-out choice tasks). Multi-step documents taught reading documents,
not reading questions.

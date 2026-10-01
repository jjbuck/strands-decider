# External benchmark: JevBench v1 public set

Paths are relative to the repository root unless they are links. A module path such as
`infer.py` is relative to `src/strands_decider/`.

The internal evaluations ([results.md](results.md)) are self-graded on corpora this repository builds. [JevBench](https://benchmarkheaven.com/jev-models)
is an independent benchmark for Jev-class typed decision models, MIT-licensed, and
its `typesafe` adapter targets `/v1/systemone` — so `strands-decider serve` runs against it
unmodified. 231 public tasks across 18 families. Rankings below use the board's own
public-task accuracy for every system (v1.4.2, 25 September 2026, 89 ranked systems);
per-family comparisons use its per-task outcomes where published (the v1.2 set).

**231/231 attempted, 0 failed, schema validity 1.000 (strict).** The server accepted
every task type and never emitted a malformed distribution: JevBench's `typesafe`
adapter ran against the server unchanged.

Figures below are **v19** unless stated, calibrated, at the 3072-token window it was
pre-registered and measured at (served at the 4096 default it scores 168/231;
[The context window is not the ceiling](#the-context-window-is-not-the-ceiling)), measured with the official harness. The headline metrics for v5 to v19 are in the table under
[Summary](results.md#summary).

## Board position (v1.4.2, 25 September 2026)

On the v1.4.2 board (89 ranked systems, public-task accuracy):

| | public accuracy | overall rank | 2B class (33) |
| --- | --- | --- | --- |
| **v19** | **0.723** (167/231) | 50th | 3rd |
| v18 | 0.710 (164) | 52nd, tied with `decider-2b` | 3rd, tied with `decider-2b` |
| v17 | 0.710 (164) | 52nd, tied with `decider-2b` | 3rd, tied with `decider-2b` |
| v16 | 0.706 (163) | 53rd | 4th |
| v14 | 0.697 (161) | 54th, tied with 2 | 4th, tied with `malkuth-2b` |
| v13 | 0.680 (157) | 57th | 5th |
| v7 | 0.667 (154) | 59th, tied with `kev 0.6B` | 5th, tied with `kev 0.6B` |

The board has grown from 52 systems to 89 since v1.2, most of the newcomers above us, so
v14's drop from 30th is the field moving, not the model. The top is reasoning models
(GPT-6 Luna 0.996), then Jev rebuilds on larger torsos, mostly 12B and up (0.83-0.93).

**At 2B parameters or fewer.** The closest competitors, by public accuracy:

| system | base | public accuracy |
| --- | --- | --- |
| `decision-2b` (FlyMy.AI) | MiniCPM5-2B + LoRA + pointer head — 2.5B dense* | 0.753 |
| `system-one-open` | Gemma 4 E2B + LoRA — ~2B effective* | 0.732 |
| **v19** | **Qwen3.5-2B-Base + LoRA + pointer head, 1.9B** | **0.723** |
| `decider-2b` (Mapika) | Qwen3.5-2B-Base + trained readout, 1.9B | 0.710 |
| v18 | Qwen3.5-2B-Base + LoRA + pointer head, 1.9B | 0.710 |
| v17 | Qwen3.5-2B-Base + LoRA + pointer head, 1.9B | 0.710 |
| v16 | Qwen3.5-2B-Base + LoRA + pointer head, 1.9B | 0.706 |
| v14 | Qwen3.5-2B-Base + LoRA + pointer head, 1.9B | 0.697 |
| `malkuth-2b` | Qwen3.8-2B-Distill + LoRA + pointer head (kev) | 0.697 |
| `kev 0.6B` | 0.6B | 0.667 |
| `Open-Jev 2B` | Qwen3.5-2B + LoRA | 0.645 |
| `decision-fast` (FlyMy.AI) | Qwen3-0.6B + LoRA + pointer head | 0.632 |
| `jeff` | 400M | 0.628 |
| `jevact` | Qwen3.5-2B | 0.619 |

\* Just over 2B: MiniCPM5-2B is 2.5B dense by the board's own description, and Gemma 4
E2B has ~2B effective parameters but more in total. `smalljev` (0.606, also MiniCPM5-2B)
is the third such model. Counting all three, v19 is third of 33 in the 2B class;
excluding them, first of 30.

The system nearest v19 is `decider-2b`, on **the same torso** — Qwen3.5-2B-Base with a
one-pass trained readout. Its board entry predates its newest release (v11), which
scores 175/231 on our harness, eight tasks ahead of v19; its 4B sibling `decider-4b-v2`
scores 0.835.

The tier split below is on this repo's partition of the public tasks (easy 48, standard
72, hard 111); v1.4 re-cut the board's own tiers, adding a "judge" tier and sealed
held-out items, so the leaders column is from the v1.2 per-task data.

| tier | v7 | v13 | v14 | v16 | v17 | v18 | v19 | leaders (v1.2 data) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| easy (48) | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** | 1.000 |
| standard (72) | 0.833 | 0.833 | 0.847 | 0.833 | 0.847 | 0.833 | 0.875 | 0.986 |
| hard (111) | 0.414 | 0.441 | 0.468 | 0.495 | 0.495 | 0.505 | 0.505 | 0.964 |

## Against the nearest open systems

Per-task outcomes, joined on all 231 public tasks. v19 is our own harness run; the
others are from the benchmark's per-task data (v1.2, plus the later `decider-2b`
addition). `decider-2b` shares v19's torso; `Open-Jev 2B` is a Qwen3.5-2B with a rank-8
LoRA, and `Open-Jev 9B` the same design scaled 4.5x.

| | **v19** | decider-2b | Open-Jev 2B | jeff 400M | Laya 421M | Open-Jev 9B |
| --- | --- | --- | --- | --- | --- | --- |
| **overall** | **0.723** | 0.710 | 0.645 | 0.628 | 0.584 | 0.775 |
| easy / standard / hard | 1.000 / 0.875 / 0.505 | 1.000 / 0.847 / 0.495 | 1.000 / 0.764 / 0.414 | 1.000 / 0.750 / 0.387 | 0.958 / 0.694 / 0.351 | 1.000 / 0.903 / 0.595 |
| `routing_hard` | 1.000 | 1.000 | 0.800 | 0.600 | 0.200 | 1.000 |
| `intent` | 0.917 | 1.000 | 0.708 | 0.708 | 0.833 | 1.000 |
| `adversarial` | 0.667 | 0.667 | 0.500 | 1.000 | 0.667 | 1.000 |
| `long_policy` | 0.368 | 0.316 | 0.263 | 0.105 | 0.211 | 0.474 |
| `multi_hop` | 0.444 | 0.556 | 0.500 | 0.389 | 0.333 | 0.667 |
| `policy` | 0.750 | 0.833 | 0.917 | 0.917 | 0.750 | 1.000 |
| `trap` | 0.875 | 1.000 | 0.875 | 0.500 | 0.000 | 1.000 |
| `judge_hard` | 0.588 | 0.529 | 0.529 | 0.471 | 0.412 | 0.765 |
| `temporal_numeric` | 0.267 | 0.333 | 0.067 | 0.400 | 0.333 | 0.133 |

`decider-2b` is the most informative comparison: the same Qwen3.5-2B-Base torso, a
different recipe. v19 leads its board entry by three tasks, 167 to 164 (48, 63 and 56
by tier against 48, 61 and 55), and each gets tasks right that the other misses: 19
for v19, 16 for `decider-2b`. Net by family, v19 is ahead by three tasks in `ambiguous`, two each in
`adequacy` and `routing`, and one each in `judge_hard`, `long_policy`, `ordinal` and
`tradeoff`. `decider-2b` is ahead by two each in `intent` and `multi_hop`, and by one
each in `policy`, `probability`, `temporal_numeric` and `trap`.

v19 leads the other three 2B-class peers overall and on the hard tier, and is the best
of the five 2B-class systems in the table on `judge_hard` (where the answer turns on a
condition or a withdrawn instruction) and `long_policy`. Its weakest families against
them are `policy`, joint lowest with Laya, and `temporal_numeric`, where only
`Open-Jev 2B` scores lower.

Scaling Open-Jev 2B to 9B gains +0.130 overall, nearly all of it in
the hard tier (0.414 -> 0.595). Our own Qwen3 4B reproduces that curve at a smaller
jump (+0.043, hard 0.396 -> 0.450; [Scaling, and the measurement that hid it](../research/history.md#scaling-and-the-measurement-that-hid-it)).

## The shape matters more than the rank

`easy` is saturated: v19 ties the leaders at 1.000, and so do most systems on the board.
**The separation lives in `hard`.** By family, v19:

| strong | | weak | |
| --- | --- | --- | --- |
| `extraction` | 1.000 | `temporal_numeric` | 0.267 |
| `fact` | 1.000 | `probability` | 0.300 |
| `ordinal` | 1.000 | `long_policy` | 0.368 |
| `routing_hard` | 1.000 | `multi_hop` | 0.444 |
| `tool_selection` | 1.000 | `tradeoff` | 0.500 |
| `intent` | 0.917 | `judge_hard` | 0.588 |

The strong families are classification — match a state to a category. The weak ones
need multi-step inference, arithmetic, or weighing competing considerations, and a
System One model does one forward pass and a linear readout. The top two systems are
explicitly reasoning models that spend tokens deliberating.

Multi-step inference is not out of reach, only hard to teach. `multi_hop`, the family
v14's MuSiQue rows target, rose from 0.333 (v13) to 0.500 — the first time a change of
training data has moved a hard family here. v16's generated documents then moved `trap`
(6 -> 8 of 8) and `tradeoff` (2 -> 4 of 6), and v18's weak-skill questions `long_policy`
(7 -> 10 of 19). `temporal_numeric` looks capability-bound: no non-reasoning system in
the table passes 0.533 on it, Jev itself scores 0.27, the frozen Qwen3.5-4B 3 of 15,
v15 and v16 each lost two of v14's four (v17 one), and v18's questions on rule versions
and running totals, which rose 0.16 on unseen domains, left it at two; v19 has four.
v19's adequacy rows moved `adequacy` itself (7 -> 9 of 12) and, with it, `judge_hard`
(7 -> 10 of 17), where a described action is judged against a policy; `long_policy`
gave back v18's gain (10 -> 7 of 19).

## The context window is not the ceiling

The default window is **4096 tokens**: raised from 1024 to 3072 on v5 (Qwen3-1.7B), by
the two measurements below, and to 4096 for serving from v19, by the third. The
Qwen3.5 torso handles far longer positions natively.

First, a genuine bug: the state was tokenised against the whole window before the
question, so a state that filled it left the question a floor of 8 tokens — too few
to hold the option list. All 19 `long_policy` items sat at exactly 1032 input tokens,
scoring 0.316 against a 0.301 chance rate for those option counts. The model was
picking among options it could not see. Fixed by reserving the question's tokens
first (`_fit` in `infer.py`), which recovered ECE 0.115 -> 0.079 but moved accuracy
by one item.

Second, a window sweep, inference-only — `max_length` is a config field and Qwen3
handles 32k positions natively, so no retraining was needed to test it:

| | 1024 | 3072 | 4096 |
| --- | --- | --- | --- |
| accuracy | 0.632 | **0.636** | 0.636 |
| ECE | 0.079 | 0.092 | 0.096 |
| p95 latency | 272 ms | 590 ms | 610 ms |
| `long_policy` | 0.263 | **0.368** | 0.368 |
| `multi_hop` | 0.278 | **0.333** | 0.333 |
| all 14 unaffected families | identical | identical | identical |

Three things follow. **Position generalisation is free**: the LoRA never saw a
sequence past 1811 tokens in training, yet all 14 families that already fitted are
bit-identical at 3072 and 4096 — no degradation at unseen positions. **3072 and 4096
are indistinguishable**: eight `long_policy` items were still cut at 3072 and none at
4096, they received up to 790 more tokens, their probabilities moved by up to 0.17 —
and exactly one prediction flipped, wrong to wrong. The model reads the extra text and
cannot use it. **So the context curve is flat past ~3072**, and `long_policy` sits at
0.368 with the complete document in context, barely clear of chance.

3072 was therefore the default rather than 4096: same accuracy, less latency. Overall
the sweep is worth +1 item of 231 (McNemar p = 1.000) — it is not an accuracy win. It
is the removal of a confound, and it cost p95 latency.

Third, the same sweep on v19 (Qwen3.5-2B), whose `long_policy` documents run 2,202 to
3,665 tokens against at most 2,193 for any generated training document. At 4096 only
the eight documents that 3072 cuts change at all; everything else is bit-identical.
Of those, two firmed up on answers already right (0.72 -> 0.83 and 0.52 -> 0.66 on the
right option), and one flipped, wrong to right, on a five-option question where the
right option went from 0.36 to 0.37: 167 -> 168 of 231, Brier 0.3421 -> 0.3416, ECE
0.0522 -> 0.0507. That flip is noise, not a gain. What changed is the cost: on the
hybrid torso p95 latency is 296 ms at 4096 against 299 ms at 3072. With nothing to
trade, **4096 is the default from v19 on**, so that long documents are not silently
cut. v19's figures in these documents are its pre-registered measurement, at 3072. No
training row is longer than 3,072 tokens, so training at 4096 changes nothing.

Two smaller notes. `ordinal` scores 0.917 here while `score` is our weakest primitive
internally — JevBench's rubrics are far crisper than 5-point sentiment, which supports
the label-noise reading. And JevBench's own paraphrase-consistency measure (0.833)
independently reproduces the phrasing brittleness measured in
[How the question is worded moves the answer](../research/history.md#how-the-question-is-worded-moves-the-answer).

## Calibration: v16 to v19 fixed the top band

v14's JevBench ECE is 0.086, and its top bin was overconfident: its 76 answers at 0.9
confidence or above were right 0.789 of the time at a mean claim of 0.952. v16's ECE is
0.069, and its 53 answers at that level were right 0.981 of the time at a claim of
0.961. Its middle band now claims slightly more than it delivers (0.662 right at a mean
claim of 0.737). v17 (ECE 0.083) keeps the top band honest — its 67 answers at 0.9 or
above were right 0.940 of the time at a claim of 0.948 — with the same middle-band
overclaim (0.667 at 0.746). v18 (ECE 0.079): 54 answers at 0.9 or above, right 0.963
of the time at a claim of 0.948, and a smaller middle-band overclaim (0.703 at 0.736).
v19 (ECE 0.052, the lowest recorded here): 53 answers at 0.9 or above, all right, at a
claim of 0.957, and the same small middle-band overclaim (0.707 at 0.736). Its Brier
score, 0.342, is lower than v18's on 147 of 231 tasks — the change came from the
adequacy rows loosening a yes-bias every earlier model had: on JevBench's 74 yes/no
tasks (47% "yes"), v18 said "yes" to 68% and v19 to 55%.
The temperatures were fitted on held-out classification tasks, and
JevBench's hard tier is further from that distribution; for automation on traffic
unlike the training data, fit the temperature on a sample of that traffic.

## Reproducing, and two caveats

```bash
strands-decider serve checkpoints/hobson-2b-recipe --port 8099   # inside WSL2 for v13 onwards
git clone https://github.com/fstandhartinger/jevbench && cd jevbench
git checkout 1bcc55eb6c8cffde2306b3db03ede39b61c6152a   # the commit evaluation/jevbench/jevbench.sh pins
pip install -e .
cat datasets/public/{original,easy,hard}.jsonl > /tmp/all.jsonl
python -m jevbench.cli run --tasks /tmp/all.jsonl --adapter typesafe   --endpoint http://127.0.0.1:8099 --key-env '' --model hobson-2b-recipe   --cost-basis no_billable_account_public_endpoint --reserve-usd 0   --results OUT/results.jsonl --raw-dir OUT/raw --ledger OUT/ledger.jsonl
python -m jevbench.cli summarize --tasks /tmp/all.jsonl   --results OUT/results.jsonl --ledger OUT/ledger.jsonl --public-export OUT/summary.json
```

On Windows, confirm the server actually bound port 8099 before trusting a run. A stale
server holds the port, the new one exits with `[Errno 10048]`, and the benchmark then
silently measures the *old* model. This has cost three runs here. `/health` now reports
the checkpoint path and `max_length`, so check both before a run.

`--raw-dir` must sit outside the cloned repo; the harness refuses otherwise so task
text cannot leak into a public checkout.

**The harness needed one patch to run on Windows.** `jevbench/budget.py` imports
`fcntl`, which is POSIX-only, for advisory locking on the cost ledger. Replaced with
a no-op under `ImportError`. Single process, no billing, and nothing in scoring,
accuracy or calibration is touched — but it is a modification to a benchmark, so it
is disclosed.

**This is not the leaderboard composite.** Since v1.4, JevBench's headline "JevBench
Score" combines chance-corrected Intelligence (tier-weighted, including a judge tier and
sealed held-out items), Calibration, Speed and Cost, all measured by the maintainers on
their own hardware. We ran the 231 public tasks locally, so speed and cost are not
comparable and the sealed items are absent. The accuracy comparison above *is*
like-for-like: the same 231 task ids, and the board's own public-task accuracy for every
other system.

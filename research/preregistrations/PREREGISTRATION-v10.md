# Pre-registration: v10, minimal pairs for cross-reference and requirement checking

Committed before training. Same discipline as v9, whose pre-registration correctly
named its own failure in advance: the model learned the generators (0.436 -> 0.972)
while the targeted JevBench families moved by exactly zero tasks.

## What changed since v9, and why

Three changes, each taken from evidence rather than from the benchmark's structure:

1. **No arithmetic.** On `temporal_numeric` every non-reasoning system in the JevBench
   table stalls — Jev itself scores 0.27, Gemini 3.1 Flash-Lite 0.13 — and 60% of v9's
   synthetic corpus was aimed there. In v9's own transfer eval, the held-out variants
   that did *not* transfer were the arithmetic ones (rounding mode, order of
   operations); reading an unseen clamping rule did. So the targets are now
   `multi_hop` and `judge_hard`, which non-reasoning systems reach 0.88 on and where a
   2B system (`system-one-open`) scores 0.82 on `judge_hard` against our 0.41.
2. **Minimal pairs.** Every training item is half of a pair: identical document, one
   decisive fact changed (1–3 lines), different answer. A model that learned the
   surface cannot separate the halves. kev 0.6B trains on pairs of this kind.
3. **Realistic documents.** ~2,200 characters for cross-reference, with a superseded
   table marked do-not-use, a glossary, a revision history, varying section order, and
   sometimes a requester's note naming a plausible wrong answer.

Generator: `src/hobson/data/synth_pairs.py`. Labels are re-derived from the rendered
documents by `tests/verify_synth_pairs.py`: 0% disagreement on 6,000 items across both
splits, and 0 pairs whose halves share an answer.

## What is being changed

One variable against v7: the corpus. v10 = v7's recipe (Qwen3-1.7B-Base, pointer
readout, KL anchor 0.3, 3072-token window, one epoch) on `train_v5.jsonl` plus 20,000
minimal-pair items (10,000 pairs). v9's synthetic corpus is **not** included.

## Split, fixed now

**PRIMARY** — families these generators target. v7: **14/35 = 0.400**.
`multi_hop` 7/18, `judge_hard` 7/17.

**TRANSFER** — no generator targets these: `long_policy`, `policy`, `trap`,
`ambiguous`, `tradeoff`, `adversarial`, `probability`. No numeric prediction.

**NOT TARGETED** — `temporal_numeric` (2/15). Dropped as a target on the evidence
above; expected not to move.

**CONTROL** — v7 103/113 = 0.911, easy tier 48/48.

## Held out of training entirely

- the `claim` domain skin for cross-reference (unseen vocabulary, same lookup skill)
- the `date` requirement type for requirement checking (never listed as a requirement
  in training, while date formats vary freely there — a deliberately hard test)

## v7 baselines on the synthetic evals

| eval | item | pair (both halves right) | same answer to both halves |
| --- | --- | --- | --- |
| in-distribution, reqcheck | 0.497 | 0.091 | 0.803 |
| in-distribution, xref | 0.385 | 0.141 | — |
| transfer, reqcheck (`date`) | 0.473 | **0.004** | **0.992** |
| transfer, xref (`claim`) | 0.422 | 0.177 | — |
| **in-distribution overall** | **0.441** | **0.116** | |
| **transfer overall** | **0.447** | **0.091** | |

On the held-out date requirement v7 gives the same answer to both halves 99.2% of the
time: it cannot see the decisive fact. That is the behaviour this corpus targets.

## Predictions

1. **PRIMARY ≥ 18/35 = 0.514** (+4 tasks over v7).
2. **CONTROL ≥ 0.89** and the easy tier stays at 48/48.
3. **Overall JevBench ≥ 0.6667.**
4. **Synthetic in-distribution: item ≥ 0.85 and pair ≥ 0.70.**
5. **Synthetic transfer: item ≥ 0.70 and pair ≥ 0.45.**

Pair accuracy is part of 4 and 5 because it is the direct test for surface learning:
a model answering both halves the same way can get at most one right.

## What would count as failure

- **4 holds but 1 does not** — v9 again: learned the generators, not the capability.
- **4's item accuracy holds but its pair accuracy does not** — learned the surface
  even of the generator; the minimal-pair design failed at its one job.
- **1 and 4 hold but 5 does not** — memorised the training skins and requirement types
  rather than learning to read an unfamiliar document.
- **2 fails** — a net loss regardless.

## Power

As before: 231 tasks cannot resolve an overall difference below ~+0.043, so prediction
3 is a guard. PRIMARY at n=35 needs roughly +5–6 tasks to be clearly outside noise, so
prediction 1 is set at a level that would be suggestive rather than conclusive. The
synthetic evals (n=2,500 and 3,000, with 1,250 and 1,500 pairs) are where the real
statistical power is.

## Outcome (added after the run)

Nothing above this section was edited after training. Scored against the predictions
as written:

| | prediction | result | |
| --- | --- | --- | --- |
| 1 | PRIMARY >= 18/35 | 14/35 -> **11/35** | FAIL |
| 2 | CONTROL >= 0.89, easy 48/48 | 0.911 -> 0.903, easy 48/48 | pass |
| 3 | overall >= 0.6667 | 154 -> 145/231 (-9, McNemar p = 0.150) | FAIL |
| 4 | synthetic in-distribution: item >= 0.85, pair >= 0.70 | 0.951 / 0.902 | pass |
| 5 | synthetic transfer: item >= 0.70, pair >= 0.45 | 0.747 / 0.504 | pass |

The first failure listed — *4 holds but 1 does not* — again, and worse than v9: the
targeted families went backwards. `multi_hop` fell from 7/18 to 4/18 (`judge_hard`
7/17 unchanged), and all nine lost tasks are in the hard tier, which fell from 46 to
38 of 111. TRANSFER fell from 35/68 to 31/68. `temporal_numeric`, dropped as a target,
went from 2/15 to 1/15.

The second failure — item accuracy holding while pair accuracy does not — did **not**
happen. The model separates the halves of a pair, including in an unseen domain:

| | item | pair (both halves right) | same answer to both halves |
| --- | --- | --- | --- |
| xref, training skins | 0.999 | 0.998 | 0.002 |
| xref, held-out `claim` skin | 0.901 | 0.821 | 0.020 |
| reqcheck, held-out `date` | 0.593 | 0.187 | 0.813 |

So this is not surface learning of the generator. The model learned to find the
decisive fact in these documents, and that did not carry to JevBench's. The skins vary
vocabulary but share one document structure; the most likely reading is that the
structure is what was learned, and that the procedure learned for it displaced some of
what v7 did on real procedure documents.

With RuleTaker (v5) and v9, this is the third templated synthetic corpus that reached
near its ceiling on its own generator and moved JevBench by zero or less. v7 remains
the recommended model.

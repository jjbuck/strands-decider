# Pre-registration: v9, synthetic stated-rule execution

Written and committed **before** the corpus was generated or any training run started.
Its purpose is to stop the result being interpreted after the fact. Nine of the last
ten interventions in this project produced differences too small for 231 tasks to
resolve, and the temptation in that situation is to find the reading that makes the
run look successful. The predictions below are the reading, fixed in advance.

## What is being changed

One variable against v7: the corpus. v9 = v7's recipe (Qwen3-1.7B-Base, pointer
readout `dim=256`, `kl_frozen_weight: 0.3`, 3072-token window, one epoch) trained on
`train_v5.jsonl` **plus** synthetic stated-rule items.

Nothing about the architecture, the optimiser or the window changes.

## Why synthetic data at all

JevBench's hard tier is dominated by one structure: a precisely stated rule plus a
specific case, answered by executing the rule. Ground truth for that structure is
*computable*, so a generator produces it with no annotator and therefore no label
noise — and label noise is the documented ceiling on the existing corpus.

The generator is in `src/hobson/data/synth.py`. Its scenarios, domains and field names
are unrelated to anything in JevBench; only the abstract structure is shared.

## The honesty problem, stated plainly

Designing training data after reading benchmark items risks fitting the benchmark even
when no item is copied. Roughly fifteen JevBench items were read closely while
designing the generators, across `temporal_numeric`, `multi_hop`, `judge_hard`,
`policy`, `trap`, `probability`, `ambiguous` and `long_policy`. That information cannot
be unlearned, so the split below separates families the generators were built against
from families they were not.

## The split, fixed now

**PRIMARY** — families whose structure a generator directly targets. v7 scores
**16/50 = 0.320**.

| family | generator targeting it | v7 |
| --- | --- | --- |
| `temporal_numeric` | `syn_threshold`, `syn_units`, `syn_window` | 2/15 |
| `multi_hop` | `syn_bands` | 7/18 |
| `judge_hard` | `syn_verify` | 7/17 |

**TRANSFER** — families no generator targets. If the model learned a capability rather
than a task, these should move too. v7 scores **35/68 = 0.515**: `long_policy` 8/19,
`probability` 3/10, `policy` 10/12, `ambiguous` 1/7, `trap` 5/8, `tradeoff` 4/6,
`adversarial` 4/6.

**CONTROL** — families that have nothing to do with stated-rule execution and must not
regress. v7 scores **103/113 = 0.911**, and the easy tier is 48/48.

## Predictions

Recorded before the run. Each is falsifiable.

1. **PRIMARY improves to at least 0.44** (22/50, up from 16/50). Below that, the
   synthetic corpus did not teach what it was built to teach.
2. **CONTROL does not fall below 0.89** (100/113) and the easy tier stays at 48/48.
   A larger drop means the corpus displaced capability rather than adding it.
3. **Overall JevBench does not regress below 0.6667** (154/231).
4. **Synthetic in-distribution** (`syn_bands`, `syn_threshold`; same distribution as
   training) rises from ~0.45 to **at least 0.75**. This only shows the data is
   learnable at all.
5. **Synthetic transfer** (`syn_units` with *round-to-nearest*, `syn_window` with
   *month-end clamping*, `syn_verify` with *wrong-order*, all three held out of
   training entirely) rises from ~0.52 to **at least 0.65**.

TRANSFER carries no numeric prediction. Any movement there is the interesting result
and will be reported as exploratory, not as a confirmed effect.

## What would count as failure

- **Prediction 4 holds but 1 does not** — the model learned the generator's surface and
  nothing else. This is exactly what RuleTaker did in v5: held-out depths improved by
  0.173 while JevBench moved 0.000. It is the most likely failure mode and the reason
  prediction 5 exists.
- **Predictions 1 and 4 hold but 5 does not** — the model memorised the specific rule
  variants seen in training rather than learning to read the rule from the document.
- **Prediction 2 fails** — a net loss regardless of anything else.

## Power, acknowledged in advance

231 tasks cannot resolve an overall difference below roughly +0.043, so prediction 3 is
a guard rather than a test. PRIMARY at n=50 can resolve about +0.14; the synthetic
evals at n≈2,500 resolve far less and are where the real statistical power lives.

A result that satisfies 1, 2, 4 and 5 will still be reported with its McNemar p-value
on the overall number, and that p-value will not be significant.

## Outcome (added after the run)

Nothing above this section was edited after training. Scored against the predictions
as written:

| | prediction | result | |
| --- | --- | --- | --- |
| 1 | PRIMARY >= 22/50 | 16/50 -> **16/50** | FAIL |
| 2 | CONTROL >= 100/113, easy 48/48 | 103 -> 105/113, easy 48/48 | pass |
| 3 | overall >= 0.6667 | 0.6667 -> 0.6580 (-2 tasks, McNemar p = 0.815) | FAIL |
| 4 | synthetic in-distribution >= 0.75 | 0.436 -> 0.972 | pass |
| 5 | synthetic transfer >= 0.65 | 0.533 -> 0.673 | pass, narrowly |

The first failure listed under *What would count as failure* — prediction 4 holds but
1 does not — is what happened. PRIMARY did not move by a single task: 48 of 50
verdicts are identical to v7's and the two that flipped cancel (`temporal_numeric`
2/15, `multi_hop` 7/18, `judge_hard` 7/17, all unchanged).

Prediction 5 passes on the aggregate only. Of the three held-out variants, one
transferred and two did not move:

| held out of training | v7 | v9 |
| --- | --- | --- |
| `syn_window`, month-end clamping | 0.503 | **0.908** |
| `syn_units`, round-to-nearest | 0.612 | 0.613 |
| `syn_verify`, wrong-order fault | 0.483 | 0.498 |

The one that transferred is reading a stated rule; the two that did not are
arithmetic.

TRANSFER, which carried no prediction, fell from 35/68 to 31/68. `trap` rose from 5/8
to 8/8, unexplained; `tradeoff`, `adversarial`, `ambiguous`, `long_policy` and
`probability` each fell. ECE (0.0678 -> 0.0580) and ordinal MAE (0.5922 -> 0.5560)
improved. The in-training validation accuracy rose to 0.895 against v7's 0.835, which
reflects how easily the synthetic rows fit, not the benchmark.

v7 remains the recommended model. What changed for v10: `temporal_numeric` was dropped
as a target (no non-reasoning system in the JevBench table passes 0.533 on it), and
items became minimal pairs in longer, messier documents.

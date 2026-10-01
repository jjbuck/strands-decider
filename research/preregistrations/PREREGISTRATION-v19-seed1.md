# Pre-registration: v19-seed1, the seed replicate

Committed before training. A measurement, not an intervention: no default changes on
its result.

## Why

Every run since v13 has been read against a single training run of the recipe before
it. JevBench cannot resolve differences below about 10 tasks from sampling alone, and
how much a retrain of the *same* recipe moves on its own has never been measured.
Several recent readings depend on it:

- v18 raised `long_policy` from 7 to 10 of 19 and v19 took it back to 7. Eight of the
  family's 19 tasks sit near 50/50 and have flipped between runs; whether the swing was
  the data or the seed is not known.
- v17 -> v18 was +9 / -9 tasks on JevBench and v18 -> v19 +14 / -11: how many of those
  flips a retrain produces by itself is not known.
- The held-out evaluations (MuSiQue 0.892 -> 0.879, p = 0.05; HotpotQA 0.705 -> 0.726,
  p = 0.06) have been read with sign tests that treat the model as fixed, which ignores
  training noise.

## What is being changed

Nothing but the seed: v19's config, data and teacher file, with `seed: 1` in place of
`seed: 0` (`configs/experiments/v19-seed1.yaml`). The seed sets the LoRA and readout
initialisation, which 3% of rows are held out for validation, the data order, and the
per-batch option shuffles and question phrasings. Trained under WSL2 like v19,
evaluated with the same chain, and served for JevBench at 3072 tokens, the window v19
was measured at.

## What will be reported

For every evaluation v19 has, v19-seed1's score, the difference from v19, and the
per-row agreement (tasks or rows right in one run and wrong in the other):

- JevBench (231): total, tiers, per family, Brier, ECE; task flips between the seeds.
- HelpSteer2 adequacy (234) and generated adequacy (302, balanced accuracy).
- MuSiQue, ContractNLI, BoardgameQA, HotpotQA; the v16 and v18 generated sets.
- Held-out short tasks; the question-sensitivity probe.

## Expectations (not a pass/fail bar)

Written down so that the result can surprise us:

1. JevBench total within 5 tasks of v19's 167, with 10-20 tasks flipping between seeds
   (v17 -> v18, a data change, flipped 18).
2. `long_policy` within 3 tasks of v19's 7, the flips concentrated in the eight tasks
   that have already flipped between v16 and v19.
3. Held-out evaluations of 900-1,200 rows within about 0.015 of v19; the adequacy sets
   (234 and 302 rows) within about 0.03.

## How it will be used

The seed-to-seed differences become the noise floor for reading future runs: a change
on a measure that is no larger than the spread between these two seeds of the same
recipe is not read as an effect. One replicate gives one draw of that spread, not its
distribution; if the two seeds differ by more than the expectations above, a third
seed is the next step before any further recipe change is read.

v19 stays the default whatever the result.

## Status

Backlogged on 2026-09-28: training was stopped at step 1,420 of 3,738 to free the GPU
for other work, and no results were taken from the partial run. When it is resumed it
restarts from scratch with the config and expectations above, unchanged.

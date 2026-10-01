# The recipe step by step

The six core steps of `training/recipe.sh`, with the command and what it writes. [README.md](README.md#usage) gives the one-line form. Paths are relative to the repository
root unless they are links.

## 1. Build the corpora

Recipes convert public classification datasets into `(state, question, answer)` triples.
The mix is deliberately varied — topic, intent, sentiment, entailment, ordinal ratings,
yes/no, rule chaining — because a corpus drawn from one task teaches memorisation, not
binding.

```bash
strands-decider data build --out data/train_v5.jsonl --max-options 24 \
  -r ag_news -r banking77 -r clinc150 -r dbpedia -r lang_id -r yahoo_topics \
  -r spam -r toxicity -r mnli_entail -r boolq \
  -r paws -r vitaminc -r wnli -r pubmed_qa \
  -r yelp_stars -r sst5_sentiment -r app_reviews -r formality \
  -r ruletaker_d0 -r ruletaker_d1 -r ruletaker_d2 \
  -r emotion -r massive_intent -r sarcasm -r hate_severity \
  -r ruletaker_d3 -r ruletaker_d5 -r ruletaker_natlang \
  --holdout emotion --holdout massive_intent --holdout sarcasm --holdout hate_severity \
  --holdout ruletaker_d3 --holdout ruletaker_d5 --holdout ruletaker_natlang
```

The build writes `data/train_v5.jsonl` (100,449 rows, 21 tasks) and
`data/train_v5.holdout.jsonl` in about two minutes. It is deterministic: a fresh build
gives the same files on a Mac and on AWS (`data/train_v5.jsonl` sha256 `3d17d148…`, full hashes in
[Reproduction contract](../data/README.md#reproduction-contract)). The committed teacher targets align with this build row by row. Whether it is byte-identical
to the original v5 file that v7 trained on is not verified.

`--holdout` routes whole tasks to the `.holdout.jsonl` file. Held-out *tasks* — not
just held-out rows — are the honest test of whether the readout learned to bind to
prompt-described options, since in-distribution accuracy can be reached by memorising
label sets. Of the seven held out, the four with unseen label sets and rubrics are the
generalisation test; the three RuleTaker depths share a generator with the trained ones,
so `recipe_v7.sh` drops them into `data/holdout_v5_norule.jsonl` for calibration and
evaluation.

Datasets with large label sets (banking77's 77 intents) are subsampled per example to
`--max-options`, always keeping the gold option. Varying N across examples is itself
useful signal: it stops the readout from assuming a fixed number of live options.

```bash
strands-decider data recipes                     # what is available
strands-decider data stats data/train_v5.jsonl   # composition
strands-decider data peek  data/train_v5.jsonl   # exactly what the model will see
```

The multi-step rows come from two downloaded releases and two HuggingFace datasets:

```bash
training/recipe.sh fetch                  # ContractNLI, MuSiQue and HelpSteer2 into data/raw/ (~350 MB)
python -m strands_decider.data.multistep    # data/multistep_v14.jsonl and _eval.jsonl
```

That writes 12,909 training rows and 4,084 evaluation rows from splits never trained
on, including 959 from HotpotQA, a source held out of training entirely. ContractNLI is
balanced per claim, because its 17 claims are asked of every contract and answering each
with its usual label scores 0.679 without reading; MuSiQue keeps each question's
answerable and unanswerable versions together. Rows longer than the window are
dropped, never truncated. `tests/test_multistep.py` checks both properties.

## 2. Label with the teacher

```bash
python -m strands_decider.data.teacher --src data/multistep_v14.jsonl \
  --out data/teacher_multistep_v14.jsonl --shift-by data/train_v5.jsonl
```

A frozen Qwen3.5-4B reads each multi-step row the way SemIf reads it — chat template,
JSON, one forward pass, softmax over option letters — with prompts checked
byte-identical to SemIf's own (`tests/test_teacher.py`, which needs SemIf's code and skips
without it). Rows are written as they are
labelled and a rerun resumes. `--shift-by` also writes the labels indexed into the
concatenated training files, which is what training reads. About an hour under WSL2.

## 3. Train

Before this step, `recipe.sh parent` trains the parent checkpoint from
`configs/train-parent.yaml`, and `recipe.sh replay` labels the multi-step rows with it,
writing `data/replay_parent_multistep.jsonl`, which `configs/train.yaml` reads.

```bash
strands-decider train --config configs/train.yaml       # v19, under WSL2, ~6 h
strands-decider train --config configs/train-v7.yaml    # v7, on Windows, ~3 h
```

LoRA on the torso, the pointer readout and the KL terms (the teacher's only for `configs/train.yaml`);
bf16, gradient checkpointing,
one epoch, length-grouped batching. Each writes to the `output_dir` of its config,
`checkpoints/hobson-*-recipe`.
Head-only training (`freeze_torso: true`) is much cheaper and much weaker —
useful to sanity-check a pipeline, not to produce a usable model.

## 4. Calibrate

```bash
strands-decider calibrate checkpoints/hobson-2b-recipe --data data/holdout_v5_norule.jsonl
```

Fits a temperature **per primitive**, minimising **ECE**, and writes them into the
checkpoint. **Run this before serving.**

It cannot change a `choice` argmax, so choice accuracy is untouched. It *does* change
the value returned for `noul` and `score`, since those are read off the distribution
itself rather than its argmax — softening a score pulls it toward the midpoint. Skip
calibration entirely and the thresholds of the
[routing convention](../docs/architecture.md#the-routing-convention) will not hold, which is the failure
mode that matters: a model that is 70% accurate but reports 0.99 confidence fails
silently in exactly the place you were relying on it.

Fitted against ECE rather than NLL on purpose. The two disagree: fitting `noul` to NLL
here chose T=2.63, which left it under-confident by 0.29 — worse calibrated than no
scaling at all, even though likelihood improved.

`calibrate` uses only the `calib` half of the file and `eval` defaults to the `test`
half — a deterministic, disjoint partition of the same file, so the reported
calibration error is one the model would actually achieve on rows the temperature was
never fitted on. Fitting and scoring on the same rows reports a flatteringly low ECE
that evaporates in production. Pass `--split all` to either command to override.

## 5. Evaluate

`training/recipe.sh eval` runs the held-out classification and multi-step evaluations. The commands
and what they measure are in [Internal evaluations](../evaluation/README.md#internal-evaluations).

## 6. Serve

`strands-decider serve` answers requests over HTTP. See [Serve](../docs/inference.md#serve).


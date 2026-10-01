#!/usr/bin/env bash
# The v7 recipe, end to end: build the corpus, train, calibrate, evaluate.
#
# v7 led until v13 tied it on a Qwen3.5 torso (configs/experiments/v13.yaml, trained under WSL2).
# v7 trains on Windows, and v13 uses the same corpus. Every step here is the command
# that produced v7:
#
#   - the corpus build (~2 min): a fresh build today gives the sha256 that
#     data/SHA256SUMS records for data/train_v5.jsonl and data/train_v5.holdout.jsonl.
#     Identity with v7's original files is not verified.
#   - calibration is fitted on the held-out file *without* the RuleTaker depths, which
#     share a generator with the trained depths and would fit the temperature to an
#     easier distribution than a genuinely unseen task;
#   - training uses configs/train-v7.yaml (~2-3 h on one RTX 3090). The one difference
#     from v7 is length-grouped batching; set group_by_length: false there for an exact
#     reproduction.
#
# Writes to checkpoints/hobson-1.7b-recipe.
#
# Usage: training/recipe_v7.sh [build|train|calibrate|eval|all]   (default: all)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."  # every path below is relative to the repo root
STEP="${1:-all}"
CKPT="checkpoints/hobson-1.7b-recipe"

# PY is the interpreter (recipe.sh sets it); the default is the active environment's python.
PY="${PY:-python}"
export PYTHONUNBUFFERED=1 HF_HUB_DISABLE_SYMLINKS_WARNING=1 HF_HUB_DISABLE_PROGRESS_BARS=1
strands-decider() { "$PY" -u -m strands_decider.cli "$@"; }

build() {
  echo "############ build the v5 corpus ############"
  # Seven whole tasks are held out: four with label sets and rubrics never seen in
  # training, and three RuleTaker depths (in-family transfer, reported separately).
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

  # The calibration / evaluation file: held-out tasks minus RuleTaker. Filtered as
  # bytes so each line keeps its own ending.
  "$PY" - <<'EOF'
import json
src = open("data/train_v5.holdout.jsonl", "rb").read().splitlines(keepends=True)
keep = [ln for ln in src if not json.loads(ln)["task"].startswith("ruletaker")]
open("data/holdout_v5_norule.jsonl", "wb").write(b"".join(keep))
print(f"data/holdout_v5_norule.jsonl: {len(keep):,} of {len(src):,} held-out rows")
EOF
}

train() {
  echo "############ train ############"
  strands-decider train --config configs/train-v7.yaml
}

calibrate() {
  echo "############ calibrate (calib half of the unseen tasks) ############"
  strands-decider calibrate "$CKPT" --data data/holdout_v5_norule.jsonl
}

evaluate() {
  # Same files and sample sizes as v7's reported numbers. Held-out eval reads the
  # `test` half, disjoint from the rows calibration was fitted on.
  mkdir -p reports
  echo "############ eval: unseen tasks ############"
  strands-decider eval "$CKPT" --data data/holdout_v5_norule.jsonl --limit 6000 \
    --out reports/recipe_heldout.json
  echo "############ eval: tasks in the training mix ############"
  strands-decider eval "$CKPT" --data data/train_v5.jsonl --limit 6000 --split all \
    --out reports/recipe_indist.json
}

case "$STEP" in
  build) build ;;
  train) train ;;
  calibrate) calibrate ;;
  eval) evaluate ;;
  all) build; train; calibrate; evaluate ;;
  *) echo "unknown step: $STEP (build|train|calibrate|eval|all)" >&2; exit 2 ;;
esac

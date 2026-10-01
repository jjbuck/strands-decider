# Legacy: the v7-era evaluation script

`run_eval.sh` calibrated and evaluated a v7-era checkpoint on Windows. These are its three
commands, with its variable defaults and without its Windows interpreter lines. Set `PY`
to your Python interpreter before you run them.

```bash
CKPT="${1:-checkpoints/hobson-1.7b-recipe}"
HOLDOUT="${2:-data/holdout_v5_norule.jsonl}"
TRAIN="${3:-data/train_v5.jsonl}"
"$PY" -u -m strands_decider.cli calibrate "$CKPT" --data "$HOLDOUT" --limit 6000
"$PY" -u -m strands_decider.cli eval "$CKPT" --data "$HOLDOUT" --limit 6000 --out reports/heldout.json
"$PY" -u -m strands_decider.cli eval "$CKPT" --data "$TRAIN" --limit 6000 \
  --split all --out reports/indist.json
```

Its calibration sample is 6,000 rows. `recipe_v7.sh calibrate` uses the default of 4,000
rows, as v7 itself did, and `recipe.sh calibrate` does the same. The third argument must
be the corpus that the checkpoint trained on.

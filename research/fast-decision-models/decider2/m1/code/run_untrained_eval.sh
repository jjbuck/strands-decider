#!/bin/bash
source ~/venv/bin/activate; cd ~/work/m1; mkdir -p preds
python m1eval.py evalkit ~/work/m1/preds/untrained_full.jsonl --untrained full --tau_json ~/work/m1/untrained2.json > eval_untrained.log 2>&1

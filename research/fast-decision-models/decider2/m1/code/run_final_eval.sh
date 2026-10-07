#!/bin/bash
# final evaluation of one checkpoint: all 3,227 evalkit questions + the train_v5 holdout probe (RuleTaker d3/d5 etc.)
# usage: run_final_eval.sh CKPT TAG
source ~/venv/bin/activate; cd ~/work/m1; mkdir -p preds
python m1eval.py evalkit ~/work/m1/preds/$2.jsonl --ckpt $1 > eval_$2.log 2>&1
python m1eval.py holdout ~/work/m1/preds/holdout_$2.json --ckpt $1 --n 300 > holdout_$2.log 2>&1

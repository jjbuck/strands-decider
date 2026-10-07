#!/bin/bash
# run the untrained-J2 (= hobson) eval once qag_all's eval has finished (pipeline validation)
source ~/venv/bin/activate; cd ~/work/j2
while [ ! -f preds/qag_all.json ]; do sleep 60; done
[ -f preds/base.json ] || python j2eval.py preds/base.json --mode causal > logs/eval_base.log 2>&1

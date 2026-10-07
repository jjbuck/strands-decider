#!/bin/bash
# pipeline validation on a subset: untrained J2 (gates 0 = hobson) through j2eval, once qag_all's eval has finished
source ~/venv/bin/activate; cd ~/work/j2
while [ ! -f preds/qag_all.json ]; do sleep 60; done
[ -f preds/base_sub.json ] || python j2eval.py preds/base_sub.json --mode causal --suites JB-all,CF-probe > logs/eval_base.log 2>&1

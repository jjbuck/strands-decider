#!/bin/bash
# baseline evals through the deployed kernels (no training)
source ~/venv/bin/activate; cd ~/work/h6; mkdir -p preds
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"; K48="map:~/work/h2/precmap_w4a4_k48.json"
python h6eval.py preds/plainqb_k48.jsonl --prec $K48 --codes $C --layout plainqb >> evals.log 2>&1
python h6eval.py preds/schema_bf16_s0.jsonl --prec bf16 --layout schema >> evals.log 2>&1
python h6eval.py preds/schemamix_k48_s0.jsonl --prec $K48 --codes $C --layout schemamix >> evals.log 2>&1
python h6eval.py preds/plainqb_w4a4.jsonl --prec w4a4 --codes $C --layout plainqb >> evals.log 2>&1
echo BASEDONE >> evals.log

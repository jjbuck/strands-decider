#!/bin/bash
# continuation of chain3 (replaced while check2 was running): CUTLASS-int8 latency, post-hoc ternary hobson, BitDistill-lite run + eval
source ~/venv/bin/activate; cd ~/work/j10
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
CK=$(ls -d ck_a8/step* | sort -V | tail -1)
python bench_j10.py lat $CK c8,mixc 64,128,256,400,1000,4000 1,4 >> lat.log 2>&1
echo "$(date +%T) lat c8 done" >> chain.log
python ptq_hob.py w158a8 JB-all REAL-agree CF CF-probe >> ptq.log 2>&1
echo "$(date +%T) ptq done" >> chain.log
python tern_hob.py train --out ck_th --max_steps 3000 --max_minutes ${THMIN:-50} > th_train.log 2>&1
echo "$(date +%T) th train rc $?" >> chain.log
python tern_hob.py eval ck_th/final th JB-all REAL-agree CF CF-probe LONG >> th_eval.log 2>&1
echo "$(date +%T) CHAIN3DONE" >> chain.log

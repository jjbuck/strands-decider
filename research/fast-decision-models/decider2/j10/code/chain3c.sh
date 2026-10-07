#!/bin/bash
source ~/venv/bin/activate; cd ~/work/j10
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
while pgrep -f "ptq_hob.py" > /dev/null; do sleep 10; done
echo "$(date +%T) ptq done" >> chain.log
CK=$(ls -d ck_a8/step* | sort -V | tail -1)
python bench_j10.py lat $CK c8,mixc 64,128,256,400,1000,4000 1,4 >> lat.log 2>&1
echo "$(date +%T) lat c8 done (clean)" >> chain.log
python tern_hob.py train --out ck_th --max_steps 3000 --max_minutes ${THMIN:-45} > th_train.log 2>&1
echo "$(date +%T) th train rc $?" >> chain.log
python tern_hob.py eval ck_th/final th JB-all REAL-agree CF CF-probe LONG >> th_eval.log 2>&1
echo "$(date +%T) CHAIN3DONE" >> chain.log

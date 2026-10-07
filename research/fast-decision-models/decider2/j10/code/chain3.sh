#!/bin/bash
# after chain2: ternary-hobson smoke test, runtime fidelity check (fixed code extraction), CUTLASS-int8 latency, post-hoc ternary hobson,
# then the BitDistill-lite ternary-hobson run and its eval
source ~/venv/bin/activate; cd ~/work/j10
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
while ! grep -q CHAIN2DONE chain.log; do sleep 20; done
python tern_hob.py train --out ck_th_smoke --max_steps 3 > th_smoke.log 2>&1
echo "$(date +%T) th smoke rc $?" >> chain.log
CK=$(ls -d ck_a8/step* | sort -V | tail -1)
NCHECK=40 python bench_j10.py check $CK i8,c8,w2,mixc >> check_full2.log 2>&1
echo "$(date +%T) check2 done" >> chain.log
python bench_j10.py lat $CK c8,mixc 64,128,256,400,1000,4000 1,4 >> lat.log 2>&1
echo "$(date +%T) lat c8 done" >> chain.log
python ptq_hob.py w158a8 JB-all REAL-agree CF CF-probe >> ptq.log 2>&1
echo "$(date +%T) ptq done" >> chain.log
python tern_hob.py train --out ck_th --max_steps 3000 --max_minutes ${THMIN:-50} > th_train.log 2>&1
echo "$(date +%T) th train rc $?" >> chain.log
python tern_hob.py eval ck_th/final th JB-all REAL-agree CF CF-probe LONG >> th_eval.log 2>&1
echo "$(date +%T) CHAIN3DONE" >> chain.log

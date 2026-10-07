#!/bin/bash
source ~/venv/bin/activate; cd ~/work/j1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
echo "$(date +%T) start" >> chain_prep.log
(N_STATES=6000 PER_STATE=1 RMAX=9216 python prep_j1.py real ~/work/evalkit/train_pool.jsonl > prep_real.log 2>&1; echo "$(date +%T) prep real done" >> chain_prep.log) &
python prep_j1.py corpus > prep_corpus.log 2>&1
echo "$(date +%T) prep corpus done" >> chain_prep.log
wait
for n in rows_corpus rows_lceval rows_real; do python prep_j1.py merge $n >> merge.log 2>&1; done
echo "$(date +%T) merged" >> chain_prep.log

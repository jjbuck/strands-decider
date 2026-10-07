#!/bin/bash
source ~/venv/bin/activate; cd ~/work/g3
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
echo "$(date +%T) start" >> chain_teacher.log
(N_STATES=6000 PER_STATE=1 RMAX=9216 python prep_g3.py real ~/work/evalkit/train_pool.jsonl > prep_real.log 2>&1; echo "$(date +%T) prep real done" >> chain_teacher.log) &
python prep_g3.py corpus > prep_corpus.log 2>&1
echo "$(date +%T) prep corpus done" >> chain_teacher.log
python teacher.py rows_corpus_q.pt 24000 > teacher_corpus.log 2>&1
echo "$(date +%T) teacher corpus done" >> chain_teacher.log
python teacher.py rows_lceval_q.pt 24000 > teacher_lc.log 2>&1
wait
python teacher.py rows_real_q.pt 24000 > teacher_real.log 2>&1
echo "$(date +%T) teacher real done" >> chain_teacher.log
for n in rows_corpus rows_lceval rows_real; do python prep_g3.py merge $n >> merge.log 2>&1; done
echo "$(date +%T) merged" >> chain_teacher.log

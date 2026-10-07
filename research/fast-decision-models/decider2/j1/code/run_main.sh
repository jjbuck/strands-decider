#!/bin/bash
source ~/venv/bin/activate; cd ~/work/j1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
echo "$(date +%T) train start $*" >> chain_main.log
python train_j1.py --rows rows_corpus_e.pt rows_real_e.pt --out ck_main --mode masked --resume "$@" >> train_main.log 2>&1
echo "$(date +%T) train exit $?" >> chain_main.log

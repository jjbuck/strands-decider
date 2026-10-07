#!/bin/bash
# A4 QAT continuation of the A8 decider in the deployable Ampere 'mix' format (qkv + gate/up inputs int4 per token, o / down int8), then eval
source ~/venv/bin/activate; cd ~/work/j10
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
while ! grep -q CHAINDONE chain.log; do sleep 30; done
CK=$(ls -d ck_a8/step* | sort -V | tail -1)
echo "$(date +%T) qat start from $CK" >> chain.log
python train_j10.py --rows rows_corpus_s.pt rows_real_s.pt --out ck_mixa4 --frac_real 0.6 --tokb 3072 --ckpt_above 3072 --init $CK \
   --aq4 qkv,gu --max_steps ${QSTEPS:-500} --lr 5e-5 --head_lr 2e-4 --norm_lr 5e-5 --seed 1 --save_at 1.0 >> train_mixa4.log 2>&1
CK2=$(ls -d ck_mixa4/step* | sort -V | tail -1)
echo "$(date +%T) qat done $CK2" >> chain.log
python ev_j10.py $CK2 mixa4qat JB-all REAL-agree CF CF-probe LONG >> ev_qat.log 2>&1
echo "$(date +%T) CHAIN2DONE" >> chain.log

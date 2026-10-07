#!/bin/bash
source ~/venv/bin/activate; cd ~/work/h3; export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
log() { echo "$(date +%T) $*" >> logs/main.log; }
while ! grep -q 'main7 done' logs/main.log; do sleep 30; done
ev() { [ -f STOP ] && exit 0; log "ev $*"; python ev_h3.py "$@" >> logs/ev.log 2>&1; log "ev done $*"; }
ev --ckpt ck/qatnv/final.pt --cfgs nvfp4-qb16 --sub dev --tag qatnv
ev --ckpt ck/v1/final.pt --cfgs nvfp4-qb16 --sub dev --tag v1 --merge
ev --ckpt ck/v2/final.pt --cfgs w4a4r-qb16 --sub dev --tag v2
log "tail2 done"

#!/bin/bash
# H3 chain v6 (replaces run_main5.sh; resumable)
source ~/venv/bin/activate; cd ~/work/h3; export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
log() { echo "$(date +%T) $*" >> logs/main.log; }
while pgrep -f 'ev_h3.py --ckpt|train_h3.py' > /dev/null; do sleep 15; done
log "main6 start"
ev() { [ -f STOP ] && exit 0; log "ev $*"; python ev_h3.py "$@" >> logs/ev.log 2>&1; log "ev done $*"; }
tr() { [ -f STOP ] && exit 0; arm=$1; shift; if [ ! -f ck/$arm/final.pt ]; then log "train $arm"; python train_h3.py --arm $arm "$@" > logs/train_$arm.log 2>&1; log "train done $arm"; fi; }
tr qat --head std --norm rms --qat int4 --steps 1200
ev --ckpt ck/qat/final.pt --cfgs w4a4,bf16 --sub dev --tag qat
tr qatnv --head std --norm rms --qat nvfp4 --steps 1200
ev --ckpt ck/qatnv/final.pt --cfgs nvfp4,bf16 --sub dev --tag qatnv
ev --cfgs bf16,nvfp4 --sub long --tag hobson
ev --ckpt ck/v1/final.pt --cfgs bf16,nvfp4 --sub long --tag v1 --merge
ev --cfgs nvfp4r --sub dev --tag hobson
tr v1x3 --head hyp --norm rms --steps 3600
ev --ckpt ck/v1x3/final.pt --cfgs bf16,nvfp4 --sub dev --tag v1x3
log "main6 done"

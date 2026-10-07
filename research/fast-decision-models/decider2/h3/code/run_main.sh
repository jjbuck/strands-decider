#!/bin/bash
# H3 main chain (GPU exclusive, sequential). Each step skips if its output exists; safe to relaunch.
source ~/venv/bin/activate; cd ~/work/h3; export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
log() { echo "$(date +%T) $*" >> logs/main.log; }
# wait for step-1 runs I need (errB int4, errA int4 rot), then stop the optional errB nvfp4
while ! grep -q '^done' logs/errA_int4rot.log 2>/dev/null; do sleep 20; done
pkill -f run_err.sh; pkill -f 'errprop.py B --n 48 --fmt nvfp4'; sleep 5
log "step1 done"
ev() { [ -f STOP ] && exit 0; log "ev $*"; python ev_h3.py "$@" >> logs/ev.log 2>&1; log "ev done $*"; }
tr() { [ -f STOP ] && exit 0; arm=$1; shift; if [ ! -f ck/$arm/final.pt ]; then log "train $arm"; python train_h3.py --arm $arm "$@" > logs/train_$arm.log 2>&1; log "train done $arm"; fi; }
ev --cfgs bf16,w4a4,w4a4r,nvfp4 --sub dev --tag hobson
tr v1 --head hyp --norm rms --steps 1200
ev --ckpt ck/v1/final.pt --cfgs bf16,w4a4,w4a4r,nvfp4 --sub dev --tag v1
tr v2 --head hyp --norm l2s --steps 1200
ev --ckpt ck/v2/final.pt --cfgs bf16,w4a4,w4a4r,nvfp4 --sub dev --tag v2
tr qat --head std --norm rms --qat int4 --steps 1200
ev --ckpt ck/qat/final.pt --cfgs w4a4,bf16 --sub dev --tag qat
ev --cfgs w4a4-ab16,w16a4,w4a16,mxfp4,nvfp4r,w8a8 --sub dev --tag hobson
log "main done"

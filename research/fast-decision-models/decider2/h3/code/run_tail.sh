#!/bin/bash
# after run_main4.sh: 3x-budget v1 (does the bf16 offset close?), v2 literal, LONG for v1.
source ~/venv/bin/activate; cd ~/work/h3; export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
log() { echo "$(date +%T) $*" >> logs/main.log; }
while ! grep -q 'main4 done' logs/main.log; do sleep 30; done
ev() { [ -f STOP ] && exit 0; log "ev $*"; python ev_h3.py "$@" >> logs/ev.log 2>&1; log "ev done $*"; }
tr() { [ -f STOP ] && exit 0; arm=$1; shift; if [ ! -f ck/$arm/final.pt ]; then log "train $arm"; python train_h3.py --arm $arm "$@" > logs/train_$arm.log 2>&1; log "train done $arm"; fi; }
tr v1x3 --head hyp --norm rms --steps 3600
ev --ckpt ck/v1x3/final.pt --cfgs bf16,nvfp4 --sub dev --tag v1x3
tr v2lit --head hyp --norm l2 --steps 1200
ev --ckpt ck/v2lit/final.pt --cfgs bf16 --sub dev --tag v2lit
ev --ckpt ck/v1/final.pt --cfgs bf16,nvfp4 --sub long --tag v1 --merge
log "tail done"

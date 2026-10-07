#!/bin/bash
# final exclusive-GPU latency window (nothing else on the GPU)
source ~/venv/bin/activate; cd ~/work/j1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CK=${1:-ck_main/step3392}
echo "$(date +%T) final latency window start ($CK)" >> chain_post.log
python lat_enc.py check $CK > latcheck_final.log 2>&1; cp latcheck_enc.json latcheck_final.json
python gemm_bench.py > gemm_bench.log 2>&1
python lat_enc.py time $CK lat_enc_v2.json > lat_enc_v2.log 2>&1
LOCAL=$(cat q3_local.txt) WINDOW=512 TS=256,1000,2000,4000 python lat_enc.py time $CK lat_enc_loc512.json > lat_enc_loc512.log 2>&1
AUTOTUNE=1 python lat_hob.py lat_hob_auto.json > lat_hob_auto.log 2>&1
echo "$(date +%T) final latency window done" >> chain_post.log

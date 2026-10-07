#!/bin/bash
# after ck_a8 finishes: full-suite eval, then exclusive-GPU latency, then the A4 emulation evals
source ~/venv/bin/activate; cd ~/work/j10
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
while ! grep -q TRAINDONE train_a8.log; do sleep 30; done
CK=$(ls -d ck_a8/step* | sort -V | tail -1)
echo "$(date +%T) eval $CK" >> chain.log
python ev_j10.py $CK a8 JB-all REAL-agree CF CF-probe LONG >> ev_a8.log 2>&1
echo "$(date +%T) eval done" >> chain.log
python bench_j10.py check $CK bf16,i8,w2,mix >> check_full.log 2>&1
echo "$(date +%T) check done" >> chain.log
python bench_j10.py lat $CK bf16,i8,w2,mix,s4 64,128,256,400,1000,4000 1,4 >> lat.log 2>&1
echo "$(date +%T) lat done" >> chain.log
python hob_lat.py 64,128,256,400,1000,4000 >> hob_lat.log 2>&1
echo "$(date +%T) hob lat done" >> chain.log
python bench_j10.py gemm bf16,i8,w2,s4 64,128,256,400,1000,4000 >> gemm.log 2>&1
echo "$(date +%T) gemm done" >> chain.log
AQ=qkv:4:0,gu:4:0 python ev_j10.py $CK a8_mixA4 JB-all REAL-agree CF CF-probe >> ev_a4.log 2>&1
echo "$(date +%T) mixA4 done" >> chain.log
AQ=qkv:4:16,o:4:16,gu:4:16,down:4:16 python ev_j10.py $CK a8_allA4b16 JB-all REAL-agree CF CF-probe >> ev_a4.log 2>&1
echo "$(date +%T) allA4b16 done" >> chain.log
AQ=qkv:4:0,o:4:0,gu:4:0,down:4:0 python ev_j10.py $CK a8_allA4 JB-all REAL-agree CF CF-probe >> ev_a4.log 2>&1
echo "$(date +%T) CHAINDONE" >> chain.log

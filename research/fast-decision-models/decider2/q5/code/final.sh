#!/bin/bash
# runs alone at the end: exclusive GEMM timing for B12 shapes, then the B7 precision references for Q1's 4-bit w4q8 (two weight copies)
source ~/venv/bin/activate; cd ~/work/q5
while [ ! -f QA_DONE ] || [ ! -f QB_DONE ]; do sleep 20; done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader > final_gpu_before.txt
python q5gemmtime.py --alloc res_alloc.json --M 1125 --Mq 125 --out ~/work/q5/res_gemmtime_1125.json > final_gemm_1125.log 2>&1
python q5gemmtime.py --alloc res_alloc.json --M 4125 --Mq 125 --out ~/work/q5/res_gemmtime_4125.json > final_gemm_4125.log 2>&1
python q5gemmtime.py --alloc res_alloc.json --M 381 --Mq 125 --out ~/work/q5/res_gemmtime_381.json > final_gemm_381.log 2>&1
Q5_MEMFRAC=0.85 python q5dev.py --cfgs cfgs_q5.json --tags w4q8.fbank,w4q8.fret > final_b7w4.log 2>&1
touch FINAL_DONE

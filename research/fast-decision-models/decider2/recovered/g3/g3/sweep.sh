#!/bin/bash
# expert-kernel config sweep on a recorded routing: bash sweep.sh results/route_st4b_1000.pt
source ~/venv/bin/activate; cd ~/work/g3
R=$1
for BM in 16 32 64 128; do for BN1 in 64 128; do for w in 4 8; do for s in 2 3 4; do
  CFG="{\"BM\":$BM,\"BN1\":$BN1,\"BK1\":64,\"BN2\":128,\"BK2\":64,\"w1\":$w,\"s1\":$s,\"w2\":$w,\"s2\":$s}" python vllm_bench.py $R mine 2>&1 | grep '^{' || echo "fail $BM $BN1 $w $s"
done; done; done; done

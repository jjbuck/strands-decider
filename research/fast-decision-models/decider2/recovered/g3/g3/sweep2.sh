#!/bin/bash
source ~/venv/bin/activate; cd ~/work/g3
R=$1
for nf in 0 1; do for BM in 32 64 128; do for BN1 in 64 128; do for BN2 in 64 128; do
  CFG="{\"BM\":$BM,\"BN1\":$BN1,\"BK1\":64,\"BN2\":$BN2,\"BK2\":64,\"w1\":4,\"s1\":3,\"w2\":4,\"s2\":3,\"nf\":$nf}" python vllm_bench.py $R mine 2>&1 | grep '^{' || echo "fail $nf $BM $BN1 $BN2"
done; done; done; done

#!/bin/bash
# J5 accuracy chain: Hessians -> GPTQ weight-only codes -> all evalkit questions per format (resumable: each step skips finished work)
cd ~/work/j5; source ~/venv/bin/activate; export HF_HUB_OFFLINE=1
[ -f hess/meta.json ] || python wq.py calib 128
for f in w8 w4g128 w4g64 w3g128 w4g128rtn; do [ -f wq_$f.pt ] || python wq.py quant $f; done
python wq.py eval ${FMTS:-bf16,w8,w4g128,w4g64,w3g128,w4g128rtn,mix:w4g128:w8}
echo CHAIN DONE

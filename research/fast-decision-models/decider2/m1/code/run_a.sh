#!/bin/bash
# M1 variant (a): transfer (all 18 mixers, teacher-forced) then end-to-end distillation with all 18 converted at once.
source ~/venv/bin/activate; cd ~/work/m1
python m1train.py --stage transfer --ck ~/work/m1/ck_transfer --hours ${TH:-0.33} --variant full --untrained ~/work/m1/untrained2.json --lr_big 1e-4 --lr_small 1e-3 > transfer.log 2>&1
python m1train.py --stage e2e --ck ~/work/m1/ck_a --init ~/work/m1/ck_transfer/transfer.pt --hours ${EH:-4.5} --every_min 30 --variant full > e2e_a.log 2>&1

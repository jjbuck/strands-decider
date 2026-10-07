#!/bin/bash
# M1 variant (b): bottom-up schedule from the same transfer init: stage 1 converts GDN layers 0-6, stage 2 adds 8-14, stage 3 adds 16-22;
# each stage distilled for EH/3 hours (equal split). Compared with (a) at equal training time / tokens.
source ~/venv/bin/activate; cd ~/work/m1
python m1train.py --stage e2e --ck ~/work/m1/ck_b --init ~/work/m1/ck_transfer/transfer.pt --hours ${EH:-2.0} --every_min 30 --variant full \
  --schedule '0,1,2,4,5,6|8,9,10,12,13,14|16,17,18,20,21,22' > e2e_b.log 2>&1

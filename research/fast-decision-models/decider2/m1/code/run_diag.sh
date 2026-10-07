#!/bin/bash
# short single-factor e2e runs from the transfer init (about 60 updates each); compare the training-batch hid / kl_real trends
source ~/venv/bin/activate; cd ~/work/m1
D="--stage e2e --init ~/work/m1/ck_transfer/transfer.pt --hours 0.11 --every_min 1000 --variant full --ckpt_rows 1200 --warm 10 --ndev 20 --probe_n 8"
python m1train.py $D --ck ~/work/m1/diag/d1_mix   --lr_big 1e-5 --lr_small 0 --lr_lora 0    --p_v5 0 --p_cf 0 > diag_d1.log 2>&1
python m1train.py $D --ck ~/work/m1/diag/d2_lora  --lr_big 0    --lr_small 0 --lr_lora 5e-5 --p_v5 0 --p_cf 0 > diag_d2.log 2>&1
python m1train.py $D --ck ~/work/m1/diag/d3_small --lr_big 0    --lr_small 1e-4 --lr_lora 0 --p_v5 0 --p_cf 0 > diag_d3.log 2>&1

#!/bin/bash
# e2e (a) only (transfer done). Lower learning rates than the first attempt (which drifted away from the transferred init).
source ~/venv/bin/activate; cd ~/work/m1
python m1train.py --stage e2e --ck ~/work/m1/ck_a --init ~/work/m1/ck_transfer/transfer.pt --hours ${EH:-4.5} --every_min 30 --variant full --ckpt_rows ${CR:-1200} \
  --lr_big ${LB:-1e-5} --lr_small ${LS:-1e-4} --lr_lora ${LL:-5e-5} --warm 100 >> e2e_a.log 2>&1

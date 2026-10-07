#!/bin/bash
# M1 final distillation (engineer-approved): from the transfer checkpoint, full-rank mixer weights frozen, LoRA r32 (5e-5) + small mixer params (1e-4),
# train_v5 CE + KL (p .25), cf_aug CE + 0.3 KL (p .10), real states KL; dense rows 5/11/17/23; 8 micro-batches per update; dev every 30 min.
# Then the dev-best checkpoint (lowest dev TV) gets the full evalkit + holdout.
source ~/venv/bin/activate; cd ~/work/m1
python m1train.py --stage e2e --ck ~/work/m1/ck_a4 --init ~/work/m1/ck_transfer/transfer.pt --hours ${EH:-5.8} --every_min 30 --variant full --ckpt_rows 1200 \
  --lr_big 0 --lr_small 1e-4 --lr_lora 5e-5 --warm 30 --accum 8 --p_v5 0.25 --p_cf 0.10 --w_ce 1.0 >> e2e_a4.log 2>&1
BEST=$(python select_best.py ~/work/m1/ck_a4); echo "best $BEST" >> e2e_a4.log
bash run_final_eval.sh $BEST a4best

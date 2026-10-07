# q3b (big GPU): GPTQ init re-run here (no checkpoint copies), then Q1's row-role candidate w4q8 (state rows W4A4 trainable,
# question rows W8A8 with fixed GPTQ8 codes): B4.1a scales (1.5M tokens) -> QAD (TOK tokens, default 100M), no activation checkpointing.
cd ~/work/q3; source ~/venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
[ -f gptq4.pt ] || python q3gptq.py > gptq.log 2>&1
[ -f gptq8.pt ] || python q3gptq.py --bits 8 > gptq8.log 2>&1
python q3train.py --mode scale --q8 ~/work/q3/gptq8.pt --lr 1e-3 --warm 10 --tokens 1.5e6 --dev_every 0.75e6 --accum 4 --mem_gb ${MEM:-70} --ckpt 0 \
  --ck ~/work/q3/ck_scale_w4q8 > ck_scale_w4q8.out 2>&1
python q3train.py --mode qad --q8 ~/work/q3/gptq8.pt --seed_ck ~/work/q3/ck_scale_w4q8/best.pt --lat_init center --lr 2e-5 --warm 30 \
  --tokens ${TOK:-100e6} --dev_every 10e6 --accum 4 --l2sp 100 --w_hid 0.5 --mem_gb ${MEM:-70} --ckpt 0 --max_rows 9000 \
  --ck ~/work/q3/ck_qad_w4q8 --curve CF-probe,REAL-agree --run qadw4q8 > ck_qad_w4q8.out 2>&1

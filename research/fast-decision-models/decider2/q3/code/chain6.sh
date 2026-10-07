# Q1-recommended QAT base w4q8 (state rows W4A4 trainable, question rows W8A8 GPTQ8 fixed): B4.1a scales (3M tokens) -> QAD 20M tokens
cd ~/work/q3; source ~/venv/bin/activate
python q3train.py --mode scale --q8 ~/work/q3/gptq8.pt --lr 1e-3 --warm 10 --tokens 3e6 --dev_every 1e6 --accum 4 --max_rows 6000 \
  --ck ~/work/q3/ck_scale_w4q8 > ck_scale_w4q8.out 2>&1
python q3train.py --mode qad --q8 ~/work/q3/gptq8.pt --seed_ck ~/work/q3/ck_scale_w4q8/best.pt --lat_init center --lr 2e-5 --warm 30 \
  --tokens 20e6 --dev_every 5e6 --accum 4 --l2sp 100 --w_hid 0.5 --max_rows 6000 \
  --ck ~/work/q3/ck_qad_w4q8 --curve CF-probe,REAL-agree --run qadw4q8 > ck_qad_w4q8.out 2>&1
echo CHAIN6_DONE

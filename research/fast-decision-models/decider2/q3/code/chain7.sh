# w4q8 QAD seeded from the w4q8 scale run's best (dev TV), 12M tokens (A10G time left), dev + CF-probe/REAL-agree every 4M
cd ~/work/q3; source ~/venv/bin/activate
python q3train.py --mode qad --q8 ~/work/q3/gptq8.pt --seed_ck ~/work/q3/ck_scale_w4q8/best.pt --lat_init center --lr 2e-5 --warm 30 \
  --tokens 12e6 --dev_every 4e6 --accum 4 --l2sp 100 --w_hid 0.5 --max_rows 6000 \
  --ck ~/work/q3/ck_qad_w4q8 --curve CF-probe,REAL-agree --run qadw4q8 > ck_qad_w4q8.out 2>&1
echo CHAIN7_DONE

cd ~/work/q3; source ~/venv/bin/activate
python q3train.py --mode qad --seed_ck ~/work/q3/ck_scale/best.pt --lr 2e-5 --warm 30 --tokens 50e6 --dev_every 10e6 --accum 4 --l2sp 100 --w_hid 0.5 \
  --ck ~/work/q3/ck_qad --curve CF-probe,REAL-agree --run qad > ck_qad.out 2>&1

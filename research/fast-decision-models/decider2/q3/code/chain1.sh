# after B4.1a (scales) finishes: B4.1b (rounding), seeded from the scale result
cd ~/work/q3; source ~/venv/bin/activate
while pgrep -f "q3train.py --mode scale" >/dev/null; do sleep 10; done
python q3train.py --mode round --seed_ck ~/work/q3/ck_scale/best.pt --lr 3e-3 --warm 10 --tokens 2e6 --dev_every 1e6 --round_h0 0.95 --accum 4 \
  --reg 1.0 --reg_warm 0.0 --beta0 20 --beta1 2 --ck ~/work/q3/ck_round > ck_round.out 2>&1

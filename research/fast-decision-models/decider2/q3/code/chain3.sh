# B4.1b rounding (seeded from the scale result), then B4.2 QAD plain W4A4 seeded from the better B4.1 result (dev TV). Sequential, one GPU job at a time.
cd ~/work/q3; source ~/venv/bin/activate
python q3train.py --mode round --seed_ck ~/work/q3/ck_scale/best.pt --lr 3e-3 --warm 10 --tokens 2e6 --dev_every 1e6 --round_h0 0.95 --accum 4 \
  --reg 1.0 --reg_warm 0.0 --beta0 20 --beta1 2 --ck ~/work/q3/ck_round > ck_round.out 2>&1
SEED=$(python - <<'PY'
import json, os
c = {}
for d in ('ck_scale', 'ck_round'):
    p = os.path.expanduser(f'~/work/q3/{d}/state.json')
    if os.path.exists(p): c[d] = json.load(open(p))['best']
b = min(c, key=c.get); print(os.path.expanduser(f'~/work/q3/{b}/best.pt'))
PY
)
echo "seed $SEED"
python q3train.py --mode qad --seed_ck $SEED --lr 2e-5 --warm 30 --tokens 50e6 --dev_every 10e6 --accum 4 --l2sp 100 --w_hid 0.5 \
  --ck ~/work/q3/ck_qad --curve CF-probe,REAL-agree --run qad > ck_qad.out 2>&1

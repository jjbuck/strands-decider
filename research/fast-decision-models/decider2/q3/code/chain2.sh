# after B4.1b: GPTQ8 codes (for the w4q8 row-role arm), then B4.2 QAD plain W4A4 seeded from the better B4.1 result (dev TV)
cd ~/work/q3; source ~/venv/bin/activate
while pgrep -f "q3train.py --mode round" >/dev/null || pgrep -f "chain1.sh" >/dev/null; do sleep 10; done
[ -f gptq8.pt ] || python q3gptq.py --bits 8 > gptq8.log 2>&1
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

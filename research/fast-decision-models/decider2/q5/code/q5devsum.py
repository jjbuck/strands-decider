"""Laptop: table of dev screening results (dev_res.json from the box). python3 q5devsum.py [file] [filter-substring]"""
import sys, json, os
f = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser('~/decider2/q5/res/dev_res.json')
flt = sys.argv[2] if len(sys.argv) > 2 else ''
R = json.load(open(f))
print(f"{'tag':44s} {'bankKL':>9s} {'bkF':>4s} {'retKL':>9s} {'rtF':>4s} {'TV':>7s} | {'vsfmt bKL':>9s} {'F':>3s} | sparse_s  sp_q  removed")
for t, r in R.items():
    if flt and flt not in t: continue
    v = r['vs_dense']; w = r['work']; vf = r.get('vs_fmt')
    fs = f"{vf['bank']['kl']:9.2e} {vf['bank']['flips']+vf['ret']['flips']:3d}" if vf else ' ' * 13
    print(f"{t:44s} {v['bank']['kl']:9.2e} {v['bank']['flips']:4d} {v['ret']['kl']:9.2e} {v['ret']['flips']:4d} {v['all']['tv']:7.4f} | {fs} | {w['sparse_state']:.3f} {w['sparse_q']:.3f} {w['removed']:.3f}")

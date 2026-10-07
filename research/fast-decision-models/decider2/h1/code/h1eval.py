"""H1 eval: run configs on evalkit subsets, write preds JSON (resumable).
python h1eval.py --sub main --cfg 'dense;w8a8;w8a8,ba16' --out res/x.json
config grammar: 'dense' | BASE[,opt...]   BASE in w8a8 w4a4 w4a8
   opts: gptq (4-bit weights by GPTQ) | gptq8 (8-bit weights by GPTQ too) | ba16 (GDN beta/decay rows bf16) | ohead (per-head Hadamard for Wo)
         clip=0.85 (A4 clip) | seed=N (rotation seed) | k8=FILE:N (top-N GEMMs of a sensitivity ranking -> w8a8) | k16=FILE:N (-> bf16)
         ex=i.k+i.k (explicit GEMMs -> bf16) | e8=i.k+... (-> w8a8) | lora=PATH (merge a LoRA delta, in UNROTATED space, before quantization)
"""
import os, sys, json, time, argparse, re
sys.path[:0] = [os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/evalkit')]
import torch
import h1lib as HL

ap = argparse.ArgumentParser(); ap.add_argument('--sub', default='main'); ap.add_argument('--cfg', required=True); ap.add_argument('--out', required=True)
ap.add_argument('--limit', type=int, default=0); ap.add_argument('--shard', default='0/1')
a = ap.parse_args()
SUBS = json.load(open(os.path.expanduser('~/work/g2/subsets.json')))
PLANS = {
    'main': ['REAL-ALL', 'CF-T', 'CF-probe-T'],
    'quick': ['REAL-agree-SD', 'CF-T', 'CF-probe-T'],
    'jb': ['JB-hard'],
    'long': ['LONG-ALL'],
    'rest': ['CF-ALL', 'CF-probe-ALL', 'JB-hard', 'JB-long', 'LONG-ALL'],
    'all': ['REAL-ALL', 'CF-ALL', 'CF-probe-ALL', 'JB-hard', 'JB-long', 'LONG-ALL'],
}
import evalkit as EK
SUBS['JB-long'] = [['JB-long', it['id'], q] for it in EK.load_suite('JB-long') for q in it['questions']]
rows = []; seen = set()
for nm in PLANS[a.sub]:
    for r in SUBS[nm]:
        k = (r[1], r[2])
        if k in seen: continue
        seen.add(k); rows.append(r)
si, sn = map(int, a.shard.split('/')); rows = rows[si::sn]
if a.limit: rows = rows[:a.limit]
items = {}
for s in ('REAL-agree', 'LONG', 'CF', 'CF-probe', 'JB-hard', 'JB-long'):
    for it in EK.load_suite(s): items[it['id']] = it

g = HL.H1()
base_opt = dict(g.opt)
BASEW = {(i, k): g.L[i][k] for i in range(24) for k in HL.GEMMS}


def parse(c):
    if c == 'dense': return None
    parts = c.split(','); base = parts[0]
    opt = dict(base_opt); P = g.uniform(base); lora = None
    for p in parts[1:]:
        if p == 'gptq': opt['wq'] = 'gptq'
        elif p == 'gptq8': opt['wq8'] = 'gptq'; opt['wq'] = 'gptq'
        elif p == 'ba16': opt['ba16'] = True
        elif p == 'ohead': opt['ohead'] = True
        elif p.startswith('clip='): opt['aclip4'] = float(p[5:])
        elif p.startswith('seed='): opt['rseed'] = int(p[5:])
        elif p.startswith('k8=') or p.startswith('k16='):
            f, n = p.split('=')[1].rsplit(':', 1); rk = json.load(open(f))['rank']
            for key in rk[:int(n)]: P[tuple([int(key.split('.')[0]), key.split('.')[1]])] = 'w8a8' if p.startswith('k8=') else 'bf16'
        elif p.startswith('ex=') or p.startswith('e8='):
            for key in p[3:].split('+'):
                P[(int(key.split('.')[0]), key.split('.')[1])] = 'bf16' if p.startswith('ex=') else 'w8a8'
        elif p.startswith('map='):
            for key, v in json.load(open(p[4:])).items(): P[(int(key.split('.')[0]), key.split('.')[1])] = v
        elif p == 'norout': opt['rout'] = False
        elif p == 'qb16': opt['qb16'] = True
        elif p.startswith('lora='): lora = p[5:]
        elif p.startswith('lrot='): opt['_lrot'] = p[5:]
        else: raise ValueError(p)
    return dict(opt=opt, P=P, lora=lora)


cfgs = a.cfg.split(';'); spec = {c: parse(c) for c in cfgs}
print('configs', cfgs, flush=True)
res = {c: {} for c in cfgs}
if os.path.exists(a.out):
    old = json.load(open(a.out))['preds']
    for c in cfgs: res[c] = old.get(c, {})
done = lambda iid, qn: all(qn in res[c].get(iid, {}) for c in cfgs)
cur_lora = None


def activate(sp):
    global cur_lora
    lo = sp['lora'] if sp else None
    if lo != cur_lora:
        for (i, k), W in BASEW.items(): g.L[i][k] = W
        if lo:
            sd = torch.load(lo, map_location=g.dev)
            for i in range(24):
                for k in HL.GEMMS:
                    if f'{i}.{k}.A' in sd:
                        g.L[i][k] = (BASEW[(i, k)].float() + sd[f'{i}.{k}.B'].float() @ sd[f'{i}.{k}.A'].float()).to(torch.bfloat16)
        g.drop_cache(); cur_lora = lo
    if sp is None: g.set_prec({}); g.load_lrot(None); return
    g.opt = sp['opt']
    lr = sp['opt'].get('_lrot')
    if lr != g.LRtag: g.load_lrot(lr)
    g.set_prec(sp['P'])


t0 = time.time(); n_new = 0
for c in cfgs:                       # config-outer: one config's quantized weights in memory at a time
    activate(spec[c])
    for ri, (suite, iid, qn) in enumerate(rows):
        if qn in res[c].get(iid, {}): continue
        it = items[iid]
        pr = g.prep(it['state'], it['questions'][qn]); ids = pr['s'] + pr['q']
        h, _ = g.fwd(ids, q0=pr['q0'])
        res[c].setdefault(iid, {})[qn] = g.pdict(g.logits(h, pr), pr)
        n_new += 1
        if n_new % 50 == 0:
            print(f'{c} {ri}/{len(rows)} {suite} T={len(ids)} {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}G', flush=True)
            json.dump(dict(preds=res, cfgs=cfgs), open(a.out + '.tmp', 'w')); os.replace(a.out + '.tmp', a.out)
    json.dump(dict(preds=res, cfgs=cfgs), open(a.out + '.tmp', 'w')); os.replace(a.out + '.tmp', a.out)
    print('config done', c, f'{time.time()-t0:.0f}s', flush=True)
    g.drop_cache()
print('done', f'{time.time()-t0:.0f}s', flush=True)

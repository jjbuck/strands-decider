"""H5: build a quantized hobson (rotation + RTN or GPTQ weights) and score it on the train-split dev set; optionally save it.
python h5build.py --R had|rot/R1_a_s200.pt --w gptq --preset w4a4 [--set Wd=4,8 ...] --ncal 128 --save q/NAME.pt --name NAME"""
import os, sys, json, time, random, argparse
sys.path.insert(0, os.path.expanduser('~/work/h5'))
import torch
import h5lib as H

ap = argparse.ArgumentParser()
ap.add_argument('--R', default='had'); ap.add_argument('--w', default='rtn'); ap.add_argument('--preset', default='w4a4')
ap.add_argument('--set', nargs='*', default=[]); ap.add_argument('--aclip', type=float, default=1.0); ap.add_argument('--ncal', type=int, default=128)
ap.add_argument('--calmax', type=int, default=2048); ap.add_argument('--save', default=''); ap.add_argument('--name', default='')
ap.add_argument('--ndev', type=int, default=240); ap.add_argument('--noba', action='store_true'); ap.add_argument('--load', default=''); ap.add_argument('--fp4', action='store_true', help='NVFP4 for every 4-bit quantizer'); ap.add_argument('--kron', default='', help='rot/kron_TAG.pt: per-GEMM learned Kronecker rotations + clips'); ap.add_argument('--map', default='', help='JSON {"<layer>.<Win|Wo|Wgu|Wd>": "w8a8"|"w4a8"|"bf16"} overrides (H1 precmap format)')
a = ap.parse_args()
W = os.path.expanduser('~/work/h5/'); os.makedirs(W + 'q', exist_ok=True); os.makedirs(W + 'res', exist_ok=True)
CL = ('Win_g', 'Win_a', 'Wo_g', 'Wo_a', 'Wgu', 'Wd')
P = {'w4a4': (4, 4), 'w4a8': (4, 8), 'w8a8': (8, 8), 'w16a4': (16, 4)}
t0 = time.time()
m = H.Q5(lean=True)
m.cfg = {c: P[a.preset] for c in CL}
for s in a.set:
    k, v = s.split('='); m.cfg[k] = tuple(int(x) for x in v.split(','))
m.aclip = a.aclip; m.ba_hp = not a.noba
if a.fp4: H.FMT['a'] = 'nvfp4'; H.FMT['w'] = 'nvfp4'
if a.map:
    PB = {'w8a8': (8, 8), 'w4a8': (4, 8), 'w4a4': (4, 4), 'bf16': (16, 16)}
    for k, v in json.load(open(a.map)).items():
        li, site = k.split('.'); li = int(li)
        m.layer_cfg[(li, H.site_class(m.L, li, site))] = PB[v]
    print('map', a.map, len(m.layer_cfg), 'overrides', flush=True)
if a.R == 'none':
    m.R1 = torch.eye(2048, device=m.dev); m.Ho = H.Ident(); m.Hd = H.Ident()
elif a.R != 'had':
    m.R1 = torch.load(W + a.R if not a.R.startswith('/') else a.R).to(m.dev).float()
if a.kron:
    KD = torch.load(W + a.kron)
    m.K = {k: (v[0].to(m.dev), v[1].to(m.dev)) for k, v in KD['K'].items()}; m.clipA = dict(KD.get('clipA', {}))
    print('kron', a.kron, len(m.K), 'GEMMs', flush=True)
print('cfg', m.cfg, 'aclip', m.aclip, 'R', a.R, 'w', a.w, flush=True)

pool = H.load_pool(150, 3000)
rng = random.Random(11); dev_pool = list(pool); rng.shuffle(dev_pool)
dev_rids = {r['rid'] for r in dev_pool[:240]}
DEV = []
for r in dev_pool[:a.ndev]:
    qn = rng.choice(sorted(r['questions'])); pr = m.prep(r['state'], r['questions'][qn]); DEV.append((pr, pr['s'] + pr['q']))

if a.load:
    D = torch.load(a.load)
    m.Wr = [{k: v.to(m.dev) for k, v in d.items()} for d in D['Wr']]; m.Ws = D.get('Ws'); m.R1 = D['R1'].to(m.dev); m.cfg = D['cfg']; m.aclip = D['aclip']; m.ba_hp = D['ba_hp']
elif a.w == 'rtn':
    m.build_rtn()
else:
    cal_pool = [r for r in H.load_pool(300, 100000) if r['rid'] not in dev_rids]
    random.Random(21).shuffle(cal_pool)
    seqs = []
    for r in cal_pool[:a.ncal]:
        qn = sorted(r['questions'])[0]; pr = m.prep(r['state'], r['questions'][qn]); ids = pr['s'] + pr['q']
        if len(ids) > a.calmax: ids = ids[:a.calmax // 4] + ids[-(a.calmax - a.calmax // 4):]
        seqs.append(ids)
    print('gptq calibration', len(seqs), 'seqs', sum(map(len, seqs)), 'tokens', flush=True)
    tg = time.time(); m.build_gptq(seqs, log=lambda s: print(s, round(time.time() - tg), 's', flush=True))
print('built', round(time.time() - t0), 's', flush=True)

refp = W + f'res/devref_{a.ndev}.pt'
if os.path.exists(refp): REF = torch.load(refp)
else:
    REF = []
    with torch.no_grad():
        for pr, ids in DEV:
            h, _ = m.forward(ids, 'ref'); REF.append(torch.softmax(m.logits(h, pr).float(), -1).cpu())
    torch.save(REF, refp)
fl = 0; tv = 0.0; kl = 0.0
with torch.no_grad():
    for (pr, ids), r in zip(DEV, REF):
        h, _ = m.forward(ids, 'q'); p = torch.softmax(m.logits(h, pr).float(), -1).cpu()
        fl += int(p.argmax() != r.argmax()); tv += H.tv(p, r); kl += H.kl(r, p)
n = len(DEV)
res = dict(name=a.name, R=a.R, w=a.w, map=a.map, kron=a.kron, fp4=a.fp4, cfg={k: list(v) for k, v in m.cfg.items()}, aclip=m.aclip, ba_hp=m.ba_hp, flips=fl, n=n,
           flip_rate=round(fl / n, 4), tv=round(tv / n, 4), kl=round(kl / n, 4), s=round(time.time() - t0))
print('DEV', json.dumps(res), flush=True)
with open(W + 'res/build.jsonl', 'a') as f: f.write(json.dumps(res) + '\n')
if a.save:
    torch.save(dict(Wr=[{k: v.cpu() for k, v in d.items()} for d in m.Wr], Ws=[{k: v.cpu() for k, v in d.items()} for d in m.Ws],
                    R1=m.R1.cpu(), cfg=m.cfg, layer_cfg=m.layer_cfg, aclip=m.aclip, ba_hp=m.ba_hp,
                    K={k: (v[0].cpu(), v[1].cpu()) for k, v in m.K.items()}, clipA=m.clipA, fmt=dict(H.FMT), norot=(a.R == 'none')), W + a.save)
    print('saved', a.save, flush=True)

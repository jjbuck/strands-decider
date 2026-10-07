"""H5 sensitivity study on a TRAIN-split dev set (evalkit/train_pool, eval tasks excluded): which GEMM sites carry the W4A4 error.
python h5sens.py [n_dev] [configs...]   -> res/sens.json (appends)"""
import os, sys, json, time, random
sys.path.insert(0, os.path.expanduser('~/work/h5'))
import torch
import h5lib as H

N = int(sys.argv[1]) if len(sys.argv) > 1 else 240
only = sys.argv[2:]
os.makedirs(os.path.expanduser('~/work/h5/res'), exist_ok=True)
OUT = os.path.expanduser('~/work/h5/res/sens.json')
t0 = time.time()
m = H.Q5()
print('loaded', round(time.time() - t0), 's', flush=True)


def dev_set(n, seed=11, max_tok=3000):
    pool = H.load_pool(150, max_tok)
    rng = random.Random(seed); rng.shuffle(pool)
    out = []
    for r in pool[:n]:
        qn = rng.choice(sorted(r['questions']))
        pr = m.prep(r['state'], r['questions'][qn])
        out.append(dict(rid=r['rid'], q=qn, pr=pr, ids=pr['s'] + pr['q']))
    return out


DEV = dev_set(N)
print('dev', len(DEV), 'tokens', sum(len(d['ids']) for d in DEV), flush=True)


@torch.no_grad()
def run(mode):
    ps = []
    for d in DEV:
        h, _ = m.forward(d['ids'], mode)
        ps.append(torch.softmax(m.logits(h, d['pr']).float(), -1).cpu())
    return ps


t1 = time.time(); REF = run('ref'); print('ref pass', round(time.time() - t1), 's', flush=True)


def metrics(ps):
    fl = sum(int(p.argmax() != r.argmax()) for p, r in zip(ps, REF))
    return dict(flips=fl, n=len(ps), flip_rate=round(fl / len(ps), 4), tv=round(sum(H.tv(p, r) for p, r in zip(ps, REF)) / len(ps), 4),
                kl=round(sum(H.kl(r, p) for p, r in zip(ps, REF)) / len(ps), 4))


CL = ('Win_g', 'Win_a', 'Wo_g', 'Wo_a', 'Wgu', 'Wd')


def cfg_all(b): return {c: b for c in CL}


CONFIGS = {
    'q16_equiv': (cfg_all((16, 16)), 1.0, True),
    'w4a4': (cfg_all((4, 4)), 1.0, True),
    'w4a4_c09': (cfg_all((4, 4)), 0.9, True),
    'w4a4_noba': (cfg_all((4, 4)), 1.0, False),
    'w16a4': (cfg_all((16, 4)), 1.0, True),
    'w4a16': (cfg_all((4, 16)), 1.0, True),
    'w4a8': (cfg_all((4, 8)), 1.0, True),
    'w8a8': (cfg_all((8, 8)), 1.0, True),
}
for c in CL:
    d = cfg_all((16, 16)); d[c] = (4, 4); CONFIGS[f'only_{c}_w4a4'] = (d, 1.0, True)
for c in CL:
    d = cfg_all((4, 4)); d[c] = (4, 8); CONFIGS[f'all_but_{c}_a8'] = (d, 1.0, True)

res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for name, (cfg, clip, ba) in CONFIGS.items():
    if only and name not in only: continue
    m.cfg = dict(cfg); m.aclip = clip; m.ba_hp = ba
    t1 = time.time(); m.build_rtn()
    r = metrics(run('q')); r['s'] = round(time.time() - t1)
    res[f'{name}|n{N}'] = r
    print(name, json.dumps(r), flush=True)
    json.dump(res, open(OUT, 'w'), indent=1)
print('done', round(time.time() - t0), flush=True)

"""G2 map: per-token width / precision configs on frozen hobson over evalkit subsets -> preds per config.
python g2map.py --sub pilot --cfg 're' --out res/map_pilot.json
Subsets: g2/subsets.json (laptop-built). Signals per (item, question): qa7 (exact hobson question->state attention at layer 7, oracle locator),
loc4 / loc8 / locN (same attention from an approximate pass of layers 0-7: all rows W4A4-Hadamard / W8A8-Hadamard / width 1/4), surp (0.8B-Base surprisal), rand."""
import os, sys, json, time, re, argparse, hashlib, random
sys.path[:0] = [os.path.expanduser('~/work/g2'), os.path.expanduser('~/work/evalkit')]
import torch
import g2lib as GL

ap = argparse.ArgumentParser(); ap.add_argument('--sub', default='pilot'); ap.add_argument('--cfg', default='.*'); ap.add_argument('--out', required=True)
ap.add_argument('--signals', default='qa7,loc4,rand'); ap.add_argument('--lora', default=None); ap.add_argument('--skipdense', action='store_true'); ap.add_argument('--limit', type=int, default=0); ap.add_argument('--shard', default='0/1')
a = ap.parse_args()
W = os.path.expanduser('~/work/g2/')
SUBS = json.load(open(W + 'subsets.json'))
PILOT = dict(REALSD=('REAL-agree-SD', 80), CFT=('CF-T', 120), CFPT=('CF-probe-T', 100), JB=('JB-hard', 0), LONGSD=('LONG-SD', 0))
FULL = dict(REALSD=('REAL-agree-SD', 346), CFT=('CF-T', 218), CFPT=('CF-probe-T', 210), JB=('JB-hard', 130), LONGSD=('LONG-SD', 165))
MID = dict(REALSD=('REAL-agree-SD', 200), CFT=('CF-T', 218), CFPT=('CF-probe-T', 210), JB=('JB-hard', 130), LONGSD=('LONG-SD', 60))
QALL = dict(REAL=('REAL-ALL', 1083), CF=('CF-ALL', 812), CFP=('CF-probe-ALL', 640), JB=('JB-hard', 130), LONG=('LONG-ALL', 471))
QSD = dict(REALSD=('REAL-agree-SD', 346), CFT=('CF-T', 218), CFPT=('CF-probe-T', 210), JB=('JB-hard', 130), LONGSD=('LONG-SD', 165))
QMAIN = dict(REAL=('REAL-ALL', 1083), CFT=('CF-T', 218), CFPT=('CF-probe-T', 210), JB=('JB-hard', 130), LONG=('LONG-ALL', 471))
REALCF = dict(REAL=('REAL-ALL', 1083), CFT=('CF-T', 218), CFPT=('CF-probe-T', 210))
QATF = dict(REAL=('REAL-ALL', 1083), CFT=('CF-T', 218), CFPT=('CF-probe-T', 210), JB=('JB-hard', 130), LONGSD=('LONG-SD', 165))
QSD3 = dict(REALSD=('REAL-agree-SD', 346), CFT=('CF-T', 218), CFPT=('CF-probe-T', 210))
LONGALL = dict(LONG=('LONG-ALL', 471)); REALONLY = dict(REAL=('REAL-ALL', 1083))
plan = {'qatf': QATF, 'qsd3': QSD3, 'longall': LONGALL, 'realonly': REALONLY, 'realcf': REALCF, 'qmain': QMAIN, 'pilot': PILOT, 'full': FULL, 'mid': MID, 'qall': QALL, 'qsd': QSD}[a.sub]
rows = []
for k, (nm, n) in plan.items():
    rows += SUBS[nm][:n]
si, sn = map(int, a.shard.split('/')); rows = rows[si::sn]
if a.limit: rows = rows[:a.limit]
items = {}
import evalkit as EK
for s in ('REAL-agree', 'LONG', 'CF', 'CF-probe', 'JB-hard'):
    for it in EK.load_suite(s): items[it['id']] = it

E = lambda **k: dict(k)
X = E()                                   # exact class
Q8, Q8P, Q4 = E(prec='r8'), E(prec='p8'), E(prec='r4')
N2, N4, N8 = E(mlp=.5, heads=.5), E(mlp=.25, heads=.25), E(mlp=.125, heads=.125)
NM4, NM0, DROP = E(mlp=.25), E(mlp=0.0), E(drop=True)
CFGS = {}   # name -> (signal or None, plan [(class, frac)], classes)
CFGS['dense'] = (None, None, None)
for pc in ('p8', 'r8', 'r4', 'r4c', 'r48'):
    CFGS['Q:' + pc] = ('allq', [(1, 1.0)], [X, E(prec=pc)])      # every row (state + question) quantized: the uniform model
    CFGS['S:' + pc] = ('all', [(1, 1.0)], [X, E(prec=pc)])       # state rows only
FAM = {
    'P8-10': ([(0, .10), (1, .90)], [X, Q8]),
    'P8-25': ([(0, .25), (1, .75)], [X, Q8]),
    'P4-10': ([(0, .10), (1, .90)], [X, Q4]),
    'P4-25': ([(0, .25), (1, .75)], [X, Q4]),
    'P84-10': ([(0, .10), (1, .15), (2, .75)], [X, Q8, Q4]),
    'P8r4-25': ([(1, .25), (2, .75)], [X, Q8, Q4]),
    'P8r4-50': ([(1, .50), (2, .50)], [X, Q8, Q4]),
    'N4-10': ([(0, .10), (1, .90)], [X, N4]),
    'N4-25': ([(0, .25), (1, .75)], [X, N4]),
    'N24-10': ([(0, .10), (1, .20), (2, .70)], [X, N2, N4]),
    'N8-20': ([(0, .20), (1, .80)], [X, N8]),
    'NM4-10': ([(0, .10), (1, .90)], [X, NM4]),
    'NM0-10': ([(0, .10), (1, .90)], [X, NM0]),
    'D-10': ([(0, .10), (1, .90)], [X, DROP]),
    'D-40': ([(0, .40), (1, .60)], [X, DROP]),
    'N4P4-10': ([(0, .10), (1, .90)], [X, E(mlp=.25, heads=.25, prec='r4')]),
    'N2P4-25': ([(0, .25), (1, .75)], [X, E(mlp=.5, heads=.5, prec='r4')]),
    'M2-50': ([(0, .50), (1, .50)], [X, E(mlp=.5)]),
    'M4-50': ([(0, .50), (1, .50)], [X, E(mlp=.25)]),
    'M42-25': ([(0, .25), (1, .25), (2, .50)], [X, E(mlp=.5), E(mlp=.25)]),
    'M2-25': ([(0, .25), (1, .75)], [X, E(mlp=.5)]),
}
for sig in a.signals.split(','):
    for f, (pl, cl) in FAM.items():
        CFGS[f'{f}/{sig}'] = (sig, pl, cl)
sel = [c for c in CFGS if re.fullmatch(a.cfg, c) or (c == 'dense' and not a.skipdense)]
print('configs', len(sel), sel, flush=True)

g = GL.G2()
rp = W + 'ranks.pt'
if os.path.exists(rp): g.load_ranks(rp)
else:
    # calibration: 8 train-pool states (train-split tasks only), state rows
    seqs = []
    with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
        for li, l in enumerate(f):
            if li % 300 != 7: continue
            from strands_decider.prompting import render_state
            r = json.loads(l)
            if not 1000 <= r['n_state_tok'] <= 4000: continue
            ids = g.p.tok(render_state(r['state']), add_special_tokens=True)['input_ids'][:3000]
            seqs.append(ids)
            if len(seqs) >= 8: break
    t0 = time.time(); g.calibrate(seqs, rp); print('calibrated on', len(seqs), 'states', sum(map(len, seqs)), 'tokens', f'{time.time()-t0:.0f}s', flush=True)

if a.lora: g.merge_lora(a.lora); print('merged', a.lora, flush=True)
surp_m = None
def surprisal(ids):
    global surp_m
    if surp_m is None:
        from transformers import AutoModelForCausalLM
        import glob
        pth = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*'))[0]
        surp_m = AutoModelForCausalLM.from_pretrained(pth, torch_dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        t = torch.tensor([ids], device='cuda')
        hs = surp_m.model(input_ids=t).last_hidden_state[0]
        out = torch.zeros(len(ids), device='cuda')
        for s in range(0, len(ids) - 1, 1024):
            lg = surp_m.lm_head(hs[s:s + 1024]).float().log_softmax(-1)
            tgt = t[0, s + 1:s + 1025]
            out[s + 1:s + 1 + len(tgt)] = -lg[:len(tgt)].gather(1, tgt[:, None])[:, 0] / 0.6931
        return out

res = {c: {} for c in sel}; meta = {c: dict(cost=[], n=0) for c in sel}; rec = {}
sigcache = {}
t0 = time.time()
for ri, (suite, iid, qn) in enumerate(rows):
    it = items[iid]
    pr = g.prep(it['state'], it['questions'][qn])
    ids = pr['s'] + pr['q']; q0 = pr['q0']; T = len(ids)
    sig = {}
    need = {CFGS[c][0] for c in sel}
    if not a.skipdense or not need <= {None, 'all', 'allq'}:
        h, caps = g.forward(ids, None, capture=(3, 7), q0=q0)
        dense_p = g.probs(h, pr)
        sig['qa7'] = caps[7]; sig['qa3'] = caps[3]
    else:
        dense_p = None; sig['qa7'] = torch.zeros(q0, device='cuda')
    if 'loc4' in need:
        cls = torch.ones(T, dtype=torch.long, device='cuda'); _, cp = g.forward(ids, dict(rowcls=cls, classes=[X, Q4]), stop=8, capture=(7,), q0=q0); sig['loc4'] = cp[7]
    if 'loc8' in need:
        cls = torch.ones(T, dtype=torch.long, device='cuda'); _, cp = g.forward(ids, dict(rowcls=cls, classes=[X, Q8]), stop=8, capture=(7,), q0=q0); sig['loc8'] = cp[7]
    if 'locN' in need:
        cls = torch.ones(T, dtype=torch.long, device='cuda'); _, cp = g.forward(ids, dict(rowcls=cls, classes=[X, N4]), stop=8, capture=(7,), q0=q0); sig['locN'] = cp[7]
    if 'surp' in need:
        sk = hashlib.sha1(json.dumps(pr['s']).encode()).hexdigest()
        if sk not in sigcache: sigcache[sk] = surprisal(pr['s'])
        sig['surp'] = sigcache[sk]
    if 'rand' in need:
        gg = torch.Generator(device='cuda'); gg.manual_seed(int(hashlib.sha1((iid + qn).encode()).hexdigest()[:8], 16))
        sig['rand'] = torch.rand(q0, device='cuda', generator=gg)
    # locator recall vs the exact signal (top 10%)
    k10 = max(1, int(0.1 * q0)); top = set(torch.topk(sig['qa7'], k10).indices.tolist())
    rc = {s: len(top & set(torch.topk(v, k10).indices.tolist())) / k10 for s, v in sig.items() if s not in ('qa7',) and v.numel() == q0}
    rec[f'{iid}|{qn}'] = dict(T=T, q0=q0, recall10=rc)
    for c in sel:
        s_, pl, cl = CFGS[c]
        if s_ is None: p = dense_p; cst = 1.0
        else:
            if s_ in ('all', 'allq'):
                rowcls = torch.zeros(T, dtype=torch.long, device='cuda')
                if s_ == 'all': rowcls[:q0] = 1
                else: rowcls[:] = 1
                cst = GL.cost([(1.0, cl[1])])
            else:
                rowcls = GL.assign(sig[s_], q0, T, pl)
                cst = GL.cost([(f, cl[k]) for k, f in pl])
            hh, _ = g.forward(ids, dict(rowcls=rowcls, classes=cl))
            p = g.probs(hh, pr)
        res[c].setdefault(iid, {})[qn] = {lab: p[j] for j, lab in enumerate(pr['rq'].slot_labels)}
        meta[c]['cost'].append(cst); meta[c]['n'] += 1
    if ri % 20 == 0:
        print(f'{ri}/{len(rows)} {suite} T={T} {time.time()-t0:.0f}s  mem {torch.cuda.max_memory_allocated()/1e9:.1f}G', flush=True)
        json.dump(dict(preds=res, meta={c: dict(cost=sum(m['cost']) / max(1, len(m['cost'])), n=m['n']) for c, m in meta.items()}, recall=rec), open(a.out, 'w'))
json.dump(dict(preds=res, meta={c: dict(cost=sum(m['cost']) / max(1, len(m['cost'])), n=m['n']) for c, m in meta.items()}, recall=rec), open(a.out, 'w'))
print('done', f'{time.time()-t0:.0f}s', flush=True)

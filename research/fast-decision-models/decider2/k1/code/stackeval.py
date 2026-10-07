"""K1 stackeval: inference side of every run (box only).

python stackeval.py run   OUT.jsonl --ck CK|none --flags F [--qz QZ.pt] --suites JB-all,REAL-agree,LONG,CF,CF-probe|DEV|EXIT|HOLDOUT [--taps 16]
      every question of every item, all questions of an item as branches over one state pass; rows {suite,id,q,T,kind,probs,logits};
      taps (residual after L layers at the answer row then the option rows, bf16) -> OUT.taps.NNN.pt shards. Resumable.
python stackeval.py calib HDIR --ck CK|none --flags F      GPTQ input Hessians (mean x^T x, unrotated) of every GEMM the runtime quantizes,
      from H1's 64 train-split calibration questions (h1lib.cal_items(64, seed=0): train_pool line % 97 == 13), state truncated to 3000 tokens.
python stackeval.py quant QZ.pt --ck CK|none --flags F --hdir HDIR       act-order GPTQ8 codes (H1 recipe) for every GEMM outside precmap_w8a8_b8.
python stackeval.py exit  HEAD.pt --train EXIT.jsonl --dev DEV.jsonl --apply A.jsonl,B.jsonl,...
      J15 exit head at layer 16, trained by KL to the run's own final distribution on EXIT taps, early-stopped on DEV KL (patience 6, <= 40
      epochs); writes A.exit.jsonl for every tap file in --apply.
"""
import os, sys, json, time, argparse, collections, random, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import torch, torch.nn.functional as F
W = os.path.expanduser('~/work/')


def load_model(ck, flags, qz=None, train=False):
    import stacklib as SL
    m = SL.Stack()
    m.free_hf()
    if ck and ck != 'none':
        meta = m.load_student(os.path.expanduser(ck))
        assert SL.flagset(meta['flags']) == SL.flagset(flags), (meta['flags'], flags)
    else:
        m.load_assets(SL.flagset(flags) or ('C', 'V', 'Q'))
        m.head = None
        if 'Q' in SL.flagset(flags): raise SystemExit('Q needs trained adapters')
    if qz:
        d = torch.load(os.path.expanduser(qz), map_location=m.dev, weights_only=False)
        m.qz = dict(codes={tuple(k): v for k, v in d['codes'].items()} if isinstance(d['codes'], dict) else d['codes'], bf16=set(d['bf16']))
    return m


def suite_items(suites):
    """-> list of (suite, item) ; DEV/EXIT from k1/data; HOLDOUT from train_v5.holdout (ruletaker d3/d5/natlang, 150 each as J3)"""
    import evalkit as EK
    out = []; seen = set()
    for s in suites:
        if s in ('DEV', 'EXIT'):
            for l in open(W + f'k1/data/{s}.jsonl'):
                it = json.loads(l); out.append((s, it))
        elif s == 'HOLDOUT':
            rows = [json.loads(l) for l in open(W + 'training/data/train_v5.holdout.jsonl')]
            random.Random(11).shuffle(rows)
            byt = collections.defaultdict(list)
            for r in rows:
                if r['task'] in ('ruletaker_d3', 'ruletaker_d5', 'ruletaker_natlang') and len(byt[r['task']]) < 150: byt[r['task']].append(r)
            for t in sorted(byt):
                for k, r in enumerate(byt[t]):
                    ins = r['instructions']
                    if r['kind'] in ('choice', 'noul'): qd = {'type': r['kind'], 'instructions': ins, 'criteria': {n: d for n, d in r['options']}}; gold = r['options'][r['label']][0]
                    else: qd = {'type': 'score', 'instructions': ins, 'criteria': [d for _, d in r['options']]}; gold = str(r['label'])
                    out.append(('HOLDOUT', dict(id=f'HO-{t}-{k}', task=t, state=r['state'], questions={'q': qd}, expected={'q': gold})))
        else:
            for it in EK.load_suite(s):
                qs = [q for q in it['questions'] if (it['id'], q) not in seen]
                if not qs: continue
                for q in qs: seen.add((it['id'], q))
                out.append((s, dict(it, questions={q: it['questions'][q] for q in qs})))
    return out


def cmd_run(a):
    m = load_model(a.ck, a.flags, a.qz)
    items = suite_items(a.suites.split(','))
    if a.limit: items = items[:a.limit]
    out = os.path.expanduser(a.out); done = set()
    if os.path.exists(out):
        for l in open(out):
            try: r = json.loads(l); done.add((r['id'], r['q']))
            except Exception: pass
    taps = tuple(int(x) for x in a.taps.split(',')) if a.taps else ()
    nshard = len(glob.glob(out + '.taps.*.pt')); shard = {}
    print('items', len(items), 'done q', len(done), 'flags', a.flags, 'qz', bool(a.qz), flush=True)
    t0 = time.time(); n = 0; buf = []

    def flush():
        nonlocal shard, nshard
        with open(out, 'a') as f:
            for l in buf: f.write(l + '\n')
        if taps and shard:
            torch.save(shard, f'{out}.taps.{nshard:03d}.pt'); nshard += 1; shard = {}
        buf.clear()
    from strands_decider.prompting import render_state
    with torch.inference_mode():
        for suite, it in items:
            qn = [q for q in it['questions'] if (it['id'], q) not in done]
            if not qn: continue
            st = it['state'] if isinstance(it['state'], str) else json.dumps(it['state'], indent=2, ensure_ascii=False)
            b = m.prep(render_state(it['state']), [(q, it['questions'][q], None) for q in qn])
            v = m.view(b, a.flags)
            o = m.forward(v, taps=taps)
            for j, qd in enumerate(b.qs):
                lg = o['logits'][j].float(); p = torch.softmax(lg, -1).tolist()
                pd = {lab: p[k] for k, lab in enumerate(qd['rq'].slot_labels)}
                buf.append(json.dumps(dict(suite=suite, id=it['id'], q=qd['name'], T=len(b.s), rows=o['P'].T, kind=qd['kind'], probs=pd,
                                           logits=[round(x, 5) for x in lg.tolist()])))
                if taps:
                    rec = {L: o['taps'][L][j].to(torch.bfloat16).cpu() for L in taps}
                    rec.update(kind=qd['kind'], n=qd['rq'].n_slots, T=len(b.s), temp=m.temp(qd['kind']), labels=list(pd))
                    shard[(it['id'], qd['name'])] = rec
                n += 1
            if len(buf) >= 300: flush(); print(n, f'{time.time() - t0:.0f}s', flush=True)
    flush()
    print('done', n, f'{time.time() - t0:.0f}s', flush=True)


def cal_items(n=64, seed=0, minT=300, maxT=3500):
    EV = set(json.load(open(W + 'evalkit/split.json'))['eval_tasks'])
    out = []; rng = random.Random(seed)
    with open(W + 'evalkit/train_pool.jsonl') as f:
        for li, l in enumerate(f):
            if li % 97 != 13 + seed: continue
            r = json.loads(l)
            if r['task'] in EV or not minT <= r['n_state_tok'] <= maxT: continue
            qn = rng.choice(sorted(r['questions']))
            out.append((r['state'], qn, r['questions'][qn]))
    return out[:n]


def cmd_calib(a):
    import stacklib as SL
    m = load_model(a.ck, a.flags)
    from strands_decider.prompting import render_state
    keys = m.gemm_keys(a.flags)
    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    its = cal_items(64)
    views = []
    for st, qn, qd in its:
        txt = render_state(st)
        b = m.prep(txt, [(qn, qd, None)])
        k = 3000 - len(b.qs[0]['q'])
        if len(b.s) > k: b = m.prep(SL.trunc_text(m.tok, txt, k), [(qn, qd, None)])
        views.append(m.view(b, a.flags))
    t0 = time.time()
    for grp in (range(0, 6), range(6, 12), range(12, 18), range(18, 24)):
        m.capture = {kk for kk in keys if kk[0] in grp}; m.Hacc = {}; m.Hn = {}
        with torch.inference_mode():
            for v in views: m.forward(v, stop=max(grp) + 1)
        for kk, Hs in m.Hacc.items():
            torch.save((Hs / m.Hn[kk]).cpu(), os.path.join(os.path.expanduser(a.out), f'H_{kk[0]}_{kk[1]}.pt'))
        print('group', list(grp), len(m.Hacc), f'{time.time() - t0:.0f}s', flush=True)
        m.Hacc = {}; m.Hn = {}; torch.cuda.empty_cache()
    m.capture = None
    json.dump(dict(n=len(views), rows=sum(1 for _ in views), keys=[list(k) for k in keys]), open(os.path.join(os.path.expanduser(a.out), 'meta.json'), 'w'))
    print('done', f'{time.time() - t0:.0f}s', flush=True)


def cmd_quant(a):
    import stacklib as SL
    m = load_model(a.ck, a.flags)
    bf = SL.load_bf16_map()
    hd = os.path.expanduser(a.hdir)
    hess = {}
    for kk in m.gemm_keys(a.flags):
        if kk in bf: continue
        hess[kk] = torch.load(os.path.join(hd, f'H_{kk[0]}_{kk[1]}.pt'), map_location='cpu')
    t0 = time.time()
    with torch.no_grad():
        codes = m.quantize(hess, bf, a.flags)
    torch.save(dict(codes={k: (q.cpu(), s.cpu()) for k, (q, s) in codes.items()}, bf16=sorted(bf), flags=a.flags), os.path.expanduser(a.out))
    print('quantized', len(codes), f'{time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- exit head (J15)
def load_preds(p):
    out = {}
    for l in open(p):
        r = json.loads(l); out[(r['id'], r['q'])] = r
    return out


def load_taps(prefix, L):
    feats = {}
    for f in sorted(glob.glob(prefix + '.taps.*.pt')):
        for k, v in torch.load(f).items():
            feats[k] = dict(x=v[L], n=v['n'], kind=v['kind'], T=v['T'], temp=v['temp'], labels=v['labels'])
    return feats


def batches(keys, feats, teach, bs, shuffle, dev='cuda'):
    keys = list(keys)
    if shuffle: random.shuffle(keys)
    for i in range(0, len(keys), bs):
        kb = keys[i:i + bs]; Kmax = max(feats[k]['n'] for k in kb)
        d = torch.stack([feats[k]['x'][0].float() for k in kb]).to(dev)
        o = torch.zeros(len(kb), Kmax, 2048, device=dev); mask = torch.zeros(len(kb), Kmax, dtype=torch.bool, device=dev)
        tv = torch.zeros(len(kb), Kmax, device=dev); temp = torch.tensor([feats[k]['temp'] for k in kb], device=dev)
        for j, k in enumerate(kb):
            n = feats[k]['n']; o[j, :n] = feats[k]['x'][1:1 + n].float().to(dev); mask[j, :n] = True
            if teach is not None: tv[j, :n] = torch.tensor([teach[k]['probs'][l] for l in feats[k]['labels']], device=dev)
        yield kb, d, o, mask, tv, temp


def run_head(h, keys, feats, teach=None, bs=64):
    out = {}; kl = 0.0; agree = 0; n = 0; h.eval()
    with torch.no_grad():
        for kb, d, o, mask, tv, temp in batches(keys, feats, teach, bs, False):
            lg = (h(d, o) / temp[:, None]).masked_fill(~mask, -1e9); p = torch.softmax(lg, -1)
            for j, k in enumerate(kb): out[k] = p[j, :feats[k]['n']].cpu()
            if teach is not None:
                kl += (tv * (torch.log(tv.clamp_min(1e-9)) - torch.log_softmax(lg, -1))).masked_fill(~mask, 0).sum().item()
                agree += (p.argmax(-1) == tv.argmax(-1)).sum().item(); n += len(kb)
    return out, (kl / max(n, 1), agree / max(n, 1))


def cmd_exit(a):
    import stacklib as SL
    from h3lib import StdHead
    random.seed(0); torch.manual_seed(0)
    m = load_model(a.ck, a.flags)
    base = (m.head if m.head is not None else m.head0).float().cpu()
    L = SL.EXIT_L
    tx = load_preds(os.path.expanduser(a.train)); td = load_preds(os.path.expanduser(a.dev))
    fx = load_taps(os.path.expanduser(a.train), L); fd = load_taps(os.path.expanduser(a.dev), L)
    kx = [k for k in fx if k in tx]; kd = [k for k in fd if k in td]
    h = SL.EH(base).cuda()
    opt = torch.optim.AdamW(h.parameters(), lr=3e-4, weight_decay=0.01)
    best = (1e9, None, -1, 0); logl = []; t0 = time.time()
    for ep in range(40):
        h.train()
        for kb, d, o, mask, tv, temp in batches(kx, fx, tx, 32, True):
            lg = (h(d, o) / temp[:, None]).masked_fill(~mask, -1e9)
            loss = (tv * (torch.log(tv.clamp_min(1e-9)) - torch.log_softmax(lg, -1))).masked_fill(~mask, 0).sum(-1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        _, (dkl, dag) = run_head(h, kd, fd, td)
        logl.append((ep, round(dkl, 5), round(dag, 4)))
        if dkl < best[0]: best = (dkl, {k: v.detach().clone() for k, v in h.state_dict().items()}, ep, dag)
        if ep - best[2] >= 6: break
    h.load_state_dict(best[1])
    torch.save(best[1], os.path.expanduser(a.out))
    info = dict(L=L, best_dev_kl=best[0], best_dev_agree=best[3], best_ep=best[2], log=logl, n_train=len(kx), n_dev=len(kd), secs=round(time.time() - t0))
    json.dump(info, open(os.path.expanduser(a.out) + '.json', 'w'))
    print('exit head', json.dumps(info), flush=True)
    for pth in a.apply.split(','):
        pth = os.path.expanduser(pth)
        f = load_taps(pth, L); P = load_preds(pth)
        pe, _ = run_head(h, list(f), f)
        with open(pth[:-6] + '.exit.jsonl', 'w') as fo:
            for k, p in pe.items():
                fo.write(json.dumps(dict(suite=P[k]['suite'] if k in P else None, id=k[0], q=k[1], probs={l: float(x) for l, x in zip(f[k]['labels'], p.tolist())})) + '\n')
        print('applied', pth, len(pe), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd'); ap.add_argument('out'); ap.add_argument('--ck', default='none'); ap.add_argument('--flags', default='')
    ap.add_argument('--qz', default=''); ap.add_argument('--suites', default='JB-all,REAL-agree,LONG,CF,CF-probe'); ap.add_argument('--taps', default='')
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--hdir', default=''); ap.add_argument('--train', default=''); ap.add_argument('--dev', default='')
    ap.add_argument('--apply', default='')
    a = ap.parse_args()
    dict(run=cmd_run, calib=cmd_calib, quant=cmd_quant, exit=cmd_exit)[a.cmd](a)

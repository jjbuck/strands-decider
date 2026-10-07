"""LM-native decision probes, scored by candidate likelihood (identical inputs for every arm).
  build: python lmeval.py build            -> data/lmeval.pkl  (CPU)
  score: python lmeval.py score RUNNAME OUT.json [--ckpt path]
Layout of every sequence (1024 + 1 tokens): [EOS pad][state][query][candidate][EOS pad]; pad chosen so that the query starts at a multiple
of 128: for the slot arm the query+candidate block is exactly one deep block (deep stack over query rows + 16 slots, state via reader K/V).
recall (CF-probe generator, eval-split states): query = the bound field of the target record; candidates = its true value vs the counterfactual
  value (for *_distract kinds that value sits in a distractor record bound to another id) / the 3 statuses.
squad (SQuAD-MC-long): query = '\\nQuestion: ...\\nAnswer:' candidates = ' ' + option."""
import os, sys, re, json, copy, pickle, inspect, random, collections, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
W = os.path.expanduser('~/work/g4'); D = f'{W}/data'
N = 1024; T = 128


def build(split='eval', n_pairs=400, n_squad=1500, seed=5):
    import prep_ft as PF
    src = inspect.getsource(PF.probe).replace('return sa_, ea, sb_, eb, ques', 'return sa_, ea, sb_, eb, ques, A, B, key, tool, (amt, lim, d1, d2, idf, stats)').replace('def probe(', 'def probe_rec(')
    exec(compile(src, 'probe_rec', 'exec'), PF.__dict__)
    enc = lambda s: PF.enc(s).astype(np.int64)
    states = [json.loads(l) for l in open(f'{D}/{split}_states.jsonl')]
    rng = random.Random(seed); items = []; cnt = collections.Counter(); i = 0; per = n_pairs // 8
    while sum(cnt.values()) < n_pairs and i < 50 * n_pairs:
        st = states[i % len(states)]; i += 1
        kk = sum(cnt.values()); kind = PF.KINDS[kk % 4]; distract = (kk // 4) % 2 == 1
        if cnt[(kind, distract)] >= per: continue
        w = PF.window(st['state'], rng, rng.randint(540, 560) if split == 'eval' else rng.randint(150, 560))
        if not w: continue
        r = PF.probe_rec(rng, w, st['domain'], kind, distract)
        if not r: continue
        sa, ea, sb, eb, ques, A, B, key, tool, (amt, lim, d1, d2, idf, stats) = r
        field = {'amount_vs_limit': amt, 'date_order': d1, 'status_equal': 'status', 'id_match': idf}[kind]
        q = f'\nLooking at the {tool} result for {key}: "{field}": '
        ok = True; new = []
        for side, (s, R_, O_) in enumerate([(sa, A, B), (sb, B, A)]):
            if kind == 'status_equal': cands = [json.dumps(x) for x in stats]; y = stats.index(R_['status'])
            else: cands = [json.dumps(R_[field]), json.dumps(O_[field])]; y = 0
            se = enc(s)
            if len(se) > 880: ok = False; break
            new.append(dict(task='recall', kind=kind + ('_distract' if distract else ''), pair=len(items) // 2 + side * 0, s=se, q=enc(q), c=[enc(c) for c in cands], y=y))
        if not ok: continue
        for e in new: e['pair'] = sum(cnt.values())
        items += new; cnt[(kind, distract)] += 1
    print('recall items', len(items), dict(cnt), flush=True)
    # squad: reuse PF.squad with a text-capturing ex()
    cap = []
    PF.ex = lambda task, qt, state, y, **m: cap.append((qt, state, y)) or True
    PF.squad('validation' if split == 'eval' else 'train', n_squad, 4 if split == 'eval' else 31, 680 if split == 'eval' else 200, 700)
    sq = []
    for qt, state, y in cap:
        m = re.match(r'Question: (.*)\nOptions: A\) (.*)\nB\) (.*)\nC\) (.*)\nD\) (.*)\nAnswer:$', qt, re.S)
        if not m: continue
        se = enc(state)
        if len(se) > 880: continue
        sq.append(dict(task='squad', s=se, q=enc('\nQuestion: ' + m.group(1) + '\nAnswer:'), c=[enc(' ' + m.group(k)) for k in range(2, 6)], y=y))
    print('squad items', len(sq), flush=True)
    return items, sq


def layout(e, ci):
    """-> (ids[N+1], score positions (indices into targets), ok)"""
    s, q, c = e['s'], e['q'], e['c'][ci]
    pre = (-len(s)) % T
    if len(s) + pre < T: pre += T          # at least one state block before the query block
    qc = np.concatenate([q, c])[:T]
    seq = np.concatenate([np.full(pre, EOS_C), s, qc])
    ids = np.full(N + 1, EOS_C, dtype=np.int64); ids[:len(seq)] = seq[:N + 1]
    start = pre + len(s) + len(q)            # first candidate token index in ids
    pos = np.arange(start - 1, start - 1 + len(c))   # target index t predicts ids[t+1]
    return ids, pos


if __name__ == '__main__':
    EOS_C = int(json.load(open(f'{D}/meta.json'))['eos_c'])
    if sys.argv[1] == 'build':
        rec, sq = build('eval')
        rec_tr, sq_tr = build('train', n_pairs=8000, n_squad=16000, seed=6)
        pickle.dump(dict(recall=rec, squad=sq, recall_train=rec_tr, squad_train=sq_tr), open(f'{D}/lmeval.pkl', 'wb'))
        sys.exit()
    import torch, common as C
    run, out = sys.argv[2], sys.argv[3]
    ck = sys.argv[sys.argv.index('--ckpt') + 1] if '--ckpt' in sys.argv else f'{W}/runs/{run}/final.pt'
    sc, arm = run.split('_')[0], run.split('_')[1]
    cfg = C.Cfg(sc, arm); m = C.build(cfg)
    sd = torch.load(ck, map_location='cpu'); m.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()}, strict=False)
    m = m.cuda().eval()
    if cfg.kind == 'slot': m._pt_setup(N, 'cuda')
    f = torch.compile(lambda a, b: m.lm_loss(a, b))
    data = pickle.load(open(f'{D}/lmeval.pkl', 'rb'))
    res = {}
    for task in ('recall', 'squad'):
        jobs = [(ii, ci) + layout(e, ci) for ii, e in enumerate(data[task]) for ci in range(len(e['c']))]
        scores = collections.defaultdict(dict)
        B = 16
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            for b0 in range(0, len(jobs), B):
                ch = jobs[b0:b0 + B]; mlen = len(ch); ch = ch + [ch[0]] * (B - mlen)
                x = torch.from_numpy(np.stack([j[2] for j in ch])).cuda()
                l = f(x[:, :-1], x[:, 1:]).float().cpu().numpy()
                for k, (ii, ci, ids, pos) in enumerate(ch[:mlen]):
                    lp = -l[k, pos]
                    scores[ii][ci] = (float(lp.sum()), float(lp.mean()))
        its = data[task]; y = np.array([e['y'] for e in its])
        ssum = np.array([max(scores[i], key=lambda c: scores[i][c][0]) for i in range(len(its))])
        smean = np.array([max(scores[i], key=lambda c: scores[i][c][1]) for i in range(len(its))])
        r = dict(n=len(its), acc_sum=float((ssum == y).mean()), acc_mean=float((smean == y).mean()),
                 correct_sum=(ssum == y).astype(int).tolist(), correct_mean=(smean == y).astype(int).tolist())
        if task == 'recall':
            byk = collections.defaultdict(list); byp = collections.defaultdict(list)
            for e, ok in zip(its, ssum == y): byk[e['kind']].append(ok); byp[e['pair']].append(ok)
            r['by_kind'] = {k: round(float(np.mean(v)), 3) for k, v in sorted(byk.items())}
            r['pair_both'] = float(np.mean([all(v) for v in byp.values() if len(v) == 2]))
            dk = [ok for e, ok in zip(its, ssum == y) if e['kind'].endswith('_distract')]
            r['acc_distract'] = float(np.mean(dk))
        res[task] = r
        print(run, task, {k: v for k, v in r.items() if not k.startswith('correct')}, flush=True)
    json.dump(dict(run=run, ckpt=ck, **res), open(out, 'w'))

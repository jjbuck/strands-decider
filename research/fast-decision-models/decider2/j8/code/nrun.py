"""nrun.py (inf2): latency and eval with traced NEFF modules.
  python nrun.py lat  TAG T M [REPS]        exact-length latency, fresh states per rep -> neff/lat_TAG.json
  python nrun.py eval TAG1,TAG2,... [LIMITFILE]   eval items bucketed (right-padded) into the Single graphs -> npreds_<...>.jsonl
"""
import os, sys, json, time, statistics as st
sys.path.insert(0, os.path.expanduser('~/work/j8'))
import torch, torch_neuronx
import hob

D = os.path.expanduser('~/work/j8'); N = f'{D}/neff'
mode = sys.argv[1] if __name__ == "__main__" else ""


def head():
    from safetensors.torch import load_file
    hc, _ = hob.ckpt_dirs()
    return hob.Head(load_file(f'{hc}/head.safetensors'), json.load(open(f'{hc}/hobson_config.json')))


if mode == 'lat':
    tag, T, M = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); REPS = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    LI = json.load(open(f'{D}/lat_inputs.json')); qs = LI['questions']; H = head()
    t0 = time.time(); m = torch.jit.load(f'{N}/{tag}.pt'); tl = time.time() - t0
    S_SEL = None

    def inp(rep):
        s = LI['states'][str(T)][rep % len(LI['states'][str(T)])]
        if M == 1:
            q = qs[0]; ids = s + q['q']; r = [T + o for o in q['opt']] + [len(ids) - 1]
            if PADL: ids = ids + [0] * (PADL - len(ids))
            return (torch.tensor(ids), torch.tensor(r + [r[-1]] * (SEL - len(r))))
        qq = qs[:M]; Lq = max(len(q['q']) for q in qq); S = max(len(q['opt']) for q in qq) + 1
        Lq = ((Lq + CC - 1) // CC) * CC; Ls = ((T + CC - 1) // CC) * CC
        b = torch.zeros(M, Lq, dtype=torch.long); sel = torch.zeros(M, S, dtype=torch.long)
        for i, q in enumerate(qq):
            b[i, :len(q['q'])] = torch.tensor(q['q']); r = q['opt'] + [len(q['q']) - 1]; r += [r[-1]] * (S - len(r)); sel[i] = torch.tensor(r)
        return (torch.tensor(s + [0] * (Ls - T)), b, sel)

    SEL = int(tag.split('_S')[1].split('_')[0]) if '_S' in tag else 3
    PADL = int(tag[1:].split('_')[0]) if tag.startswith('L') else 0
    CC = int(tag.split('_C')[1].split('_')[0]) if '_C' in tag else 64

    def call(rep):
        x = inp(rep); rows = m(*x)
        if M == 1: return [H.probs(rows[:len(qs[0]['opt']) + 1], qs[0]['kind'])]
        return [H.probs(rows[i, :len(q['opt']) + 1], q['kind']) for i, q in enumerate(qs[:M])]

    for r in range(3): call(r)
    if os.environ.get('START_AT'):
        while time.time() < float(os.environ['START_AT']): time.sleep(0.01)
    tw = time.time()
    ts = []
    for r in range(3, 3 + REPS):
        t = time.perf_counter(); p = call(r); ts.append((time.perf_counter() - t) * 1000)
    # device-only time: same input repeated (no host prep), for the split
    x = inp(0); td = []
    for _ in range(10):
        t = time.perf_counter(); m(*x); td.append((time.perf_counter() - t) * 1000)
    ts.sort(); td.sort()
    out = dict(window=[round(tw, 2), round(time.time(), 2)], tag=tag, T=T, M=M, median=round(st.median(ts), 2), p95=round(ts[int(0.95 * (len(ts) - 1) + 0.5)], 2), min=round(ts[0], 2),
               call_only_median=round(st.median(td), 2), n=len(ts), load_s=round(tl, 1), probs=[x.tolist() for x in p])
    print(json.dumps(out), flush=True)
    json.dump(out, open(f'{N}/lat_{tag}.json', 'w'))

elif mode == 'eval':
    tags = sys.argv[2].split(',')
    items = [json.loads(l) for l in open(f'{D}/ids.jsonl')]
    if len(sys.argv) > 3:
        keep = set(json.load(open(sys.argv[3]))); items = [it for it in items if f"{it['id']}|{it['qn']}" in keep]
    H = head()
    mods = {}
    for t in tags:
        if t.startswith('L'): L = int(t[1:].split('_')[0])
        else:
            L = int(t.split('_')[0][1:]) if t.startswith('T') else int(t[5:].split('_')[0])
            if t.startswith('T'): L = L + 93
        SEL = int(t.split('_S')[1].split('_')[0])
        mods[L] = (t, SEL)
    Ls = sorted(mods)
    out = os.environ.get('OUT', f'{D}/npreds.jsonl')
    done = set()
    if os.path.exists(out):
        for l in open(out): done.add(json.loads(l)['k'])
    by = {}
    for it in items:
        L = len(it['s']) + len(it['q'])
        b = next((x for x in Ls if x >= L), None)
        if b is None or len(it['opt']) + 1 > mods[b][1]: continue
        by.setdefault(b, []).append(it)
    print({b: len(v) for b, v in by.items()}, 'of', len(items), flush=True)
    with open(out, 'a') as f:
        for b in Ls:
            if b not in by: continue
            tag, SEL = mods[b]; m = torch.jit.load(f'{N}/{tag}.pt'); t0 = time.time()
            for it in by[b]:
                k = f"{it['id']}|{it['qn']}"
                if k in done: continue
                ids = it['s'] + it['q']; Lr = len(ids)
                ids = ids + [0] * (b - Lr)
                r = [len(it['s']) + o for o in it['opt']] + [Lr - 1]; r += [r[-1]] * (SEL - len(r))
                rows = m(torch.tensor(ids), torch.tensor(r))
                p = H.probs(rows[:len(it['opt']) + 1], it['kind'])
                f.write(json.dumps(dict(k=k, suite=it['suite'], id=it['id'], qn=it['qn'], p=dict(zip(it['labels'], p.tolist())), L=Lr, b=b)) + '\n'); f.flush()
            print('bucket', b, len(by[b]), '%.0fs' % (time.time() - t0), flush=True)
            del m
    print('done', flush=True)

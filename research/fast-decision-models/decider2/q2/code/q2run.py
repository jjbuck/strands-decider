"""Q2 runner (J15 j15run.py interface) for Q2's runtimes: questions through the deployed kernels, hobson layout, references' exact token ids,
optional residual taps for depth-exit heads. python q2run.py OUT.jsonl --fmt FMT [--suites ...|DEV|EXIT] [--taps 16] [--limit n]
FMT: c:<fmt> (QRT2C; e.g. c:map:~/work/q2/q2map_k64rr.json) | h2:<prec> (H2 QRT via J15's QRTJ, e.g. h2:map:~/work/h1/precmap_w8a8_b8.json:w8a8).
Taps: rows the pointer head reads (answer row, then option rows) of the residual after L layers, UNROTATED, bf16 -> OUT.taps.NNN.pt shards."""
import sys, os, json, time, argparse, torch
sys.path[:0] = [os.path.expanduser('~/work/q2'), os.path.expanduser('~/work/j15'), os.path.expanduser('~/work/h2'), os.path.expanduser('~/work/evalkit'),
                os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import qrt as Q, q2rt as R

CODES = os.environ.get('CODES', 'w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt')


def load_codes(spec):
    return {kv.split('=')[0]: torch.load(os.path.expanduser(kv.split('=')[1])) for kv in spec.split(',')} if spec else None


def build(fmt, P=None):
    from kitrun import load_P
    from lean2 import Lean2
    P = P or load_P()
    ln = Lean2(P.tm, fuse=''); head = P.model.head.float().eval()
    wc = load_codes(CODES)
    if fmt.startswith('h2:'):
        from j15run import QRTJ
        m = QRTJ(ln, head=head, prec=fmt[3:], wcodes=wc); m.tune = False
    elif fmt.startswith('c:'):
        m = R.QRT2C(ln, head=head, fmt=fmt[2:], wcodes=wc, tune=False); m.tune = False
    else:
        m = R.QRT2(ln, head=head, fmt=fmt, wcodes=wc, tune=False); m.tune = False
    import gc
    for d in ln.layers:
        for k_ in ('Wgu_il', 'Win_f', 'Wgu_f'): d.pop(k_, None)
    del wc; gc.collect(); torch.cuda.empty_cache()
    return P, m, head


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('out'); ap.add_argument('--fmt', required=True); ap.add_argument('--suites', default='all'); ap.add_argument('--taps', default='')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    import evalkit as EK
    from kitrun import prep_question, probdict
    P, m, head = build(a.fmt)
    print('built', a.fmt, 'mem', round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
    suites = None if a.suites == 'all' else a.suites.split(',')
    items = list(EK.all_question_items(suites))
    if a.limit: items = items[:a.limit]
    byid = {}
    for s in (suites or ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']):
        for it in EK.load_suite(s): byid[it['id']] = it
    out = os.path.expanduser(a.out); os.makedirs(os.path.dirname(out), exist_ok=True)
    done = set()
    if os.path.exists(out):
        for l in open(out): r = json.loads(l); done.add((r['id'], r['q']))
    taps = tuple(int(x) for x in a.taps.split(',')) if a.taps else ()
    m.taps = taps
    shard = {}; nshard = len([f for f in os.listdir(os.path.dirname(out)) if f.startswith(os.path.basename(out) + '.taps.')])
    print('questions', len(items), 'done', len(done), 'taps', taps, flush=True)
    t0 = time.time(); n = 0; buf = []

    def flush():
        global shard, nshard
        with open(out, 'a') as f:
            for l in buf: f.write(l + '\n')
        if taps and shard:
            torch.save(shard, f'{out}.taps.{nshard:03d}.pt'); nshard += 1; shard = {}
        buf.clear()

    with torch.inference_mode():
        for suite, iid, qn, st, spec in items:
            if (iid, qn) in done: continue
            pr = prep_question(P, byid[iid], qn)
            ids = torch.tensor(pr['s'] + pr['q'], device='cuda'); T = ids.shape[0]
            rows = torch.tensor([T - 1] + [pr['q0'] + o for o in pr['opt']], device='cuda')
            m.taprows = rows; m.tapped = {}; m.q0 = pr['q0']
            hn = m.forward(ids, Q.Lay('single', T))
            h = m.unrot(hn[rows]).float()
            lg = (m.head(h[:1], h[1:][None]) / P.temp_for(pr['rq'].kind))[0, :pr['rq'].n_slots].float()
            pd = probdict(pr['rq'], torch.softmax(lg, -1).tolist())
            if taps:
                rec = {L: m.unrot_raw(v).to(torch.bfloat16).cpu() for L, v in m.tapped.items()}
                rec.update(kind=pr['rq'].kind, n=pr['rq'].n_slots, T=T, temp=P.temp_for(pr['rq'].kind), labels=list(pd))
                shard[(iid, qn)] = rec
            buf.append(json.dumps(dict(suite=suite, id=iid, q=qn, T=T, kind=pr['rq'].kind, probs=pd, logits=[round(x, 5) for x in lg.tolist()]))); n += 1
            if n % 200 == 0:
                flush(); print(a.fmt, n, f'{time.time() - t0:.0f}s', flush=True)
    flush()
    print('done', a.fmt, n, f'{time.time() - t0:.0f}s', flush=True)

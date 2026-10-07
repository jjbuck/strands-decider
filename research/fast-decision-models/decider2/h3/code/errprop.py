"""H3 step 1: quantization-error propagation in hobson-v19 (GDN hybrid), after the paper's figs 2/3/5.
  A) quantize ONE layer (all 4 GEMMs, W4A4) at a time; capture every sub-layer tensor of every layer; per position report
     rel = mean_t ||y~_t - y_t|| / ||y_t||  (the paper's metric)  and  abs = ||Y~ - Y||_F / sqrt(T) (RMS error per token), ref norm.
  B) quantize ONE sub-layer GEMM of one layer at a time (GDN: in_proj qkv / z / ab rows, out_proj; attention: qkv(+gate) / o; MLP: gate_up / down);
     report final-hidden error at the readout rows and the decision effect (TV, argmax flips) over many questions.
Samples: train-split real states (evalkit/train_pool.jsonl, eval tasks excluded), one question each.
python errprop.py A --n 12 --fmt int4      python errprop.py B --n 64 --fmt int4
"""
import os, sys, json, time, argparse, random
sys.path.insert(0, os.path.expanduser('~/work/h3'))
import torch
import h3lib as H

ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--n', type=int, default=12); ap.add_argument('--fmt', default='int4')
ap.add_argument('--maxT', type=int, default=1600); ap.add_argument('--minT', type=int, default=700); ap.add_argument('--out', default=None)
ap.add_argument('--layers', default='all'); ap.add_argument('--clip', type=float, default=1.0); ap.add_argument('--rot', action='store_true'); ap.add_argument('--tag', default='')
a = ap.parse_args()
OUT = os.path.expanduser('~/work/h3/res'); os.makedirs(OUT, exist_ok=True)


def samples(n, minT, maxT, seed=7):
    EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
    rows = []
    with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
        for li, l in enumerate(f):
            if li % 7 != 3: continue
            r = json.loads(l)
            if r['task'] in EV or not (minT <= r['n_state_tok'] <= maxT): continue
            rows.append(r)
    random.Random(seed).shuffle(rows)
    out = []
    for r in rows:
        qn = sorted(r['questions'])[len(out) % len(r['questions'])]
        out.append((r['rid'], qn, r['state'], r['questions'][qn]))
        if len(out) >= n: break
    return out


m = H.H3(); m.free_hf()
S = samples(a.n, a.minT, a.maxT)
LAYERS = list(range(24)) if a.layers == 'all' else [int(x) for x in a.layers.split(',')]
print('samples', len(S), flush=True)


@torch.inference_mode()
def run_A():
    res = {}  # qlayer -> {layer -> {pos -> [rel_sum, abs_sq_sum, ref_sq_sum, T]}}
    for si, (rid, qn, st, qd) in enumerate(S):
        pr = m.prep(st, qd); ids = pr['s'] + pr['q']; T = len(ids)
        ref = {}
        def cap_ref(i, pos, t): ref[(i, pos)] = t.detach().clone()
        h0 = m.forward(ids, H.DENSE, cap=cap_ref); p0 = m.probs(h0, pr)
        for ql in LAYERS:
            acc = res.setdefault(ql, {})
            def cap_q(i, pos, t):
                if i < ql: return
                r_ = ref[(i, pos)].float(); d = t.float() - r_
                rel = (d.norm(dim=-1) / r_.norm(dim=-1).clamp_min(1e-6)).mean().item()
                e = acc.setdefault(i, {}).setdefault(pos, [0.0, 0.0, 0.0, 0])
                e[0] += rel * T; e[1] += d.pow(2).sum().item(); e[2] += r_.pow(2).sum().item(); e[3] += T
            qc = H.Q(a.fmt, layers=[ql], clip=a.clip, rot=a.rot)
            h = m.forward(ids, qc, cap=cap_q); p = m.probs(h, pr)
            e = acc.setdefault('dec', [0.0, 0, 0])
            e[0] += 0.5 * (p - p0).abs().sum().item(); e[1] += int(p.argmax() != p0.argmax()); e[2] += 1
            rd = acc.setdefault('readout', [0.0, 0])
            rows = [T - 1] + [pr['q0'] + o for o in pr['opt']]
            rd[0] += ((h[rows].float() - h0[rows].float()).norm(dim=-1) / h0[rows].float().norm(dim=-1)).mean().item(); rd[1] += 1
        del ref
        print(f'A {si + 1}/{len(S)} T={T} {time.time() - t0:.0f}s mem {torch.cuda.max_memory_allocated() / 1e9:.1f}G', flush=True)
        dump_A(res)
    return res


def dump_A(res):
    out = {}
    for ql, acc in res.items():
        o = {}
        for i, v in acc.items():
            if i == 'dec': o['dec'] = dict(tv=v[0] / v[2], flip=v[1] / v[2], n=v[2]); continue
            if i == 'readout': o['readout_rel'] = v[0] / v[1]; continue
            o[str(i)] = {pos: dict(rel=e[0] / e[3], abs=(e[1] / e[3]) ** 0.5, ref=(e[2] / e[3]) ** 0.5) for pos, e in v.items()}
        out[str(ql)] = o
    json.dump(dict(fmt=a.fmt, n=len(S), clip=a.clip, res=out), open(a.out or f'{OUT}/errA_{a.fmt}{a.tag}.json', 'w'))


@torch.inference_mode()
def run_B():
    subs = {}
    for l in LAYERS:
        gdn = m.L[l]['type'] == 'linear_attention'
        names = (['Win.qkv', 'Win.z', 'Win.ab'] if gdn else ['Win']) + ['Wo', 'Wgu', 'Wd']
        for nm in names:
            subs[f'{l}:{nm}'] = H.Q(a.fmt, layers=[l], names=[nm], clip=a.clip, rot=a.rot)
    # whole-model references
    subs['all:ALL'] = H.Q(a.fmt, clip=a.clip)
    subs['all:noab'] = H.QCfg([dict(w=a.fmt, a=a.fmt, clip=a.clip), dict(w='bf16', a='bf16', names=['Win.ab'])])
    subs['all:W'] = H.Q(a.fmt, 'bf16'); subs['all:A'] = H.Q('bf16', a.fmt, clip=a.clip)
    subs['all:fold'] = H.Q(a.fmt, clip=a.clip, fold=True)
    acc = {k: [0.0, 0, 0.0, 0] for k in subs}
    for si, (rid, qn, st, qd) in enumerate(S):
        pr = m.prep(st, qd); ids = pr['s'] + pr['q']; T = len(ids)
        rows = [T - 1] + [pr['q0'] + o for o in pr['opt']]
        h0 = m.forward(ids, H.DENSE); p0 = m.probs(h0, pr)
        for k, qc in subs.items():
            h = m.forward(ids, qc); p = m.probs(h, pr)
            e = acc[k]
            e[0] += 0.5 * (p - p0).abs().sum().item(); e[1] += int(p.argmax() != p0.argmax())
            e[2] += ((h[rows].float() - h0[rows].float()).norm(dim=-1) / h0[rows].float().norm(dim=-1)).mean().item(); e[3] += 1
        if si % 4 == 3 or si == len(S) - 1:
            json.dump(dict(fmt=a.fmt, n=si + 1, res={k: dict(tv=v[0] / v[3], flip=v[1] / v[3], readout_rel=v[2] / v[3]) for k, v in acc.items()}),
                      open(a.out or f'{OUT}/errB_{a.fmt}{a.tag}.json', 'w'))
            print(f'B {si + 1}/{len(S)} T={T} {time.time() - t0:.0f}s', flush=True)


t0 = time.time()
if a.mode == 'A': run_A()
else: run_B()
print('done', time.time() - t0, flush=True)

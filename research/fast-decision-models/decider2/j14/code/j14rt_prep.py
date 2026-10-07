"""Build compiled question caches for the runtime (latency sets + fidelity items) with j14lib (bf16, frozen conv semantics).
python j14rt_prep.py OUT.pt [--ckpt CK] [--lv oa+sfx]"""
import os, sys, json, argparse, collections
sys.path[:0] = [os.path.expanduser('~/work/j14'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from j14lib import J14, live_positions
from h3lib import chunk_gated_delta_rule
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--ckpt', default=None); ap.add_argument('--lv', default='oa+sfx')
ap.add_argument('--nfid', type=int, default=24)
a = ap.parse_args()
m = J14(); m.setup(); m.conv_mode = 'frozen'
if a.ckpt:
    import j14train_util as TU; TU.load_ckpt(m, a.ckpt)
m.free_hf()
Q15 = ['cc_asked_for_human', 'cc_insists', 'cc_offers_transfer', 'cc_refuses', 'cc_can_still_help', 'cc_procedure_found', 'cc_claims_done',
       'cc_ends', 'details_match', 'failure_cause', 'rule_bound_values', 'identity_established', 'identity_verified', 'leaks_internal', 'wants_change']
specs = {}
for s, iid, q, st, spec in EK.all_question_items(['REAL-agree', 'LONG']):
    specs.setdefault(q, spec)
pool = [json.loads(l) for l in open(os.path.expanduser('~/work/evalkit/train_pool.jsonl'))][:4000]
for r in pool:
    for q, spec in r['questions'].items(): specs.setdefault(q, spec)
missing = [q for q in Q15 if q not in specs]; print('missing', missing, flush=True)
jb = [x for x in EK.all_question_items(['JB-all'])]


def compile_q(spec, state=''):
    pr = m.prep(state, spec); lp = live_positions(m.tok, pr, a.lv)
    with torch.inference_mode():
        cache, T = m.prefix(pr['s'])
        m.suffix(cache, T, pr['q'], lp)
        comp = m.comp_out
        l0 = lp[0]
        out = []
        for i in range(24):
            c = comp[i]
            if 'cv' in c:
                cv = c['cv']
                need = sorted({p - 3 + j for p in lp for j in range(3) if 0 <= p - 3 + j and (p - 3 + j) not in set(lp)})
                d = dict(cv=cv.cpu(), ab=c['ab'].cpu(), raw={p: c['raw'][p].cpu() for p in need})
                # affine transfer of the leading compiled run [0, l0)
                qq, kk, vv = cv[:l0].split(2048, -1)
                L0 = l0
                I0 = torch.eye(128, device=m.dev, dtype=torch.float32)[None, None].expand(1, 16, 128, 128).contiguous()
                kw = dict(use_qk_l2norm_in_kernel=True, output_final_state=True)
                _, A = chunk_gated_delta_rule(qq.reshape(1, L0, 16, 128), kk.reshape(1, L0, 16, 128), torch.zeros_like(vv).reshape(1, L0, 16, 128), c['g'][:l0][None],
                                              c['beta'][:l0][None].to(qq.dtype), initial_state=I0, **kw)
                _, B = chunk_gated_delta_rule(qq.reshape(1, L0, 16, 128), kk.reshape(1, L0, 16, 128), vv.reshape(1, L0, 16, 128), c['g'][:l0][None],
                                              c['beta'][:l0][None].to(qq.dtype), initial_state=torch.zeros_like(I0), **kw)
                d['A'] = {l0: A[0].cpu()}; d['B'] = {l0: B[0].cpu()}
                out.append(d)
            else:
                out.append(dict(K=c['K'].cpu(), V=c['V'].cpu()))
        del cache
    return dict(q=list(pr['q']), lp=lp, opt=list(pr['opt']), temp=m.temp(pr['rq'].kind), n_slots=pr['rq'].n_slots, comp=out,
                labels=list(pr['rq'].slot_labels))


res = dict(sets={}, fid=[])
res['sets']['bk15'] = [compile_q(specs[q]) for q in Q15]
print('bk15 live/q', [(len(x['lp']), len(x['q'])) for x in res['sets']['bk15']], flush=True)
res['sets']['bk4'] = res['sets']['bk15'][:4]; res['sets']['bk1'] = res['sets']['bk15'][:1]
jl = sorted(jb, key=lambda x: len(m.prep('', x[4])['q']))
med = jl[len(jl) // 2]
try:
    res['sets']['jb1'] = [compile_q(med[4])]
    res['sets']['jb4'] = [compile_q(x[4]) for x in jl[len(jl) // 2 - 2: len(jl) // 2 + 2]]
except TypeError:
    print('JB questions fully live under this live set: nothing to compile', flush=True)
print({k: [len(x['q']) for x in v] for k, v in res['sets'].items()}, {k: [len(x['lp']) for x in v] for k, v in res['sets'].items()}, flush=True)
# fidelity items: real REAL-agree states (<= 1500 tokens), lib probabilities in the same layout
its = [x for x in EK.all_question_items(['REAL-agree'])]
k = 0
for s, iid, q, st, spec in its[::17]:
    pr = m.prep(st, spec)
    if len(pr['s']) > 1500: continue
    cq = compile_q(spec)
    with torch.inference_mode():
        cache, T = m.prefix(pr['s'])
        hl = m.suffix(cache, T, pr['q'], cq['lp'])
        p = torch.softmax(m.readout(hl, cq['lp'], pr['opt'], len(pr['q']), pr['rq'].kind, pr['rq'].n_slots).float(), -1).cpu()
    res['fid'].append(dict(s=list(pr['s']), qc=cq, p_lib=p, iid=iid, qn=q))
    k += 1
    if k >= a.nfid: break
torch.save(res, a.out)
print('saved', a.out, flush=True)

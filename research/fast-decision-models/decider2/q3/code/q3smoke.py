"""Q3 smoke tests (box): 1) packed multi-question teacher == per-question teacher; 2) rotated-domain bf16 student == teacher;
3) W4A4 GPTQ student vs teacher on a few train questions; 4) one backward per mode: finite, nonzero grads; 5) step timing.
python q3smoke.py [--gptq ~/work/q3/gptq4.pt]"""
import os, sys, json, time, argparse
sys.path[:0] = [os.path.expanduser('~/work/q3')]
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch
import q3lib as QL

ap = argparse.ArgumentParser(); ap.add_argument('--gptq', default='~/work/q3/gptq4.pt'); ap.add_argument('--n', type=int, default=6); a = ap.parse_args()
m = QL.Q3(); dev = m.dev
pool, devset, v5 = QL.load_data(5000, 20, 0)
res = {}


def tv(l1, l2): return float(0.5 * (l1.float().exp() - l2.float().exp()).abs().sum())


reqs = []
for r in pool[:40]:
    if len(r['questions']) >= 3 and r['n'] < 2500:
        names = sorted(r['questions'])[:4]; reqs.append((r, names))
    if len(reqs) >= a.n: break
with torch.no_grad():
    # 1) packed vs single (teacher)
    d1 = []
    for r, names in reqs:
        rq = m.prep(r['state'], [r['questions'][n] for n in names])
        (lp, _), _ = m.run(rq, student=False, keep=())
        for j, n in enumerate(names):
            rq1 = m.prep(r['state'], [r['questions'][n]])
            (l1, _), _ = m.run(rq1, student=False, keep=())
            d1.append(tv(lp[j], l1[0]))
    res['packed_vs_single_teacher_tv_max'] = max(d1); res['packed_vs_single_teacher_tv_mean'] = sum(d1) / len(d1)
    print('1', res, flush=True)
    # 2) bf16 student in the rotated domain (all GEMMs 'bf16' pmap) vs teacher
    pm = {(i, k): 'bf16' for i in range(24) for k in QL.GEMMS}
    m.set_student('lat', None, pmap=pm)
    d2 = []
    for r, names in reqs:
        rq = m.prep(r['state'], [r['questions'][n] for n in names])
        (lt, kt), _ = m.run(rq, student=False); (ls, ks), _ = m.run(rq, student=True)
        d2 += [tv(x, y) for x, y in zip(lt, ls)]
        res.setdefault('rot_bf16_hid_rel', []).append({i: float(((ks[i] - kt[i]).pow(2).sum(-1) / kt[i].pow(2).sum(-1)).mean()) for i in ks})
    res['rot_bf16_vs_teacher_tv_mean'] = sum(d2) / len(d2); res['rot_bf16_vs_teacher_tv_max'] = max(d2)
    print('2', json.dumps(res)[:2000], flush=True)
    m.S = {}; torch.cuda.empty_cache()
init = torch.load(os.path.expanduser(a.gptq), map_location='cpu')
src = {key: dict(q=e['q'], s=e['s'], Wc=e['Wc']) for key, e in init.items()}; del init
params = m.set_student('lat', src)
with torch.no_grad():
    d3 = []; fl = 0; kls = []
    for r, names in reqs:
        rq = m.prep(r['state'], [r['questions'][n] for n in names])
        (lt, kt), _ = m.run(rq, student=False); (ls, ks), _ = m.run(rq, student=True)
        for x, y in zip(lt, ls):
            d3.append(tv(x, y)); fl += int(x.argmax() != y.argmax()); kls.append(float((x.exp() * (x - y)).sum()))
        res.setdefault('w4_hid_rel', []).append({i: float(((ks[i] - kt[i]).pow(2).sum(-1) / kt[i].pow(2).sum(-1)).mean()) for i in ks})
    res['w4a4_gptq_tv_mean'] = sum(d3) / len(d3); res['w4a4_gptq_flips'] = fl; res['w4a4_n'] = len(d3); res['w4a4_kl_mean'] = sum(kls) / len(kls)
    print('3', json.dumps({k: v for k, v in res.items() if 'w4' in k}), flush=True)
# 4) backward + timing (qad mode)
opt = QL.AdamSR(params, [p_.detach().clone() for p_ in params], lr=2e-5, l2sp=100.0)
print('mem after opt anchors', torch.cuda.memory_allocated() / 1e9, flush=True)
tt = []
for it, (r, names) in enumerate(reqs[:4]):
    rq = m.prep(r['state'], [r['questions'][n] for n in names])
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad(): (lt, kt), _ = m.run(rq, student=False)
    torch.cuda.synchronize(); t1 = time.time()
    (ls, ks), lay = m.run(rq, student=True)
    loss, kl, hid = QL.loss_fn(lt, kt, ls, ks, 0.5, None)
    loss.backward(); torch.cuda.synchronize(); t2 = time.time()
    gn = [float(p_.grad.float().norm()) for p_ in params[:8]]
    finite = all(torch.isfinite(p_.grad).all().item() for p_ in params)
    opt.step(); opt.zero_grad(set_to_none=True); torch.cuda.synchronize(); t3 = time.time()
    tt.append(dict(T=lay['T'], teacher=t1 - t0, student_fb=t2 - t1, opt=t3 - t2, kl=kl, hid=hid, gnorm0=gn[:4], finite=finite,
                   mem=torch.cuda.max_memory_allocated() / 1e9))
    print('4', json.dumps(tt[-1]), flush=True)
res['timing'] = tt
res['tok_per_s_student_plus_teacher'] = sum(x['T'] for x in tt[1:]) / sum(x['teacher'] + x['student_fb'] for x in tt[1:])
# soft / code modes: one backward each
for mode in ('soft', 'code'):
    m.S = {}; torch.cuda.empty_cache()
    init = torch.load(os.path.expanduser(a.gptq), map_location='cpu')
    src = {}
    for key, e in init.items():
        u = e['Wc'].float() / e['s'][:, None]; cf = torch.floor(u).clamp(-7, 6); rest = (u - cf).clamp(0, 1)
        p_ = ((rest - QL.GAMMA) / (QL.ZETA - QL.GAMMA)).clamp(1e-4, 1 - 1e-4)
        src[key] = dict(q=e['q'], s=e['s'], Wc=e['Wc'], cf=cf.to(torch.int8), V=torch.log(p_ / (1 - p_)).to(torch.bfloat16))
    del init
    ps = m.set_student(mode, src); del src
    r, names = reqs[0]
    rq = m.prep(r['state'], [r['questions'][n] for n in names])
    with torch.no_grad(): (lt, kt), _ = m.run(rq, student=False)
    (ls, ks), lay = m.run(rq, student=True)
    loss, kl, hid = QL.loss_fn(lt, kt, ls, ks, 0.5, None); loss.backward()
    res[mode] = dict(kl=kl, hid=hid, finite=all(torch.isfinite(p_.grad).all().item() for p_ in ps), gnorm=[float(p_.grad.float().norm()) for p_ in ps[:4]])
    print('5', mode, json.dumps(res[mode]), flush=True)
json.dump(res, open(os.path.expanduser('~/work/q3/smoke.json'), 'w'), indent=1)
print('SMOKE_DONE', flush=True)

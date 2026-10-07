"""Laptop analysis of q1fo outputs (~/decider2/q1/res/fo/{tag}.json): B1.3 predictor, B6 certificate, B2 coherence, B8 classes, positions, B10.
python3 q1fo_report.py tag [tag...]  -> prints; writes ~/decider2/q1/res/fo_summary.json"""
import json, sys, os, math, random, statistics as St
D = os.path.expanduser('~/decider2/q1/res/fo')
outp = os.path.expanduser('~/decider2/q1/res/fo_summary.json')
allo = json.load(open(outp)) if os.path.exists(outp) else {}


def corr(x, y):
    mx, my = St.mean(x), St.mean(y); c = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x); vy = sum((b - my) ** 2 for b in y)
    return c / math.sqrt(vx * vy) if vx > 0 and vy > 0 else float('nan'), c / vx if vx > 0 else float('nan')


def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg: return float('nan')
    w = 0.0
    for p in pos:
        for n in neg: w += 1.0 if p > n else (0.5 if p == n else 0.0)
    return w / (len(pos) * len(neg))


def certify(rq, key, rng_seed=0, cover=1.0):
    """split requests in half (calibration / test); alpha = max (cover=1) of |dm| / stat on calibration; certified if m0 > alpha*stat"""
    idx = list(range(len(rq))); random.Random(rng_seed).shuffle(idx)
    cal = [rq[i] for i in idx[:len(idx) // 2]]; te = [rq[i] for i in idx[len(idx) // 2:]]
    stat = lambda r: key(r)
    ratios = sorted(abs(r['dm']) / max(stat(r), 1e-12) for r in cal)
    alpha = ratios[min(len(ratios) - 1, int(math.ceil(cover * len(ratios))) - 1)]
    cert = [r for r in te if r['m0'] > alpha * stat(r)]
    return dict(alpha=alpha, certified=len(cert) / len(te), resid_flips=sum(r['flip'] for r in cert), test_flips=sum(r['flip'] for r in te), n_test=len(te))


def margin_rule(rq, rng_seed=0):
    idx = list(range(len(rq))); random.Random(rng_seed).shuffle(idx)
    cal = [rq[i] for i in idx[:len(idx) // 2]]; te = [rq[i] for i in idx[len(idx) // 2:]]
    tau = max([r['m0'] for r in cal if r['flip']] + [0.0])
    cert = [r for r in te if r['m0'] > tau]
    return dict(tau=tau, certified=len(cert) / len(te), resid_flips=sum(r['flip'] for r in cert), test_flips=sum(r['flip'] for r in te), n_test=len(te))


for tag in sys.argv[1:]:
    d = json.load(open(f'{D}/{tag}.json')); rq = d['reqs']; n = len(rq)
    o = dict(n=n, flips=sum(r['flip'] for r in rq), flip_rate=sum(r['flip'] for r in rq) / n, tv=St.mean(r['tv'] for r in rq), work=d.get('work'))
    xs = [r['pred'] for r in rq]; ys = [r['dm'] for r in rq]
    o['pred_corr'], o['pred_slope'] = corr(xs, ys)
    o['rms_dm'] = math.sqrt(St.mean(y * y for y in ys)); o['rms_pred'] = math.sqrt(St.mean(x * x for x in xs))
    o['rms_pred_state'] = math.sqrt(St.mean(r['pred_s'] ** 2 for r in rq)); o['rms_pred_question'] = math.sqrt(St.mean(r['pred_q'] ** 2 for r in rq))
    o['flip_auc_pred'] = auc([-(r['pred']) / max(r['m0'], 1e-9) for r in rq], [r['flip'] for r in rq])
    o['flip_pred_first_order'] = dict(predicted=sum(1 for r in rq if r['m0'] + r['pred'] < 0), hit=sum(1 for r in rq if r['m0'] + r['pred'] < 0 and r['flip']))
    # per-layer first-order variance share
    pl = [St.mean(r['per_layer'][i] ** 2 for r in rq) for i in range(24)]; t = sum(pl) or 1
    o['layer_share_first_order'] = [round(v / t, 4) for v in pl]
    # B6 certificates
    o['cert_corr_absdm_sqrtc'] = corr([math.sqrt(sum(r['cert'])) for r in rq], [abs(r['dm']) for r in rq])[0]
    o['cert_corr_absdm_enorm'] = corr([math.sqrt(sum(r['enorm'])) for r in rq], [abs(r['dm']) for r in rq])[0]
    o['b6'] = dict(cert=certify(rq, lambda r: math.sqrt(sum(r['cert']))), cert_enorm=certify(rq, lambda r: math.sqrt(sum(r['enorm']))),
                   oracle_first_order=certify(rq, lambda r: abs(r['pred']) + 1e-3), margin_only=margin_rule(rq))
    # B2 coherence: kappa = E[S^2] / E[Q] per role, aggregated by layer band
    coh = d.get('coh', {}); kap = {}
    for role in ('s', 'q'):
        for band, lo, hi in (('0-8', 0, 8), ('9-12', 9, 12), ('13-23', 13, 23), ('all', 0, 23)):
            S2 = sum(v[role][0] for k, v in coh.items() if role in v and lo <= int(k.split('.')[0]) <= hi)
            Q = sum(v[role][1] for k, v in coh.items() if role in v and lo <= int(k.split('.')[0]) <= hi)
            kap[f'{role}_{band}'] = S2 / Q if Q > 0 else float('nan')
    o['b2_coherence_kappa'] = kap
    # first-order variance by role summed over GEMMs (coherent, per GEMM)
    o['first_order_var_by_role'] = {role: sum(v[role][0] for v in coh.values() if role in v) / n for role in ('s', 'q')}
    # B8 token classes (state rows): per-row mean of sum_GEMM a^2, relative to the state average
    cl = d.get('cls', {}); tot_r = sum(v[0] for k, v in cl.items() if k != '_question'); tot_e = sum(v[1] for k, v in cl.items() if k != '_question')
    o['b8_classes'] = {k: dict(rows_share=v[0] / tot_r, energy_share=v[1] / tot_e, per_row_rel=(v[1] / v[0]) / (tot_e / tot_r)) for k, v in cl.items() if k != '_question' and v[0] > 0}
    if '_question' in cl: o['b8_question_per_row_rel_to_state'] = (cl['_question'][1] / cl['_question'][0]) / (tot_e / tot_r)
    ps = d.get('pos', {})
    if 'all_state' in ps:
        te = ps['all_state'][1]; tr = ps['all_state'][0]
        o['positions'] = {k: dict(rows_share=v[0] / tr, energy_share=v[1] / te) for k, v in ps.items() if k not in ('question',)}
        o['question_vs_state_energy'] = ps['question'][1] / (te + ps['question'][1])
    if 'b10' in d: o['b10'] = d['b10']
    allo[tag] = o
    print(f"### {tag}: n {n}, flips {o['flips']} ({100*o['flip_rate']:.2f}%), mean TV {o['tv']:.4f}; first-order predictor corr {o['pred_corr']:.3f} slope {o['pred_slope']:.3f}, "
          f"rms dm {o['rms_dm']:.4f} vs pred {o['rms_pred']:.4f} (state {o['rms_pred_state']:.4f}, question {o['rms_pred_question']:.4f}); flip AUC {o['flip_auc_pred']:.3f}; "
          f"first-order flips predicted {o['flip_pred_first_order']}")
    print('   layer share of first-order var:', ' '.join(f'{v:.3f}' for v in o['layer_share_first_order']))
    print('   B6:', json.dumps(o['b6']), f"corr(|dm|, sqrt cert) {o['cert_corr_absdm_sqrtc']:.3f}, corr(|dm|, |e|) {o['cert_corr_absdm_enorm']:.3f}")
    print('   B2 kappa:', {k: round(v, 2) for k, v in kap.items()}, ' first-order var by role', {k: f'{v:.3e}' for k, v in o['first_order_var_by_role'].items()})
    if o.get('positions'): print('   positions:', {k: (round(v['rows_share'], 3), round(v['energy_share'], 3)) for k, v in o['positions'].items()}, 'question share of row energy', round(o['question_vs_state_energy'], 3))
    print('   B8:', {k: (round(v['rows_share'], 3), round(v['per_row_rel'], 2)) for k, v in sorted(o['b8_classes'].items(), key=lambda x: -x[1]['per_row_rel'])})
json.dump(allo, open(outp, 'w'), indent=1)

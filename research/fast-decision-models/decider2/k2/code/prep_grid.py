"""Grid inputs -> ~/work/k2/grid_cfg.json: question sets (exact hobson token ids, option-end offsets, K), which are deployed (Q), the 16k merges
of the in-context questions and the measured median state compression of the 16k vocabulary on REAL-agree states.
python prep_grid.py [SUPER_JSON] [DEPLOYED_JSON]"""
import os, sys, json, statistics as st
sys.path.insert(0, os.path.expanduser('~/work/k2'))
for p in ('~/work/evalkit', '~/work/tokens', '~/work/j6', '~/work/j7'):
    sys.path.append(os.path.expanduser(p))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import evalkit as EK
from kitrun import load_P
from strands_decider.prompting import render_state

SUP = sys.argv[1] if len(sys.argv) > 1 else ''
DEP = sys.argv[2] if len(sys.argv) > 2 else ''
P = load_P(); tok = P.tok
Q4 = ['details_match', 'failure_cause', 'needed_procedure', 'rule_bound_values']
Q15 = ['code_abusive_customer_behavior', 'code_accessibility_or_special_needs', 'code_account_closure_request', 'code_account_ownership_dispute',
       'code_complex_billing_dispute', 'code_customer_demands_after_unavailable_offer_refusal', 'code_deceased_account_holder',
       'code_fraud_or_security_concern', 'code_kb_search_unsuccessful_customer_requests_transfer', 'code_legal_or_regulatory_matter',
       'code_specialized_department_required', 'code_technical_system_error', 'code_third_party_inquiry',
       'code_unconfirmed_external_communication', 'identity_verified']
spec = {}
for s in ('REAL-agree', 'LONG'):
    for it in EK.load_suite(s):
        for q, sp in it['questions'].items(): spec.setdefault(q, sp)
dep = json.load(open(os.path.expanduser(DEP))) if DEP and os.path.exists(os.path.expanduser(DEP)) else None
if dep is None:
    import qtab
    dep = qtab.deployed_specs(qtab.load_pool())
sup = None
if SUP and os.path.exists(os.path.expanduser(SUP)):
    from superbpe import Super
    sup = Super(os.path.expanduser(SUP), tok=tok)


def qrec(name, sp, state='S'):
    pr = P.prep(state, sp)
    r = dict(name=name, ids=pr['q'], opt=pr['opt'], K=len(pr['opt']), n_slots=pr['rq'].n_slots, kind=pr['rq'].kind,
             deployed=bool(name in dep and json.dumps(dep[name], sort_keys=True) == json.dumps(sp, sort_keys=True)))
    if sup is not None:
        mi, en = sup.merge_ids(list(pr['q']))
        e2r = {e: k for k, e in enumerate(en)}
        r['ids16k'] = mi; r['ends16k'] = en; r['opt16k'] = [e2r[o] if o in e2r else next(k for k, e in enumerate(en) if e >= o) for o in pr['opt']]
    return r


jb = []
for it in EK.load_suite('JB-all'):
    q = list(it['questions'])[0]; pr = P.prep(it['state'], it['questions'][q]); jb.append((len(pr['q']), it, q))
jb.sort(key=lambda x: x[0]); L, it, q = jb[len(jb) // 2]
cfg = dict(qsets=dict(JB1=[dict(qrec('jb:' + it['id'], it['questions'][q], it['state']), deployed=False)],
                      BK4=[qrec(n, spec[n]) for n in Q4], BK15=[qrec(n, spec[n]) for n in Q15]))
cfg['jb_item'] = it['id']
ratio = None
if sup is not None:
    rs = []
    for it in EK.load_suite('REAL-agree'):
        ids = tok(render_state(it['state']), add_special_tokens=True)['input_ids']
        mi, _ = sup.merge_ids(ids); rs.append(len(ids) / len(mi))
    ratio = st.median(rs); cfg['ratio_state_n'] = len(rs)
    qr = [len(q['ids']) / len(q['ids16k']) for qs in cfg['qsets'].values() for q in qs]
    cfg['ratio_q_sets'] = st.median(qr)
cfg['ratio_state'] = ratio or 1.97
cfg['ratio_state_src'] = 'measured, K1 sb16k, median over REAL-agree states' if ratio else 'J7 report (16k, REAL states 1.97x)'
cfg['share'] = 0.46; cfg['frame'] = 45
json.dump(cfg, open(os.path.expanduser('~/work/k2/grid_cfg.json'), 'w'))
print({k: [(q['name'], len(q['ids']), q['K'], q['deployed']) for q in v] for k, v in cfg['qsets'].items()}, cfg['ratio_state'], flush=True)

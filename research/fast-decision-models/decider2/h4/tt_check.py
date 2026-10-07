"""H4: correctness of this-that in the fused lean runtime (vs the HF/thisthat path) and of the schema-first prefix cache (vs the full pass)."""
import os, sys, json, random
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.expanduser('~/work/h4')); sys.path.insert(0, os.path.expanduser('~/work/evalkit'))
import numpy as np, torch
import evalkit as EK
from tt_lean import TTL, load_tt
from thisthat.prompt import build, option_label_ids
from thisthat.systemone_protocol import _typed

m, tok = load_tt()
lab = option_label_ids(tok)
ttl = TTL(m.model, lab)
dev = 'cuda'


def kit_items():
    its = {}
    for s, iid, q, st, spec in EK.all_question_items():
        its.setdefault(iid, (s, st, {}))[2][q] = spec
    return its


def tq(specs):
    return [_typed(k, s) for k, s in specs.items()]


@torch.inference_mode()
def lean_probs(ids, slots, nopt):
    ttl.set_fuse(len(ids))
    h, _ = ttl.fwd(torch.tensor([ids], device=dev))
    return ttl.head(h, torch.tensor(slots, device=dev), torch.tensor(nopt, device=dev)).cpu().numpy()


@torch.inference_mode()
def hf_probs(ids, slots, nopt):
    hid = m.model(input_ids=torch.tensor([ids], device=dev)).last_hidden_state[0]
    lg = (hid[torch.tensor(slots, device=dev)] @ m.lm_head.weight[torch.tensor(lab[:10], device=dev)].t()).float()
    lg = lg.masked_fill(torch.arange(10, device=dev)[None, :] >= torch.tensor(nopt, device=dev)[:, None], float('-inf'))
    return torch.softmax(lg, -1).cpu().numpy()


its = kit_items(); keys = sorted(its); random.seed(0); random.shuffle(keys)
res = {'lean_vs_hf': [], 'cache_vs_full': []}
# 1) fused runtime vs HF, state_first, one question per sequence (the eval layout)
for iid in keys[:40]:
    s, st, specs = its[iid]
    st = st if isinstance(st, str) else json.dumps(st, ensure_ascii=False)
    q, spec = next(iter(specs.items()))
    t = tq({q: spec})
    b = build(tok, st, [x.question for x in t], max_state_tokens=10 ** 7)
    pl = lean_probs(b['ids'], b['slots'], b['n_options']); ph = hf_probs(b['ids'], b['slots'], b['n_options'])
    res['lean_vs_hf'].append(dict(id=iid, L=len(b['ids']), maxdp=float(np.abs(pl - ph).max()), same=bool((pl.argmax(-1) == ph.argmax(-1)).all())))
r = res['lean_vs_hf']
print('lean vs HF: n', len(r), 'argmax same', sum(x['same'] for x in r), 'max|dp| median %.4f max %.4f' % (np.median([x['maxdp'] for x in r]), max(x['maxdp'] for x in r)), 'L range', min(x['L'] for x in r), max(x['L'] for x in r), flush=True)

# 2) schema-first prefix cache vs the full schema-first pass, multi-question items
CTX = tok.encode('\n\nContext:\n', add_special_tokens=False)
multi = [k for k in keys if len(its[k][2]) >= 3][:12]
for iid in multi:
    s, st, specs = its[iid]
    t = tq(specs)
    b = build(tok, st, [x.question for x in t], layout='schema_first', max_state_tokens=10 ** 7)
    P = b['prefix_len'] + len(CTX)
    assert b['ids'][b['prefix_len']:P] == CTX
    full = lean_probs(b['ids'], b['slots'], b['n_options'])
    with torch.inference_mode():
        pre = torch.tensor([b['ids'][:P]], device=dev); suf = torch.tensor([b['ids'][P:]], device=dev)
        ttl.set_fuse(P); _, cache = ttl.fwd(pre, want_cache=True)
        ttl.set_fuse(suf.shape[1]); ttl.set_prefix_mask(P, suf.shape[1])
        h, _ = ttl.fwd(suf, pos0=P, cache=cache)
        pc = ttl.head(h, torch.tensor([x - P for x in b['slots']], device=dev), torch.tensor(b['n_options'], device=dev)).cpu().numpy()
    res['cache_vs_full'].append(dict(id=iid, P=P, S=suf.shape[1], nq=len(t), maxdp=float(np.abs(pc - full).max()), same=bool((pc.argmax(-1) == full.argmax(-1)).all())))
r = res['cache_vs_full']
print('cache vs full: n', len(r), 'argmax same', sum(x['same'] for x in r), 'max|dp| median %.4f max %.4f' % (np.median([x['maxdp'] for x in r]), max(x['maxdp'] for x in r)), 'P range', min(x['P'] for x in r), max(x['P'] for x in r), flush=True)
json.dump(res, open(os.path.expanduser('~/work/h4/check.json'), 'w'), indent=1)

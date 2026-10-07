import os, sys, json
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.expanduser('~/work/h4'))
exec(open(os.path.expanduser('~/work/h4/tt_check.py')).read().split("its = kit_items()")[0])
its = kit_items(); keys = sorted(its); random.seed(0); random.shuffle(keys)
CTX = tok.encode('\n\nContext:\n', add_special_tokens=False)
multi = [k for k in keys if len(its[k][2]) >= 3][:6]
for iid in multi:
    s, st, specs = its[iid]; t = tq(specs)
    b = build(tok, st, [x.question for x in t], layout='schema_first', max_state_tokens=10 ** 7)
    ids, slots, nopt = b['ids'], b['slots'], b['n_options']
    ph = hf_probs(ids, slots, nopt); pl = lean_probs(ids, slots, nopt)
    out = dict(L=len(ids), lean_hf=float(abs(pl - ph).max()))
    for nm, P in (('P', b['prefix_len'] + len(CTX)), ('P64', (b['prefix_len'] // 64) * 64)):
        with torch.inference_mode():
            pre = torch.tensor([ids[:P]], device=dev); suf = torch.tensor([ids[P:]], device=dev)
            ttl.set_fuse(P); _, cache = ttl.fwd(pre, want_cache=True)
            ttl.set_fuse(suf.shape[1]); ttl.set_prefix_mask(P, suf.shape[1])
            h, _ = ttl.fwd(suf, pos0=P, cache=cache)
            pc = ttl.head(h, torch.tensor([x - P for x in slots], device=dev), torch.tensor(nopt, device=dev)).cpu().numpy()
        out[nm + '_vs_lean'] = float(abs(pc - pl).max()); out[nm + '_vs_hf'] = float(abs(pc - ph).max())
    print(iid, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}), flush=True)

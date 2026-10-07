"""G1 projection study.
  python proj_study.py cov                 -> covs.pt (input second moments over 48 train-pool states, 1 question each, <= 4096 tok)
  python proj_study.py layer OUT.jsonl     -> per-matrix relative output error, structures x fractions x methods, layers 2/11/20
  python proj_study.py model OUT.jsonl SPEC [SPEC..]  -> no-training model-level KL/agreement vs hobson on 96 held-out real rows
        SPEC = which:kind:b:frac:method   e.g. mlp:btt:4:0.5:white
"""
import os, sys, json, time, math, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1lib as G, slib as S

mode = sys.argv[1]
t0 = time.time()
m = G.load_hobson()
print("loaded", round(time.time() - t0), flush=True)

if mode == "cov":
    items = G.pool_items(48, seed=11)
    rows = G.fit_rows(items, maxlen=4096, per_item=1, seed=1)
    covs, n = G.collect_cov(m, rows, "all")
    torch.save({k: v.cpu() for k, v in covs.items()}, "covs.pt")
    print("cov tokens", n, "groups", len(covs), round(time.time() - t0), "s", flush=True)
    sys.exit()

covs = torch.load("covs.pt")

if mode == "layer":
    fo = open(sys.argv[2], "a")
    for (i, an, par, attr, lin) in G.targets(m, "all"):
        if i not in (2, 11, 20): continue
        if an in ("in_proj_a", "in_proj_b", "k_proj", "v_proj"): continue
        W = lin.weight.data.float()
        C = covs[(i, G.GROUP[an])].cuda()
        n_out, n_in = W.shape
        for frac in (0.5, 0.25, 0.125):
            confs = [("btt", b) for b in (1, 2, 4, 8, 16)] + [("lrbd", 4), ("lrbd", 8), ("kron", 0)]
            for kind, b in confs:
                if kind == "lrbd" and 1.0 / b >= frac: continue
                for meth in ("frob", "white", "refine"):
                    if kind == "kron" and meth == "white": continue
                    if meth == "refine" and kind == "btt" and b not in (1, 4): continue
                    if meth == "frob" and kind == "btt" and b in (2, 16): continue
                    t1 = time.time()
                    try:
                        What, fac, mac = G.project_one(W, C, kind, frac, b, meth, refine_steps=100)
                    except Exception as e:
                        print("fail", i, an, kind, b, frac, meth, e, flush=True); continue
                    rec = dict(layer=i, mat=an, n_in=n_in, n_out=n_out, kind=kind, b=b, frac=frac, real_frac=round(mac / (n_in * n_out), 4),
                               method=meth, out_err=round(S.out_err(W, What, C), 4), frob_err=round(S.frob_err(W, What), 4), s=round(time.time() - t1, 2))
                    fo.write(json.dumps(rec) + "\n"); fo.flush()
                    print(json.dumps(rec), flush=True)
        del C
    print("done", round(time.time() - t0), flush=True)

if mode == "model":
    fo = open(sys.argv[2], "a")
    items = G.pool_items(96, seed=23)
    rows = G.fit_rows(items, maxlen=4096, per_item=1, seed=2)
    G.run_rows(m, rows, key="p0")
    for spec in sys.argv[3:]:
        which, kind, b, frac, meth = spec.split(":"); b = int(b); frac = float(frac)
        sw = G.DenseSwap(); errs = []; mac_s = 0; mac_d = 0
        t1 = time.time()
        for (i, an, par, attr, lin) in G.targets(m, which):
            W = lin.weight.data.float(); C = covs[(i, G.GROUP[an])].cuda()
            What, fac, mac = G.project_one(W, C, kind, frac, b, meth, refine_steps=150)
            errs.append(S.out_err(W, What, C)); mac_s += mac; mac_d += W.numel()
            sw.set(lin, What)
        tp = time.time() - t1
        G.run_rows(m, rows, key="p")
        kl, ag = G.kl_and_agree(rows, "p", "p0")
        tot_d = sum(l.in_features * l.out_features for ly in G.layers_of(m) for l in ly.modules() if isinstance(l, torch.nn.Linear))
        rec = dict(spec=spec, kl=round(kl, 4), agree=round(ag, 4), mean_out_err=round(sum(errs) / len(errs), 4),
                   struct_mac_frac=round(mac_s / mac_d, 4), model_mac_frac=round((tot_d - mac_d + mac_s) / tot_d, 4), proj_s=round(tp), n=len(rows))
        fo.write(json.dumps(rec) + "\n"); fo.flush(); print(json.dumps(rec), flush=True)
        sw.restore()
    print("done", round(time.time() - t0), flush=True)

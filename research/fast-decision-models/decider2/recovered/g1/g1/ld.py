"""G1 sequential block-wise local distillation (an ALS-like retrofit; no end-to-end backprop).
Layer by layer, on calibration states, with the STUDENT's own inputs (so upstream error is seen):
  'all': in-projections (qkv, z | q) projected with the student's input covariance (whitened BTT/low-rank) + Adam refinement on the exact
         output error; then the out-projection the same way on its student-side input covariance;
  MLP:   init by whitened projection with the student's covariance, then Adam on gate/up/down factors jointly to minimise the error of the
         SwiGLU block output against the dense MLP on the same inputs (loss = 0.5 global relative + 0.5 per-token relative squared error).
usage: python ld.py OUT.pt WHICH KIND B FRAC [steps]     (KIND btt; B=1 is low rank)"""
import os, sys, json, time, math, random, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1lib as G, slib as S
torch.backends.cuda.matmul.allow_tf32 = True

out, which, kind, b, frac = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
steps = int(sys.argv[6]) if len(sys.argv) > 6 else 250
t0 = time.time()
m = G.load_hobson()
for p in m.parameters(): p.requires_grad_(False)
tm = getattr(m.torso, "model", m.torso)
items = G.pool_items(48, seed=11)
rows = G.fit_rows(items, maxlen=4096, per_item=1, seed=1)
dev = "cuda"
with torch.no_grad():
    H, PE = [], []
    for r in rows:
        ids = r["ids"][None].to(dev)
        e = tm.embed_tokens(ids)
        pos = torch.arange(ids.shape[1], device=dev).view(1, 1, -1).expand(3, 1, -1)
        PE.append(tm.rotary_emb(e, pos)); H.append(e)
ntok = sum(h.shape[1] for h in H)
print("calib", len(H), "seqs", ntok, "tokens", round(time.time() - t0), "s", flush=True)


def mixer(ly, x, pe):
    if hasattr(ly, "linear_attn"):
        return ly.linear_attn(hidden_states=x, cache_params=None, attention_mask=None)
    return ly.self_attn(hidden_states=x, attention_mask=None, position_ids=None, past_key_values=None, position_embeddings=pe)[0]


def cov_of(xs):
    C = None; n = 0
    for x in xs:
        x2 = x.reshape(-1, x.shape[-1]).float()
        C = x2.t() @ x2 if C is None else C.addmm_(x2.t(), x2); n += x2.shape[0]
    return C / n


def struct_proj(lin, C, refine_steps=150):
    W = lin.weight.data.float()
    What, fac, mac = G.project_one(W, C, kind, frac, b, "refine" if refine_steps else "white", refine_steps=refine_steps)
    return S.BTTLinear(fac["R"], fac["L"]).to(dev), S.out_err(W, What, C), mac, W.numel()


log = []
mac_s = mac_d = 0
for i, ly in enumerate(tm.layers):
    t1 = time.time(); rec = dict(layer=i)
    lin_attn = hasattr(ly, "linear_attn")
    with torch.no_grad():
        R_in = [ly.input_layernorm(h) for h in H]
    if which == "all":
        par = ly.linear_attn if lin_attn else ly.self_attn
        names = ["in_proj_qkv", "in_proj_z"] if lin_attn else ["q_proj"]
        C = cov_of(R_in)
        for nm in names:
            mod, err, mac, dn = struct_proj(getattr(par, nm), C); setattr(par, nm, mod); rec[nm] = round(err, 4); mac_s += mac; mac_d += dn
        del C
        # out-projection on its student-side input
        oname = "out_proj" if lin_attn else "o_proj"
        olin = getattr(par, oname); acc = {"C": None, "n": 0}
        def hk(mod, inp, outp):
            x2 = inp[0].reshape(-1, inp[0].shape[-1]).float()
            acc["C"] = x2.t() @ x2 if acc["C"] is None else acc["C"].addmm_(x2.t(), x2); acc["n"] += x2.shape[0]
        hh = olin.register_forward_hook(hk)
        with torch.no_grad():
            for r_, pe in zip(R_in, PE): mixer(ly, r_, pe)
        hh.remove()
        mod, err, mac, dn = struct_proj(olin, acc["C"] / acc["n"]); setattr(par, oname, mod); rec[oname] = round(err, 4); mac_s += mac; mac_d += dn
    with torch.no_grad():
        Xmid = [h + mixer(ly, r_, pe) for h, r_, pe in zip(H, R_in, PE)]
        U = torch.cat([ly.post_attention_layernorm(x).reshape(-1, 2048) for x in Xmid])
        mlp = ly.mlp
        T = torch.cat([mlp(U[s:s + 8192]) for s in range(0, U.shape[0], 8192)]).float()
    del R_in
    def ccov(X):
        C = torch.zeros(X.shape[1], X.shape[1], device=dev)
        for s_ in range(0, X.shape[0], 16384):
            xf = X[s_:s_ + 16384].float(); C.addmm_(xf.t(), xf)
        return C / X.shape[0]
    Cu = ccov(U)
    mods = {}
    for nm in ("gate_proj", "up_proj"):
        mods[nm], e_, mac, dn = struct_proj(getattr(mlp, nm), Cu, refine_steps=0); rec[nm] = round(e_, 4); mac_s += mac; mac_d += dn
    with torch.no_grad():
        Mid = torch.cat([(F.silu(mlp.gate_proj(U[s:s + 8192])) * mlp.up_proj(U[s:s + 8192])) for s in range(0, U.shape[0], 8192)])
        Cd = ccov(Mid)
    mods["down_proj"], e_, mac, dn = struct_proj(mlp.down_proj, Cd, refine_steps=0); rec["down_proj"] = round(e_, 4); mac_s += mac; mac_d += dn
    del Mid, Cd, Cu
    # joint local distillation of the SwiGLU block
    ps = []
    for nm in ("gate_proj", "up_proj", "down_proj"):
        for p in mods[nm].parameters():
            p.data = p.data.float(); p.requires_grad_(True); ps.append(p)
    opt = torch.optim.Adam([dict(params=[p], lr=1e-3 * float(p.detach().abs().mean())) for p in ps])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    N = U.shape[0]; perm = torch.randperm(N, device=dev); nval = min(8192, N // 10)
    val, trn = perm[:nval], perm[nval:]
    def mlp_s(u):
        u = u.float()
        return mods["down_proj"](F.silu(mods["gate_proj"](u)) * mods["up_proj"](u))
    def lossf(u, t):
        e = mlp_s(u) - t
        g = (e * e).sum() / (t * t).sum()
        tn = (t * t).sum(-1)
        pt = ((e * e).sum(-1) / (tn + 1e-3 * tn.mean())).mean()
        return 0.5 * g + 0.5 * pt, g, pt
    with torch.no_grad():
        _, g0, p0 = lossf(U[val], T[val])
    for st in range(steps):
        idx = trn[torch.randint(0, trn.numel(), (4096,), device=dev)]
        loss, _, _ = lossf(U[idx], T[idx])
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    with torch.no_grad():
        _, g1, p1 = lossf(U[val], T[val])
    rec.update(mlp_rel0=round(float(g0) ** 0.5, 4), mlp_rel=round(float(g1) ** 0.5, 4), mlp_tok0=round(float(p0) ** 0.5, 4), mlp_tok=round(float(p1) ** 0.5, 4))
    for nm in ("gate_proj", "up_proj", "down_proj"):
        for p in mods[nm].parameters():
            p.requires_grad_(False); p.data = p.data.to(torch.bfloat16)
        setattr(mlp, nm, mods[nm])
    with torch.no_grad():
        H = [x + mlp(ly.post_attention_layernorm(x)) for x in Xmid]
    del Xmid, U, T
    torch.cuda.empty_cache()
    rec["s"] = round(time.time() - t1, 1)
    print(json.dumps(rec), flush=True); log.append(rec)

sd = {n: p.detach().cpu() for n, p in m.named_parameters() if n.endswith(".R") or n.endswith(".L")}
info = dict(which=which, kind=kind, b=b, frac=frac, struct_mac_frac=mac_s / mac_d, init="localdistill", steps=steps)
torch.save(dict(state=sd, args=dict(which=which, kind=kind, b=b, frac=frac), info=info, log=log), out)
print("saved", out, json.dumps(info), round(time.time() - t0), "s", flush=True)

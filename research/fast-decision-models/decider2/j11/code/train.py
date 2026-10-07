"""J11 FLOP-budgeted from-scratch training of one arm at one scale on the shared decision stream.
  python train.py --arm dec --d 512 --L 12 --budget_pf 18 --out runs/S_dec
All arms walk the SAME step sequence (same examples, same order); each arm stops when its own analytic training FLOPs
(3 x forward, models.fwd_flops) reach the budget, so cheaper arms see more steps. LR schedule is set per arm from its step count.
"""
import os, sys, json, time, math, argparse, random, glob
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as Mo

DATA = os.path.expanduser("~/work/j11/data/prep")
SLOT_ARMS = ("slot", "vslot", "belief")


# ------------------------------------------------------------------ data
def load_rows(name):
    p = f"{DATA}/{name}.pt"
    return torch.load(p, weights_only=False) if os.path.exists(p) else []


def strip(rows, arm):
    """drop the fields an arm never reads (RAM)."""
    drop = {"dec": ("pl", "qp_pl", "op_pl", "qp", "op"), "decv": ("qp_pl", "op_pl", "qp", "op"),
            "slot": ("pl", "qp_pl", "op_pl"), "belief": ("pl", "qp_pl", "op_pl"), "vslot": ()}[arm]
    for r in rows:
        for k in drop:
            r.pop(k, None)
    return rows


def build_vocab(sources):
    seen = np.zeros(300000, bool)
    for rows in sources:
        for r in rows:
            seen[r["ids"]] = True
            if r.get("qp") is not None:
                seen[r["qp"]] = True
                for o in r["op"]: seen[o] = True
    vmap = np.ones(300000, np.int64)  # 1 = UNK, 0 = PAD
    idx = np.nonzero(seen)[0]; vmap[idx] = np.arange(2, 2 + len(idx))
    return vmap, 2 + len(idx)


class Stream:
    """deterministic stream of steps, each a list of (src_idx, row_idx) with ~step_tok tokens; identical prefix for every arm."""
    def __init__(self, sources, shares, step_tok, seed=0):
        self.S = sources; self.rng = np.random.default_rng(seed); self.step_tok = step_tok
        mlen = [np.mean([len(r["ids"]) for r in rows]) for rows in sources]
        p = np.array([s / m for s, m in zip(shares, mlen)]); self.p = p / p.sum()
        self.perms = [list(self.rng.permutation(len(rows))) for rows in sources]; self.ptr = [0] * len(sources)
        self.choices = iter(())
    def _src(self):
        try: return next(self.choices)
        except StopIteration:
            self.choices = iter(self.rng.choice(len(self.S), size=4096, p=self.p).tolist()); return next(self.choices)
    def next_step(self):
        cur = []; tok = 0
        while tok < self.step_tok:
            si = self._src()
            if self.ptr[si] >= len(self.perms[si]): self.perms[si] = list(self.rng.permutation(len(self.S[si]))); self.ptr[si] = 0
            ri = self.perms[si][self.ptr[si]]; self.ptr[si] += 1
            cur.append((si, ri)); tok += len(self.S[si][ri]["ids"])
        return cur


def row_flops(arm, a, r):
    """forward FLOPs of the decision pass + the dense auxiliary (tied-head CE at 10% of positions; encoders also pay a masked state pass)."""
    K = r["n"]; Ts = r["nstate"]; Tq = len(r["ids"]) - Ts; V = getattr(a, "V", 85000); aux = getattr(a, "aux", 1.0) > 0
    if arm in SLOT_ARMS:
        if r.get("qp") is None: return 0.0
        f = Mo.fwd_flops(arm, a.d, a.L, Ts, Tq, K, R=a.R, To=[len(o) for o in r["op"]], Tqp=len(r["qp"]))
        if aux:
            Ls = max(2, a.L // 3); h = Mo.ffn_hidden(a.d)
            f += Ls * (2 * (4 * a.d * a.d + 3 * a.d * h) * Ts + 4 * Ts * Ts * a.d) + 0.1 * Ts * 2 * V * a.d
        return f
    f = Mo.fwd_flops(arm, a.d, a.L, Ts, Tq, K)
    return f + (0.1 * (Ts + Tq) * 2 * V * a.d if aux else 0.0)


def pad2(arrs, fill, dtype, T=None):
    T = T or max(len(x) for x in arrs); out = np.full((len(arrs), T), fill, dtype)
    for i, x in enumerate(arrs): out[i, :len(x)] = x
    return out


def pl_tensors(pls, T=None):
    """list of payload tuples -> dict of padded tensors (cpu)."""
    typ = pad2([p[0] for p in pls], 0, np.int64, T); lh = pad2([p[1] for p in pls], 0, np.int64, T)
    kh = pad2([p[2] for p in pls], 0, np.int64, T); rh = pad2([p[3] for p in pls], 0, np.int64, T)
    val = pad2([p[4] for p in pls], np.nan, np.float32, T); st = pad2([p[5] for p in pls], 0, np.float32, T)
    return dict(typ=torch.from_numpy(typ), lh=torch.from_numpy(lh), kh=torch.from_numpy(kh), rh=torch.from_numpy(rh),
                val=torch.from_numpy(val), st=torch.from_numpy(st))


def mem_payload(parts):
    """parts: list of payload dicts [B,T_i] (state, stem, options-flattened-per-row) -> pointer payload P on the memory axis."""
    cat = {k: torch.cat([p[k] for p in parts], 1) for k in parts[0]}
    typ = cat["typ"]; lit = (typ >= 1) & (typ <= 4)
    v = cat["val"]; mv = torch.isfinite(v)
    v0 = torch.where(mv, v, torch.zeros_like(v))
    return dict(lh=torch.where(lit, cat["lh"], torch.zeros_like(cat["lh"])), rh=cat["rh"],
                hb=Mo.hash_bits(torch.where(lit, cat["lh"], torch.zeros_like(cat["lh"]))), v0=v0, mv=mv.float(),
                z=torch.sign(v0) * torch.log1p(v0.abs()), t1=F.one_hot(typ.clamp(0, 5), 6).float(), ls=cat["st"] * lit.float())


def collate(arm, rows, vmap, dev):
    B = len(rows); b = {}
    n = torch.tensor([r["n"] for r in rows]); K = int(n.max()); b["n"] = n
    if arm not in SLOT_ARMS:
        ids = [vmap[r["ids"]] for r in rows]
        b["ids"] = torch.from_numpy(pad2(ids, 0, np.int64))
        b["ans"] = torch.tensor([len(r["ids"]) - 1 for r in rows])
        b["opt"] = torch.tensor([r["opt"] + [-1] * (K - r["n"]) for r in rows])
        if arm == "decv": b["pl"] = pl_tensors([r["pl"] for r in rows])
    else:
        s = [vmap[r["ids"][:r["nstate"]]] for r in rows]; q = [vmap[r["qp"]] for r in rows]
        b["s_ids"] = torch.from_numpy(pad2(s, 0, np.int64)); b["s_len"] = torch.tensor([len(x) for x in s])
        b["q_ids"] = torch.from_numpy(pad2(q, 0, np.int64)); b["q_len"] = torch.tensor([len(x) for x in q])
        To = max(len(o) for r in rows for o in r["op"])
        o = np.zeros((B, K, To), np.int64); ol = np.zeros((B, K), np.int64)
        for i, r in enumerate(rows):
            for k, x in enumerate(r["op"]): o[i, k, :len(x)] = vmap[x]; ol[i, k] = len(x)
        b["o_ids"] = torch.from_numpy(o); b["o_len"] = torch.from_numpy(ol)
        if arm == "vslot":
            Ts = b["s_ids"].shape[1]; Tq = b["q_ids"].shape[1]
            sp = pl_tensors([tuple(x[:r["nstate"]] for x in r["pl"]) for r in rows], Ts)
            qp = pl_tensors([r["qp_pl"] for r in rows], Tq)
            empty = tuple(np.zeros(0, t) for t in (np.int8, np.int64, np.int64, np.int64, np.float32, np.int8))
            opl = [r["op_pl"][k] if k < r["n"] else empty for r in rows for k in range(K)]
            op = pl_tensors(opl, To)  # [B*K, To]
            b["s_pl"], b["q_pl"] = sp, qp; b["o_pl"] = op
            opm = {k: v.view(B, K * To) for k, v in op.items()}
            b["mem_P"] = mem_payload([sp, qp, opm])
            Tm = Ts + Tq + K * To
            b["mem_P"]["is_state"] = torch.cat([torch.ones(B, Ts, dtype=torch.bool), torch.zeros(B, Tm - Ts, dtype=torch.bool)], 1)
            # each slot's own literals: answer + scratch slots = question-stem literals; option slot k = option k's literals
            if os.environ.get("J11_NOBIND") == "1": return_now = True
            else: return_now = False
            M = 4; SH = np.zeros((B, 1 + K + M, 8), np.int64)
            def lits(pl):
                typ, lh = pl[0], pl[1]; u = []
                for t_, h_ in zip(typ.tolist(), lh.tolist()):
                    if 1 <= t_ <= 4 and h_ != 0 and h_ not in u: u.append(h_)
                return u[:8]
            for i, r in enumerate(rows):
                ql = lits(r["qp_pl"]); SH[i, 0, :len(ql)] = ql; SH[i, 1 + K:, :len(ql)] = ql
                for k in range(r["n"]):
                    ol = lits(r["op_pl"][k]); SH[i, 1 + k, :len(ol)] = ol
            if not return_now: b["slot_h"] = torch.from_numpy(SH)
    # targets
    tg = torch.zeros(B, K); has_g = torch.zeros(B); tt = torch.zeros(B, K); has_t = torch.zeros(B)
    for i, r in enumerate(rows):
        if r["w"] > 0 and r["label"] >= 0:
            if r.get("dist") is not None: tg[i, :r["n"]] = torch.tensor(r["dist"])
            else: tg[i, r["label"]] = 1.0
            has_g[i] = r["w"]
        if r.get("t") is not None: tt[i, :r["n"]] = torch.tensor(r["t"]); has_t[i] = 1.0
    b.update(tg=tg, has_g=has_g, tt=tt, has_t=has_t)
    return to_dev(b, dev)


def to_dev(x, dev):
    if isinstance(x, dict): return {k: to_dev(v, dev) for k, v in x.items()}
    return x.to(dev, non_blocking=True) if torch.is_tensor(x) else x


def masked_logp(logits, n):
    K = logits.shape[1]; valid = torch.arange(K, device=logits.device).unsqueeze(0) < n.unsqueeze(1)
    return F.log_softmax(logits.float().masked_fill(~valid, -1e4), -1).masked_fill(~valid, 0.0), valid


def loss_fn(outs, b, belief_w=(0.2, 0.3, 0.5)):
    ws = belief_w[-len(outs):] if len(outs) > 1 else (1.0,)
    tot = 0.0
    for wr, lg in zip(ws, outs):
        lp, valid = masked_logp(lg, b["n"])
        ce = -(b["tg"] * lp).sum(-1)
        tt = b["tt"]; kl = (tt * (torch.log(tt.clamp_min(1e-9)) - lp)).masked_fill(tt <= 0, 0).sum(-1)
        tot = tot + wr * (b["has_g"] * ce + b["has_t"] * kl)
    return tot  # per-example [B]


def micro_batches(rows, arm, tok_budget):
    """sort by length, greedy pack under a padded-token budget."""
    def L(r):
        if arm in SLOT_ARMS: return r["nstate"] + len(r["qp"]) + sum(len(o) for o in r["op"]) + 8
        return len(r["ids"])
    rows = sorted(rows, key=L); out = []; cur = []; mx = 0
    for r in rows:
        l = L(r)
        if cur and max(mx, l) * (len(cur) + 1) > tok_budget: out.append(cur); cur = []; mx = 0
        cur.append(r); mx = max(mx, l)
    if cur: out.append(cur)
    return out


@torch.no_grad()
def predict(model, arm, rows, vmap, dev, tok_budget=16000):
    """-> list of prob lists (each row's rendered option order); final round for belief."""
    model.eval(); res = [None] * len(rows)
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
    i = 0
    while i < len(order):
        j = i; mx = 0
        while j < len(order):
            l = len(rows[order[j]]["ids"]) + (sum(len(o) for o in rows[order[j]]["op"]) if arm in SLOT_ARMS and rows[order[j]].get("qp") is not None else 0)
            if j > i and max(mx, l) * (j - i + 1) > tok_budget: break
            mx = max(mx, l); j += 1
        idx = order[i:j]; i = j
        ok = [k for k in idx if arm not in SLOT_ARMS or rows[k].get("qp") is not None]
        if not ok: continue
        b = collate(arm, [rows[k] for k in ok], vmap, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(b)[-1]
        lp, valid = masked_logp(lg, b["n"])
        p = lp.exp().cpu()
        for t, k in enumerate(ok): res[k] = p[t, :rows[k]["n"]].tolist()
    model.train(); return res


def synth_acc(preds, rows):
    by = {}
    for p, r in zip(preds, rows):
        if p is None: continue
        f = r["meta"]["family"]; by.setdefault(f, []).append(int(np.argmax(p) == r["label"]))
    return {f: float(np.mean(v)) for f, v in sorted(by.items())}


def init_weights(model, n_res):
    for name, p in model.named_parameters():
        if p.dim() == 2 and "emb" not in name and "ptr.out" not in name and "bel" not in name and "typ" not in name and "scratch" not in name:
            std = 0.02 / math.sqrt(2 * n_res) if name.endswith(("o.weight", "w2.weight")) else 0.02
            torch.nn.init.normal_(p, std=std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--d", type=int, default=512); ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--R", type=int, default=3); ap.add_argument("--budget_pf", type=float, required=True, help="training PFLOPs")
    ap.add_argument("--step_tok", type=int, default=32768); ap.add_argument("--micro_tok", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--out", required=True); ap.add_argument("--eval_every", type=int, default=400)
    ap.add_argument("--shares", default="0.45,0.25,0.30"); ap.add_argument("--max_steps", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--aux", type=float, default=1.0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = "cuda"
    torch.manual_seed(a.seed); random.seed(a.seed)
    t0 = time.time()
    synth = [r for f in sorted(glob.glob(f"{DATA}/synth_train_*.pt")) for r in torch.load(f, weights_only=False)]; corpus = load_rows("corpus"); real = load_rows("real") + load_rows("real2")
    stest = [r for r in load_rows("synth_test") if r["meta"].get("rot") is None]
    sources = [synth, corpus, real]; shares = [float(x) for x in a.shares.split(",")]
    vmap, V = build_vocab(sources); V = V + 1; a.V = V; MASK = V - 1  # last id = [MASK] for the encoders' auxiliary
    for rows in (synth, corpus, real, stest): strip(rows, a.arm)
    print(f"loaded {[len(s) for s in sources]} test {len(stest)} V {V} in {time.time() - t0:.0f}s", flush=True)
    stream = Stream(sources, shares, a.step_tok, seed=0); steps = []; fl = []; tot = 0.0
    while tot < a.budget_pf * 1e15 and len(steps) < a.max_steps:
        st = stream.next_step(); f = 3 * sum(row_flops(a.arm, a, sources[s][i]) for s, i in st)
        steps.append(st); fl.append(f); tot += f
    fl = np.array(fl); cum = np.cumsum(fl); n_steps = len(steps)
    print(f"arm {a.arm}: {n_steps} steps for {a.budget_pf} PF ({cum[n_steps - 1] / 1e15:.2f}); mean {fl[:n_steps].mean() / 1e12:.2f} TF/step", flush=True)
    model = Mo.build(a.arm, V, a.d, a.L, R=a.R).to(dev)
    n_res = (a.L if a.arm in ("dec", "decv") else len(model.enc) + len(model.slots) * (a.R if a.arm == "belief" else 1)) * 2
    init_weights(model, n_res)
    n_all = sum(p.numel() for p in model.parameters()); n_emb = model.emb.weight.numel()
    print(f"params total {n_all / 1e6:.1f}M non-emb {(n_all - n_emb) / 1e6:.1f}M", flush=True)
    dec_p = [p for nm, p in model.named_parameters() if p.dim() >= 2 and "emb" not in nm]
    nodec = [p for nm, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in nm)]
    opt = torch.optim.AdamW([dict(params=dec_p, weight_decay=0.1), dict(params=nodec, weight_decay=0.0)], lr=a.lr, betas=(0.9, 0.95), fused=True)
    warm = max(50, int(0.03 * n_steps))
    sched = lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, n_steps - warm)))
    start = 0; ck = f"{a.out}/ckpt.pt"
    if os.path.exists(ck):
        st = torch.load(ck, weights_only=False); model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); start = st["step"]
        print("resumed at", start, flush=True)
    meta = dict(vars(a), V=V, n_steps=n_steps, params=n_all, params_nonemb=n_all - n_emb, Ls=getattr(model, "enc", None) and len(model.enc),
                Ld=getattr(model, "slots", None) and len(model.slots))
    json.dump(meta, open(f"{a.out}/meta.json", "w"), indent=1)
    np.save(f"{a.out}/vmap.npy", vmap)
    log = open(f"{a.out}/log.jsonl", "a"); tl = time.time(); last_ck = time.time(); lsum = 0.0; lcnt = 0; ntok = 0; asum = 0.0; dsum = 0.0
    curve = []
    for step in range(start, n_steps):
        for g in opt.param_groups: g["lr"] = a.lr * sched(step)
        rows = [sources[s][i] for s, i in steps[step]]
        if a.arm in SLOT_ARMS: rows = [r for r in rows if r.get("qp") is not None]
        mbs = micro_batches(rows, a.arm, a.micro_tok); nex = len(rows)
        for mb in mbs:
            b = collate(a.arm, mb, vmap, dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outs = model(b)
                l = loss_fn(outs, b).sum() / nex; dsum += float(l)
                if a.aux > 0:
                    al, na = Mo.aux_dec(model, b) if a.arm in ("dec", "decv") else Mo.aux_enc(model, b, MASK)
                    l = l + a.aux * al * len(mb) / nex; asum += float(al) * len(mb) / nex
            l.backward(); lsum += float(l); ntok += sum(len(r["ids"]) for r in mb)
        lcnt += 1
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if (step + 1) % 25 == 0:
            el = time.time() - tl
            rec = dict(step=step + 1, loss=lsum / lcnt, dec_loss=dsum / lcnt, aux=asum / lcnt, gn=float(gn), lr=a.lr * sched(step), tok_s=ntok / el, pf=float(cum[step] / 1e15), t=time.time() - t0)
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + "\n"); log.flush(); lsum = 0; lcnt = 0; ntok = 0; asum = 0.0; dsum = 0.0; tl = time.time()
        if (step + 1) % a.eval_every == 0 or step + 1 == n_steps:
            sub = stest[::3] if step + 1 < n_steps else stest
            acc = synth_acc(predict(model, a.arm, sub, vmap, dev), sub)
            rec = dict(step=step + 1, pf=float(cum[step] / 1e15), synth=acc, mean_iid=float(np.mean([v for k, v in acc.items() if not k.startswith("H_")])),
                       mean_held=float(np.mean([v for k, v in acc.items() if k.startswith("H_")])))
            print("EVAL", json.dumps(rec), flush=True); log.write(json.dumps(dict(eval=rec)) + "\n"); log.flush()
        if time.time() - last_ck > 600 and step + 1 < n_steps:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step + 1), ck); last_ck = time.time()
    torch.save(dict(model=model.state_dict(), meta=meta), f"{a.out}/final.pt")
    if os.path.exists(ck): os.remove(ck)
    print("DONE", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()

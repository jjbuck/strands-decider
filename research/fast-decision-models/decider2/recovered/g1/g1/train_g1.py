"""G1 heal-by-self-distillation trainer.
Student = hobson-v19 (merged) with target projections replaced by structured layers initialised by projection (or, for the dense-FT
control, unchanged hobson + LoRA on the same projections). Only the structured factors / LoRA train (fp32 master copies, AdamW,
3% warmup + cosine). Loss per row: KL(hobson || student) on hobson's calibrated (per-kind temperature) distribution, every row;
+ GW * CE(gold) on train_v5 rows. Rows per step BS, micro-batches under a padded-token budget; per-layer activation checkpointing.
usage: python train_g1.py --arm NAME --which mlp|all --kind btt|lrbd|lora --b 4 --frac 0.5 --method white|refine --steps N --out DIR
"""
import os, sys, json, math, time, random, argparse
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1lib as G, slib as S

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True); ap.add_argument("--which", default="mlp"); ap.add_argument("--kind", default="btt")
ap.add_argument("--b", type=int, default=4); ap.add_argument("--frac", type=float, default=0.5); ap.add_argument("--method", default="white")
ap.add_argument("--steps", type=int, default=600); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--tokb", type=int, default=12288)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--gw", type=float, default=0.3); ap.add_argument("--rank", type=int, default=64)
ap.add_argument("--out", required=True); ap.add_argument("--save_at", default="0.33,0.67,1.0"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--refine_steps", type=int, default=150); ap.add_argument("--init", default=""); ap.add_argument("--rel_lr", type=float, default=0.0)
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
m = G.load_hobson()
for p in m.parameters(): p.requires_grad_(False)


class LoRALinear(nn.Module):
    def __init__(self, lin, r):
        super().__init__()
        self.lin = lin; self.in_features, self.out_features = lin.in_features, lin.out_features
        self.A = nn.Parameter(torch.randn(r, lin.in_features, device=lin.weight.device, dtype=lin.weight.dtype) / math.sqrt(lin.in_features))
        self.B = nn.Parameter(torch.zeros(lin.out_features, r, device=lin.weight.device, dtype=lin.weight.dtype))
        self.scale = 1.0
    def forward(self, x):
        return self.lin(x) + (x @ self.A.t()) @ self.B.t() * self.scale


covs = torch.load("covs.pt") if (a.kind != "lora" and not a.init) else None
INIT = torch.load(a.init) if a.init else None
if INIT: a.which, a.kind, a.b, a.frac = INIT["args"]["which"], INIT["args"]["kind"], INIT["args"]["b"], INIT["args"]["frac"]
names = {id(mm): n for n, mm in m.named_modules()}
mac_s = mac_d = 0; errs = []
for (i, an, par, attr, lin) in G.targets(m, a.which):
    W = lin.weight.data.float()
    if a.kind == "lora":
        setattr(par, attr, LoRALinear(lin, a.rank)); continue
    if INIT:
        pref = names[id(lin)]
        fac = dict(R=INIT["state"][pref + ".R"], L=INIT["state"][pref + ".L"]); mac_d += W.numel()
        mac_s += S.BTTLinear(fac["R"], fac["L"]).macs()
        mod = S.BTTLinear(fac["R"], fac["L"]); setattr(par, attr, mod.cuda()); continue
    C = covs[(i, G.GROUP[an])].cuda()
    What, fac, mac = G.project_one(W, C, a.kind, a.frac, a.b, a.method, refine_steps=a.refine_steps)
    errs.append(S.out_err(W, What, C)); mac_s += mac; mac_d += W.numel()
    mod = S.BTTLinear(fac["R"], fac["L"]) if a.kind == "btt" else S.LRBDLinear(fac["U"], fac["V"], fac["D"])
    setattr(par, attr, mod.cuda())
    del C, What
torch.cuda.empty_cache()
dense_total = 0
for ly in G.layers_of(m):
    for mod in ly.modules():
        if isinstance(mod, nn.Linear): dense_total += mod.in_features * mod.out_features
info = dict(arm=a.arm, which=a.which, kind=a.kind, b=a.b, frac=a.frac, method=a.method, mean_out_err=(sum(errs) / len(errs)) if errs else 0.0,
            struct_mac_frac=(mac_s / mac_d) if mac_d else 1.0)
print("built", json.dumps(info), round(time.time() - t0), "s", flush=True)

# gradient checkpointing on the torso layers
torso = getattr(m.torso, "model", m.torso)
m.torso.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}) if hasattr(m.torso, "gradient_checkpointing_enable") else None
m.train()
params = [p for p in m.parameters() if p.requires_grad]
master = [p.detach().float().clone().requires_grad_(True) for p in params]
print("trainable", sum(p.numel() for p in params), flush=True)
if a.rel_lr > 0:   # scale-aware: per-tensor lr = rel_lr * rms(tensor)  (relative update size per step)
    opt = torch.optim.AdamW([dict(params=[mp], lr=a.rel_lr * float(mp.detach().pow(2).mean().sqrt())) for mp in master], betas=(0.9, 0.95), weight_decay=0.0)
else:
    opt = torch.optim.AdamW(master, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
warm = max(1, int(0.03 * a.steps))
lam = lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, a.steps - warm))))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)

D = torch.load("rows_train.pt"); rows = D["rows"]; TT = D["TT"]; T0 = D["T0"]
dev = torch.load("rows_dev.pt")["rows"]
pad = m.tokenizer.pad_token_id
rng = random.Random(a.seed)
idx = list(range(len(rows))); rng.shuffle(idx)
need = a.steps * a.bs
order = (idx * (need // len(idx) + 1))[:need]
steps = []
for s in range(0, need, a.bs * 25):   # length-grouped within megabatches
    mb = sorted(order[s:s + a.bs * 25], key=lambda i: len(rows[i]["ids"]))
    steps += [mb[j:j + a.bs] for j in range(0, len(mb), a.bs)]
rng.shuffle(steps)
save_at = sorted({max(1, int(round(float(f) * a.steps))) for f in a.save_at.split(",")})


def temps(rs):
    return torch.tensor([TT.get(r["kind"], T0) for r in rs], dtype=torch.float32)


def tens(rs):
    ids = pad_sequence([r["ids"].long() for r in rs], batch_first=True, padding_value=pad)
    am = pad_sequence([torch.ones(len(r["ids"]), dtype=torch.long) for r in rs], batch_first=True)
    W = max(r["n"] for r in rs)
    opt_ = torch.tensor([r["opt"] + [-1] * (W - r["n"]) for r in rs]); ns = torch.tensor([r["n"] for r in rs])
    tl = torch.full((len(rs), W), -1e4)
    for j, r in enumerate(rs): tl[j, :r["n"]] = torch.tensor(r["t_logits"])
    lab = torch.tensor([r["label"] for r in rs]); w = torch.tensor([r["w"] for r in rs])
    dist = torch.zeros(len(rs), W); has_d = torch.zeros(len(rs), dtype=torch.bool)
    for j, r in enumerate(rs):
        if r.get("dist") is not None: dist[j, :r["n"]] = torch.tensor(r["dist"]); has_d[j] = True
    return ids, am, opt_, ns, tl, lab, w, dist, has_d


def micro(rs, tokb):
    rs = sorted(rs, key=lambda r: len(r["ids"])); out, cur = [], []
    for r in rs:
        if cur and (len(cur) + 1) * max(len(x["ids"]) for x in cur + [r]) > tokb: out.append(cur); cur = []
        cur.append(r)
    out.append(cur); return out


@torch.no_grad()
def dev_eval():
    m.eval(); kl = []; ag = []
    for mb in micro(dev, 12288):
        ids, am, opt_, ns, tl, lab, w, dist, has_d = [t.cuda() for t in tens(mb)]
        T = temps(mb).cuda()
        lp = m(input_ids=ids, attention_mask=am, n_slots=ns, opt_idx=opt_, temperature=T)["log_probs"].float()
        tlp = F.log_softmax(tl / T[:, None], -1)
        valid = torch.arange(lp.shape[1], device=lp.device)[None] < ns[:, None]
        k = (tlp.exp() * (tlp - lp.masked_fill(~valid, 0)) * valid).sum(-1)
        kl += k.tolist(); ag += (lp.masked_fill(~valid, -1e9).argmax(-1) == tl.argmax(-1)).tolist()
    m.train()
    return dict(dev_kl=round(sum(kl) / len(kl), 4), dev_agree=round(sum(ag) / len(ag), 4), n=len(kl))


def save(tag):
    sd = {n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}
    torch.save(dict(state=sd, info=info, args=vars(a)), os.path.join(a.out, f"{tag}.pt"))


logf = open(os.path.join(a.out, "log.jsonl"), "a")
e = dict(step=0, **dev_eval(), el=round(time.time() - t0)); print(json.dumps(e), flush=True); logf.write(json.dumps(e) + "\n"); logf.flush()
ntok = 0; acc = dict(kl=0., ce=0., n=0, ng=0); tt = time.time()
for step in range(1, a.steps + 1):
    srows = [rows[i] for i in steps[step - 1]]
    for mb in micro(srows, a.tokb):
        ids, am, opt_, ns, tl, lab, w, dist, has_d = [t.cuda(non_blocking=True) for t in tens(mb)]
        T = temps(mb).cuda()
        lp = m(input_ids=ids, attention_mask=am, n_slots=ns, opt_idx=opt_, temperature=T)["log_probs"].float()
        valid = torch.arange(lp.shape[1], device=lp.device)[None] < ns[:, None]
        safe = lp.masked_fill(~valid, 0.0)
        tlp = F.log_softmax(tl / T[:, None], -1)
        kl = (tlp.exp() * (tlp - safe) * valid).sum(-1)
        ce_h = -safe.gather(1, lab[:, None]).squeeze(1); ce_s = -(dist * safe).sum(-1)
        ce = torch.where(has_d, ce_s, ce_h)
        per = kl + a.gw * (w > 0).float() * ce
        (per.sum() / len(srows)).backward()
        acc["kl"] += float(kl.detach().sum()); acc["n"] += len(mb); acc["ce"] += float(((w > 0).float() * ce).detach().sum()); acc["ng"] += int((w > 0).sum())
        ntok += int(am.sum())
    for p, mp in zip(params, master):
        mp.grad = None if p.grad is None else p.grad.float(); p.grad = None
    gn = torch.nn.utils.clip_grad_norm_(master, 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    with torch.no_grad():
        for p, mp in zip(params, master): p.copy_(mp)
    if step % 10 == 0 or step == a.steps:
        el = time.time() - tt
        e = dict(step=step, kl=round(acc["kl"] / max(1, acc["n"]), 4), ce=round(acc["ce"] / max(1, acc["ng"]), 4), gn=round(float(gn), 3),
                 lr=sched.get_last_lr()[0], tok_s=round(ntok / el), eta_min=round(el / step * (a.steps - step) / 60, 1),
                 mem=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        print(json.dumps(e), flush=True); logf.write(json.dumps(e) + "\n"); logf.flush()
        acc = dict(kl=0., ce=0., n=0, ng=0)
    if step in save_at:
        save(f"step{step}")
        e = dict(step=step, **dev_eval(), el=round(time.time() - t0)); print(json.dumps(e), flush=True); logf.write(json.dumps(e) + "\n"); logf.flush()
print("done", round(time.time() - t0), "s", flush=True)

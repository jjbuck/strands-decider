"""Recovery fine-tune of a depth-truncated hobson-v19: keep first L layers, add fresh LoRA (all linear projections) + train final norm + pointer head,
loss = CE(label) + KL(teacher calibrated probs from the stock 24-layer model), then eval on val (fit T) and JevBench public."""
import sys, argparse
from lib import *
import numpy as np, csv
from items import build
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, inject_adapter_in_model
ap = argparse.ArgumentParser()
ap.add_argument("--L", type=int, required=True); ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--accum", type=int, default=8); ap.add_argument("--maxtok", type=int, default=2400)
ap.add_argument("--kl", type=float, default=0.5); ap.add_argument("--tag", default=None); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--nval", type=int, default=150); ap.add_argument("--probe", default=None); ap.add_argument("--headlr", type=float, default=1e-4); ap.add_argument("--noinit", action="store_true")
a = ap.parse_args(); tag = a.tag or f"L{a.L}"
torch.manual_seed(0)
eng = load(); R = Runner(eng); tm = eng.model.torso; dev = eng.device
import torch.nn as nn
from strands_decider.modeling import PointerHead
class Probe(nn.Module):
    def __init__(self, mu, sd, dim=256):
        super().__init__(); self.register_buffer("mu", mu); self.register_buffer("sd", sd); self.head = PointerHead(2048, dim=dim, dropout=0.0)
    def forward(self, D, O): return self.head((D - self.mu) / self.sd, (O - self.mu) / self.sd)
pk = torch.load(a.probe); st = pk["state"]
head = Probe(st["mu"].clone(), st["sd"].clone()).to(dev); head.load_state_dict(st)
tr = torch.load("feats_train.pt")["data"]; items = build("train")
assert len(tr) == len(items) and all(f["id"] == it["id"] for f, it in zip(tr, items)), "train item order mismatch"
va_f = torch.load("feats_val.pt")["data"]; va_items = build("val")
assert len(va_f) == len(va_items) and all(f["id"] == it["id"] for f, it in zip(va_f, va_items))
jbf = torch.load("feats_jb.pt")["data"]
tasks = jevbench_tasks(); tier = {r["task_id"]: r["tier"] for r in csv.DictReader(open(os.path.expanduser("~/work/shrink/jevbench_tasks.csv")))}
Tk = {k: temp_of(eng.model, k) for k in ("noul", "choice", "score")}
L = a.L
tm.layers = torch.nn.ModuleList(list(tm.layers)[:L])
for p in tm.parameters(): p.requires_grad_(False)
cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, bias="none",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj", "gate_proj", "up_proj", "down_proj"])
inject_adapter_in_model(cfg, tm)
lora_params = [p for n, p in tm.named_parameters() if "lora_" in n]
for p in lora_params: p.data = p.data.float(); p.requires_grad_(True)
for p in head.parameters(): p.requires_grad_(True)
head.mu.requires_grad_(False); head.sd.requires_grad_(False)
print(f"L={L} trainable lora {sum(p.numel() for p in lora_params)/1e6:.1f}M, mem {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
params = [dict(params=lora_params, lr=a.lr), dict(params=list(head.head.parameters()), lr=a.headlr)]
opt = torch.optim.AdamW(params, weight_decay=0.01)

def fwd(ids, train):
    B, T = ids.shape
    emb = tm.embed_tokens(ids)
    pos = torch.arange(T, device=ids.device).view(1, 1, -1).expand(4, B, -1)
    am = torch.ones(B, T, dtype=torch.long, device=ids.device)
    mk = dict(config=R.cfg, inputs_embeds=emb, attention_mask=am, past_key_values=None, position_ids=pos[0])
    masks = {"full_attention": create_causal_mask(**mk), "linear_attention": create_recurrent_attention_mask(**mk)}
    pe = tm.rotary_emb(emb, pos[1:]); h = emb
    for i in range(L):
        layer = tm.layers[i]; m = masks[R.types[i]]
        def f(h, layer=layer, m=m):
            o = layer(h, position_embeddings=pe, attention_mask=m, position_ids=pos[0], past_key_values=None, use_cache=False)
            return o[0] if isinstance(o, tuple) else o
        h = checkpoint(f, h, use_reentrant=False) if train else f(h)
    return h

def logits_for(ids, opt_idx):
    hid = fwd(ids, train=torch.is_grad_enabled())
    pooled = hid[0, -1].float().unsqueeze(0); options = hid[0, opt_idx].float().unsqueeze(0)
    return head(pooled, options)[0]

def evaluate(fs, its, Tfit=None):
    for m in [head]: m.eval()
    out = []
    with torch.inference_mode():
        for it in its:
            e = encode(eng, it["state"], it["q"], it["order"]) if "q" in it else it["enc"]
            out.append(logits_for(e["ids"], e["opt"]).float().cpu())
            if len(out) % 100 == 0: print("  eval", len(out), flush=True)
    return out

def metrics(lgs, labels, Tv):
    acc = []; nll = []; conf = []; bri = []
    for lg, y in zip(lgs, labels):
        p = torch.softmax(lg / Tv, -1); acc.append(int(p.argmax()) == y); nll.append(-math.log(max(float(p[y]), 1e-9))); conf.append(float(p.max()))
        bri.append(float(((p - F.one_hot(torch.tensor(y), len(p)).float()) ** 2).sum()))
    acc = np.array(acc)
    return acc, float(np.mean(nll)), ece(conf, acc), float(np.mean(bri))

def fitT(lgs, labels):
    best = (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(0.3), np.log(4), 50)):
        best = min(best, (metrics(lgs, labels, float(T))[1], float(T)))
    return best[1]

jb_enc = []
for t in tasks:
    e = encode(eng, t["state"], jb_question(t)); jb_enc.append(dict(enc=e, label=jb_label(t, e["rq"]), tier=tier.get(t["id"], "?")))
jb_tiers = np.array([x["tier"] for x in jb_enc]); jb_lab = [x["label"] for x in jb_enc]
# validation set encoded once
va_enc = []
for it in va_items[:a.nval]:
    e = encode(eng, it["state"], it["q"], it["order"]); va_enc.append(dict(enc=e))
va_items = va_items[:a.nval]; va_enc = va_enc[:a.nval]; va_lab = [it["label"] for it in va_items]

def full_eval(name):
    t0 = time.time()
    lv = evaluate(None, va_enc); Tv = fitT(lv, va_lab); accv, nllv, ecev, briv = metrics(lv, va_lab, Tv)
    lj = evaluate(None, jb_enc); accj, nllj, ecej, brij = metrics(lj, jb_lab, Tv)
    r = dict(name=name, L=L, T=Tv, val_acc=float(accv.mean()), val_nll=nllv, jb_acc=float(accj.mean()), jb_nll=nllj, jb_ece=ecej, jb_brier=brij,
             **{k: float(accj[jb_tiers == k].mean()) for k in ("easy", "standard", "hard")})
    print(f"[{name}] L={L} T={Tv:.2f} | val acc {r['val_acc']:.3f} nll {nllv:.3f} | JB acc {r['jb_acc']:.3f} (easy {r['easy']:.2f} std {r['standard']:.2f} hard {r['hard']:.2f}) nll {nllj:.3f} ece {ecej:.3f} brier {brij:.3f} ({time.time()-t0:.0f}s)", flush=True)
    return r

results = [] if a.noinit else [full_eval("init")]
order_idx = [i for i, it in enumerate(items)]
if a.limit: order_idx = order_idx[:a.limit]
N = len(order_idx); total_steps = a.epochs * (N // a.accum); step = 0
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 15) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))) )
t0 = time.time(); ntok = 0
for ep in range(a.epochs):
    random.Random(ep).shuffle(order_idx); head.train(); run_loss = []
    for k, i in enumerate(order_idx):
        it, f = items[i], tr[i]
        e = encode(eng, it["state"], it["q"], it["order"])
        if e["T"] > a.maxtok: continue
        lg = logits_for(e["ids"], e["opt"])
        y = torch.tensor(it["label"], device=dev)
        ce = F.cross_entropy(lg.unsqueeze(0), y.unsqueeze(0))
        tp = torch.softmax(f["logits"].to(dev) / Tk[it["kind"]], -1)
        kl = F.kl_div(F.log_softmax(lg, -1), tp, reduction="sum")
        loss = ((1 - a.kl) * ce + a.kl * kl) / a.accum
        loss.backward(); run_loss.append(float(loss) * a.accum); ntok += e["T"]
        if (k + 1) % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(lora_params + list(head.head.parameters()), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 25 == 0:
                print(f"ep{ep} step {step}/{total_steps} loss {np.mean(run_loss[-100:]):.4f} tok/s {ntok/(time.time()-t0):.0f} elapsed {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)
    head.eval(); results.append(full_eval(f"ep{ep}"))
    json.dump(results, open(f"train_{tag}.json", "w"))
torch.save(dict(lora={n: p.detach().cpu() for n, p in tm.named_parameters() if "lora_" in n}, head=head.state_dict()), f"ckpt_{tag}.pt")

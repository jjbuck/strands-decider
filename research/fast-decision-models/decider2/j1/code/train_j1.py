"""J1 trainer = F7's recipe (as replicated by g3's train_g3.py) with the T5Gemma-2B encoder torso:
pointer head dim 256 (+LayerNorm, dropout .05) + LoRA r16/alpha32 on q,k,v,o,gate,up,down of all 26 layers, lr 1e-4 / head 1e-3,
32 rows/step, 3% warmup + cosine, AdamW(0.9,0.95), clip 1.0, 1 epoch, length-grouped megabatches of 50 steps,
CE(gold, ordinal smoothing) + 1.0*KL(hobson||student) on labelled rows + 1.0*KL on real-state KL-only rows, hobson calibrated temps.
Attention: --mode masked (e1b: state never sees the question) | full (e1a). Optional --local L1,L2.. --window W for state->state windows.
Resumable: --resume picks up OUT/last.pt (LoRA, head, optimiser, scheduler, step); touch OUT/STOP to save + exit at the next step.
"""
import os, sys, json, math, time, random, argparse
import torch, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
sys.path.insert(0, os.path.expanduser("~/work/sd/src"))
from strands_decider.modeling import StrandsDeciderConfig, build_head, masked_log_softmax, gather_options, pool_last_token
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encj1 as E

ap = argparse.ArgumentParser()
ap.add_argument("--rows", nargs="+", default=["rows_corpus_e.pt", "rows_real_e.pt"])
ap.add_argument("--out", required=True); ap.add_argument("--max_steps", type=int, default=0)
ap.add_argument("--tw", type=float, default=1.0); ap.add_argument("--rw", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--head_lr", type=float, default=1e-3)
ap.add_argument("--rank", type=int, default=16); ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--tokb", type=int, default=4600); ap.add_argument("--ckpt_above", type=int, default=4600); ap.add_argument("--mega", type=int, default=50)
ap.add_argument("--save_at", default="0.25,0.5,0.75,1.0"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--limit_rows", type=int, default=0); ap.add_argument("--mode", default="masked")
ap.add_argument("--softcap", type=float, default=0.0); ap.add_argument("--local", default=""); ap.add_argument("--window", type=int, default=0)
ap.add_argument("--resume", action="store_true"); ap.add_argument("--init", default="", help="start from a saved LoRA+head (heal runs)")
ap.add_argument("--resume_every", type=int, default=100); ap.add_argument("--lc", type=int, default=1)
ap.add_argument("--stop_after", type=int, default=0, help="exit after this many steps in this process (smoke)")
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
os.makedirs(a.out, exist_ok=True)

Wt, ecfg = E.load_weights()
local = [int(x) for x in a.local.split(",") if x != ""]
torso = E.EncTorso(Wt, r=a.rank, alpha=2 * a.rank, softcap=(a.softcap or None), local_layers=local, window=a.window).cuda()
torso.mode = a.mode
if os.environ.get('NOCOMPILE') != '1': E.use_compiled()
del Wt; torch.cuda.empty_cache()
hcfg = StrandsDeciderConfig(base_model="t5gemma-2b-2b-ul2-encoder", head_type="pointer", pointer_dim=256, head_dropout=0.05, max_length=16384,
                            use_lora=True, lora_r=a.rank, lora_alpha=2 * a.rank, ordinal_smoothing=0.1, lora_dropout=0.0)
head = build_head(hcfg, E.D).cuda()
if a.init:
    sd = torch.load(os.path.join(a.init, "lora.pt")); torso.lora.load_state_dict({k: v.cuda() for k, v in sd.items()})
    head.load_state_dict(torch.load(os.path.join(a.init, "slot_head.pt")))
    print("init from", a.init, flush=True)
torso.train(); head.train()
lora = [p for p in torso.lora.parameters()]
print(f"mode {a.mode} softcap {a.softcap} local {local} window {a.window} lora params {sum(p.numel() for p in lora):,} head {sum(p.numel() for p in head.parameters()):,}", flush=True)
PAD = 0


def fwd(ids, meta, ns, opt):
    hid = torso.forward_packed(ids, meta)
    pooled = hid[meta["last"]].float()
    logits = head(pooled, hid[opt].float())
    return masked_log_softmax(logits, ns)


def kl_rows(t_logits, s_lp, ns):
    Wd = s_lp.shape[-1]
    valid = torch.arange(Wd, device=s_lp.device)[None, :] < ns[:, None]
    t_lp = F.log_softmax(t_logits.masked_fill(~valid, -1e4), -1)
    s_safe = s_lp.masked_fill(~valid, 0.0)
    p = t_lp.exp() * valid
    return (p * (t_lp - s_safe)).sum(-1)


TT = {}


def batch_tensors(rows):
    """packed stream: rows back to back, no padding"""
    lens = [len(r["ids"]) for r in rows]; nst = [r["nstate"] for r in rows]
    ids = torch.cat([r["ids"].long() for r in rows])
    starts = [0]
    for L in lens[:-1]: starts.append(starts[-1] + L)
    Wd = max(r["n"] for r in rows)
    opt = torch.tensor([[st + o for o in r["opt"]] + [st] * (Wd - r["n"]) for st, r in zip(starts, rows)])
    ns = torch.tensor([r["n"] for r in rows]); lab = torch.tensor([r["label"] for r in rows])
    tl = torch.zeros((len(rows), Wd)); has_t = torch.zeros(len(rows), dtype=torch.bool)
    dist = torch.zeros(len(rows), Wd); has_d = torch.zeros(len(rows), dtype=torch.bool)
    for j, r in enumerate(rows):
        if r.get("t_logits") is not None:
            tl[j, :r["n"]] = torch.tensor(r["t_logits"]) / TT.get(r["kind"], 1.0); has_t[j] = True
        if r.get("dist") is not None:
            dist[j, :r["n"]] = torch.tensor(r["dist"]); has_d[j] = True
    w = torch.tensor([r["w"] for r in rows])
    return (lens, nst), ids, opt, ns, lab, tl, has_t, dist, has_d, w


@torch.no_grad()
def lc_eval(rows, tokb=12288):
    torso.eval(); head.eval(); res = {}
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]["ids"]))
    k = 0
    while k < len(order):
        ch = []; tot = 0
        while k < len(order) and (not ch or tot + len(rows[order[k]]["ids"]) <= tokb):
            ch.append(rows[order[k]]); tot += len(rows[order[k]]["ids"]); k += 1
        (lens, nst), ids, opt, ns, lab, tl, has_t, *_ = batch_tensors(ch)
        meta = E.pack_meta(lens, nst, "cuda")
        lp = fwd(ids.cuda(), meta, ns.cuda(), opt.cuda()).float().cpu()
        for j, r in enumerate(ch):
            d = res.setdefault(r["src"], dict(n=0, acc=0, agree=0, nll=0.0))
            d["n"] += 1; pred = int(lp[j, :r["n"]].argmax())
            d["acc"] += int(pred == r["label"]); d["nll"] += -float(lp[j, r["label"]])
            if r.get("t_logits") is not None:
                d["agree"] += int(pred == int(torch.tensor(r["t_logits"]).argmax()))
    torso.train(); head.train()
    return {s: dict(n=d["n"], acc=round(d["acc"] / d["n"], 4), agree_hobson=round(d["agree"] / d["n"], 4), nll=round(d["nll"] / d["n"], 4)) for s, d in res.items()}


def save(path, extra):
    os.makedirs(path, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in torso.lora.state_dict().items()}, os.path.join(path, "lora.pt"))
    torch.save(head.state_dict(), os.path.join(path, "slot_head.pt"))
    with open(os.path.join(path, "strands_decider_config.json"), "w") as f: f.write(hcfg.to_json())
    json.dump(extra, open(os.path.join(path, "j1_meta.json"), "w"))


rows = []
for p in a.rows:
    Dd = torch.load(p); rr = Dd["rows"]
    if Dd.get("teacher_temps"): TT = dict(Dd["teacher_temps"]["by_kind"])
    rows += rr; print(p, len(rr), "rows", flush=True)
if a.limit_rows: random.Random(3).shuffle(rows); rows = rows[:a.limit_rows]
lc_rows = torch.load("rows_lceval_e.pt")["rows"] if (a.lc and os.path.exists("rows_lceval_e.pt")) else []
print("teacher temps", TT, flush=True)
print("train rows", len(rows), "teacher-covered", sum(r.get("t_logits") is not None for r in rows), "KL-only", sum(r["w"] == 0 for r in rows),
      "tokens", sum(len(r["ids"]) for r in rows), "lc-eval rows", len(lc_rows), flush=True)

rng = random.Random(a.seed)
idx = list(range(len(rows))); rng.shuffle(idx)
MB = a.mega * a.bs; steps = []
for s in range(0, len(idx), MB):
    mb = sorted(idx[s:s + MB], key=lambda i: len(rows[i]["ids"]))
    steps += [mb[j:j + a.bs] for j in range(0, len(mb) - a.bs + 1, a.bs)]
rng.shuffle(steps)
total = a.max_steps or len(steps); steps = steps[:total]
warm = max(1, int(0.03 * total))
save_at = sorted({max(1, int(round(float(f) * total))) for f in a.save_at.split(",")})
print(f"{total} steps, warmup {warm}, save at {save_at}", flush=True)

hd, hnd = [], []
for n, p in head.named_parameters(): (hnd if p.ndim <= 1 else hd).append(p)
opt = torch.optim.AdamW([dict(params=hd, lr=a.head_lr, weight_decay=0.01), dict(params=hnd, lr=a.head_lr, weight_decay=0.0),
                         dict(params=lora, lr=a.lr, weight_decay=0.0)], betas=(0.9, 0.95), eps=1e-8)
lam = lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, total - warm))))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
start = 1; ntok_prev = 0; el_prev = 0.0
LAST = os.path.join(a.out, "last.pt")
if a.resume and os.path.exists(LAST):
    ck = torch.load(LAST, weights_only=False)
    torso.lora.load_state_dict({k: v.cuda() for k, v in ck["lora"].items()}); head.load_state_dict(ck["head"])
    opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
    start = ck["step"] + 1; ntok_prev = ck.get("ntok", 0); el_prev = ck.get("el", 0.0)
    print("resumed at step", start, flush=True)


def save_last(step, ntok, el):
    tmp = LAST + ".tmp"
    torch.save(dict(lora={k: v.detach().cpu() for k, v in torso.lora.state_dict().items()}, head=head.state_dict(), opt=opt.state_dict(),
                    sched=sched.state_dict(), step=step, ntok=ntok, el=el), tmp)
    os.replace(tmp, LAST)


logf = open(os.path.join(a.out, "train_log.jsonl"), "a")
t0 = time.time() - el_prev; ntok = ntok_prev; acc = dict(loss=0., ce=0., kl=0., rkl=0., n=0, nl=0, nt=0, nr=0); done_here = 0
for step in range(start, total + 1):
    srows = [rows[i] for i in steps[step - 1]]
    micro, cur = [], []
    for r in srows:
        if cur and sum(len(x["ids"]) for x in cur) + len(r["ids"]) > a.tokb:
            micro.append(cur); cur = []
        cur.append(r)
    micro.append(cur)
    for mb in micro:
        (lens, nst), *tt = batch_tensors(mb)
        ids, optx, ns, lab, tl, has_t, dist, has_d, w = [t.cuda(non_blocking=True) for t in tt]
        meta = E.pack_meta(lens, nst, "cuda")
        torso.ckpt = ids.numel() > a.ckpt_above
        lp = fwd(ids, meta, ns, optx)
        safe = lp.masked_fill(torch.arange(lp.shape[-1], device=lp.device)[None, :] >= ns[:, None], 0.0)
        ce_hard = -safe.gather(1, lab[:, None]).squeeze(1)
        ce_soft = -(dist * safe).sum(-1)
        ce = torch.where(has_d, ce_soft, ce_hard)
        kl = kl_rows(tl, lp, ns)
        lab_rows = (w > 0).float(); kl_only = (w == 0).float()
        per = lab_rows * ce + has_t.float() * kl * (lab_rows * a.tw + kl_only * a.rw)
        loss = per.sum() / len(srows)
        loss.backward()
        with torch.no_grad():
            acc["loss"] += float(per.detach().sum()); acc["n"] += len(mb)
            acc["ce"] += float((lab_rows * ce).sum()); acc["nl"] += int(lab_rows.sum())
            acc["kl"] += float((lab_rows * has_t.float() * kl).sum()); acc["nt"] += int((lab_rows * has_t.float()).sum())
            acc["rkl"] += float((kl_only * has_t.float() * kl).sum()); acc["nr"] += int((kl_only * has_t.float()).sum())
        ntok += int(ids.numel())
    torch.nn.utils.clip_grad_norm_(hd + hnd + lora, 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    done_here += 1
    if step % 20 == 0 or step == total or step in (1, 5) or done_here <= 3:
        el = time.time() - t0
        e = dict(step=step, loss=acc["loss"] / max(1, acc["n"]), ce=acc["ce"] / max(1, acc["nl"]), kl=acc["kl"] / max(1, acc["nt"]),
                 rkl=acc["rkl"] / max(1, acc["nr"]), lr=sched.get_last_lr()[-1], el=round(el), tok_s=round(ntok / el),
                 eta_min=round(el / step * (total - step) / 60, 1), mem=round(torch.cuda.max_memory_allocated() / 2**30, 1))
        print(json.dumps(e), flush=True); logf.write(json.dumps(e) + "\n"); logf.flush()
        acc = dict(loss=0., ce=0., kl=0., rkl=0., n=0, nl=0, nt=0, nr=0)
    if step in save_at:
        ckd = os.path.join(a.out, f"step{step}")
        save(ckd, dict(step=step, total=total, rows_seen=step * a.bs, tokens_seen=ntok, mode=a.mode, softcap=a.softcap, local=local, window=a.window))
        if lc_rows:
            te = time.time(); r = lc_eval(lc_rows)
            e = dict(step=step, lc=r, eval_s=round(time.time() - te))
            print("LC", json.dumps(e), flush=True); logf.write(json.dumps(e) + "\n"); logf.flush()
            t0 += time.time() - te
    stop = os.path.exists(os.path.join(a.out, "STOP")) or (a.stop_after and done_here >= a.stop_after)
    if step % a.resume_every == 0 or stop or step == total:
        save_last(step, ntok, time.time() - t0)
    if stop:
        print("STOP at step", step, flush=True); break
print("done", round(time.time() - t0), "s", flush=True)

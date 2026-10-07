"""G1 model-side library: hobson-v19 (LoRA merged) as a StrandsDeciderModel, target projections, activation covariances,
structured replacement (dense reconstruction for eval / BTTLinear for training), evalkit eval rows and runner."""
import os, sys, json, glob, time, random, math
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
sys.path.insert(0, os.path.expanduser("~/work/sd/src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strands_decider.modeling import StrandsDeciderModel
from strands_decider.prompting import render_question, render_state
from pydantic import TypeAdapter
import strands_decider.schema as SC
import slib as S
ta = TypeAdapter(SC.Question)
HOB = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*"))[0]
KIT = os.path.expanduser("~/work/evalkit")

MLP_T = ["gate_proj", "up_proj", "down_proj"]
ATT_T = ["q_proj", "o_proj"]                      # k_proj / v_proj (2048->512) stay dense
GDN_T = ["in_proj_qkv", "in_proj_z", "out_proj"]   # in_proj_a / in_proj_b (2048->16) stay dense
GROUP = {"gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "mlp_mid", "q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in",
         "o_proj": "attn_o", "in_proj_qkv": "gdn_in", "in_proj_z": "gdn_in", "in_proj_a": "gdn_in", "in_proj_b": "gdn_in", "out_proj": "gdn_o"}


def load_hobson():
    import fast
    fast.apply(bf16_lora=False)
    m = StrandsDeciderModel.load(HOB)
    m.torso = m.torso.merge_and_unload()
    return m.cuda().eval()


def layers_of(m):
    t = m.torso
    t = getattr(t, "model", t)
    return t.layers


def targets(m, which="mlp"):
    """[(layer, short_name, parent_module, attr, linear)]"""
    names = MLP_T + ((ATT_T + GDN_T) if which == "all" else [])
    out = []
    for i, ly in enumerate(layers_of(m)):
        for pn, par in ly.named_modules():
            for an, ch in par.named_children():
                if an in names and isinstance(ch, nn.Linear):
                    out.append((i, an, par, an, ch))
    return out


def dense_macs_per_token(m):
    tot = 0
    for ly in layers_of(m):
        for mod in ly.modules():
            if isinstance(mod, nn.Linear): tot += mod.in_features * mod.out_features
            elif hasattr(mod, "macs") and not isinstance(mod, nn.Linear) and mod.__class__.__name__ in ("BTTLinear", "LRBDLinear"): tot += mod.macs()
    return tot


# ------------------------------------------------------------------ data rows
def fit_rows(items, maxlen=16384, per_item=None, seed=0):
    from prep import fit
    rng = random.Random(seed); rows = []
    for it in items:
        qs = list(it["questions"].items())
        if per_item and len(qs) > per_item: qs = rng.sample(qs, per_item)
        for qn, qd in qs:
            rq = render_question(ta.validate_python(qd))
            s, q, opt = fit(render_state(it["state"]), rq.text, rq, max_len=maxlen)
            rows.append(dict(id=it.get("id"), qn=qn, rq=rq, ids=torch.tensor(s + q), opt=opt, n=rq.n_slots))
    return rows


def pool_items(n, seed=11, max_tok=3500, min_tok=300):
    its = []
    with open(f"{KIT}/train_pool.jsonl") as f:
        for l in f: its.append(json.loads(l))
    split = json.load(open(f"{KIT}/split.json"))
    ev = set(split["eval_tasks"]) if isinstance(split.get("eval_tasks"), list) else set(sum(split["eval_tasks"].values(), []))
    its = [r for r in its if str(r.get("task")) not in ev]
    random.Random(seed).shuffle(its)
    out = []
    for r in its:
        nt = r.get("n_state_tok") or r.get("n_tok") or 0
        if nt and (nt > max_tok or nt < min_tok): continue
        out.append(dict(id=r.get("rid", len(out)), state=r["state"], questions=r["questions"]))
        if len(out) >= n: break
    return out


# ------------------------------------------------------------------ forward over rows
@torch.inference_mode()
def run_rows(m, rows, tokb=24576, key="p"):
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]["ids"]))
    pad = m.tokenizer.pad_token_id; k = 0
    while k < len(order):
        L = len(rows[order[k]]["ids"]); nb = max(1, min(32, tokb // L))
        ch = [rows[i] for i in order[k:k + nb]]; k += len(ch)
        ids = pad_sequence([r["ids"] for r in ch], batch_first=True, padding_value=pad).cuda()
        am = pad_sequence([torch.ones(len(r["ids"]), dtype=torch.long) for r in ch], batch_first=True).cuda()
        W = max(r["n"] for r in ch)
        opt = torch.tensor([r["opt"] + [-1] * (W - r["n"]) for r in ch]).cuda()
        ns = torch.tensor([r["n"] for r in ch]).cuda()
        lp = m(input_ids=ids, attention_mask=am, n_slots=ns, opt_idx=opt, temperature=1.0)["log_probs"].float().cpu()
        for j, r in enumerate(ch):
            r[key] = {lab: float(lp[j, i].exp()) for i, lab in enumerate(r["rq"].slot_labels)}


# ------------------------------------------------------------------ covariances
@torch.no_grad()
def collect_cov(m, rows, which="all", tokb=8192):
    """input second moments X^T X (fp32, GPU) per (layer, group); returns dict and token count."""
    tg = targets(m, which)
    covs, hooks, cnt = {}, [], {"n": 0}
    seen = set()
    for (i, an, par, attr, lin) in tg + [(i, an, par, attr, lin) for (i, an, par, attr, lin) in targets(m, "all") if an in ("k_proj",)]:
        key = (i, GROUP[an])
        if key in seen: continue
        seen.add(key)
        def hk(mod, inp, out, key=key):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            if key not in covs: covs[key] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
            covs[key].addmm_(x.t(), x)
        hooks.append(lin.register_forward_hook(hk))
    # attention_mask padding rows would pollute; run unpadded (one row per forward)
    for r in rows:
        ids = r["ids"][None].cuda(); am = torch.ones_like(ids)
        W = r["n"]; opt = torch.tensor([r["opt"]]).cuda(); ns = torch.tensor([r["n"]]).cuda()
        m(input_ids=ids, attention_mask=am, n_slots=ns, opt_idx=opt, temperature=1.0)
        cnt["n"] += ids.numel()
    for h in hooks: h.remove()
    for k in covs: covs[k] /= cnt["n"]
    return covs, cnt["n"]


# ------------------------------------------------------------------ structured replacement
def project_one(W, C, kind, frac, b, method, refine_steps=200):
    """-> (What dense fp32, factors dict, macs, info)"""
    n_out, n_in = W.shape
    Cw = C if method in ("white", "refine") else None
    if kind == "btt":
        r = S.btt_r(n_in, n_out, b, frac)
        R, L = S.btt_project(W, b, b, r, Cw)
        if method == "refine":
            R, L = S.refine([R, L], S.btt_from_factors, W, C, steps=refine_steps)
        What = S.btt_from_factors(R, L); fac = dict(R=R, L=L); mac = S.macs_btt(n_in, n_out, b, b, r) if hasattr(S, "macs_btt") else r * (b * n_in + b * n_out)
    elif kind == "lrbd":
        # half the budget to the block diagonal (b blocks), half to low rank -- unless the block diagonal alone exceeds the budget
        bd = 1.0 / b
        rl = max(1, int(round((frac - bd) * n_in * n_out / (n_in + n_out))))
        U, V, D = S.lrbd_project(W, b, rl, Cw)
        if method == "refine":
            U, V, D = S.refine([U, V, D], S.lrbd_from, W, C, steps=refine_steps)
        What = S.lrbd_from(U, V, D); fac = dict(U=U, V=V, D=D); mac = rl * (n_in + n_out) + n_in * n_out // b
    elif kind == "kron":
        p, q, s, t = S.kron_dims(n_in, n_out)
        per = min(q * s * t + p * q * s, p * q * t + p * t * s)
        nt = max(1, int(round(frac * n_in * n_out / per)))
        A, Bm = S.kron_project(W, nt)
        if method == "refine":
            A, Bm = S.refine([A, Bm], S.kron_from, W, C, steps=refine_steps)
        What = S.kron_from(A, Bm); fac = dict(A=A, B=Bm); mac = nt * per
    else:
        raise ValueError(kind)
    return What, fac, mac


class DenseSwap:
    """swap Linear weights for structured reconstructions (eval); restore() puts the originals back."""
    def __init__(self): self.saved = []
    def set(self, lin, What):
        self.saved.append((lin, lin.weight.data))
        lin.weight.data = What.to(lin.weight.dtype).contiguous()
    def restore(self):
        for lin, w in self.saved: lin.weight.data = w
        self.saved = []


def kl_and_agree(rows, kp="p", kq="p0"):
    """mean KL(q||p) (teacher q), argmax agreement."""
    kl, ag = [], []
    for r in rows:
        p, q = r[kp], r[kq]
        kl.append(sum(q[l] * (math.log(max(q[l], 1e-9)) - math.log(max(p[l], 1e-9))) for l in q))
        ag.append(max(p, key=p.get) == max(q, key=q.get))
    return sum(kl) / len(kl), sum(ag) / len(ag)


# ------------------------------------------------------------------ evalkit subsets
def suite_items(suite, every=1, pairs=False):
    its = [json.loads(l) for l in open(f"{KIT}/suites/{suite}.jsonl")]
    if every <= 1: return its
    if suite in ("CF", "CF-probe"):
        prs = [json.loads(l) for l in open(f"{KIT}/suites/{suite}.pairs.jsonl")][::every]
        keep = {x for p in prs for x in (p["a"], p["b"])}
        return [i for i in its if i["id"] in keep]
    return its[::every]


def write_preds(rows, path):
    out = {}
    for r in rows: out.setdefault(r["id"], {})[r["qn"]] = r["p"]
    with open(path, "w") as f:
        for i, q in out.items(): f.write(json.dumps(dict(id=i, q=q)) + "\n")

"""H3 step 4 (small scale): pretrain a standard pre-RMSNorm transformer and an nGPT-style hyperspherical transformer from scratch with
identical data, tokens and size, then compare W4A4 robustness (per-token int4, rotated int4, NVFP4, MXFP4) and GEMM-input crest factors.
Data: train-split real states (evalkit train_pool, eval tasks excluded) + train_v5 states, hobson tokenizer restricted to the top 16k ids (+UNK).
Validation: held-out train-pool TASKS (no task overlap with training).
python pretrain_tiny.py prep ; python pretrain_tiny.py train --arch std|ngpt --steps 3000 ; python pretrain_tiny.py evalq"""
import os, sys, json, math, time, random, argparse, collections
sys.path.insert(0, os.path.expanduser('~/work/h3'))
import torch, torch.nn as nn, torch.nn.functional as F
import h3lib as H

W = os.path.expanduser('~/work/h3/tiny/'); os.makedirs(W, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--arch', default='std'); ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--d', type=int, default=512); ap.add_argument('--L', type=int, default=8); ap.add_argument('--seq', type=int, default=512); ap.add_argument('--bs', type=int, default=32)
a = ap.parse_args()
V = 16384


def prep():
    from transformers import AutoTokenizer
    import glob
    ck = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*'))[0]
    tok = AutoTokenizer.from_pretrained(ck)
    EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
    tr, va = [], []
    tasks = set()
    with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
        rows = [json.loads(l) for l in f]
    tasks = sorted({r['task'] for r in rows if r['task'] not in EV}); random.Random(0).shuffle(tasks); vt = set(tasks[:len(tasks) // 10])
    seen = set()
    for r in rows:
        if r['task'] in EV: continue
        s = r['state'][:20000]
        if s[:3000] in seen: continue      # many requests share a conversation prefix; keep one per prefix
        seen.add(s[:3000])
        (va if r['task'] in vt else tr).append(s)
    with open(os.path.expanduser('~/work/training/data/train_v5.jsonl')) as f:
        for li, l in enumerate(f):
            if li % 3: continue
            tr.append(json.loads(l)['state'])
    random.Random(1).shuffle(tr)
    def enc(texts):
        ids = []
        for i in range(0, len(texts), 512):
            for e in tok(texts[i:i + 512], add_special_tokens=False)['input_ids']: ids.extend(e + [tok.eos_token_id or 0])
        return torch.tensor(ids, dtype=torch.int32)
    tr_ids = enc(tr); va_ids = enc(va)
    cnt = torch.bincount(tr_ids.long()); top = torch.argsort(cnt, descending=True)[:V - 1]
    remap = torch.full((cnt.numel() + 300000,), V - 1, dtype=torch.int32); remap[top] = torch.arange(V - 1, dtype=torch.int32)
    tr_c = remap[tr_ids.long()]; va_c = remap[va_ids.long()]
    torch.save(dict(train=tr_c, val=va_c), W + 'data.pt')
    print('train tokens', tr_c.numel(), 'val tokens', va_c.numel(), 'unk frac', float((tr_c == V - 1).float().mean()), flush=True)


# ------------------------------------------------------------------ models
def rope(x, cos, sin):  # x [B, H, T, hd]
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)


def l2(x, dim=-1): return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-6)


class QL(nn.Linear):
    """linear whose GEMM can be fake-quantized at eval (QC global) and whose input can be recorded (CREST)"""
    QC = None; CREST = None; ROT = False
    def forward(self, x):
        sh = x.shape; x2 = x.reshape(-1, sh[-1])
        if QL.CREST is not None:
            xf = x2.float(); QL.CREST.setdefault(self.tag, []).append(float((xf.abs().amax(-1) / xf.pow(2).mean(-1).sqrt().clamp_min(1e-8)).median()))
        if QL.QC is None: return F.linear(x, self.weight)
        Wt = self.weight.float()
        if QL.ROT: x2 = H.rot_rows(x2.float()); Wt = H.rot_rows(Wt)
        y = H.fq_act(x2, QL.QC) @ H.fq_w(Wt, QL.QC).t()
        return y.to(x.dtype).reshape(*sh[:-1], -1)


class Block(nn.Module):
    def __init__(self, d, nh, ffn, ngpt, li):
        super().__init__()
        self.ngpt = ngpt; self.nh = nh; self.hd = d // nh; self.d = d
        self.qkv = QL(d, 3 * d, bias=False); self.o = QL(d, d, bias=False)
        self.gu = QL(d, 2 * ffn, bias=False); self.down = QL(ffn, d, bias=False)
        for n_, m_ in (('qkv', self.qkv), ('o', self.o), ('gu', self.gu), ('down', self.down)): m_.tag = f'{li}.{n_}'
        if ngpt:
            sc = 1.0 / math.sqrt(d)
            self.alpha_a = nn.Parameter(torch.full((d,), sc)); self.alpha_m = nn.Parameter(torch.full((d,), sc))
            self.a_init = 0.05; self.a_scale = sc
            self.s_qk = nn.Parameter(torch.full((nh, 1, self.hd), sc)); self.sqk_scale = sc
            self.s_u = nn.Parameter(torch.ones(ffn)); self.s_v = nn.Parameter(torch.ones(ffn))
        else:
            self.n1 = nn.RMSNorm(d); self.n2 = nn.RMSNorm(d)

    def attn(self, x, cos, sin):
        B, T, _ = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        if self.ngpt:
            s = self.s_qk * (1.0 / self.sqk_scale)
            q = l2(q) * s; k = l2(k) * s
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=math.sqrt(self.hd))
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(o.transpose(1, 2).reshape(B, T, -1))

    def mlp(self, x):
        g, u = self.gu(x).chunk(2, -1)
        if self.ngpt:
            u = u * self.s_u; g = g * self.s_v * math.sqrt(self.d)
        return self.down(F.silu(g) * u)

    def forward(self, h, cos, sin):
        if not self.ngpt:
            h = h + self.attn(self.n1(h), cos, sin)
            return h + self.mlp(self.n2(h))
        aa = (self.alpha_a * (self.a_init / self.a_scale)).abs(); am = (self.alpha_m * (self.a_init / self.a_scale)).abs()
        h = l2(h + aa * (l2(self.attn(h, cos, sin)) - h))
        return l2(h + am * (l2(self.mlp(h)) - h))


class LM(nn.Module):
    def __init__(self, ngpt, d=512, L=8, nh=8, ffn=1536):
        super().__init__()
        self.ngpt = ngpt; self.d = d
        self.emb = nn.Embedding(V, d); self.blocks = nn.ModuleList([Block(d, nh, ffn, ngpt, i) for i in range(L)])
        self.out = nn.Linear(d, V, bias=False)
        if ngpt:
            self.s_z = nn.Parameter(torch.full((V,), 1.0 / math.sqrt(d))); self.sz_scale = 1.0 / math.sqrt(d)
        else:
            self.nf = nn.RMSNorm(d); self.out.weight = self.emb.weight
        for p_ in self.parameters():
            if p_.dim() == 2: nn.init.normal_(p_, std=0.02)
        inv = 1.0 / (10000 ** (torch.arange(0, d // nh, 2).float() / (d // nh))); self.register_buffer('inv', inv)
        if ngpt: self.normalize()

    @torch.no_grad()
    def normalize(self):
        """nGPT: every matrix unit-norm along the embedding dimension"""
        self.emb.weight.copy_(l2(self.emb.weight, 1)); self.out.weight.copy_(l2(self.out.weight, 1))
        for b in self.blocks:
            for m_ in (b.qkv, b.gu): m_.weight.copy_(l2(m_.weight, 1))
            for m_ in (b.o, b.down): m_.weight.copy_(l2(m_.weight, 0))

    def forward(self, ids):
        B, T = ids.shape
        fr = torch.arange(T, device=ids.device).float()[:, None] * self.inv[None]
        cos, sin = fr.cos()[None, None], fr.sin()[None, None]
        h = self.emb(ids)
        for b in self.blocks: h = b(h, cos, sin)
        if self.ngpt: return self.out(h) * (self.s_z * (1.0 / self.sz_scale))
        return self.out(self.nf(h))


def batches(data, bs, seq, g):
    n = data.numel() - seq - 1
    while True:
        ix = torch.randint(0, n, (bs,), generator=g)
        x = torch.stack([data[i:i + seq + 1] for i in ix.tolist()]).long().cuda()
        yield x[:, :-1], x[:, 1:]


def train():
    D = torch.load(W + 'data.pt'); torch.manual_seed(0)
    ng = a.arch == 'ngpt'
    m = LM(ng, a.d, a.L).cuda()
    if ng: opt = torch.optim.Adam(m.parameters(), lr=2e-3, betas=(0.9, 0.95))
    else:
        dec = [p_ for p_ in m.parameters() if p_.dim() == 2]; nd = [p_ for p_ in m.parameters() if p_.dim() < 2]
        opt = torch.optim.AdamW([dict(params=dec, weight_decay=0.1), dict(params=nd, weight_decay=0.0)], lr=1e-3, betas=(0.9, 0.95))
    warm = 0 if ng else 100
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
    g = torch.Generator(); g.manual_seed(0); it = batches(D['train'], a.bs, a.seq, g)
    t0 = time.time(); log = open(W + f'train_{a.arch}.log', 'w')
    for s in range(1, a.steps + 1):
        x, y = next(it)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(m(x).float().reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True); loss.backward()
        if not ng: torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
        if ng: m.normalize()
        if s % 100 == 0 or s == 1:
            r = dict(step=s, loss=round(float(loss), 4), t=round(time.time() - t0)); print(json.dumps(r), flush=True); log.write(json.dumps(r) + '\n'); log.flush()
    torch.save(m.state_dict(), W + f'{a.arch}.pt'); print('done', time.time() - t0, flush=True)


@torch.no_grad()
def evalq():
    D = torch.load(W + 'data.pt'); out = {}
    for arch in ('std', 'ngpt'):
        if not os.path.exists(W + f'{arch}.pt'): continue
        m = LM(arch == 'ngpt', a.d, a.L).cuda(); m.load_state_dict(torch.load(W + f'{arch}.pt')); m.eval()
        g = torch.Generator(); g.manual_seed(123); it = batches(D['val'], 16, a.seq, g); vb = [next(it) for _ in range(16)]
        res = {}
        for name, qc, rot in (('bf16', None, False), ('int4', 'int4', False), ('int4rot', 'int4', True), ('nvfp4', 'nvfp4', False), ('mxfp4', 'mxfp4', False), ('int8', 'int8', False)):
            QL.QC, QL.ROT = qc, rot
            if name == 'bf16': QL.CREST = {}
            tot = 0.0
            for x, y in vb:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    tot += float(F.cross_entropy(m(x).float().reshape(-1, V), y.reshape(-1)))
            res[name] = tot / len(vb)
            if name == 'bf16':
                res['crest'] = {k: sum(v) / len(v) for k, v in QL.CREST.items()}; QL.CREST = None
        QL.QC = None
        out[arch] = res
        print(arch, {k: (round(v, 4) if isinstance(v, float) else None) for k, v in res.items()}, flush=True)
        cr = res['crest']; print('  mean crest by GEMM:', {t: round(sum(v for k, v in cr.items() if k.endswith(t)) / sum(1 for k in cr if k.endswith(t)), 1) for t in ('qkv', '.o', 'gu', 'down')}, flush=True)
    json.dump(out, open(W + 'evalq.json', 'w'), indent=1)


{'prep': prep, 'train': train, 'evalq': evalq}[a.mode]()

import os, sys, time, torch, torch.nn.functional as F
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encj1 as E
dev = 'cuda'
W, _ = E.load_weights()
m = E.EncTorso(W, lora=True).to(dev); del W
with torch.no_grad():
    for p in m.lora.parameters(): p.normal_(0, 0.01)
R = torch.load('rows_corpus_e.pt')['rows']
rows = [r for r in R if 300 < len(r['ids']) < 900][:3]
lens = [len(r['ids']) for r in rows]; ns = [r['nstate'] for r in rows]
print('lens', lens, 'nstate', ns)
def cosr(a, b): return F.cosine_similarity(a.float(), b.float(), dim=-1)
for mode, local, win in (('masked', (), 0), ('full', (), 0), ('masked', tuple(range(26)), 128)):
    m.mode = mode; m.local = set(local); m.window = win
    meta = E.pack_meta(lens, ns, dev)
    ids = torch.cat([r['ids'].long() for r in rows]).to(dev)
    # padded reference input for per-layer check
    idp = torch.nn.utils.rnn.pad_sequence([r['ids'].long() for r in rows], batch_first=True).to(dev)
    am = (torch.arange(idp.shape[1])[None] < torch.tensor(lens)[:, None]).long().to(dev); qs = torch.tensor(ns, device=dev)
    with torch.no_grad():
        x0p = F.embedding(idp, m.embed) * torch.tensor(E.D ** 0.5, dtype=torch.bfloat16)
        T = idp.shape[1]; cs, sn = E.rope_cs(torch.arange(T, device=dev)[None].expand(len(rows), T), dev)
        mg = m.build_mask(am, qs, False); ml = m.build_mask(am, qs, True) if win else mg
        x0 = torch.cat([x0p[j, :L] for j, L in enumerate(lens)])
        c2, s2 = E.rope_cs(meta['pos'], dev)
        worst = 1.0
        for i in (0, 5, 13, 25):
            yp = m.layer(i, x0p, ml if i in m.local else mg, cs, sn)
            yk = m.layer_packed(i, x0, c2, s2, meta)
            ypf = torch.cat([yp[j, :L] for j, L in enumerate(lens)])
            dd = cosr(yk - x0, ypf - x0)
            worst = min(worst, float(dd.min()))
        # end-to-end (bf16 chaos expected on some rows): readout rows
        hk = m.forward_packed(ids, meta); hp = m(idp, am, qs)
        hpf = torch.cat([hp[j, :L] for j, L in enumerate(lens)])
        last = cosr(hk[meta['last']], hpf[meta['last']])
    print(mode, 'win', win, 'per-layer delta cos min %.6f' % worst, '| e2e last-row cos', [round(float(x), 5) for x in last], 'mean row cos %.5f' % float(cosr(hk, hpf).mean()), flush=True)
# gradient check: packed vs padded on the LoRA grads (masked)
m.mode = 'masked'; m.local = set(); m.window = 0; m.train()
meta = E.pack_meta(lens, ns, dev); ids = torch.cat([r['ids'].long() for r in rows]).to(dev)
hk = m.forward_packed(ids, meta); hk[meta['last']].float().pow(2).sum().backward()
gk = [p.grad.clone() for p in m.lora.parameters()]; m.zero_grad()
hp = m(idp, am, qs); torch.stack([hp[j, L - 1] for j, L in enumerate(lens)]).float().pow(2).sum().backward()
gp = [p.grad.clone() for p in m.lora.parameters()]; m.zero_grad()
cg = [float(F.cosine_similarity(a.flatten().float(), b.flatten().float(), dim=0)) for a, b in zip(gk, gp) if b.abs().sum() > 0]
print('grad cos packed vs padded: min %.4f median %.4f n %d' % (min(cg), sorted(cg)[len(cg) // 2], len(cg)), flush=True)
# speed
opt = torch.optim.AdamW(m.lora.parameters(), lr=1e-6)
def bench(rws, ck, tag):
    ln = [len(r['ids']) for r in rws]; nst = [r['nstate'] for r in rws]
    meta = E.pack_meta(ln, nst, dev); ids = torch.cat([r['ids'].long() for r in rws]).to(dev)
    m.ckpt = ck; torch.cuda.reset_peak_memory_stats()
    for it in range(3):
        torch.cuda.synchronize(); t0 = time.time()
        h = m.forward_packed(ids, meta); h[meta['last']].float().pow(2).mean().backward(); opt.zero_grad()
        torch.cuda.synchronize(); dt = time.time() - t0
    print(tag, 'rows', len(rws), 'tok', sum(ln), 'ckpt', ck, '%.3fs' % dt, 'tok/s %d' % (sum(ln) / dt), 'mem %.1fG' % (torch.cuda.max_memory_allocated() / 2**30), flush=True)
short = [r for r in R if 150 < len(r['ids']) < 260][:16]
mid = sorted(R, key=lambda r: abs(len(r['ids']) - 1000))[:4]
RR = torch.load('rows_real_e.pt')['rows']
long1 = [max(RR, key=lambda r: len(r['ids']))]
l3 = sorted(RR, key=lambda r: abs(len(r['ids']) - 3100))[:1]
for comp in ((True,) if os.environ.get('ONLYC') else (False, True)):
    if comp: E.use_compiled()
    for rws, tag in ((short, 'short'), (mid, 'mid'), (l3, 'l3k'), (long1, 'long')):
        for ck in ((True,) if tag in ('l3k', 'long') else (False, True)):
            try: bench(rws, ck, ('C ' if comp else '') + tag)
            except torch.OutOfMemoryError: print(tag, ck, 'OOM'); torch.cuda.empty_cache()

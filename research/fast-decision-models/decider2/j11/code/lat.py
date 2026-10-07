"""J11 batch-1 latency (A10G), CUDA graphs, exact lengths, fresh inputs each call, median + p95 of >=20 warm reps.
  python lat.py --d 512 --L 12 --arms dec,slot,vslot,belief --T 64,128,256,400,1000,2000,4000,8000 --Q 1,4 --out lat_S.json
dec  : state prefix + Q question rows (questions 125 tokens each) attending to the state's K/V; state rows skip the last layer's
       attention/MLP (only its K/V is read): the efficient decoder runtime for the hobson layout.
slot*: compiled question (stem + option memories, their per-slot-layer K/V and slot inits precomputed once per deployment);
       per request = shallow state encoder + state K/V for every slot layer + the slot core for Q questions batched.
"""
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models as Mo

QLEN = 125; K = 4; OLEN = 12


def dec_fn(m, ids, qids, Q):
    """ids [1,T] state; qids [Q,QLEN] -> logits [Q,K]."""
    cs = m.cs; T = ids.shape[1]
    xs = m.emb(ids); xq = m.emb(qids)
    cq = (cs[0][T:], cs[1][T:])
    mask = torch.ones(QLEN, T + QLEN, dtype=torch.bool, device=ids.device)
    mask[:, T:] = torch.tril(torch.ones(QLEN, QLEN, dtype=torch.bool, device=ids.device))
    nL = len(m.blocks)
    for li, blk in enumerate(m.blocks):
        H = blk.H
        q, k, v = blk.qkv(blk.n1(xs)).view(1, T, 3, H, 64).permute(2, 0, 3, 1, 4)
        q, k = Mo.rope(blk.qn(q), *cs), Mo.rope(blk.kn(k), *cs)
        if li < nL - 1:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            xs2 = xs + blk.o(a.transpose(1, 2).reshape(1, T, -1)); xs2 = xs2 + blk.mlp(blk.n2(xs2))
        qq, kq, vq = blk.qkv(blk.n1(xq)).view(Q, QLEN, 3, H, 64).permute(2, 0, 3, 1, 4)
        qq, kq = Mo.rope(blk.qn(qq), *cq), Mo.rope(blk.kn(kq), *cq)
        Kc = torch.cat([k.expand(Q, -1, -1, -1), kq], 2); Vc = torch.cat([v.expand(Q, -1, -1, -1), vq], 2)
        a = F.scaled_dot_product_attention(qq, Kc, Vc, attn_mask=mask)
        xq = xq + blk.o(a.transpose(1, 2).reshape(Q, QLEN, -1)); xq = xq + blk.mlp(blk.n2(xq))
        if li < nL - 1: xs = xs2
    xq = m.nf(xq)
    return m.head(xq[:, -1], xq[:, QLEN - 1 - K - 3:QLEN - 3][:, :K])


class SlotRT:
    """compiled-question runtime for a SlotModel."""
    def __init__(self, m, Q, dev):
        self.m = m; self.Q = Q; d = m.d
        g = torch.Generator(device="cpu").manual_seed(0)
        qids = torch.randint(2, 1000, (1, QLEN - K * OLEN), generator=g).to(dev)
        oids = torch.randint(2, 1000, (K, OLEN), generator=g).to(dev)
        with torch.inference_mode():
            mq = self.enc(qids, 1); mo = self.enc(oids, 2)  # [1,Tq,d], [K,OLEN,d]
            qm = torch.cat([mq, mo.reshape(1, K * OLEN, d)], 1)
            self.qkv = [blk.mem_kv(qm) for blk in m.slots]  # per layer (k,v) [1,nkv,Tqm,64]
            ans = mq[:, -1] + m.styp.weight[0]; opts = mo[:, -1].unsqueeze(0) + m.styp.weight[1]
            scr = m.scratch.unsqueeze(0) + m.qpool(mq.mean(1)).unsqueeze(1) + m.styp.weight[2]
            self.s0 = torch.cat([ans.unsqueeze(1), opts, scr], 1).expand(Q, -1, -1).contiguous()
        self.P = None
    def enc(self, ids, seg):
        m = self.m; x = m.emb(ids) + m.seg.weight[seg]
        for blk in m.enc: x = blk(x, m.cs)
        return m.nf_e(x)
    def make_P(self, T, dev):
        Tm = T + QLEN - K * OLEN + K * OLEN; g = torch.Generator(device="cpu").manual_seed(1)
        hb = (torch.randint(0, 2, (self.Q, Tm, 64), generator=g).float() * 2 - 1).to(dev)
        v = torch.rand(self.Q, Tm, generator=g).to(dev) * 1000
        self.P = dict(hb=hb, v0=v, mv=torch.ones_like(v), z=torch.log1p(v), t1=torch.zeros(self.Q, Tm, 6, device=dev), ls=torch.ones_like(v))
    def __call__(self, ids):
        m = self.m; Q = self.Q
        ms = self.enc(ids, 0)  # [1,T,d]
        s = self.s0.clone(); mkv = []
        for blk, (kq, vq) in zip(m.slots, self.qkv):
            ks, vs = blk.mem_kv(ms)
            mkv.append((torch.cat([ks, kq], 2).expand(Q, -1, -1, -1), torch.cat([vs, vq], 2).expand(Q, -1, -1, -1)))
        Tm = mkv[0][0].shape[2]; S = s.shape[1]
        mask = torch.ones(Q, 1, S, Tm + S, dtype=torch.bool, device=ids.device)
        oval = torch.ones(Q, K, dtype=torch.bool, device=ids.device)
        return m.run_slots(s, mkv, mask, self.P, oval, K)[-1]


def timeit(fn, make_input, static, reps=20, warm=5):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3): out = fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g): out = fn()
    ts = []
    for i in range(warm + reps):
        x = make_input()
        torch.cuda.synchronize(); t = time.perf_counter()
        static.copy_(x, non_blocking=True); g.replay(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    ts = np.array(ts[warm:]); return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--d", type=int); ap.add_argument("--L", type=int); ap.add_argument("--arms")
    ap.add_argument("--T", default="64,128,256,400,1000,2000,4000,8000"); ap.add_argument("--Q", default="1,4"); ap.add_argument("--out")
    ap.add_argument("--R", type=int, default=3)
    a = ap.parse_args(); dev = "cuda"; res = []
    for arm in a.arms.split(","):
        m = Mo.build(arm, 100000, a.d, a.L, R=a.R).to(dev).to(torch.bfloat16).eval()
        m.cs = Mo.rope_cache(20000, dev)
        for Q in map(int, a.Q.split(",")):
            rt = None if arm in ("dec", "decv") else SlotRT(m, Q, dev)
            for T in map(int, a.T.split(",")):
                static = torch.randint(2, 50000, (1, T), device=dev)
                if arm in ("dec", "decv"):
                    qids = torch.randint(2, 50000, (Q, QLEN), device=dev)
                    fn = lambda: dec_fn(m, static, qids, Q)
                    fl = Mo.fwd_flops("dec", a.d, a.L, T, QLEN, K) + (Q - 1) * Mo.fwd_flops("dec", a.d, a.L, 0, QLEN, K)
                else:
                    if arm == "vslot": rt.make_P(T, dev)
                    fn = lambda: rt(static)
                    fl = Mo.fwd_flops(arm, a.d, a.L, T, QLEN, K, R=a.R, compiled_q=True, To=[OLEN] * K, Tqp=QLEN - K * OLEN)
                    fl += (Q - 1) * (Mo.fwd_flops(arm, a.d, a.L, 0, QLEN, K, R=a.R, compiled_q=True, To=[OLEN] * K, Tqp=QLEN - K * OLEN))
                mk = lambda: torch.randint(2, 50000, (1, T), device="cpu").pin_memory()
                try:
                    med, p95 = timeit(fn, mk, static)
                except torch.OutOfMemoryError:
                    med, p95 = None, None
                r = dict(arm=arm, d=a.d, L=a.L, T=T, Q=Q, ms=med, p95=p95, gflops=fl / 1e9)
                print(json.dumps(r), flush=True); res.append(r)
                torch.cuda.empty_cache()
        del m; torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

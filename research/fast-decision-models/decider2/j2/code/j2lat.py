"""J2 latency: the bidirectional GDN hybrid in d1's fused runtime (lean2: folded norms, fused conv+l2norm, gated-norm kernel, SwiGLU epilogue,
residual-add epilogues), CUDA graph per exact shape, exclusive GPU, fresh random ids every rep, 3 warm-ups discarded, median + p95.
Timed span = H2D of the ids + graph replay + D2H of the final hidden rows read by the head.

Layouts (T = exact state tokens, Lq = question tokens per question):
  hob  : hobson (causal). 1 question = one sequence [S;Q]; M questions = state once + M branches continuing the state (shared prefix).
  bi   : J2 'qag' (masked state cache): + reverse GDN scans in REV layers (state reversed, then each question reversed starting from the state's
         reverse final state), + attention o = o_c + lam (o_nc - o_c) (state<->state full; question -> state + whole question). M questions share the state.
  bi_nc: as bi with pure non-causal attention (lam = 1 folded: one attention pass).
  qa   : J2 'qa' (question-aware): every question needs its own full bidirectional pass over [S;Q] (M rows batched).
python j2lat.py time [out.json]  |  python j2lat.py check REF.pt"""
import os, sys, json, time, statistics as st
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/j2')]
import numpy as np, torch, torch.nn.functional as F
from lean2 import Lean2, tgemm, conv_l2, gnorm, attn_prep, LM
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

dev = 'cuda'
GDN = tuple(i for i in range(24) if i not in (3, 7, 11, 15, 19, 23))
FUSE = 'addrms,gnorm,silu,prep,conv,gemm_swiglu,fold'


def gdr(q, k, v, a, b, d, S0=None, fin=False):
    return chunk_gated_delta_rule(q, k, v, a, b, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'],
                                  use_beta_sigmoid_in_kernel=True, initial_state=S0, output_final_state=fin)


class BL(Lean2):
    def setup(self, rev=(), nc='none', gam=0.5, lam=0.5):
        """rev: GDN layers with reverse scans. nc: 'none' (causal), 'mix' (o_c + lam (o_nc - o_c)), 'nc' (pure non-causal)."""
        self.rev = set(rev); self.nc = nc
        self.gam = {i: torch.full((16, 128), gam, device=dev, dtype=torch.bfloat16) for i in self.rev}
        self.lamv = torch.full((1, 8, 1, 1), lam, device=dev, dtype=torch.bfloat16)

    @torch.no_grad()
    def fwd(self, ids, T, M, Lq, qa=False):
        """ids [1, N]. not qa: N = T + M*Lq (state then M branches). qa: N = M*(T+Lq) (M full rows). -> final hidden rows [*, 2048]"""
        N = ids.shape[1]
        x = F.embedding(ids[0], self.embed)
        if qa:
            R = M; L = T + Lq
            pos = torch.arange(L, device=dev, dtype=torch.float32).repeat(M)
        else:
            pos = torch.cat([torch.arange(T, device=dev, dtype=torch.float32), (T + torch.arange(Lq, device=dev, dtype=torch.float32)).repeat(M)])
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        ss = x.float().pow(2).sum(-1)
        if not qa and M > 0:
            # branch conv input: [3 tail state rows + Lq branch rows] per branch
            gi = torch.cat([torch.cat([torch.arange(T - 3, T, device=dev), T + m * Lq + torch.arange(Lq, device=dev)]) for m in range(M)])
        for i, d in enumerate(self.layers):
            proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1])
            if d['type'] == 'linear_attention':
                z = proj[:, 6144:8192]
                if qa:
                    qkv3 = conv_l2(proj, d['conv_w'])                                    # (rows are equal length; 3-token leak across rows ignored)
                    q, k, v = (qkv3[j].reshape(R, L, 16, 128) for j in range(3))
                    a_ = proj[:, 8208:8224].reshape(R, L, 16); b_ = proj[:, 8192:8208].reshape(R, L, 16)
                    o, _ = gdr(q, k, v, a_, b_, d)
                    if i in self.rev:
                        orv, _ = gdr(q.flip(1), k.flip(1), v.flip(1), a_.flip(1), b_.flip(1), d)
                        o = o + orv.flip(1) * self.gam[i]
                    o = o.reshape(N, 16, 128)
                else:
                    ps = proj[:T]
                    qkv3 = conv_l2(ps, d['conv_w'])
                    q, k, v = qkv3[0][None], qkv3[1][None], qkv3[2][None]
                    a_ = ps[:, 8208:8224].reshape(1, T, 16); b_ = ps[:, 8192:8208].reshape(1, T, 16)
                    o_s, Sf = gdr(q, k, v, a_, b_, d, fin=M > 0)
                    if i in self.rev:
                        orv, Sr = gdr(q.flip(1), k.flip(1), v.flip(1), a_.flip(1), b_.flip(1), d, fin=M > 0)
                        o_s = o_s + orv.flip(1) * self.gam[i]
                    outs = [o_s.reshape(T, 16, 128)]
                    if M > 0:
                        pb = proj[gi]                                                        # [M*(3+Lq), 8224]
                        cb = conv_l2(pb, d['conv_w']).reshape(3, M, 3 + Lq, 16, 128)[:, :, 3:]
                        qb, kb, vb = cb[0], cb[1], cb[2]
                        pr = proj[T:].reshape(M, Lq, 8224)
                        ab = pr[..., 8208:8224]; bb = pr[..., 8192:8208]
                        o_b, _ = gdr(qb, kb, vb, ab, bb, d, S0=Sf.expand(M, -1, -1, -1).contiguous())
                        if i in self.rev:
                            orb, _ = gdr(qb.flip(1), kb.flip(1), vb.flip(1), ab.flip(1), bb.flip(1), d, S0=Sr.expand(M, -1, -1, -1).contiguous())
                            o_b = o_b + orb.flip(1) * self.gam[i]
                        outs.append(o_b.reshape(M * Lq, 16, 128))
                    o = torch.cat(outs, 0) if len(outs) > 1 else outs[0]
                o = gnorm(o.contiguous(), z, d['gn_w'], self.eps)
            else:
                qh, kh, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                vh = proj[:, 4608:5120].reshape(N, 2, 256)
                if qa:
                    Q_ = qh.reshape(R, L, 8, 256).transpose(1, 2); K_ = kh.reshape(R, L, 2, 256).transpose(1, 2); V_ = vh.reshape(R, L, 2, 256).transpose(1, 2)
                    if self.nc == 'nc':
                        oa = F.scaled_dot_product_attention(Q_, K_, V_, enable_gqa=True)
                    else:
                        oa = F.scaled_dot_product_attention(Q_, K_, V_, is_causal=True, enable_gqa=True)
                        if self.nc == 'mix':
                            on = F.scaled_dot_product_attention(Q_, K_, V_, enable_gqa=True); oa = oa + self.lamv * (on - oa)
                    o = oa.transpose(1, 2).reshape(N, 2048)
                else:
                    Qs = qh[:T].reshape(1, T, 8, 256).transpose(1, 2); Ks = kh[:T].reshape(1, T, 2, 256).transpose(1, 2); Vs = vh[:T].reshape(1, T, 2, 256).transpose(1, 2)
                    if self.nc == 'nc':
                        os_ = F.scaled_dot_product_attention(Qs, Ks, Vs, enable_gqa=True)
                    else:
                        os_ = F.scaled_dot_product_attention(Qs, Ks, Vs, is_causal=True, enable_gqa=True)
                        if self.nc == 'mix':
                            on = F.scaled_dot_product_attention(Qs, Ks, Vs, enable_gqa=True); os_ = os_ + self.lamv * (on - os_)
                    outs = [os_.transpose(1, 2).reshape(T, 2048)]
                    if M > 0:
                        Qb = qh[T:].reshape(M, Lq, 8, 256).transpose(1, 2)
                        Kb = torch.cat([Ks.expand(M, -1, -1, -1), kh[T:].reshape(M, Lq, 2, 256).transpose(1, 2)], 2)
                        Vb = torch.cat([Vs.expand(M, -1, -1, -1), vh[T:].reshape(M, Lq, 2, 256).transpose(1, 2)], 2)
                        if self.nc == 'nc':
                            ob = F.scaled_dot_product_attention(Qb, Kb, Vb, enable_gqa=True)
                        else:
                            ob = F.scaled_dot_product_attention(Qb, Kb, Vb, attn_mask=self.bmask, enable_gqa=True)
                            if self.nc == 'mix':
                                on = F.scaled_dot_product_attention(Qb, Kb, Vb, enable_gqa=True); ob = ob + self.lamv * (on - ob)
                        outs.append(ob.transpose(1, 2).reshape(M * Lq, 2048))
                    o = torch.cat(outs, 0) if len(outs) > 1 else outs[0]
                o = o * gate
            ss = torch.zeros(N, device=dev, dtype=torch.float32)
            tgemm(o.contiguous(), d['Wo'], epi=3, res=x, ssout=ss)
            mm = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
            ss = torch.zeros(N, device=dev, dtype=torch.float32)
            tgemm(mm, d['Wd'], epi=3, res=x, ssout=ss)
        return LM._rms_zc(x, self.norm_w, self.eps)

    def set_bmask(self, T, Lq):
        j = torch.arange(T + Lq, device=dev)[None, :]; i_ = torch.arange(Lq, device=dev)[:, None]
        self.bmask = ((j < T) | ((j - T) <= i_))[None, None]


def med(ts):
    ts = sorted(ts)
    return dict(median=round(st.median(ts), 2), p95=round(ts[int(0.95 * (len(ts) - 1))], 2), min=round(ts[0], 2), n=len(ts))


def capture(fn):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.inference_mode():
        for _ in range(3): fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode(), torch.cuda.graph(g): out = fn()
    return g, out


def timed(g, buf, out, n=20, vocab=150000):
    rng = np.random.default_rng(0)
    pins = [torch.from_numpy(rng.integers(1000, vocab, size=buf.shape[1], dtype=np.int64)).pin_memory() for _ in range(n + 3)]
    host = torch.empty(out.shape, dtype=out.dtype).pin_memory(); ts = []
    for x in pins:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        buf[0].copy_(x, non_blocking=True); g.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return med(ts[3:])


def load():
    from plib import P
    p = P()
    rt = BL(p.tm, fuse=FUSE)
    p.model.torso = None; p.tm = None
    import gc; gc.collect(); torch.cuda.empty_cache()
    return rt


def main_time(out):
    rt = load()
    Lq = 100
    TS = [int(x) for x in os.environ.get('TS', '64,128,256,400,1000,4000').split(',')]
    cfgs = [('hob', (), 'none', False), ('bi_all', GDN, 'mix', False), ('bi_all_nc', GDN, 'nc', False), ('bi_early', tuple(i for i in GDN if i <= 10), 'mix', False),
            ('qa_all', GDN, 'mix', True)]
    res = json.load(open(out)) if os.path.exists(out) else {}
    for T in TS:
        for M in (1, 4):
            for name, rev, nc, qa in cfgs:
                key = f'{name}|T{T}|M{M}'
                if key in res: continue
                rt.setup(rev, nc)
                if qa:
                    N = M * (T + Lq)
                    rows = torch.tensor([r * (T + Lq) + T + Lq - 1 for r in range(M)], device=dev)
                else:
                    N = T + M * Lq
                    rows = torch.tensor([T + m * Lq + Lq - 1 for m in range(M)], device=dev)
                    rt.set_bmask(T, Lq)
                buf = torch.randint(1000, 150000, (1, N), device=dev)
                Mb = 0 if (not qa and M == 1) else M
                Tq = T + Lq if (not qa and M == 1) else T
                # M == 1 non-qa: one plain sequence [S;Q] (state T + question Lq), like hobson's single-question path
                if not qa and M == 1:
                    fn = lambda: rt.fwd(buf, T + Lq, 0, Lq)[rows]
                else:
                    fn = lambda: rt.fwd(buf, T, M, Lq, qa=qa)[rows]
                g, o = capture(fn)
                r = timed(g, buf, o)
                res[key] = r
                print(key, r, flush=True)
                del g, o; torch.cuda.empty_cache()
                json.dump(res, open(out, 'w'), indent=1)


def main_check():
    ref = torch.load(os.path.expanduser('~/work/j2/latref.pt'))
    rt = load(); ids = torch.tensor([ref['ids']], device=dev); T = ids.shape[1]
    for mode, rev, nc in (('causal', (), 'none'), ('qa', GDN, 'mix')):
        rt.setup(rev, nc, gam=0.5, lam=0.5)
        h = rt.fwd(ids, T, 0, 0).float().cpu(); r = ref[mode]
        cos = F.cosine_similarity(h, r, dim=-1)
        print(f'{mode}: fused runtime vs j2lib: min row cosine {float(cos.min()):.5f}, last row cosine {float(cos[-1]):.5f}, max|d| last row {float((h[-1] - r[-1]).abs().max()):.3f}', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'check': main_check()
    if sys.argv[1] == 'time':
        main_time(sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser('~/work/j2/lat.json'))

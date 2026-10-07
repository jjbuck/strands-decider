"""J7 latency in the fused bf16 runtime (H4's TTL = d1 lean2 fusions + cached-history conv + lower-right causal flash), hobson's layout:
state pass once (with cache), then each question as a branch continuing the state cache; pointer head on option rows + '<answer>'.
Reader-native input path inside the timed CUDA graph: extended-vocabulary gather (248k Qwen + 64k super rows), RoPE at ORIGINAL positions,
and (optionally) the side channels as 9 extra gathers + adds from folded tables (R @ P precomputed: [65536, 2048] each).
Configs: qwen (hobson) at exact T state tokens; super-16k / super-64k at the same content = fresh real states merged and cut to exactly
T' = round(T / r) super-tokens (r = REAL-agree median state compression), questions merged; +chan.
Discipline: CUDA graph per exact shape, exclusive GPU, fresh real states every rep (3 warm-ups discarded, 20 timed), median + p95; timed span =
H2D of ids/positions/features + graph replay + D2H of the probabilities.
python lat_j7.py OUT.json"""
import os, sys, json, time, statistics as st
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.expanduser('~/work/j7'), os.path.expanduser('~/work/h4'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens')]
import numpy as np, torch, torch.nn.functional as F
from tt_lean import TTL, tgemm, conv_l2, conv_l2_tail, gnorm, attn_prep, add_rms, LM, chunk_gated_delta_rule
from torch.nn.attention.bias import causal_lower_right
import evalkit as EK, chan
from superbpe import Super
from j7lib import Prep, CH_KEYS
from strands_decider.prompting import render_state
dev = 'cuda'
REPS = int(os.environ.get('REPS', 20)); TS = [int(x) for x in os.environ.get('TS', '64,128,256,400,1000,4000').split(',')]
RATIO = {'16k': 1.95, '64k': 2.41}       # REAL-agree median state compression (measured, superbpe stat)
Q4 = ['cc_asked_for_human', 'cc_insists', 'cc_offers_transfer', 'cc_refuses']


class RJ(TTL):
    tab = None; ch = None

    @torch.no_grad()
    def fwd(self, ids, pos0=0, cache=None, want_cache=False, pos=None, feats=None):
        f = self.fuse
        B, T = ids.shape
        x = F.embedding(ids, self.tab if self.tab is not None else self.embed).reshape(T, -1)
        if feats is not None:
            Tv, Tk, Tr, Ep, Er, Et = self.ch
            x = x + Tv[feats[0]] + Tv[feats[1]] + Tk[feats[2]] + Tk[feats[3]] + Tr[feats[4]] + Tr[feats[5]] + Ep[feats[6]] + Er[feats[7]] + Et[feats[8]]
        if pos is None: pos = torch.arange(pos0, pos0 + T, device=self.dev, dtype=torch.float32)
        fr = pos[:, None] * self.inv[None, :]; fr = torch.cat([fr, fr], -1)
        cos, sin = fr.cos().to(x.dtype).contiguous(), fr.sin().to(x.dtype).contiguous()
        fold = 'fold' in f
        newc = {} if want_cache else None
        if fold: ss = x.float().pow(2).sum(-1)
        else: h = LM._rms_zc(x, self.layers[0]['in_norm'], self.eps)
        n = len(self.layers)
        for i, d in enumerate(self.layers):
            c = cache[i] if cache is not None else None
            proj = tgemm(x, d['Win_f'], epi=0, ss=ss, Kd=x.shape[1]) if fold else self._mm(h, d['Win'])
            if d['type'] == 'linear_attention':
                qkv3 = conv_l2_tail(proj, c['tail'], d['conv_w']) if c is not None else conv_l2(proj, d['conv_w'])
                q, k, v = qkv3[0][None], qkv3[1][None], qkv3[2][None]
                a = proj[:, 8208:8224].reshape(1, T, 16); b = proj[:, 8192:8208].reshape(1, T, 16)
                o, S = chunk_gated_delta_rule(q, k, v, a, b, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=True, A_log=d['A_log'], dt_bias=d['dt_bias'],
                                              use_beta_sigmoid_in_kernel=True, initial_state=None if c is None else c['S'], output_final_state=want_cache)
                if want_cache: newc[i] = {'tail': proj[-3:].contiguous().clone(), 'S': S.clone()}
                o = gnorm(o.reshape(T, 16, 128), proj[:, 6144:8192], d['gn_w'], self.eps)
            else:
                q, k, gate = attn_prep(proj, d['qn'], d['kn'], cos, sin, self.eps)
                v = proj[:, 4608:5120].reshape(T, 2, 256)
                if want_cache: newc[i] = {'k': k.clone(), 'v': v.contiguous().clone()}
                qh = q.reshape(1, T, 8, 256).transpose(1, 2)
                if c is not None:
                    kk = torch.cat([c['k'], k], 0); vv = torch.cat([c['v'], v], 0)
                    o = F.scaled_dot_product_attention(qh, kk.reshape(1, -1, 2, 256).transpose(1, 2), vv.reshape(1, -1, 2, 256).transpose(1, 2), attn_mask=self.mask, enable_gqa=True)
                else:
                    o = F.scaled_dot_product_attention(qh, k.reshape(1, T, 2, 256).transpose(1, 2), v.reshape(1, T, 2, 256).transpose(1, 2), is_causal=True, enable_gqa=True)
                o = o.transpose(1, 2).reshape(T, 2048) * gate
            if fold:
                ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
                tgemm(o, d['Wo'], epi=3, res=x, ssout=ss)
                m = tgemm(x, d['Wgu_f'], epi=1, ss=ss, Kd=x.shape[1])
                ss = torch.zeros(T, device=self.dev, dtype=torch.float32)
                tgemm(m, d['Wd'], epi=3, res=x, ssout=ss)
                continue
            d_out = self._mm(o, d['Wo'])
            x, h2 = add_rms(x, d_out, d['post1'], self.eps)
            m = tgemm(h2, d['Wgu_il'], epi=1, cfg=self.gcfg.get('swiglu'))
            d_mlp = self._mm(m, d['Wd'])
            x, h = add_rms(x, d_mlp, (self.layers[i + 1]['in1'] if i + 1 < n else self.norm1), self.eps)
        if fold: h = LM._rms_zc(x, self.norm_w, self.eps)
        return h.reshape(T, -1), newc


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


def chan_idx(fe):
    """9 index arrays from the channel features (same hashing as j7lib.Channels)"""
    out = []
    for k in ('val', 'key', 'rec'):
        h = fe[k]; a = h % 65536; b = (h >> 20) % 65536; z = h == 0
        out += [np.where(z, 0, a), np.where(z, 0, b)]
    out += [np.clip(fe['place'], 0, 31), np.clip(fe['role'], 0, 47), np.clip(fe['turn'], 0, 31)]
    return out


def main():
    from kitrun import load_P
    from h3lib import StdHead
    Pm = load_P(); tm = Pm.tm; eng = Pm.eng
    head = StdHead(Pm.model.head).to(dev).eval()
    rt = RJ(tm, list(range(10)))
    W = os.path.expanduser('~/work/')
    S64 = Super(W + 'j7/tok/sb64k.json', tok=eng.tok)
    S16 = Super(W + 'j7/tok/sb16k.json', tok=eng.tok); S16.idx = {t: S64.V + i for i, t in enumerate(S16.toks)}
    V = S64.V
    g = torch.Generator(device='cpu'); g.manual_seed(0)
    ext = torch.cat([rt.embed[:V], (torch.randn(len(S64.toks), 2048, generator=g) * 0.015).to(dev, torch.bfloat16)], 0).contiguous()
    rt.ch = [(torch.randn(65536, 2048, generator=g) * 0.002).to(dev, torch.bfloat16) for _ in range(3)] + \
            [(torch.randn(n, 2048, generator=g) * 0.002).to(dev, torch.bfloat16) for n in (32, 48, 32)]
    Pp = Prep(eng, {'16k': S16, '64k': S64})
    spec = {}; pool = []
    for s_ in ('REAL-agree', 'LONG'):
        for it in EK.load_suite(s_):
            for q, sp in it['questions'].items(): spec.setdefault(q, sp)
            if it.get('domain') == 'banking_knowledge': pool.append(it['state'])
    texts = [render_state(x) for x in pool]
    toks = [eng.tok(t, add_special_tokens=True)['input_ids'] for t in texts]
    res = {'meta': dict(gpu=torch.cuda.get_device_name(0), reps=REPS, torch=torch.__version__, ratio=RATIO, note='hobson-v19 weights; fused TTL runtime; plain layout')}
    OUT = sys.argv[1]
    temp = Pm.temp_for

    def qprep(names, level):
        out = []
        for k in names:
            pr = Pp.base('S', spec[k])
            b = Pp.build(pr, level=level, with_feats=True)
            qi = b['ids'][b['q0']:]; qf = {kk: v[b['q0']:] for kk, v in b['feats'].items()}
            qpos = [p - b['pos'][b['q0']] for p in b['pos'][b['q0']:]]
            out.append(dict(ids=qi, pos=qpos, feats=chan_idx(qf), opt=[o - b['q0'] for o in b['opt']], kind=pr['rq'].kind, n=pr['rq'].n_slots))
        return out

    for T in TS:
        for qn, names in ((('Q1s', Q4[:1]),) if os.environ.get('SINGLE') == '1' else (('Q1', Q4[:1]), ('Q4', Q4))):
            for cfg in ('qwen', 'qwen+chan', '16k', '64k', '64k+chan'):
                level = None if cfg.startswith('qwen') else cfg.split('+')[0]
                Ts = T if level is None else max(8, int(round(T / RATIO[level])))
                inputs = []
                for t_ids, tx in zip(toks, texts):
                    if len(t_ids) < T + 8: continue
                    if level is None:
                        ids = t_ids[:Ts]; pos = list(range(Ts)); fe = None
                    else:
                        mi, ends = Pp.supers[level].merge_ids(t_ids)
                        if len(mi) < Ts: continue
                        ids = mi[:Ts]; pos = ends[:Ts]; fe = None
                    if cfg.endswith('chan'):
                        enc = eng.tok(tx, add_special_tokens=True, return_offsets_mapping=True)
                        F_ = chan.token_feats(tx, enc['offset_mapping'])
                        ends_ = pos
                        fe = chan_idx({k: F_[k][np.array(ends_)] for k in CH_KEYS})
                    inputs.append((ids, pos, fe))
                    if len(inputs) == REPS + 3: break
                assert len(inputs) == REPS + 3, (T, cfg, len(inputs))
                qs = qprep(names, level)
                rt.tab = ext if level is not None else None
                rt.set_fuse(Ts)
                buf = torch.zeros(1, Ts, dtype=torch.long, device=dev); pbuf = torch.zeros(Ts, dtype=torch.float32, device=dev)
                fbuf = torch.zeros(9, Ts, dtype=torch.long, device=dev)
                qids = [torch.tensor([q['ids']], device=dev) for q in qs]
                qfe = [torch.tensor(np.stack(q['feats']), device=dev) for q in qs]
                qpos = [torch.tensor(q['pos'], device=dev, dtype=torch.float32) for q in qs]
                masks = [causal_lower_right(len(q['ids']), Ts + len(q['ids'])) for q in qs]
                oidx = [torch.tensor(q['opt'], device=dev) for q in qs]
                use_f = cfg.endswith('chan')

                if qn == 'Q1s':
                    q = qs[0]; Lq = len(q['ids']); Tt = Ts + Lq
                    sbuf = torch.zeros(1, Tt, dtype=torch.long, device=dev); spbuf = torch.zeros(Tt, dtype=torch.float32, device=dev)
                    sfbuf = torch.zeros(9, Tt, dtype=torch.long, device=dev)
                    sbuf[0, Ts:] = qids[0][0]; sfbuf[:, Ts:] = qfe[0]
                    oi_s = oidx[0] + Ts

                    def req():
                        sbuf[0, :Ts].copy_(buf[0]); spbuf[:Ts].copy_(pbuf); spbuf[Ts:].copy_(qpos[0] + pbuf[-1] + 1)
                        if use_f: sfbuf[:, :Ts].copy_(fbuf)
                        rt.set_fuse(Tt)
                        h, _ = rt.fwd(sbuf, pos=spbuf, feats=sfbuf if use_f else None)
                        lg = (head.k(head.norm(h[oi_s].float())) @ head.q(head.norm(h[-1].float()))) * head.scale / temp(q['kind'])
                        return torch.softmax(lg[:q['n']], -1)
                else:
                  def req():
                    hs, cs = rt.fwd(buf, want_cache=True, pos=pbuf, feats=fbuf if use_f else None)
                    outs = []
                    last = pbuf[-1]
                    for j, q in enumerate(qs):
                        rt.mask = masks[j]; rt.set_fuse(len(q['ids']))
                        hq, _ = rt.fwd(qids[j], cache=cs, pos=qpos[j] + last + 1, feats=qfe[j] if use_f else None)
                        lg = (head.k(head.norm(hq[oidx[j]].float())) @ head.q(head.norm(hq[-1].float()))) * head.scale / temp(q['kind'])
                        outs.append(torch.softmax(lg[:q['n']], -1))
                    rt.set_fuse(Ts)
                    return torch.cat(outs)
                pins = []
                for ids, pos, fe in inputs:
                    pins.append((torch.tensor(ids, dtype=torch.long).pin_memory(), torch.tensor(pos, dtype=torch.float32).pin_memory(),
                                 torch.tensor(np.stack(fe), dtype=torch.long).pin_memory() if fe is not None else None))
                buf[0].copy_(pins[0][0]); pbuf.copy_(pins[0][1])
                if pins[0][2] is not None: fbuf.copy_(pins[0][2])
                gr, out = capture(req)
                host = torch.empty(out.shape, dtype=out.dtype).pin_memory(); ts = []
                for x, p, fe in pins:
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    buf[0].copy_(x, non_blocking=True); pbuf.copy_(p, non_blocking=True)
                    if fe is not None: fbuf.copy_(fe, non_blocking=True)
                    gr.replay(); host.copy_(out, non_blocking=True); torch.cuda.synchronize()
                    ts.append((time.perf_counter() - t0) * 1000)
                r = dict(med(ts[3:]), state_rows=Ts, q_rows=[len(q['ids']) for q in qs])
                res[f'T{T}_{qn}_{cfg}'] = r
                print(T, qn, cfg, r, flush=True)
                del gr, out; torch.cuda.empty_cache()
                json.dump(res, open(OUT, 'w'), indent=1)
    json.dump(res, open(OUT, 'w'), indent=1)


def check(ckpt, level, preds_path, n_items=24):
    """fused-runtime path (LoRA merged into the torso, folded super/channel tables) vs the j7lib eval predictions of the same checkpoint."""
    from kitrun import load_P
    from h3lib import StdHead
    import h7lat
    Pm = load_P(); tm = Pm.tm; eng = Pm.eng
    hsd = h7lat.merge_lora(tm, ckpt)
    head = StdHead(Pm.model.head).to(dev).eval(); head.load_state_dict(hsd)
    rt = RJ(tm, list(range(10)))
    W = os.path.expanduser('~/work/')
    S64 = Super(W + 'j7/tok/sb64k.json', tok=eng.tok)
    sups = {}
    if level:
        S = Super(W + f'j7/tok/sb{level}.json', tok=eng.tok); S.idx = {t: S64.V + i for i, t in enumerate(S.toks)}; sups[level] = S
    sd = torch.load(ckpt, map_location=dev)
    use_f = any(k.startswith('chans.') for k in sd)
    if 'sup.A' in sd:
        from j7lib import SuperEmb
        se = SuperEmb(rt.embed, S64.toks, dev)
        with torch.no_grad(): se.A.copy_(sd['sup.A']); se.delta.copy_(sd['sup.delta'].float())
        se.freeze_table(); rt.tab = torch.cat([rt.embed[:S64.V], se.table], 0).contiguous()   # super ids start at len(tokenizer), not at the padded table size
    if use_f:
        from j7lib import Channels
        ch = Channels(dev, float(rt.embed.float().pow(2).mean().sqrt()))
        ch.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('chans.')}, strict=False)
        with torch.no_grad():
            sc = ch.scale
            rt.ch = [((ch.R @ ch.P[k]) * sc).to(torch.bfloat16) for k in ('val', 'key', 'rec')] + [(ch.E[k] * sc).to(torch.bfloat16) for k in ('place', 'role', 'turn')]
    Pp = Prep(eng, sups)
    ref = json.load(open(preds_path))
    its = [it for it in EK.load_suite('CF') if it['id'] in ref][:n_items]
    temp = Pm.temp_for; res = []
    for it in its:
        st_ = render_state(it['state'])
        for qn, qd in it['questions'].items():
            pr = Pp.base(st_, qd); b = Pp.build(pr, level=level or None, with_feats=use_f)
            ids = torch.tensor([b['ids']], device=dev); pos = torch.tensor(b['pos'], device=dev, dtype=torch.float32)
            fe = torch.tensor(np.stack(chan_idx(b['feats'])), device=dev) if use_f else None
            rt.set_fuse(len(b['ids']))
            h, _ = rt.fwd(ids, pos=pos, feats=fe)
            oi = torch.tensor(b['opt'], device=dev)
            lg = (head.k(head.norm(h[oi].float())) @ head.q(head.norm(h[-1].float()))) * head.scale / temp(pr['rq'].kind)
            p = torch.softmax(lg[:pr['rq'].n_slots], -1).tolist()
            r = [ref[it['id']][qn][l] for l in pr['rq'].slot_labels]
            res.append((int(np.argmax(p) == np.argmax(r)), max(abs(x - y) for x, y in zip(p, r))))
    out = dict(ckpt=ckpt, level=level, n=len(res), agree=sum(x[0] for x in res), dp_med=float(np.median([x[1] for x in res])), dp_max=float(max(x[1] for x in res)))
    print('fused runtime vs j7lib', out, flush=True)
    return out


if __name__ == '__main__':
    if sys.argv[1] == 'check':
        r = check(sys.argv[2], sys.argv[3] if sys.argv[3] != '-' else '', sys.argv[4])
        json.dump(r, open(sys.argv[5], 'w'))
    else:
        main()

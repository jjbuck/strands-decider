"""G3 synthetic MoE stack for the batch-1 prefill latency floor (coordinator request).
moe:   L24 x d1024, H16 (hd 64) / KV4, E=64 top-8, ffn_e=448  -> body total ~2.2B, active ~0.33B; 125 rows/expert at 1000 tokens
dense: same attention, dense FFN 3584 (= 8 x 448: the active-equivalent dense twin, Qwen3.5-0.8B's FFN width)
usage: python synth.py [--T 1000 4000] [--E 64 --k 8 --ffn 448] [--skew CV]  (run with box.sh --timing)
Random weights (std 0.02); routing from the random router on the real hidden states (near-balanced) or synthetic skewed routing with a target load CV.
"""
import os, sys, json, argparse, random, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe as M, moe_lean as ML
ap = argparse.ArgumentParser()
ap.add_argument("--T", nargs="+", type=int, default=[1000, 4000]); ap.add_argument("--L", type=int, default=24); ap.add_argument("--d", type=int, default=1024)
ap.add_argument("--H", type=int, default=16); ap.add_argument("--KV", type=int, default=4); ap.add_argument("--E", type=int, default=64)
ap.add_argument("--k", type=int, default=8); ap.add_argument("--ffn", type=int, default=448); ap.add_argument("--skew", type=float, nargs="*", default=[0.0, 0.5])
ap.add_argument("--bits", default="16,8,4"); ap.add_argument("--dense", type=int, default=1); ap.add_argument("--tag", default="synth")
a = ap.parse_args()


def mk(E, k, ffn):
    g = M.Cfg(kind="synthetic", d=a.d, L=a.L, H=a.H, KV=a.KV, hd=a.d // a.H, E=E, k=k, ffn=ffn, act="silu", gate_fn="softmax", router_pre=False,
              emb_mult=1.0, res_mult=1.0, attn_scale=(a.d // a.H) ** -0.5, theta=1e6, eps=1e-6, vocab=32000, hidden_size=a.d)
    torch.manual_seed(0)
    R = lambda *s: (torch.randn(*s, device="cuda") * 0.02).to(torch.bfloat16)
    W = dict(embed=R(32000, a.d), norm=torch.ones(a.d, device="cuda", dtype=torch.bfloat16), layers=[])
    for _ in range(a.L):
        W["layers"].append(dict(ln1=torch.ones(a.d, device="cuda", dtype=torch.bfloat16), ln2=torch.ones(a.d, device="cuda", dtype=torch.bfloat16),
                                Wq=R(a.H * g.hd, a.d), Wk=R(a.KV * g.hd, a.d), Wv=R(a.KV * g.hd, a.d), Wo=R(a.d, a.H * g.hd), Wr=R(E, a.d),
                                Wgu=R(E, 2 * ffn, a.d), Wd=R(E, a.d, ffn)))
    return g, W


def skew_route(T, E, k, cv, seed=0):
    """each token picks k distinct experts; expert popularity ~ lognormal with the target CV of per-expert load"""
    if cv <= 0: return None
    gen = torch.Generator(device="cpu").manual_seed(seed)
    sig = (torch.log(torch.tensor(1 + cv * cv))).sqrt()
    pop = torch.exp(torch.randn(E, generator=gen) * sig)
    idx = torch.multinomial(pop.expand(T, E), k, replacement=False, generator=gen)
    return idx.reshape(-1).cuda()


res = dict(tag=a.tag, cfg=vars(a), gpu=torch.cuda.get_device_name(0))
rows = []
for E, k, ffn, name in ([(a.E, a.k, a.ffn, "moe")] + ([(1, 1, a.k * a.ffn, "dense")] if a.dense else [])):
    g, W = mk(E, k, ffn)
    tot, act = M.body_params(g)
    for T in a.T:
        pool = [torch.randint(0, 32000, (1, T)).pin_memory() for _ in range(23)]
        ids = pool[0].cuda()
        for bits in ([16] if E == 1 else [int(b) for b in a.bits.split(",")]):
            for cv in ([0.0] if E == 1 else a.skew):
                sr = skew_route(T, E, k, cv)
                ln = ML.LeanMoE(g, W, synth_route=sr) if bits == 16 else ML.LeanMoEQ(g, W, bits, synth_route=sr)
                chk = None
                if bits != 16 and cv <= 0 and T == a.T[0]:   # quantized kernels vs bf16 kernels on the dequantized weights
                    Wd_ = dict(embed=W["embed"], norm=W["norm"], layers=[dict(d0, Wgu=n["Wgu_deq"], Wd=n["Wd_deq"]) for d0, n in zip(W["layers"], ln.L)])
                    ref = ML.LeanMoE(g, Wd_)
                    with torch.no_grad():
                        o1 = ln.forward(ids).float(); o0 = ref.forward(ids).float()
                    chk = round(torch.nn.functional.cosine_similarity(o0, o1, dim=-1).min().item(), 5); del ref, Wd_
                gr, out = ML.capture(lambda: ln.forward(ids)); w = ML.wall(gr, ids, pool)
                ks = ML.kernel_split(gr)
                fs = ln.stream_graph(); gs, _ = ML.capture(fs); ws = ML.wall(gs, torch.zeros(1, 1, dtype=torch.long, device="cuda"), [torch.zeros(1, 1, dtype=torch.long).pin_memory()])
                e = dict(name=name, T=T, bits=bits, qcheck_cos_min=chk, route=("random-router" if cv <= 0 else f"skew_cv{cv}"), body_total=tot, body_active=act,
                         weight_bytes=ln.weight_bytes(), stream_ms=ws["median"], wall_ms=w["median"], p95=w["p95"], kernels=ks)
                if bits == 16 and E > 1 and cv <= 0:
                    ln2 = ML.LeanMoE(g, W, alias=True); gr2, _ = ML.capture(lambda: ln2.forward(ids)); e["alias_compute_only_ms"] = ML.wall(gr2, ids, pool)["median"]; del gr2
                print(json.dumps(e), flush=True); rows.append(e)
                del gr, gs, ln; torch.cuda.empty_cache()
    del W; torch.cuda.empty_cache()
res["rows"] = rows
os.makedirs("results", exist_ok=True)
json.dump(res, open(f"results/{a.tag}.json", "w"), indent=1)

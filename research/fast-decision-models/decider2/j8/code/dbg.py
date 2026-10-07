import os, sys, json, torch
sys.path.insert(0, os.path.expanduser('~/work/j8'))
torch.set_num_threads(16)
import hob, torch.nn.functional as F
W, hs, cfg = hob.load_weights()
m = hob.Hob(W, dtype=torch.float32).eval()
it = json.loads(open(os.path.expanduser('~/work/j8/ids.jsonl')).readline())
ids = torch.tensor(it['s'] + it['q'])[:300]
T = ids.shape[0]
with torch.inference_mode():
    x = F.embedding(ids, m.embed)
    cos, sin = m.rope_tab(torch.arange(T))
    for i, L in enumerate(m.L):
        h = hob.rms_zc(x, L.in1)
        proj = h @ L.Win.t()
        if m.types[i] == 'gdn':
            z, beta, g = m.gdn_prep(L, proj)
            c = m.conv(proj[None, :, :6144], L.conv)
            q, k, v = m.qkv_heads(c, 1, T)
            print(i, 'g', g.min().item(), g.max().item(), 'beta', beta.min().item(), 'qkv nan', q.isnan().any().item(), k.isnan().any().item(), v.isnan().any().item(), flush=True)
            o, S = m.chunk(q, k, v, g.t()[None], beta.t()[None])
            print('  o nan', o.isnan().any().item(), 'absmax', o.abs().max().item(), 'S', S.abs().max().item())
            o = m.gated_norm(o[0].transpose(0, 1), z, L.gnw)
        else:
            q, k, v, gate = m.attn_prep(L, proj, cos, sin)
            o = m.sdpa(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], causal=True)
            o = o[0].transpose(0, 1).reshape(T, 2048) * gate
        x = x + o @ L.Wo.t()
        x = x + m.mlp(L, x)
        print(i, m.types[i], 'x nan', x.isnan().any().item(), 'absmax', x.abs().max().item(), flush=True)
        if x.isnan().any(): break

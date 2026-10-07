"""per-layer: HF T5Gemma encoder layer i vs EncTorso.layer i on the same input (bf16), softcap on; plus where the end-to-end min-cos row is."""
import os, sys, torch, torch.nn.functional as F
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.expanduser('~/work/evalkit'))
import encj1 as E, evalkit as EK
from transformers import AutoTokenizer, T5GemmaEncoderModel
from strands_decider.prompting import render_state
dev = 'cuda'; path = E.find_ckpt(); tok = AutoTokenizer.from_pretrained(path)
hf = T5GemmaEncoderModel.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='eager', is_encoder_decoder=False).to(dev).eval()
W, _ = E.load_weights(path); me = E.EncTorso(W, lora=False, softcap=50.0).to(dev); me.mode = 'full'
it = EK.load_suite('REAL-agree')[0]
ids = [tok.bos_token_id] + tok(render_state(it['state']), add_special_tokens=False)['input_ids'][:1500]
x = torch.tensor([ids], device=dev); am = torch.ones_like(x); T = x.shape[1]
enc = hf.encoder
with torch.no_grad():
    h = enc.embed_tokens(x) * torch.tensor(E.D ** 0.5, dtype=torch.bfloat16)
    pos = torch.arange(T, device=dev)[None]
    pe = enc.rotary_emb(h, pos)
    cs = E.rope_cs(pos, dev)
    print('rope maxdiff', float((pe[0].float() - cs[0].float()).abs().max()), float((pe[1].float() - cs[1].float()).abs().max()))
    mask = me.build_mask(am, torch.tensor([T], device=dev), False)
    addm = torch.zeros(1, 1, T, T, device=dev, dtype=torch.bfloat16)
    for i in range(26):
        o_hf = enc.layers[i](h, pe, addm, pos)
        o_hf = o_hf[0] if isinstance(o_hf, tuple) else o_hf
        o_me = me.layer(i, h, mask, cs[0], cs[1])
        c = F.cosine_similarity(o_hf[0].float(), o_me[0].float(), dim=-1)
        d = (o_hf.float() - h.float()); dm = (o_me.float() - h.float())
        cd = F.cosine_similarity(d[0], dm[0], dim=-1)
        if i < 4 or i % 5 == 0 or i == 25:
            print(i, 'out cos min %.6f mean %.6f | delta cos min %.5f mean %.6f argmin %d | norm %.1f' % (c.min(), c.mean(), cd.min(), cd.mean(), int(cd.argmin()), float(h[0].float().norm(dim=-1).mean())), flush=True)
        h = o_hf
    hn = E.rms(h, me.norm1)
    ref = hf(input_ids=x, attention_mask=am).last_hidden_state
    print('final check (HF layers chained with addm vs hf forward):', float(F.cosine_similarity(hn[0].float(), ref[0].float(), dim=-1).min()))
    full = me(x, am, torch.tensor([T], device=dev))
    c = F.cosine_similarity(full[0].float(), ref[0].float(), dim=-1)
    print('end-to-end per-row cos: min %.4f at %d, mean %.5f; rows<0.99: %d of %d' % (c.min(), int(c.argmin()), c.mean(), int((c < 0.99).sum()), T))

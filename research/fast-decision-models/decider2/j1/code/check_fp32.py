import os, sys, torch, torch.nn.functional as F
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.expanduser('~/work/evalkit'))
import encj1 as E, evalkit as EK
from transformers import AutoTokenizer, T5GemmaEncoderModel
from strands_decider.prompting import render_state
dev = 'cuda'; path = E.find_ckpt(); tok = AutoTokenizer.from_pretrained(path)
it = EK.load_suite('REAL-agree')[0]
ids = [tok.bos_token_id] + tok(render_state(it['state']), add_special_tokens=False)['input_ids'][:1500]
x = torch.tensor([ids], device=dev); am = torch.ones_like(x); T = x.shape[1]
outs = {}
for dt in (torch.float32, torch.bfloat16):
    hf = T5GemmaEncoderModel.from_pretrained(path, dtype=dt, attn_implementation='eager', is_encoder_decoder=False).to(dev).eval()
    with torch.no_grad(): outs[str(dt)] = hf(input_ids=x, attention_mask=am).last_hidden_state.float()
    del hf; torch.cuda.empty_cache()
W, _ = E.load_weights(path)
for sc in (50.0, None):
    me = E.EncTorso(W, lora=False, softcap=sc).to(dev); me.mode = 'full'
    with torch.no_grad(): outs[f'me_{sc}'] = me(x, am, torch.tensor([T], device=dev)).float()
    del me; torch.cuda.empty_cache()
ref = outs['torch.float32']
for k, v in outs.items():
    c = F.cosine_similarity(v[0], ref[0], dim=-1)
    print(k, 'vs HF fp32: min %.4f mean %.5f rows<0.99 %d/%d rows<0.9 %d' % (c.min(), c.mean(), int((c < 0.99).sum()), T, int((c < 0.9).sum())))
c = F.cosine_similarity(outs['torch.float32'][0], outs['torch.float32'][0][:, :], dim=-1)
n = ref[0].norm(dim=-1); print('fp32 row norm min %.1f median %.1f max %.1f' % (n.min(), n.median(), n.max()))
bad = F.cosine_similarity(outs['torch.bfloat16'][0], ref[0], dim=-1) < 0.9
print('bad rows (bf16 HF) idx', bad.nonzero().flatten()[:30].tolist(), 'tokens', [tok.decode([ids[i]]) for i in bad.nonzero().flatten()[:15].tolist()])

"""J1 checks (box): (1) EncTorso == HF T5GemmaEncoderModel (eager, softcap) on real text; (2) softcap on/off drift; (3) masked single-sequence
path == masked state-cache path (state once + M questions); (4) training step memory/throughput at a few shapes and SDPA backends."""
import os, sys, json, time, glob
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/work/evalkit'))
import encj1 as E
from transformers import AutoTokenizer
dev = 'cuda'
what = sys.argv[1] if len(sys.argv) > 1 else 'all'
path = E.find_ckpt()
tok = AutoTokenizer.from_pretrained(path)
import evalkit as EK
from strands_decider.prompting import render_question, render_state
from pydantic import TypeAdapter
import strands_decider.schema as SC
ta = TypeAdapter(SC.Question)
its = EK.load_suite('REAL-agree')[:6]
texts = []
for it in its:
    st = render_state(it['state']); qn = list(it['questions'])[0]
    texts.append((st, render_question(ta.validate_python(it['questions'][qn])).text))


def cos(a, b):
    a = a.float().flatten(); b = b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm()))


W, cfg = E.load_weights(path)
res = {}
if what in ('all', 'hf'):
    from transformers import T5GemmaEncoderModel
    hf = T5GemmaEncoderModel.from_pretrained(path, dtype=torch.bfloat16, attn_implementation='eager', is_encoder_decoder=False).to(dev).eval()
    mine = E.EncTorso(W, lora=False, softcap=50.0).to(dev)
    mine.mode = 'full'
    nos = E.EncTorso(W, lora=False, softcap=None).to(dev); nos.mode = 'full'
    for k, (st, q) in enumerate(texts[:4]):
        ids = [tok.bos_token_id] + tok(st + q, add_special_tokens=False)['input_ids']
        ids = ids[:3000]
        x = torch.tensor([ids], device=dev); am = torch.ones_like(x)
        with torch.no_grad():
            h_hf = hf(input_ids=x, attention_mask=am).last_hidden_state
            h_me = mine(x, am, torch.tensor([len(ids)], device=dev))
            h_ns = nos(x, am, torch.tensor([len(ids)], device=dev))
        r = dict(T=len(ids), cos_me_hf=cos(h_me, h_hf), maxdiff=float((h_me.float() - h_hf.float()).abs().max()), hf_norm=float(h_hf.float().norm(dim=-1).mean()),
                 cos_nosoftcap_hf=cos(h_ns, h_hf), cos_nosoftcap_last=cos(h_ns[0, -1], h_hf[0, -1]),
                 percos_min_nosoftcap=float(F.cosine_similarity(h_ns[0].float(), h_hf[0].float(), dim=-1).min()))
        print('HF', r, flush=True); res.setdefault('hf', []).append(r)
    del hf; torch.cuda.empty_cache()

if what in ('all', 'cache'):
    m = E.EncTorso(W, lora=True, softcap=None).to(dev)
    with torch.no_grad():   # random LoRA so the check covers the adapters too
        for p in m.lora.parameters(): p.normal_(0, 0.02)
    m.mode = 'masked'
    st, _ = texts[0]
    s = [tok.bos_token_id] + tok(st, add_special_tokens=False)['input_ids'][:900]
    qs = [tok(t[1], add_special_tokens=False)['input_ids'] for t in texts[:4]]
    with torch.no_grad():
        hs, hq = m.forward_cached(torch.tensor(s, device=dev), qs)
        for j, q in enumerate(qs):
            x = torch.tensor([s + q], device=dev); am = torch.ones_like(x)
            h = m(x, am, torch.tensor([len(s)], device=dev))[0]
            r = dict(q=j, cos_state=cos(h[:len(s)], hs), cos_q=cos(h[len(s):], hq[j]), last=cos(h[-1], hq[j][-1]),
                     maxdiff_q=float((h[len(s):].float() - hq[j].float()).abs().max()))
            # and: in masked mode the state rows must not depend on the question
            print('CACHE', r, flush=True); res.setdefault('cache', []).append(r)
        # padding invariance of the batched training path
        x1 = torch.tensor([s + qs[0]], device=dev)
        x2 = torch.nn.utils.rnn.pad_sequence([torch.tensor(s + qs[0]), torch.tensor(s + qs[1] + qs[2])], batch_first=True).to(dev)
        am2 = (torch.arange(x2.shape[1])[None] < torch.tensor([[len(s) + len(qs[0])], [len(s) + len(qs[1]) + len(qs[2])]])).long().to(dev)
        h1 = m(x1, torch.ones_like(x1), torch.tensor([len(s)], device=dev))[0]
        h2 = m(x2, am2, torch.tensor([len(s), len(s)], device=dev))[0, :x1.shape[1]]
        print('PAD', cos(h1, h2), float((h1.float() - h2.float()).abs().max()), flush=True)
    del m; torch.cuda.empty_cache()

if what in ('all', 'train'):
    m = E.EncTorso(W, lora=True, softcap=None).to(dev); m.mode = 'masked'; m.train()
    from torch.nn.attention import sdpa_kernel, SDPBackend
    opt = torch.optim.AdamW(m.lora.parameters(), lr=1e-5)
    for (B, T, ck) in [(16, 256, False), (8, 512, False), (2, 2048, False), (1, 3072, True), (2, 2048, True), (1, 9216, True)]:
        torch.cuda.reset_peak_memory_stats()
        x = torch.randint(10, 200000, (B, T), device=dev); am = torch.ones_like(x); qs = torch.full((B,), T - 120, device=dev)
        m.ckpt = ck
        try:
            for it in range(3):
                torch.cuda.synchronize(); t0 = time.time()
                h = m(x, am, qs); loss = h[:, -1].float().pow(2).mean(); loss.backward(); opt.zero_grad()
                torch.cuda.synchronize(); dt = time.time() - t0
            r = dict(B=B, T=T, ckpt=ck, s=round(dt, 3), tok_s=round(B * T / dt), mem=round(torch.cuda.max_memory_allocated() / 2**30, 2))
        except torch.OutOfMemoryError as e:
            r = dict(B=B, T=T, ckpt=ck, oom=True)
        print('TRAIN', r, flush=True); res.setdefault('train', []).append(r)
        torch.cuda.empty_cache()
json.dump(res, open(f'check_{what}.json', 'w'), indent=1)

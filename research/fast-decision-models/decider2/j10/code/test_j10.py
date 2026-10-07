"""J10 checks on the box.
  python test_j10.py ref     -> my torso vs HF BitNetForCausalLM (online quant) on real text: hidden cosine, LM loss of both
  python test_j10.py crest   -> per-token crest factor (max/rms) and int4 / block-int4 rounding error at every BitLinear input, on
                                64 real train-split states (+ a real question), vs hobson's 15-40 (doc section 5). -> crest_j10.json
"""
import os, sys, json, math, random, time
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.expanduser('~/work/sd/src')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn.functional as F
import bitnet_j10 as BJ
from transformers import AutoTokenizer
from strands_decider.prompting import render_question, render_state
from pydantic import TypeAdapter
import strands_decider.schema as SC
ta = TypeAdapter(SC.Question)


def pool_texts(n, seed=0, maxtok=4000):
    EV = set(json.load(open(os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
    recs = []
    with open(os.path.expanduser('~/work/evalkit/train_pool.jsonl')) as f:
        for l in f:
            r = json.loads(l)
            if r['task'] in EV or not (200 <= r['n_state_tok'] <= maxtok): continue
            recs.append(r)
    random.Random(seed).shuffle(recs)
    out = []
    for r in recs[:n]:
        qn = sorted(r['questions'])[0]
        out.append(render_state(r['state']) + render_question(ta.validate_python(r['questions'][qn])).text)
    return out


def ref():
    cfg, W = BJ.load()
    tok = AutoTokenizer.from_pretrained(BJ.snap())
    m = BJ.BitNetTorso(cfg, W, lora=False).cuda().eval()
    texts = pool_texts(4, maxtok=1500)
    from transformers import AutoModelForCausalLM
    hf = AutoModelForCausalLM.from_pretrained(BJ.snap(), dtype=torch.bfloat16).cuda().eval()
    print(type(hf), hf.model.layers[0].self_attn.q_proj, flush=True)
    res = []
    with torch.inference_mode():
        for t in texts:
            ids = tok(t, return_tensors='pt')['input_ids'].cuda()
            h = m(ids)
            o = hf(input_ids=ids, output_hidden_states=True)
            hh = o.hidden_states[-1]
            cos = F.cosine_similarity(h.float().flatten(), hh.float().flatten(), dim=0).item()
            lm_mine = F.cross_entropy((h[0, :-1].float() @ m.embed.float().t()), ids[0, 1:]).item()
            lm_hf = F.cross_entropy(o.logits[0, :-1].float(), ids[0, 1:]).item()
            res.append(dict(T=ids.shape[1], cos=round(cos, 6), lm_mine=round(lm_mine, 4), lm_hf=round(lm_hf, 4)))
            print(res[-1], flush=True)
    json.dump(res, open('ref_j10.json', 'w'), indent=1)


def crest():
    cfg, W = BJ.load()
    tok = AutoTokenizer.from_pretrained(BJ.snap())
    m = BJ.BitNetTorso(cfg, W, lora=False).cuda().eval()
    texts = pool_texts(64)
    acc = {}

    def stat(i, cls, x):
        x = x.float().reshape(-1, x.shape[-1])
        rmsv = x.pow(2).mean(-1).sqrt().clamp_min(1e-8)
        cr = x.abs().amax(-1) / rmsv
        d = acc.setdefault((i, cls), dict(cr=[], e8=[], e4=[], e4b16=[], e4b32=[], e4b64=[]))
        d['cr'].append(cr.cpu())
        for key, (b, blk) in dict(e8=(8, 0), e4=(4, 0), e4b16=(4, 16), e4b32=(4, 32), e4b64=(4, 64)).items():
            e = (BJ._fq(x, b, blk) - x).pow(2).sum(-1).sqrt() / x.pow(2).sum(-1).sqrt().clamp_min(1e-8)
            d[key].append(e.cpu())
    m.stats = stat
    with torch.inference_mode():
        for t in texts:
            ids = tok(t, return_tensors='pt')['input_ids'].cuda()
            m(ids)
    out = {}
    for cls in ('qkv', 'o', 'gu', 'down'):
        rows = []
        for i in range(m.L):
            d = acc[(i, cls)]
            cr = torch.cat(d['cr'])
            r = dict(layer=i, crest_med=float(cr.median()), crest_p99=float(cr.quantile(0.99)), crest_max=float(cr.max()))
            for k in ('e8', 'e4', 'e4b16', 'e4b32', 'e4b64'):
                r[k] = float(torch.cat(d[k]).mean())
            rows.append(r)
        out[cls] = rows
        allcr = torch.cat([torch.cat(acc[(i, cls)]['cr']) for i in range(m.L)])
        print(cls, 'crest median %.1f p99 %.1f max %.1f' % (allcr.median(), allcr.quantile(0.99), allcr.max()),
              ' rel err int8 %.4f int4 %.3f int4/b16 %.3f int4/b32 %.3f int4/b64 %.3f' % tuple(
                  sum(r[k] for r in rows) / len(rows) for k in ('e8', 'e4', 'e4b16', 'e4b32', 'e4b64')), flush=True)
    json.dump(out, open('crest_j10.json', 'w'), indent=1)


if __name__ == '__main__':
    dict(ref=ref, crest=crest)[sys.argv[1]]()

"""H4: relative low-bit sensitivity of this-that vs hobson (same torso, same recipe, no training, no rotation): fake-quant every
nn.Linear of the torso, HF/engine path, REAL-agree + CF + CF-probe; decision flips are measured against each model's own bf16.
  python tt_quant.py tt|hob w4g64|w8a8|w4a16g128
  w4g64      weight-only 4-bit affine (min/max) groups of 64 = what mlx 4-bit does (the README's 4-bit row)
  w4a16g128  weight-only 4-bit symmetric groups of 128 (RTN)
  w8a8       int8 weights per output channel + int8 activations per token, symmetric, dynamic (RTN, no rotation)
The answer head (tied embedding rows for this-that; the pointer head for hobson) stays bf16.
"""
import os, sys, json, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, os.path.expanduser('~/work/evalkit')); sys.path.insert(0, os.path.expanduser('~/work/tokens'))
import numpy as np, torch
import evalkit as EK

MODEL, CFG = sys.argv[1], sys.argv[2]
SUITES = ['REAL-agree', 'CF', 'CF-probe']
out = os.path.expanduser(f'~/work/h4/preds/q_{MODEL}_{CFG}.jsonl')


@torch.no_grad()
def fq_weight(W, cfg):
    Wf = W.float()
    if cfg == 'w4g64':
        o, i = Wf.shape; g = Wf.reshape(o, i // 64, 64)
        lo, hi = g.amin(-1, keepdim=True), g.amax(-1, keepdim=True); s = (hi - lo).clamp_min(1e-8) / 15
        q = ((g - lo) / s).round().clamp(0, 15); return (q * s + lo).reshape(o, i).to(W.dtype)
    if cfg == 'w4a16g128':
        o, i = Wf.shape; g = Wf.reshape(o, i // 128, 128)
        s = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7
        return ((g / s).round().clamp(-8, 7) * s).reshape(o, i).to(W.dtype)
    if cfg == 'w8a8':
        s = Wf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
        return ((Wf / s).round().clamp(-127, 127) * s).to(W.dtype)
    raise ValueError(cfg)


def act_hook(mod, inp):
    x = inp[0]; xf = x.float()
    s = xf.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 127
    return ((xf / s).round().clamp(-127, 127) * s).to(x.dtype),


def quantize(torso):
    n = 0
    for name, m in torso.named_modules():
        if isinstance(m, torch.nn.Linear) and 'lora' not in name:
            m.weight.data = fq_weight(m.weight.data, CFG); n += 1
            if CFG == 'w8a8': m.register_forward_pre_hook(act_hook)
    print('quantized linears', n, flush=True)


done = set()
if os.environ.get('LIMIT'): out = out + '.test'
if os.path.exists(out) and not os.environ.get('LIMIT'):
    for l in open(out):
        try: done.add(json.loads(l)['id'])
        except Exception: pass
items = [it for s in SUITES for it in EK.load_suite(s)]
if os.environ.get('LIMIT'): items = items[:int(os.environ['LIMIT'])]

if MODEL == 'tt':
    from thisthat import TypedDecider
    from thisthat.prompt import build
    from thisthat.systemone_protocol import _typed
    from thisthat.model import _softmax
    dec = TypedDecider.from_pretrained('flock-io/this-that-model-1.0', device='cuda')
    quantize(dec.model.model)
    tok = dec.tokenizer; be = dec.backend

    @torch.inference_mode()
    def answer(state, specs):
        res = {}
        for k, s in specs.items():   # one question per sequence, as in the bf16 'sf' run
            t = _typed(k, s)
            b = build(tok, state, [t.question], max_state_tokens=10 ** 7)
            ids = np.array([b['ids']]); lg = be.slot_logits(ids, np.ones_like(ids), np.array(b['slots']), np.zeros(1, dtype=np.int64), np.array(b['n_options']))
            p = _softmax(lg, 1.0)[0][:len(t.labels)]; d = dict(zip(t.labels, map(float, p)))
            res[k] = {'true': d['yes'], 'false': d['no']} if t.type == 'noul' else d
        return res
else:
    from kitrun import load_engine, engine_eval
    eng = load_engine()
    eng.model.torso = eng.model.torso.merge_and_unload().eval()
    quantize(eng.model.torso)
    answer = lambda state, specs: engine_eval(eng, state, specs)

t0 = time.time(); n = 0
with open(out, 'a') as f:
    for it in items:
        if it['id'] in done: continue
        try:
            r = answer(it['state'], it['questions'])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); print('OOM', it['id'], flush=True); continue
        f.write(json.dumps({'id': it['id'], 'q': None, 'p': r}) + '\n'); f.flush(); n += 1
        if n % 100 == 0: print(MODEL, CFG, n, len(items), '%.0fs' % (time.time() - t0), flush=True)
print('done', MODEL, CFG, n, '%.0fs' % (time.time() - t0), flush=True)

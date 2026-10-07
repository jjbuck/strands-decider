"""curriculum data: short-context versions of the training tasks (same generators): probe windows 60-260 tokens (>= 2 messages),
SQuAD-MC with the gold paragraph only.  Adds probe_train_short / squad_train_short to ft.pkl."""
import os, sys, json, pickle, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep_ft as PF
D = PF.D
data = pickle.load(open(f'{D}/ft.pkl', 'rb'))
_w = PF.window


def window_short(s, rng, budget):
    ms = PF.messages(s)
    if len(ms) < 2: return None
    header = s[:PF.conv_bounds(s)[0]]
    hl = len(PF.enc(header)); lens = [len(PF.enc(m[3])) for m in ms]
    starts = list(range(len(ms) - 1)); rng.shuffle(starts)
    for i in starts[:12]:
        tot = hl; j = i
        while j < len(ms) and tot + lens[j] <= budget: tot += lens[j]; j += 1
        if j - i >= 2:
            return header + ''.join(t if t.endswith('\n') else t + '\n' for t in (m[3] for m in ms[i:j]))
    return None


_probe = PF.probe
def probe_short(rng, s, dom, kind, distract):
    ms = PF.messages(s)
    if len(ms) < 2: return None
    # same generator; relax the >=4-message guard by padding the message count check
    return _probe.__wrapped__(rng, s, dom, kind, distract) if hasattr(_probe, '__wrapped__') else _probe2(rng, s, dom, kind, distract)


import inspect, types
src = inspect.getsource(PF.probe).replace('if len(ms) < 4: return None', 'if len(ms) < 2: return None').replace('def probe(', 'def _probe2(')
exec(compile(src, 'probe2', 'exec'), PF.__dict__)
_probe2 = PF._probe2
PF.window = window_short; PF.probe = _probe2
tr = [json.loads(l) for l in open(f'{D}/train_states.jsonl')]
data['probe_train_short'] = PF.gen_probes(tr, 20000, 21, 60, 260, balanced=False)
PF.window = _w
data['squad_train_short'] = PF.squad('train', 20000, 23, 0, 0)
for k in ('probe_train_short', 'squad_train_short'):
    import numpy as np
    L = np.array([len(e['s']) for e in data[k]]); print(k, len(data[k]), 'state tok mean', L.mean().round(), 'p90', np.percentile(L, 90))
pickle.dump(data, open(f'{D}/ft.pkl', 'wb'))

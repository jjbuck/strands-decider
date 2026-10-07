"""library statistics: distinct compiled blocks per domain, their token sizes and request frequency (train_pool + eval suites + raw logs)"""
import os, sys, json, glob, collections
sys.path[:0] = [os.path.expanduser('~/work/j9')]
from transformers import AutoTokenizer
import j9lib as J
snap = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*'))[0]
tok = AutoTokenizer.from_pretrained(snap)
LL = J.LineLib(os.path.expanduser('~/work/evalkit/train_pool.jsonl'))
blk = {}; freq = collections.Counter(); dom = {}
nreq = collections.Counter()
for li, l in enumerate(open(os.path.expanduser('~/work/j9/data/raw_states.jsonl'))):
    r = json.loads(l)
    st = f"<state>\n{r['state'].strip()}\n</state>\n"
    nreq[r['domain']] += 1
    for k, ids, key in J.tokenize_pieces(tok, J.pieces(st, LL)):
        if k == 'dyn': continue
        blk[key] = len(ids); freq[key] += 1; dom[key] = r['domain'] + ('/frame' if k == 'frame' else '')
import numpy as np
out = {}
for d in sorted(set(dom.values())):
    ks = [k for k in blk if dom[k] == d]
    sz = np.array([blk[k] for k in ks]); fr = np.array([freq[k] for k in ks])
    order = np.argsort(-fr); cum = np.cumsum(fr[order]) / fr.sum()
    top = lambda q: int(np.searchsorted(cum, q) + 1)
    out[d] = dict(n_blocks=len(ks), tok_total=int(sz.sum()), tok_mean=float(sz.mean()), tok_median=float(np.median(sz)), uses=int(fr.sum()),
                  blocks_for_90pct_uses=top(0.9), blocks_for_99pct_uses=top(0.99), kv_MB=float(sz.sum() * 12288 / 1e6),
                  state_MB_bf16_E=len(ks) * 18 * 16 * 128 * 128 * 2 / 1e6)
    print(d, out[d], flush=True)
json.dump(out, open(os.path.expanduser('~/work/j9/lib_stats%s.json' % ('_strict' if os.environ.get('J9_STRICT') == '1' else '')), 'w'), indent=1)

"""constant share on the RAW gate logs (all 31,013 records -> 24,399 unique states), hobson tokenizer, CPU only"""
import os, sys, json, glob, collections
sys.path[:0] = [os.path.expanduser('~/work/j9')]
from transformers import AutoTokenizer
import j9lib as J
snap = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*'))[0]
tok = AutoTokenizer.from_pretrained(snap)
LL = J.LineLib(os.path.expanduser('~/work/evalkit/train_pool.jsonl'))
agg = collections.defaultdict(collections.Counter); lens = collections.defaultdict(list)
for li, l in enumerate(open(os.path.expanduser('~/work/j9/data/raw_states.jsonl'))):
    r = json.loads(l)
    st = f"<state>\n{r['state'].strip()}\n</state>\n"
    tp = J.tokenize_pieces(tok, J.pieces(st, LL))
    T = sum(len(ids) for _, ids, _ in tp)
    for g in ((r['domain'], r['lifecycle']), (r['domain'], 'ALL'), ('ALL', 'ALL')):
        c = agg[g]; c['n'] += 1; c['tok'] += T
        for k, ids, _ in tp: c[k] += len(ids)
        c['nblk'] += sum(1 for k, _, _ in tp if k == 'blk')
        lens[g].append((T, sum(len(ids) for k, ids, _ in tp if k == 'dyn')))
    if li % 4000 == 0: print(li, flush=True)
import numpy as np
res = {}
for g, c in sorted(agg.items(), key=lambda x: tuple(str(y) for y in x[0])):
    T = c['tok']; L = np.array(lens[g])
    res['|'.join(map(str, g))] = dict(n=c['n'], mean_tok=T / c['n'], median_tok=float(np.median(L[:, 0])), p90_tok=float(np.percentile(L[:, 0], 90)),
                            median_live=float(np.median(L[:, 1])), p90_live=float(np.percentile(L[:, 1], 90)),
                            frame=c['frame'] / T, blk=c['blk'] / T, const=(c['frame'] + c['blk']) / T, blk_per_req=c['nblk'] / c['n'],
                            mean_reduction=1 - (c['dyn'] / T))
    r_ = res['|'.join(map(str, g))]
    print(f"{'|'.join(map(str, g)):<40} n={c['n']:<6} tok mean {r_['mean_tok']:6.0f} med {r_['median_tok']:6.0f} p90 {r_['p90_tok']:6.0f} | live med {r_['median_live']:6.0f} p90 {r_['p90_live']:6.0f} | const {r_['const']:.3f} (frame {r_['frame']:.3f} blk {r_['blk']:.3f}) blk/req {r_['blk_per_req']:.1f}")
json.dump(res, open(os.path.expanduser('~/work/j9/raw_share%s.json' % ('_strict' if os.environ.get('J9_STRICT') == '1' else '')), 'w'), indent=1)

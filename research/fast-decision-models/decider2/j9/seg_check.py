"""segmentation check (box): concat(piece tokens) == whole-state tokens; token shares by piece kind per suite / domain / hook"""
import os, sys, json, glob, collections, hashlib
sys.path[:0] = [os.path.expanduser('~/work/j9')]
from transformers import AutoTokenizer
import j9lib as J
snap = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*'))[0]
tok = AutoTokenizer.from_pretrained(snap)
LL = J.LineLib(os.path.expanduser('~/work/evalkit/train_pool.jsonl'))
print('lib', len(LL.lib), flush=True)
def rs(s): return f"<state>\n{s.strip()}\n</state>\n"
res = {}
blk_lib = collections.Counter(); blk_tok = {}
train_keys = set()
# library hit rate: blocks seen in train_pool (first 4000 requests) vs eval
for li, l in enumerate(open(os.path.expanduser('~/work/evalkit/train_pool.jsonl'))):
    if li >= 4000: break
    d = json.loads(l)
    for k, ids, key in J.tokenize_pieces(tok, J.pieces(rs(d['state']), LL)):
        if k != 'dyn': train_keys.add(key)
print('train keys', len(train_keys), flush=True)
for suite in ['REAL-agree', 'LONG', 'CF', 'CF-probe', 'JB-all']:
    agg = collections.defaultdict(collections.Counter); seen = set(); mism = 0; n = 0
    for l in open(os.path.expanduser(f'~/work/evalkit/suites/{suite}.jsonl')):
        it = json.loads(l)
        st = it['state'] if isinstance(it['state'], str) else json.dumps(it['state'], indent=2, ensure_ascii=False)
        h = hashlib.sha1(st.encode()).hexdigest()
        if h in seen: continue
        seen.add(h); n += 1
        r = rs(st)
        ps = J.pieces(r, LL); tp = J.tokenize_pieces(tok, ps)
        whole = tok(r, add_special_tokens=False)['input_ids']
        cat = [t for _, ids, _ in tp for t in ids]
        if cat != whole: mism += 1
        g = (it.get('domain', 'jb'), it.get('hook', ''))
        for gg in (g, ('ALL', '')):
            c = agg[gg]; c['n'] += 1; c['tok'] += len(whole)
            for k, ids, key in tp:
                c[k] += len(ids)
                if k == 'blk':
                    c['nblk'] += 1
                    if key in train_keys: c['blk_in_train'] += len(ids)
    out = {}
    for g, c in sorted(agg.items(), key=lambda x: (x[0][0] != 'ALL', x[0])):
        T = c['tok']
        out['|'.join(g)] = dict(n=c['n'], mean_tok=T / c['n'], frame=c['frame'] / T, blk=c['blk'] / T, dyn=c['dyn'] / T,
                                const=(c['frame'] + c['blk']) / T, blk_per_req=c['nblk'] / c['n'], blk_tok_in_train_lib=c['blk_in_train'] / max(1, c['blk']),
                                mean_live=c['dyn'] / c['n'])
    res[suite] = dict(groups=out, tok_mismatch=mism, n=n)
    print(suite, 'n', n, 'tok mismatch', mism, flush=True)
    for g, r_ in out.items():
        print(f"  {g:<40} n={r_['n']:<4} tok={r_['mean_tok']:7.0f} const={r_['const']:.3f} (frame {r_['frame']:.3f} blk {r_['blk']:.3f}) live={r_['mean_live']:6.0f} blk/req={r_['blk_per_req']:.1f} inTrainLib={r_['blk_tok_in_train_lib']:.2f}")
json.dump(res, open(os.path.expanduser('~/work/j9/seg_share%s.json' % ('_strict' if os.environ.get('J9_STRICT') == '1' else '')), 'w'), indent=1)

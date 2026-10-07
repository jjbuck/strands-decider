"""what are the distinct blocks? group 'blk' pieces of the raw-log states by the KB doc id they belong to (or by kind) and count distinct texts,
and how many distinct texts are prefixes of the longest variant of the same doc (truncations)."""
import os, sys, json, collections, re
sys.path[:0] = [os.path.expanduser('~/work/j9')]
import importlib.util
src = open(os.path.expanduser('~/work/j9/j9lib.py')).read().split('# ============================================================ model')[0]
ns = {}; exec(src.replace("import torch, torch.nn.functional as F\nimport h3lib as H\nfrom h3lib import rms_zc, DENSE, chunk_gated_delta_rule", ""), ns)
LL = ns['LineLib'](os.path.expanduser('~/work/evalkit/train_pool.jsonl'))
bydoc = collections.defaultdict(collections.Counter); kinds = collections.Counter(); chars = collections.Counter()
for li, l in enumerate(open(os.path.expanduser('~/work/j9/data/raw_states.jsonl'))):
    r = json.loads(l)
    if r['domain'] != 'banking_knowledge': continue
    st = f"<state>\n{r['state'].strip()}\n</state>\n"
    ps = ns['pieces'](st, LL)
    prev = ''
    for k, txt in ps:
        if k == 'blk':
            m = re.findall(r'ID: (doc_\S+)', prev[-400:])
            if txt.lstrip().startswith('Content:') and m: key = m[-1]; kind = 'doc_body'
            elif 'system note from the steering hook' in txt[:60] or 'GUIDANCE' in txt[:60]: key = 'note:' + txt[:80]; kind = 'note'
            elif 'Tool unlocked' in txt[:300] or 'Parameters:' in txt: key = 'tool:' + txt[:60]; kind = 'tool'
            else: key = 'other:' + txt[:60]; kind = 'other'
            bydoc[key][txt] += 1; kinds[kind] += 1; chars[kind] += len(txt)
        prev = txt
tot_texts = 0; pref = 0; per_kind = collections.Counter(); per_kind_keys = collections.Counter()
for key, c in bydoc.items():
    kind = 'doc_body' if key.startswith('doc_') else key.split(':')[0]
    per_kind[kind] += len(c); per_kind_keys[kind] += 1
    longest = max(c, key=len)
    for t in c:
        tot_texts += 1
        if longest.startswith(t.replace(' ...[truncated]', '').rstrip()[:max(1, len(t) - 40)]): pref += 1
print('uses by kind', dict(kinds)); print('chars by kind', dict(chars))
print('distinct texts by kind', dict(per_kind), 'distinct keys by kind', dict(per_kind_keys))
print('texts that are (near-)prefixes of their key\'s longest variant: %d / %d' % (pref, tot_texts))
docs = [(k, c) for k, c in bydoc.items() if k.startswith('doc_')]
docs.sort(key=lambda x: -len(x[1]))
for k, c in docs[:3]:
    print(k, 'variants', len(c), 'lengths', sorted({len(t) for t in c})[:12])
    vs = sorted(c, key=len)[:3]
    for v in vs: print('   ', repr(v[:150]), '...', repr(v[-120:]))
oth = [(k, c) for k, c in bydoc.items() if k.startswith('other')]
oth.sort(key=lambda x: -sum(x[1].values()))
for k, c in oth[:6]: print(k[:90], len(c), sum(c.values()))

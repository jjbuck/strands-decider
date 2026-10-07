"""Build stacklib's deployment assets once (box k1), then distribute the assets/ folder to every box and to K2.
  assets/linelib.json      J9 LineLib: normalised lines that recur in >= 3 train-split tasks (C segmentation, loose)
  assets/deployed_q.json   the 38 deployed question specs of the train pool (Q)
  assets/sb16k.json(+.tok.json)  J7 super-BPE, 16,384 merges over Qwen ids (V)
  assets/precmap_w8a8_b8.json    H1's 8 bf16 GEMMs (P)
python mkassets.py"""
import os, sys, json, shutil, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
W = os.path.expanduser('~/work/')
A = os.path.join(HERE, 'assets'); os.makedirs(A, exist_ok=True)
t0 = time.time()
import j9lib as J
ll = J.LineLib(W + 'evalkit/train_pool.jsonl')
json.dump(sorted(ll.lib), open(f'{A}/linelib.json', 'w'))
print('linelib', len(ll.lib), f'{time.time() - t0:.0f}s', flush=True)
EV = set(json.load(open(W + 'evalkit/split.json'))['eval_tasks'])
spec = {}
for l in open(W + 'evalkit/train_pool.jsonl'):
    r = json.loads(l)
    if r['task'] in EV or r['n_state_tok'] < 16: continue
    for q, s in r['questions'].items():
        js = json.dumps(s, sort_keys=True)
        if q in spec: assert spec[q][1] == js, q
        else: spec[q] = (s, js)
json.dump({q: v[0] for q, v in sorted(spec.items())}, open(f'{A}/deployed_q.json', 'w'), indent=1)
print('deployed', len(spec), flush=True)
shutil.copy(os.path.join(HERE, 'precmap_w8a8_b8.json'), f'{A}/precmap_w8a8_b8.json')
import superbpe
superbpe.train(16384, f'{A}/sb16k.json')
print('done', f'{time.time() - t0:.0f}s', flush=True)

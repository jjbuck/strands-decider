"""Laptop: from the per-layer dev scan (s24.L{i}.<method>.bf16n[.s] in dev_res.json), rank layers by dev KL (bank + ret) per sparse MAC and add
configs that sparsify the k least sensitive layers (state rows / all rows), in bf16 and in b8 + int8 natural basis (smoothing 0.5).
python3 gen_greedy.py dev_res.json [method] -> appends to cfgs_q5.json, prints the layer orders"""
import json, sys, os
R = json.load(open(sys.argv[1])); m = sys.argv[2] if len(sys.argv) > 2 else 'sgptd'
C = json.load(open(os.path.expanduser('~/decider2/q5/code/cfgs_q5.json')))
B8 = C['b8']['fmt']
MAC = {i: (8224 if i not in (3, 7, 11, 15, 19, 23) else 5120) * 2048 + 2048 * 2048 + 12288 * 2048 + 2048 * 6144 for i in range(24)}
for role, sfx in (('s', '.s'), ('all', '')):
    sc = {}
    for i in range(24):
        r = R.get(f's24.L{i}.{m}.bf16n{sfx}')
        if r is None: continue
        v = r['vs_dense']; sc[i] = (v['bank']['kl'] * v['bank']['n'] + v['ret']['kl'] * v['ret']['n']) / (v['bank']['n'] + v['ret']['n'])
    if not sc: continue
    order = sorted(sc, key=lambda i: sc[i] / MAC[i])
    print(role, 'order (least sensitive first):', [(i, f'{sc[i]:.1e}') for i in order])
    for k in (8, 11, 14, 16, 18, 20):
        if k > len(order): continue
        lay = sorted(order[:k])
        nm = f'g{k}{m}{sfx}'
        C[f's24.{nm}.bf16n'] = {"s24": dict(layers=lay, gemms=["Win", "Wo", "Wgu", "Wd"], role=role, prec='bf16n', method=m, cal='gen')}
        C[f'b8+s24.{nm}.int8n0.5'] = {"s24": dict(layers=lay, gemms=["Win", "Wo", "Wgu", "Wd"], role=role, prec='int8n', method=m, cal='gen', smooth=0.5), "fmt": B8, "fmt_name": "b8"}
        C[f'b8+s24.{nm}.int4'] = {"s24": dict(layers=lay, gemms=["Win", "Wo", "Wgu", "Wd"], role=role, prec='int4', method=m, cal='gen'), "fmt": B8, "fmt_name": "b8"}
        # MLP-only variant on the same layers
        C[f'b8+s24.{nm}.mlp.int8n0.5'] = {"s24": dict(layers=lay, gemms=["Wgu", "Wd"], role=role, prec='int8n', method=m, cal='gen', smooth=0.5), "fmt": B8, "fmt_name": "b8"}
        for kk in [f's24.{nm}.bf16n', f'b8+s24.{nm}.int8n0.5', f'b8+s24.{nm}.int4', f'b8+s24.{nm}.mlp.int8n0.5']:
            v2 = json.loads(json.dumps(C[kk])); v2['s24']['cal'] = 'bank'; C[kk + '.bank'] = v2
        print(' ', nm, lay, f'share of GEMM MACs {sum(MAC[i] for i in lay) / sum(MAC.values()):.3f}')
json.dump(C, open(os.path.expanduser('~/decider2/q5/code/cfgs_q5.json'), 'w'), indent=0)

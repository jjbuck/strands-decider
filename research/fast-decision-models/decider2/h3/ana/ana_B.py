import json, sys
d = json.load(open(sys.argv[1])); R = d['res']
FA = (3, 7, 11, 15, 19, 23)
print(sys.argv[1], d['fmt'], 'n', d['n'])
cols = ['Win.qkv', 'Win.z', 'Win.ab', 'Win', 'Wo', 'Wgu', 'Wd']
print('layer type ' + ' '.join(f'{c:>16}' for c in cols) + '   (TV / flip-rate)')
tot = {c: [0, 0] for c in cols}
for l in range(24):
    cells = []
    for c in cols:
        k = f'{l}:{c}'
        if k in R:
            cells.append(f"{R[k]['tv']:.3f}/{R[k]['flip']:.2f}"); tot[c][0] += R[k]['tv']; tot[c][1] += 1
        else: cells.append('')
    print(f"{l:>5} {'ATT' if l in FA else 'GDN'}  " + ' '.join(f'{x:>16}' for x in cells))
print('mean TV per GEMM type:', {c: round(v[0] / v[1], 4) for c, v in tot.items() if v[1]})
for k, v in R.items():
    if k.startswith('all:'): print(k, v)

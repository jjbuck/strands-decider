"""rank file for keeping GEMMs in bf16 under W8A8, from an fp32-mode single-GEMM sensitivity (hidden-error metric, additive), by gain per us"""
import sys, json
T = {('gdn', 'Win'): (676, 287), ('att', 'Win'): (349, 186), ('*', 'Wo'): (183, 105), ('*', 'Wgu'): (843, 423), ('*', 'Wd'): (428, 261)}
ATT = (3, 7, 11, 15, 19, 23)
def dt(c):
    i, k = int(c.split('.')[0]), c.split('.')[1]; key = (('att' if i in ATT else 'gdn'), k) if k == 'Win' else ('*', k)
    return T[key][0] - T[key][1]
d = json.load(open(sys.argv[1])); acc = d['acc']; keys = [c for c in acc if c != 'ALL']
for m in ('kl', 'hid'):
    tot = sum(acc[c][m] for c in keys); print(m, 'ALL', acc['ALL'][m], 'sum singles', tot)
rank = sorted(keys, key=lambda c: -acc[c]['hid'] / dt(c))
tot = sum(acc[c]['hid'] for c in keys); cum = 0; ext = 0
for n, c in enumerate(rank, 1):
    cum += acc[c]['hid']; ext += dt(c)
    if n in (1, 2, 4, 8, 12, 16, 24, 32, 48): print(n, c, 'hid share removed', round(cum / tot, 3), 'extra GEMM ms/1k tok', round(ext / 1000, 2))
json.dump(dict(rank=rank, src=sys.argv[1]), open(sys.argv[1].replace('.json', '_rank.json'), 'w'))

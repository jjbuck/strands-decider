"""ptl.py: coarse timeline of a profile json: per 200us bin, busy fraction per engine + most common hlo_name; and idle gaps."""
import sys, json, collections
d = json.load(open(sys.argv[1]))
ins = d['instruction']
s = d['summary'][0] if isinstance(d['summary'], list) else d['summary']
print('total_ms', s['total_time'] * 1e3, 'n_ins', len(ins))
BIN = float(sys.argv[2]) if len(sys.argv) > 2 else 200e3
tot = s['total_time'] * 1e9
nb = int(tot // BIN) + 1
busy = collections.defaultdict(lambda: [0.0] * nb)
names = [collections.Counter() for _ in range(nb)]
for x in ins:
    b = int(x['timestamp'] // BIN)
    if b >= nb: continue
    busy[x['label']][b] += x['duration']
    nm = (x.get('hlo_name') or '')[:40] + '|' + (x.get('opcode') or '')[:14]
    names[b][nm] += x['duration']
engs = ['Tensor', 'TensorMatrix', 'Vector', 'Scalar', 'GpSimd', 'Sync']
print('bin_us ' + ' '.join(f'{e[:6]:>6s}' for e in engs) + '  top')
for b in range(nb):
    row = ' '.join(f'{busy[e][b] / BIN:6.2f}' for e in engs)
    top = names[b].most_common(2)
    print(f'{b * BIN / 1e3:7.0f} {row}  {top}')

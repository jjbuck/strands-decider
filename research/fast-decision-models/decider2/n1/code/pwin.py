"""pwin.py: effective compute window of a profile = from the first TensorE instruction to the last instruction end; also gaps.
python pwin.py DIR [DIR...]"""
import sys, json, glob, os
for p in sys.argv[1:]:
    q = glob.glob(f'{p}/json_reports/*.json')[0] if os.path.isdir(p) else p
    d = json.load(open(q))
    ins = sorted(d['instruction'], key=lambda x: x['timestamp'])
    s = d['summary'][0] if isinstance(d['summary'], list) else d['summary']
    pe = [x for x in ins if x['label'] in ('Tensor', 'TensorMatrix') and x['opcode'] in ('MATMUL', 'LDWEIGHTS')]
    t0 = pe[0]['timestamp']; t1 = max(x['timestamp'] + x['duration'] for x in ins)
    gaps = []; prev = ins[0]['timestamp']
    for x in ins:
        if x['timestamp'] - prev > 20000: gaps.append((prev, x['timestamp']))
        prev = max(prev, x['timestamp'] + x['duration'])
    mm = [x for x in ins if x['opcode'] == 'MATMUL']
    print(json.dumps(dict(prof=os.path.basename(p.rstrip('/')), total_us=round(s['total_time'] * 1e6, 1), window_us=round((t1 - t0) / 1e3, 1),
                          first_pe_us=round(t0 / 1e3, 1), gaps_us=[(round(a / 1e3, 1), round(b / 1e3, 1)) for a, b in gaps][:5],
                          n_matmul_traced=len(mm), mm_first_us=round(mm[0]['timestamp'] / 1e3, 1) if mm else None)))

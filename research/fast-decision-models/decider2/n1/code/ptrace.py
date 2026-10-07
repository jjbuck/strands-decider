"""ptrace.py: instruction-level view of a neuron-profile full json.  python ptrace.py DIR [t0_us t1_us]"""
import sys, json, glob, os, collections
p = sys.argv[1]
if os.path.isdir(p): p = glob.glob(f'{p}/json_reports/*.json')[0]
d = json.load(open(p))
ins = d['instruction']
s = d['summary'][0] if isinstance(d['summary'], list) else d['summary']
print({k: v for k, v in s.items() if 'throttle' in k and 'nc0' in k and v})
by = collections.defaultdict(list)
for x in ins: by[x['label']].append(x)
for eng, L in by.items():
    L.sort(key=lambda x: x['timestamp'])
    busy = sum(x['duration'] for x in L); wait = sum(x.get('evt_wait_time', 0) or 0 for x in L)
    ops = collections.Counter(x['opcode'] for x in L)
    dur = collections.defaultdict(int)
    for x in L: dur[x['opcode']] += x['duration']
    span = L[-1]['timestamp'] + L[-1]['duration'] - L[0]['timestamp']
    print(f'== {eng}: n={len(L)} busy={busy/1e3:.1f}us span={span/1e3:.1f}us evt_wait={wait/1e3:.1f}us')
    for op, c in ops.most_common(12): print(f'   {op:28s} n={c:6d} total={dur[op]/1e3:8.1f}us avg={dur[op]/c:7.1f}ns')
if len(sys.argv) > 3:
    t0, t1 = float(sys.argv[2]) * 1e3, float(sys.argv[3]) * 1e3
    W = sorted([x for x in ins if t0 <= x['timestamp'] <= t1], key=lambda x: x['timestamp'])
    for x in W:
        print(f"{x['timestamp']/1e3:9.3f} {x['duration']:6d} {x['label'][:10]:10s} {x['opcode'][:22]:22s} w={x.get('evt_wait_time',0)} {x['operands'][:110]}")

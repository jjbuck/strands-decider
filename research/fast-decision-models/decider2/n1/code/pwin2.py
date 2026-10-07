"""pwin2.py JSON t0_us t1_us: per-hlo_name busy time (sum of instruction durations per engine) inside a window, top 30, and
the window's wall span covered by each hlo (first..last instruction)."""
import sys, json, collections
d = json.load(open(sys.argv[1])); t0, t1 = float(sys.argv[2]) * 1e3, float(sys.argv[3]) * 1e3
ins = [x for x in d['instruction'] if t0 <= x['timestamp'] < t1]
agg = collections.defaultdict(lambda: collections.Counter()); span = {}
for x in ins:
    nm = (x.get('hlo_name') or '<none>').split(' = ')[0][:28]
    op = (x.get('hlo_name') or '').split(' = ')[1].split('(')[0][:18] if ' = ' in (x.get('hlo_name') or '') else ''
    key = nm + ' ' + op
    agg[key][x['label']] += x['duration']
    a, b = span.get(key, (1e18, 0)); span[key] = (min(a, x['timestamp']), max(b, x['timestamp'] + x['duration']))
rows = sorted(agg.items(), key=lambda kv: -max(kv[1].values()))
for k, c in rows[:30]:
    a, b = span[k]
    print(f'{k:50s} span {(b - a) / 1e3:7.1f}us [{a / 1e3:8.0f}-{b / 1e3:8.0f}] ' + ' '.join(f'{e[:6]}={v / 1e3:.0f}' for e, v in c.most_common(4)))

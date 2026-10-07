import sys, json
for l in sys.stdin:
    if l.startswith('{'):
        d = json.loads(l); print(d['experts_total_ms'], d['cfg'], flush=True)
    elif l.strip(): print(l.strip()[:200], flush=True)

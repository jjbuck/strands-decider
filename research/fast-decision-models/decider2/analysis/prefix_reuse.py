"""prefix_reuse.py: how much of each gate request's state was already sent by an earlier request (S11).

For every train-pool request, find the longest common text prefix with any shorter request on the same task
(and, for the second measure, the same task and source model, comparing only the conversation body after the
hook-specific header). Reports the share of state tokens that an exact prefix cache could have reused.
Text comparison only; no model or tokenizer.

    python3 prefix_reuse.py            # reads ~/decider2/evalkit/train_pool.jsonl
"""
import collections, json, os
from os.path import commonprefix

POOL = os.path.expanduser('~/decider2/evalkit/train_pool.jsonl')


def load():
    with open(POOL) as fh:
        for line in fh:
            o = json.loads(line)
            s = o['state'] if isinstance(o['state'], str) else json.dumps(o['state'])
            yield o, s


def reuse(groups):
    tot = new = 0
    fracs = []
    for v in groups.values():
        v.sort(key=lambda x: len(x[0]))
        for i, (s, nt) in enumerate(v):
            best = 0
            for j in range(i):
                t = v[j][0]
                if len(t) <= best:
                    continue
                best = max(best, len(commonprefix([s, t])))
            f = best / max(1, len(s))
            tot += nt
            new += nt * (1 - f)
            fracs.append(f)
    fracs.sort()
    n = len(fracs)
    return 1 - new / tot, fracs[n // 2], fracs[3 * n // 4], fracs[int(.9 * n)]


def main():
    by_task, by_body = collections.defaultdict(list), collections.defaultdict(list)
    n = trunc = 0
    for o, s in load():
        n += 1
        trunc += s.startswith('[earlier text omitted]')
        by_task[(o['domain'], o['task'])].append((s, o['n_state_tok']))
        body = s.split('--- CONVERSATION ---', 1)[-1]
        by_body[(o['domain'], o['task'], o.get('src_model'))].append((body, o['n_state_tok']))
    print(f'requests {n}; front-truncated {trunc / n:.2f}')
    for name, g in (('full state, same task', by_task), ('conversation body, same task and model', by_body)):
        share, med, p75, p90 = reuse(g)
        print(f'{name}: reusable share of state tokens {share:.3f}; per-request shared fraction median {med:.2f}, p75 {p75:.2f}, p90 {p90:.2f}')


if __name__ == '__main__':
    main()

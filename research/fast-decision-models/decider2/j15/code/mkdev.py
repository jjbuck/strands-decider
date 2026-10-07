"""Train-split dev sets (pure python). DEV: tau / uncertainty fitting. EXIT: depth-exit head training. Disjoint by tau task.
Items in evalkit suite format: {id, suite, domain, task, n_state_tok, state, questions}. Never touches eval tasks (train_pool is train-split)."""
import json, random, collections
KIT = '~/decider2/evalkit'
tp = [json.loads(l) for l in open(f'{KIT}/train_pool.jsonl')]
split = json.load(open(f'{KIT}/split.json'))
ev = set(split['eval_tasks']) if isinstance(split['eval_tasks'], list) else set(sum(split['eval_tasks'].values(), []))
tp = [r for r in tp if r['task'] not in ev and r['n_state_tok'] < 8000]
rng = random.Random(20261006)
tasks = sorted({r['task'] for r in tp}); rng.shuffle(tasks)
dev_tasks = set(tasks[:int(0.3 * len(tasks))])
def mk(r, suite, maxq):
    qs = list(r['questions']); rng.shuffle(qs); qs = sorted(qs[:maxq])
    return dict(id=f"{suite}-{r['rid']}", suite=suite, domain=r['domain'], task=r['task'], n_state_tok=r['n_state_tok'], state=r['state'],
                questions={q: r['questions'][q] for q in qs})
def pick(pool, n_short, n_long, suite, maxq):
    sh = [r for r in pool if r['n_state_tok'] < 4000]; lo = [r for r in pool if r['n_state_tok'] >= 4000]
    rng.shuffle(sh); rng.shuffle(lo)
    # stratify short by question-set so frequent qsets do not dominate (cap per qset)
    cap = collections.Counter(); out = []
    for r in sh:
        if len(out) >= n_short: break
        if cap[r['qset']] >= max(8, n_short // 8): continue
        cap[r['qset']] += 1; out.append(mk(r, suite, maxq))
    out += [mk(r, suite, maxq) for r in lo[:n_long]]
    return out
dev = pick([r for r in tp if r['task'] in dev_tasks], 700, 150, 'DEV', 4)
ext = pick([r for r in tp if r['task'] not in dev_tasks], 3000, 600, 'EXIT', 2)
for name, its in (('DEV', dev), ('EXIT', ext)):
    with open(f'~/decider2/j15/suites/{name}.jsonl', 'w') as f:
        for it in its: f.write(json.dumps(it) + '\n')
    nq = sum(len(it['questions']) for it in its)
    print(name, 'requests', len(its), 'questions', nq, 'tasks', len({it['task'] for it in its}), 'long', sum(it['n_state_tok'] >= 4000 for it in its),
          collections.Counter(q for it in its for q in it['questions']).most_common(12))

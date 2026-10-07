"""Fine-tune / eval data for the decision probes (box side; CPU only).
 probe : F0's CF-probe generator (evalkit; f0/build_cf.py: amount_vs_limit, date_order, status_equal, id_match, each +/- a distractor
         record for another id with the opposite answer) applied to real states windowed to <= ~560 tokens.
         train: train-split states; eval: eval-split states (evalkit split.json), 50 pairs per kind x distract (400 pairs).
 cfp_official : the official evalkit CF-probe pairs whose items fit [Q][state][Q] <= 1024 tokens.
 squad : SQuAD-MC-long: SQuAD v1.1 question, options = gold + 3 answers of other questions whose paragraph is in the passage,
         passage = gold paragraph + other paragraphs of the same article (<= 700 tokens), gold paragraph at a random place.
Writes ~/work/g4/data/ft.pkl"""
import os, re, json, copy, random, glob, pickle, collections, numpy as np
from tokenizers import Tokenizer
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

W = os.path.expanduser('~/work/g4'); D = f'{W}/data'
vmap = np.load(f'{D}/vmap.npy')
p = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/tokenizer.json'))[0]
TOK = Tokenizer.from_file(p)
MAXT = 1024


def enc(s):
    return vmap[np.array(TOK.encode(s, add_special_tokens=False).ids, dtype=np.int64)].astype(np.uint16)


# ------------------------------------------------------------------ conversation parsing (verbatim from f0/build_cf.py)
MSG = re.compile(r'^(user: |assistant: |assistant called |assistant also calls |tool result|system note from the steering hook|\[\.\.\.)', re.M)


def conv_bounds(s):
    m = re.search(r'^--- (CONVERSATION(?: SO FAR)?) ---\n', s, re.M)
    if not m: return None
    st = m.end(); nx = re.search(r'^--- ', s[st:], re.M)
    return st, (st + nx.start() if nx else len(s))


def messages(s):
    cb = conv_bounds(s)
    if not cb: return []
    st, en = cb; seg = s[st:en]
    starts = [m.start() for m in MSG.finditer(seg)]
    out = []
    for i, p in enumerate(starts):
        q = starts[i + 1] if i + 1 < len(starts) else len(seg.rstrip('\n'))
        t = seg[p:q]
        k = 'user' if t.startswith('user: ') else 'asst' if t.startswith('assistant: ') else 'call' if t.startswith('assistant called') or t.startswith('assistant also') else 'tool' if t.startswith('tool result') else 'note'
        out.append((st + p, st + q, k, t))
    return out


def insert_at(s, pos, line):
    return s[:pos] + line + '\n' + s[pos:]


def window(s, rng, budget):
    """header + a contiguous run of >= 4 conversation messages, <= budget tokens"""
    ms = messages(s)
    if len(ms) < 4: return None
    header = s[:conv_bounds(s)[0]]
    hl = len(enc(header)); lens = [len(enc(m[3])) for m in ms]
    starts = list(range(len(ms) - 3)); rng.shuffle(starts)
    for i in starts[:12]:
        tot = hl; j = i
        while j < len(ms) and tot + lens[j] <= budget: tot += lens[j]; j += 1
        if j - i >= 4:
            body = ''.join(t if t.endswith('\n') else t + '\n' for t in (m[3] for m in ms[i:j]))
            return header + body
    return None


# ------------------------------------------------------------------ CF-probe generator (f0/build_cf.py probe(), rng made explicit)
def rid6(rng): return ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(6))
def num_id(rng, n): return ''.join(rng.choice('0123456789') for _ in range(n))
def mkdate(y, m, d): return f'{y:04d}-{m:02d}-{d:02d}'


def perturb_digit(rng, v):
    idx = [i for i, ch in enumerate(v) if ch.isdigit()]
    if not idx: return None
    i = idx[-1] if rng.random() < 0.6 else rng.choice(idx[-4:])
    d = str((int(v[i]) + rng.choice([1, 2, 3, 7])) % 10)
    return v[:i] + d + v[i + 1:]


def record(rng, dom):
    if dom == 'banking_knowledge':
        key = 'acct_' + num_id(rng, 6)
        return 'get_account_details', {'account_id': key}, key, dict(account_id=key, owner_user_id=num_id(rng, 10)[:10], status=None, balance=None, daily_transfer_limit=None,
                                                                    statement_date=None, payment_due_date=None, account_class='Green Account (checking)')
    if dom == 'retail':
        key = '#W' + num_id(rng, 7)
        return 'get_order_details', {'order_id': key}, key, dict(order_id=key, user_id='user_' + num_id(rng, 4), status=None, total_paid=None, refund_limit=None,
                                                                order_date=None, return_deadline=None, payment_method_id='credit_card_' + num_id(rng, 7))
    key = rid6(rng)
    return 'get_reservation_details', {'reservation_id': key}, key, dict(reservation_id=key, user_id='user_' + num_id(rng, 4), status=None, total_paid=None, refund_limit=None,
                                                                       booking_date=None, cancellation_deadline=None, cabin='economy')


FIELDS_BY = {'banking_knowledge': ('balance', 'daily_transfer_limit', 'statement_date', 'payment_due_date', 'owner_user_id', ['open', 'frozen', 'closed']),
             'retail': ('total_paid', 'refund_limit', 'order_date', 'return_deadline', 'user_id', ['delivered', 'pending', 'cancelled']),
             'airline': ('total_paid', 'refund_limit', 'booking_date', 'cancellation_deadline', 'user_id', ['confirmed', 'cancelled', 'pending'])}


def noul(instr, t, f):
    return {'type': 'noul', 'instructions': instr, 'criteria': {'true': t, 'false': f}}


def probe(rng, s, dom, kind, distract):
    ms = messages(s)
    if len(ms) < 4: return None
    amt, lim, d1, d2, idf, stats = FIELDS_BY[dom]
    tool, args, key, base = record(rng, dom)
    rec = dict(base)
    L = round(rng.choice([250, 500, 750, 1000, 1500, 2000, 2500]) + 0.0, 2)
    y = 2024; mo = rng.randint(2, 10); dd = rng.randint(10, 19)
    rec[amt] = round(L + rng.choice([-1, 1]) * rng.randint(10, 90) + rng.randint(0, 99) / 100, 2); rec[lim] = L
    rec[d2] = mkdate(y, mo, dd); rec[d1] = mkdate(y, mo, dd - rng.randint(2, 8))
    rec['status'] = rng.choice(stats)
    A = copy.deepcopy(rec); B = copy.deepcopy(rec)
    keyname = {'get_account_details': 'account', 'get_order_details': 'order', 'get_reservation_details': 'reservation'}[tool]
    lab = f'the {keyname} {key}'
    if kind == 'amount_vs_limit':
        A[amt] = round(L - rng.randint(1, 9) * 10 - 0.5, 2); B[amt] = round(A[amt] + 2 * (L - A[amt]), 2)
        ques = noul(f"In the {tool} result for {lab}, is the {amt} greater than the {lim}?", f"the {amt} shown for {key} is larger than its {lim}", f"the {amt} shown for {key} is at most its {lim}")
        ea, eb = 'false', 'true'
    elif kind == 'date_order':
        A[d1] = mkdate(y, mo, dd - rng.randint(1, 6)); B[d1] = mkdate(y, mo, dd + rng.randint(1, 6))
        ques = noul(f"In the {tool} result for {lab}, is the {d1} earlier than the {d2}?", f"the {d1} of {key} comes before its {d2}", f"the {d1} of {key} is on or after its {d2}")
        ea, eb = 'true', 'false'
    elif kind == 'status_equal':
        sa = rng.choice(stats); sb = rng.choice([x for x in stats if x != sa]); A['status'] = sa; B['status'] = sb
        ques = noul(f"In the {tool} result for {lab}, is the status '{sa}'?", f"the status shown for {key} is exactly '{sa}'", f"the status shown for {key} is something else")
        ea, eb = 'true', 'false'
    else:
        v = A[idf]; nv = perturb_digit(rng, v); B[idf] = nv
        ques = noul(f"In the {tool} result for {lab}, is the {idf} {v}?", f"the {idf} shown for {key} is exactly {v}", f"the {idf} shown for {key} differs from {v} in any character")
        ea, eb = 'true', 'false'
    cand = [m for m in ms[1:] if m[0] < ms[0][0] + 0.6 * (ms[-1][1] - ms[0][0])]
    if not cand: return None
    m = rng.choice(cand)
    tool2, args2, key2, _ = record(rng, dom)
    ms_d = None

    def build(recd):
        out = insert_at(s, m[0], f'assistant called {tool}({json.dumps(args)})\ntool result ({tool}): {json.dumps(recd)}')
        if distract:
            D_ = copy.deepcopy(B if recd is A else A)
            D_[list(args2)[0]] = key2
            ms2 = messages(out); c2 = [x for x in ms2[1:] if abs(x[0] - m[0]) > 200]
            if not c2: return None
            p2 = c2[rng2_idx[0] % len(c2)]
            out = insert_at(out, p2[0], f'assistant called {tool}({json.dumps(args2)})\ntool result ({tool}): {json.dumps(D_)}')
        return out
    rng2_idx = [rng.randrange(10 ** 6)]          # same distractor slot for a and b
    sa_, sb_ = build(A), build(B)
    if not sa_ or not sb_: return None
    return sa_, ea, sb_, eb, ques


def render_q(q):
    return f"Question: {q['instructions']}\nAnswer true if {q['criteria']['true']}. Answer false if {q['criteria']['false']}.\nAnswer:"


def ex(task, qtext, state, y, **meta):
    q = enc(qtext); s = enc(state)
    if len(q) > 128 or 2 * len(q) + len(s) > MAXT: return None
    return dict(task=task, q=q, s=s, y=int(y), **meta)


KINDS = ['amount_vs_limit', 'date_order', 'status_equal', 'id_match']


def gen_probes(states, n_pairs, seed, budget_lo, budget_hi, balanced):
    rng = random.Random(seed); out = []; cnt = collections.Counter(); i = 0
    per = n_pairs // 8
    while len(out) < 2 * n_pairs and i < 50 * n_pairs:
        st = states[i % len(states)]; i += 1
        kk = (len(out) // 2)
        kind = KINDS[kk % 4]; distract = (kk // 4) % 2 == 1
        if balanced and cnt[(kind, distract)] >= per: continue
        w = window(st['state'], rng, rng.randint(budget_lo, budget_hi))
        if not w: continue
        r = probe(rng, w, st['domain'], kind, distract)
        if not r: continue
        sa, ea, sb, eb, ques = r
        qt = render_q(ques); pid = len(out) // 2
        a = ex('probe', qt, sa, ea == 'true', kind=kind + ('_distract' if distract else ''), pair=pid, domain=st['domain'])
        b = ex('probe', qt, sb, eb == 'true', kind=kind + ('_distract' if distract else ''), pair=pid, domain=st['domain'])
        if a is None or b is None: continue
        out += [a, b]; cnt[(kind, distract)] += 1
    print('probes', len(out), dict(cnt), flush=True)
    return out


def official_probe():
    EK = os.path.expanduser('~/work/g4/evalkit_suites')
    items = {json.loads(l)['id']: json.loads(l) for l in open(f'{EK}/CF-probe.jsonl')}
    out = []
    for l in open(f'{EK}/CF-probe.pairs.jsonl'):
        p = json.loads(l); A, B = items[p['a']], items[p['b']]
        qt = render_q(A['questions']['probe'])
        a = ex('cfp_official', qt, A['state'], p['ea'] == 'true', kind=p['kind'].replace('probe_', ''), pair=p['pair'], domain=p['domain'], iid=p['a'])
        b = ex('cfp_official', qt, B['state'], p['eb'] == 'true', kind=p['kind'].replace('probe_', ''), pair=p['pair'], domain=p['domain'], iid=p['b'])
        if a is not None and b is not None: out += [a, b]
    print('official CF-probe pairs that fit', len(out) // 2, flush=True)
    return out


def squad(split, n, seed, budget_lo, budget_hi):
    path = hf_hub_download('rajpurkar/squad', f'plain_text/{split}-00000-of-00001.parquet', repo_type='dataset')
    rows = pq.read_table(path).to_pylist()
    art = collections.defaultdict(lambda: collections.OrderedDict())
    for r in rows:
        art[r['title']].setdefault(r['context'], []).append((r['question'], r['answers']['text'][0]))
    qs = [(t, c, q, a) for t in art for c in art[t] for (q, a) in art[t][c]]
    rng = random.Random(seed); rng.shuffle(qs)
    plen = {}
    out = []
    norm = lambda x: re.sub(r'\W+', ' ', x.lower()).strip()
    for (t, c, q, a) in qs:
        if len(out) >= n: break
        budget = rng.randint(budget_lo, budget_hi)
        others = [x for x in art[t] if x != c]; rng.shuffle(others)
        if c not in plen: plen[c] = len(enc(c))
        paras = [c]; tot = plen[c]
        for o in others:
            if o not in plen: plen[o] = len(enc(o))
            if tot + plen[o] + 2 > budget: continue
            paras.append(o); tot += plen[o] + 2
        ga = norm(a)
        cands = []
        for pi, pc in enumerate(paras):
            pool = [aa for (qq, aa) in art[t][pc] if qq != q]
            rng.shuffle(pool)
            for aa in pool:
                na = norm(aa)
                if not na or na == ga or na in ga or ga in na or any(na == norm(x) for x in cands): continue
                cands.append(aa)
                if pi == 0 and len(cands) >= 3: break
            if len(cands) >= 3 and pi == 0: break
        if len(cands) < 3: continue
        opts = [a] + rng.sample(cands[:6], 3) if len(cands) >= 3 else None
        order = list(range(4)); rng.shuffle(order)
        opts = [opts[i] for i in order]; y = order.index(0)
        rng.shuffle(paras)
        clip = lambda x: x if len(x) <= 60 else x[:60]
        qt = f"Question: {q}\nOptions: A) {clip(opts[0])}\nB) {clip(opts[1])}\nC) {clip(opts[2])}\nD) {clip(opts[3])}\nAnswer:"
        e = ex('squad', qt, '\n\n'.join(paras), y, npara=len(paras))
        if e is not None: out.append(e)
    print('squad', split, len(out), flush=True)
    return out


if __name__ == '__main__':
    tr = [json.loads(l) for l in open(f'{D}/train_states.jsonl')]
    ev = [json.loads(l) for l in open(f'{D}/eval_states.jsonl')]
    data = dict(
        probe_train=gen_probes(tr, 20000, 1, 220, 560, balanced=False),
        probe_eval=gen_probes(ev, 400, 2, 540, 560, balanced=True),
        cfp_official=official_probe(),
        squad_train=squad('train', 40000, 3, 200, 700),
        squad_eval=squad('validation', 1500, 4, 680, 700),
    )
    for k, v in data.items():
        if v:
            L = np.array([len(e['s']) for e in v]); Q = np.array([len(e['q']) for e in v])
            print(k, len(v), 'state tok mean/p50/p90', L.mean().round(), np.percentile(L, 50), np.percentile(L, 90), 'q mean', Q.mean().round(),
                  'label mean', np.mean([e['y'] for e in v]).round(3))
    pickle.dump(data, open(f'{D}/ft.pkl', 'wb'))
    # print 2 examples for inspection
    inv = np.load(f'{D}/vmap.npy'); keep = np.full(32768, -1); keep[inv[inv < 32767]] = np.nonzero(inv < 32767)[0]
    for k in ('probe_eval', 'squad_eval'):
        e = data[k][0]
        print('=== EXAMPLE', k, 'y=', e['y'], e.get('kind'))
        print(TOK.decode([int(keep[i]) for i in e['q'] if keep[i] >= 0]))
        print(TOK.decode([int(keep[i]) for i in e['s'] if keep[i] >= 0])[:3000])

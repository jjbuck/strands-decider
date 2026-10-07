"""J7 detail-reading augmentation on TRAIN-split states only (evalkit/train_pool.jsonl minus split.json eval_tasks).
Written for J7: no template, field name, tool name or question wording is taken from the evalkit CF / CF-probe generators, and no
identity kind of CF (customer-stated details vs the user record, 'details_match' / 'identity_established') is generated, so CF and
CF-probe stay out-of-template tests.  Every pair (a, b) differs in exactly one value; labels by construction.
Kinds:
  stated_match : a customer message states a value, a later lookup result holds the same value or a one-character variant
                 (often in another format: 'April 17, 1979' vs '04/17/1979', '(617) 555-0834' vs '617-555-0834').
  lookup_value : the question names a value; the record for a named key holds it or a one-character variant.
  above_amount : the question gives a threshold; the record's amount is just above or just below it (formats vary).
  before_date  : the question gives a date in one format; the record's date (another format) is a few days before or after.
Half of the pairs carry a distractor record for another key with the opposite answer.  Records render as JSON, an indented
'key: value' block, or a markdown table.
python3 gen_aug.py OUT.jsonl [n_per_kind]"""
import os, re, sys, json, copy, random, collections

KIT = os.path.expanduser('~/work/evalkit') if os.path.exists(os.path.expanduser('~/work/evalkit')) else os.path.expanduser('~/decider2/evalkit')
EV = set(json.load(open(f'{KIT}/split.json'))['eval_tasks'])
MSG = re.compile(r'^(user: |assistant: |assistant called |assistant also calls |tool result|system note from the steering hook|\[\.\.\.)', re.M)
MON = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


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
    return [st + p for p in starts]


def digits(rng, n): return ''.join(rng.choice('0123456789') for _ in range(n))


def one_char(rng, v):
    idx = [i for i, ch in enumerate(v) if ch.isalnum()]
    i = rng.choice(idx[-6:]) if rng.random() < 0.7 else rng.choice(idx)
    ch = v[i]
    if ch.isdigit(): nc = str((int(ch) + rng.choice([1, 2, 3, 5, 7])) % 10)
    else: nc = rng.choice([c for c in 'abcdefghjkmnpqrstuvwxyz' if c != ch.lower()]); nc = nc.upper() if ch.isupper() else nc
    return v[:i] + nc + v[i + 1:]


def fmt_date(rng, y, m, d, style=None):
    style = style if style is not None else rng.randrange(4)
    if style == 0: return f'{y:04d}-{m:02d}-{d:02d}'
    if style == 1: return f'{m:02d}/{d:02d}/{y:04d}'
    if style == 2: return f'{MON[m - 1]} {d}, {y}'
    return f'{d} {MON[m - 1][:3]} {y}'


def fmt_amt(rng, v, style=None):
    style = style if style is not None else rng.randrange(3)
    if style == 0: return f'${v:,.2f}'
    if style == 1: return f'{v:.2f}'
    return f'{v:,.2f} USD'


# field generators: name, (value, variant) maker, how a customer says it
def f_phone(rng):
    v = f'{digits(rng, 3)}-{digits(rng, 3)}-{digits(rng, 4)}'
    return v, one_char(rng, v), (lambda x: f'({x[:3]}) {x[4:7]}-{x[8:]}' if rng.random() < 0.4 else x)


def f_zip(rng):
    v = digits(rng, 5); return v, one_char(rng, v), (lambda x: x)


def f_dob(rng):
    y, m, d = rng.randint(1950, 2002), rng.randint(1, 12), rng.randint(2, 27)
    d2 = d + rng.choice([-1, 1]); rs = rng.randrange(4)
    return fmt_date(rng, y, m, d, rs), fmt_date(rng, y, m, d2, rs), (lambda x: x)


def f_card4(rng):
    v = digits(rng, 4); return v, one_char(rng, v), (lambda x: x)


def f_acct(rng):
    v = digits(rng, rng.choice([8, 9, 10])); return v, one_char(rng, v), (lambda x: x)


def f_conf(rng):
    v = ''.join(rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ23456789') for _ in range(6)); return v, one_char(rng, v), (lambda x: x)


def f_email(rng):
    a = rng.choice(['jordan', 'maria', 'li', 'sam', 'priya', 'omar', 'kate', 'diego', 'anya', 'tom']); b = rng.choice(['lee', 'garcia', 'chen', 'patel', 'smith', 'novak', 'kim', 'silva'])
    v = f'{a}.{b}{digits(rng, 2)}@{rng.choice(["gmail.com", "outlook.com", "example.org", "mail.net"])}'
    i = v.index('@'); w = one_char(rng, v[:i]) + v[i:]
    return v, w, (lambda x: x)


def f_track(rng):
    v = digits(rng, 12); return v, one_char(rng, v), (lambda x: x)


FIELDS = [('phone on file', 'contact_phone', f_phone, 'my phone number is {x}'), ('postal code', 'postal_code', f_zip, 'my zip code is {x}'),
          ('date of birth', 'birth_date', f_dob, "I was born on {x}"), ('card ending', 'card_last_four', f_card4, 'the card ends in {x}'),
          ('account number', 'acct_number', f_acct, 'my account number is {x}'), ('confirmation code', 'confirmation', f_conf, 'the confirmation code is {x}'),
          ('email', 'email_address', f_email, 'you can reach me at {x}'), ('tracking number', 'tracking_no', f_track, 'the tracking number is {x}')]
ENT = {'banking_knowledge': ('member profile', 'lookup_member', 'member_ref', lambda r: 'M' + digits(r, 7)),
       'retail': ('customer file', 'fetch_customer', 'customer_ref', lambda r: 'C-' + digits(r, 6)),
       'airline': ('traveler profile', 'find_traveler', 'traveler_ref', lambda r: 'T' + digits(r, 6))}
AMT = [('balance_due', 'balance due'), ('credit_available', 'available credit'), ('order_value', 'order value'), ('pending_charge', 'pending charge')]
DTS = [('renewal_on', 'renewal date'), ('shipped_on', 'ship date'), ('opened_on', 'opening date'), ('expires_on', 'expiry date')]


def render_record(rng, tool, rec, style=None):
    style = style if style is not None else rng.randrange(3)
    if style == 0: return f'tool result ({tool}): ' + json.dumps(rec)
    if style == 1: return f'tool result ({tool}):\n' + '\n'.join(f'  {k}: {v}' for k, v in rec.items())
    return f'tool result ({tool}):\n| field | value |\n|---|---|\n' + '\n'.join(f'| {k} | {v} |' for k, v in rec.items())


def noul(instr, t, f): return {'type': 'noul', 'instructions': instr, 'criteria': {'true': t, 'false': f}}


def insert_at(s, pos, block): return s[:pos] + block + '\n' + s[pos:]


def make_pair(rng, r, kind, distract):
    s = r['state']; dom = r['domain'] if r['domain'] in ENT else 'retail'
    ms = messages(s)
    if len(ms) < 4: return None
    span = ms[-1] - ms[0]
    cand = [p for p in ms[1:] if p < ms[0] + 0.7 * span]
    if len(cand) < 2: return None
    ent, tool, refk, mkref = ENT[dom]
    key = mkref(rng); key2 = mkref(rng)
    rec = {refk: key}
    styl = rng.randrange(3)
    pre = []                                   # extra (pos, text) inserts shared by both items
    if kind == 'stated_match':
        fname, fkey, gen, say = rng.choice(FIELDS)
        v, w, sayf = gen(rng)
        if fkey == 'birth_date':               # the customer says it in another format half of the time
            y, mo, d = _parse_any(v)
            sv = fmt_date(rng, y, mo, d) if rng.random() < 0.5 else v
        else:
            sv = sayf(v)
        A = dict(rec); A[fkey] = v; B = dict(rec); B[fkey] = w
        A['status'] = B['status'] = rng.choice(['active', 'verified', 'on hold'])
        q = noul(rng.choice([f'Does the {fname} the customer gave in the chat agree exactly with the {fname} in the {tool} lookup?',
                             f'Is the {fname} that the customer typed identical to the one stored in their {ent}?']),
                 f'the customer stated a {fname} and it is identical to the stored one', f'the stored {fname} is different from what the customer stated, even by one character')
        p1, p2 = sorted(rng.sample(cand, 2))
        pre.append((p1, 'user: ' + rng.choice(['Sure, ', 'OK - ', 'Yes, ', '']) + say.replace('{x}', sv) + rng.choice(['.', '. Can you check?', ', thanks.'])))
        ea, eb = 'true', 'false'
    elif kind == 'lookup_value':
        fname, fkey, gen, _ = rng.choice(FIELDS)
        v, w, _ = gen(rng)
        A = dict(rec); A[fkey] = v; B = dict(rec); B[fkey] = w
        q = noul(f'Does the {ent} returned for {key} list {v} as the {fname}?', f'the {fname} stored for {key} is exactly {v}', f'the {fname} stored for {key} is not {v}')
        ea, eb = 'true', 'false'; p2 = rng.choice(cand)
    elif kind == 'above_amount':
        fk, fname = rng.choice(AMT)
        X = round(rng.choice([150, 300, 450, 800, 1200, 2500, 4000]) + rng.randint(0, 99) + rng.choice([0, 0.5, 0.25]), 2)
        dlt = round(rng.choice([rng.randint(1, 9), rng.randint(10, 60)]) + rng.randint(0, 99) / 100, 2)
        A = dict(rec); A[fk] = fmt_amt(rng, X + dlt); B = dict(rec); B[fk] = fmt_amt(rng, X - dlt)
        q = noul(f'Is the {fname} recorded for {key} higher than {fmt_amt(rng, X, rng.randrange(3))}?', f'the {fname} of {key} exceeds the amount in the question',
                 f'the {fname} of {key} is at most the amount in the question')
        ea, eb = 'true', 'false'; p2 = rng.choice(cand)
    else:  # before_date
        fk, fname = rng.choice(DTS)
        y, mo, d = rng.randint(2022, 2026), rng.randint(1, 12), rng.randint(8, 20)
        k = rng.randint(1, 6)
        st_q = rng.randrange(4); st_r = rng.choice([x for x in range(4) if x != st_q])
        A = dict(rec); A[fk] = fmt_date(rng, y, mo, d - k, st_r); B = dict(rec); B[fk] = fmt_date(rng, y, mo, d + k, st_r)
        q = noul(f"Is {key}'s {fname} earlier than {fmt_date(rng, y, mo, d, st_q)}?", f'the {fname} of {key} falls before the date in the question',
                 f'the {fname} of {key} is on or after the date in the question')
        ea, eb = 'true', 'false'; p2 = rng.choice(cand)
    if rng.random() < 0.5:
        A, B, ea, eb = B, A, eb, ea
    p3 = None
    if distract:                               # a record for another key that carries the OTHER item's value
        far = [p for p in ms[1:] if abs(p - p2) > 300 and all(abs(p - x) > 0 for x, _ in pre)]
        if not far: return None
        p3 = rng.choice(far)

    def build(recd, other):
        ins = list(pre) + [(p2, f'assistant called {tool}({json.dumps({refk: key})})\n' + render_record(rng, tool, recd, styl))]
        if p3 is not None:
            D = dict(other); D[refk] = key2
            ins.append((p3, f'assistant called {tool}({json.dumps({refk: key2})})\n' + render_record(rng, tool, D, (styl + 1) % 3)))
        o = s
        for pos, txt in sorted(ins, key=lambda t: -t[0]):
            o = insert_at(o, pos, txt)
        return o
    return build(A, B), ea, build(B, A), eb, q


def rng_fixed(seed): return random.Random(seed)


def _parse_any(v):
    m = re.match(r'(\d{4})-(\d{2})-(\d{2})$', v)
    if m: return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.match(r'(\d{2})/(\d{2})/(\d{4})$', v)
    if m: return int(m.group(3)), int(m.group(1)), int(m.group(2))
    m = re.match(r'([A-Za-z]+) (\d+), (\d{4})$', v)
    if m: return int(m.group(3)), MON.index(m.group(1)) + 1, int(m.group(2))
    m = re.match(r'(\d+) ([A-Za-z]{3}) (\d{4})$', v)
    return int(m.group(3)), [x[:3] for x in MON].index(m.group(2)) + 1, int(m.group(1))


def main():
    out = sys.argv[1]; npk = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    rng = random.Random(int(os.environ.get('SEED', 20261006 + 7)))
    pool = []
    with open(f'{KIT}/train_pool.jsonl') as f:
        for l in f:
            r = json.loads(l)
            if r['task'] in EV: continue
            if 250 <= r['n_state_tok'] <= 2600 and len(messages(r['state'])) >= 4: pool.append(r)
    rng.shuffle(pool)
    items = []; cnt = collections.Counter(); pid = 0; i = 0
    for kind in ['stated_match', 'lookup_value', 'above_amount', 'before_date']:
        for distract in (False, True):
            got = 0
            while got < npk // 2 and i < 10 ** 6:
                r = pool[i % len(pool)]; i += 1
                res = make_pair(rng, r, kind, distract)
                if not res: continue
                sa, ea, sb, eb, q = res
                for tag, st, e in (('a', sa, ea), ('b', sb, eb)):
                    items.append(dict(id=f'j7aug{pid}{tag}', kind=kind + ('_distract' if distract else ''), pair=pid, task=r['task'], domain=r['domain'],
                                      state=st, questions={'detail': q}, expected={'detail': e}))
                pid += 1; got += 1; cnt[kind + ('_d' if distract else '')] += 1
    with open(out, 'w') as f:
        for it in items: f.write(json.dumps(it) + '\n')
    print('pairs', pid, dict(cnt), 'items', len(items))


if __name__ == '__main__':
    main()

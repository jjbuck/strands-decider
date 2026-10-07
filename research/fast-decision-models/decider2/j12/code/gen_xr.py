"""J12: synthetic exact-reasoning pairs on TRAIN-split real states (evalkit/train_pool.jsonl minus split.json eval_tasks).
Templates, tool names, field names and question phrasings are written here and are DISJOINT from the CF-probe generator's
(no get_account/order/reservation_details, no balance/daily_transfer_limit/statement_date/payment_due_date/total_paid/refund_limit/
order_date/return_deadline/booking_date/cancellation_deadline, no "In the X result for ..." phrasing), so CF-probe is a template-OOD test.
Each pair (a, b) differs in one value and has opposite labels by construction. 'ptr' gives the gold pointer literals for the XR aux loss.
Kinds: num_cmp, num_const, date_cmp, date_window, date_const, status_eq, id_eq, ident1, ident2, user_amt, exists.
python3 gen_xr.py OUT.jsonl NPAIRS
"""
import os, re, sys, json, copy, random, collections, datetime

KIT = os.path.expanduser('~/decider2/evalkit') if os.path.exists(os.path.expanduser('~/decider2/evalkit')) else os.path.expanduser('~/work/evalkit')
EV = set(json.load(open(f'{KIT}/split.json'))['eval_tasks'])
MSG = re.compile(r'^(user: |assistant: |assistant called |assistant also calls |tool result|system note from the steering hook|\[\.\.\.)', re.M)
BANNED = {'balance', 'daily_transfer_limit', 'statement_date', 'payment_due_date', 'total_paid', 'refund_limit', 'order_date',
          'return_deadline', 'booking_date', 'cancellation_deadline', 'get_account_details', 'get_order_details', 'get_reservation_details'}

ENT = [  # (tool, entity word, key field, id maker)
    ('lookup_invoice', 'invoice', 'invoice_no', lambda R: 'INV-' + dg(R, 5)),
    ('fetch_ledger_entry', 'ledger entry', 'entry_ref', lambda R: 'LG' + dg(R, 4) + R.choice('ABCDEFGHJK')),
    ('get_card_summary', 'card', 'card_token', lambda R: 'ct_' + an(R, 6)),
    ('check_subscription', 'subscription', 'sub_id', lambda R: 'sub_' + dg(R, 5)),
    ('get_shipment_status', 'shipment', 'tracking_no', lambda R: '1Z' + an(R, 8).upper()),
    ('read_wallet', 'wallet', 'wallet_id', lambda R: 'wlt-' + an(R, 5)),
    ('get_loan_terms', 'loan', 'loan_no', lambda R: 'LN' + dg(R, 6)),
    ('query_policy_record', 'policy', 'policy_no', lambda R: 'POL-' + dg(R, 4) + '-' + dg(R, 2)),
    ('get_claim_file', 'claim', 'claim_id', lambda R: 'CLM' + dg(R, 6)),
    ('get_membership', 'membership', 'member_no', lambda R: 'M-' + dg(R, 5)),
    ('get_payment_plan', 'payment plan', 'plan_ref', lambda R: 'PP' + dg(R, 5)),
    ('inspect_voucher', 'voucher', 'voucher_code', lambda R: 'V' + an(R, 7).upper()),
    ('get_transfer_request', 'transfer request', 'transfer_ref', lambda R: 'TR' + dg(R, 7)),
    ('get_store_credit', 'store credit', 'credit_id', lambda R: 'SC-' + dg(R, 5)),
    ('fetch_contract', 'contract', 'contract_no', lambda R: 'K-' + dg(R, 6)),
    ('get_device_plan', 'device plan', 'line_id', lambda R: 'DL' + dg(R, 6)),
    ('get_rental_agreement', 'rental agreement', 'agreement_no', lambda R: 'RA' + dg(R, 5)),
    ('get_ticket_record', 'ticket', 'ticket_ref', lambda R: 'TK-' + dg(R, 6)),
    ('get_reward_summary', 'rewards account', 'rewards_id', lambda R: 'RW' + dg(R, 6)),
    ('view_case_file', 'case', 'case_no', lambda R: 'CASE-' + dg(R, 5)),
]
AMT = [('amount_due', 'credit_available'), ('requested_amount', 'approved_limit'), ('spent_this_cycle', 'cycle_cap'),
       ('refund_requested', 'max_refundable'), ('withdrawal_amount', 'withdrawal_cap'), ('claim_amount', 'coverage_limit'),
       ('outstanding_principal', 'payoff_quote'), ('current_usage', 'usage_quota'), ('points_needed', 'points_available'),
       ('transfer_amount', 'per_day_allowance'), ('order_value', 'free_shipping_minimum'), ('deposit_held', 'damage_charge'),
       ('price_quoted', 'price_ceiling'), ('fee_charged', 'fee_cap'), ('credit_used', 'credit_line'), ('reimbursement_sought', 'reimbursement_max')]
DAT = [('issued_on', 'expires_on'), ('shipped_at', 'promised_by'), ('claim_filed', 'policy_end'), ('start_date', 'end_date'),
       ('purchased_on', 'warranty_until'), ('requested_on', 'approval_deadline'), ('opened_on', 'last_activity'), ('renewal_date', 'cancel_by'),
       ('delivered_on', 'return_window_closes'), ('booked_on', 'free_change_until'), ('enrolled_on', 'grace_period_ends'), ('signed_on', 'effective_from')]
STAT = ['active', 'suspended', 'lapsed', 'under_review', 'approved', 'declined', 'in_transit', 'on_hold', 'pending_review', 'expired', 'settled', 'disputed']
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']


def dg(R, n): return ''.join(R.choice('0123456789') for _ in range(n))
def an(R, n): return ''.join(R.choice('abcdefghjkmnpqrstuvwxyz23456789') for _ in range(n))
def hum(f): return f.replace('_', ' ')


def fdate(R, o, fmt=None):
    d = datetime.date.fromordinal(o)
    fmt = fmt or R.choices(['iso', 'us', 'long', 'dmy'], [4, 2, 2, 1])[0]
    if fmt == 'iso': return d.isoformat()
    if fmt == 'us': return f'{d.month:02d}/{d.day:02d}/{d.year}'
    if fmt == 'long': return f'{MONTHS[d.month - 1]} {d.day}, {d.year}'
    return f'{d.day} {MONTHS[d.month - 1]} {d.year}'


def famt(R, v, style=None):
    style = style or R.choice(['plain', 'plain', 'dollar', 'comma'])
    if style == 'plain': return f'{v:.2f}'
    if style == 'dollar': return f'${v:,.2f}'
    return f'{v:,.2f}'


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
        k = 'user' if t.startswith('user: ') else 'asst' if t.startswith('assistant: ') else 'other'
        out.append((st + p, st + q, k, t))
    return out


class Doc:
    """a state under construction: insertions tracked so gold literal char offsets stay valid"""

    def __init__(self, s):
        self.s = s; self.marks = []   # (name, start, end)

    def insert(self, pos, text, marks=()):
        """insert text (+ '\n') at pos; marks: list of (name, substring) located inside text"""
        add = len(text) + 1
        self.marks = [(n, a + add if a >= pos else a, b + add if a >= pos else b) for n, a, b in self.marks]
        self.s = self.s[:pos] + text + '\n' + self.s[pos:]
        for n, sub in marks:
            i = text.find(sub[1]) if isinstance(sub, tuple) else text.find(sub)
            sub_ = sub[1] if isinstance(sub, tuple) else sub
            if isinstance(sub, tuple):   # (occurrence offset hint, substring): search after hint
                i = text.find(sub_, sub[0])
            assert i >= 0, (sub_, text)
            self.marks.append((n, pos + i, pos + i + len(sub_)))

    def mark(self, n):
        for nm, a, b in self.marks:
            if nm == n: return [a, b, self.s[a:b]]
        return None


def record_text(R, tool, key_field, key, fields, fmt=None):
    """-> (text, {field: (offset_hint, value_string)}) in one of 4 formats"""
    fmt = fmt or R.choices(['json', 'kv', 'jsonml', 'prose'], [4, 3, 2, 1])[0]
    loc = {}
    if fmt == 'json':
        d = {key_field: key}; d.update(fields)
        js = json.dumps(d)
        t = f'assistant called {tool}({json.dumps({key_field: key})})\ntool result ({tool}): {js}'
        base = t.find('tool result')
        for f, v in fields.items():
            i = t.find(json.dumps(f), base); loc[f] = (i + len(json.dumps(f)), str(v) if not isinstance(v, str) else v)
        return t, loc
    if fmt == 'jsonml':
        d = {key_field: key}; d.update(fields)
        js = json.dumps(d, indent=2)
        t = f'assistant called {tool}({json.dumps({key_field: key})})\ntool result ({tool}):\n{js}'
        base = t.find('tool result')
        for f, v in fields.items():
            i = t.find(json.dumps(f), base); loc[f] = (i + len(json.dumps(f)), str(v) if not isinstance(v, str) else v)
        return t, loc
    if fmt == 'kv':
        lines = [f'assistant called {tool}({key_field}="{key}")', f'tool result ({tool}): Found 1 record:', f'   {key_field}: {key}']
        for f, v in fields.items(): lines.append(f'   {f}: {v}')
        t = '\n'.join(lines)
        for f, v in fields.items():
            i = t.find(f'   {f}: '); loc[f] = (i + len(f) + 4, str(v))
        return t, loc
    parts = [f'the {hum(f)} is {v}' for f, v in fields.items()]
    t = f'assistant: I pulled up {key} with {tool}: ' + '; '.join(parts) + '.'
    for f, v in fields.items():
        i = t.find(f'the {hum(f)} is '); loc[f] = (i + len(hum(f)) + 8, str(v))
    return t, loc


def noul(instr, t, f):
    return {'type': 'noul', 'instructions': instr, 'criteria': {'true': t, 'false': f}}


def rec_fields(R, kind_fields):
    return kind_fields


def place(R, ms, frac=(0.0, 0.85), avoid=None):
    lo = ms[0][0]; hi = lo + frac[1] * (ms[-1][1] - lo)
    cand = [m for m in ms[1:] if m[0] <= hi and (avoid is None or abs(m[0] - avoid) > 150)]
    return R.choice(cand)[0] if cand else None


def gen_record_pair(R, s, kind):
    """record-based kinds. returns (state_a, ea, state_b, eb, question, ptr_a, ptr_b) with ptr = {'a': mark|qlit, 'b': ...}"""
    ms = messages(s)
    if len(ms) < 3: return None
    tool, ent, kf, mk = R.choice(ENT)
    key = mk(R)
    fields = {}; extra = {}
    st = R.choice(STAT); fields['status'] = st
    # fill a few generic extra fields for realism
    fa, fb = R.choice(AMT); da, db = R.choice(DAT)
    L = round(R.choice([150, 250, 400, 500, 750, 1000, 1200, 1500, 2000, 2500, 3000, 5000]) * R.choice([1, 1, 0.5, 2]), 2)
    fields[fa] = round(L * R.uniform(0.3, 1.7), 2); fields[fb] = L
    base_o = datetime.date(R.choice([2023, 2024, 2025]), R.randint(1, 12), R.randint(1, 28)).toordinal()
    fields[da] = base_o; fields[db] = base_o + R.randint(5, 120)
    fields['holder_zip'] = dg(R, 5)
    dfmt = R.choices(['iso', 'us', 'long', 'dmy'], [4, 2, 2, 1])[0]; afmt = R.choice(['plain', 'plain', 'dollar', 'comma'])
    A = dict(fields); B = dict(fields)
    qlit = None   # (name, literal) gold question-literal pointers
    hm = R.random() < 0.4
    F1 = (lambda f: hum(f) if hm else f)
    if kind == 'num_cmp':
        v2 = L; d = R.choice([0.01, 0.5, 1, 5, 10, 25, 50, 100]) * R.choice([1, 1, 2]); d = min(d, L * 0.5)
        if HARD: d = R.choice([0.01, 0.05, 0.1, 0.5])
        A[fa] = round(v2 - d, 2); B[fa] = round(v2 + d, 2)
        eq_case = R.random() < 0.2
        if eq_case: A[fa] = v2
        tpl = R.choice([
            ('Does the {f1} on {ent} {key} exceed its {f2}?', 'gt'), ('Is the {f1} recorded for {key} below the {f2}?', 'lt'),
            ('For {ent} {key}, is the {f1} at least the {f2}?', 'ge'), ("Going by the {tool} output, is {key}'s {f2} higher than its {f1}?", 'rgt'),
            ("Is {key}'s {f1} no more than its {f2}?", 'le'), ('According to the {tool} lookup, is the {f1} for {key} larger than the {f2}?', 'gt')])
        q = tpl[0].format(f1=F1(fa), f2=F1(fb), ent=ent, key=key, tool=tool)
        def lab(r):
            x, y = r[fa], r[fb]
            return {'gt': x > y, 'lt': x < y, 'ge': x >= y, 'le': x <= y, 'rgt': y > x}[tpl[1]]
        ques = noul(q, f'yes: for {key} the comparison holds', f'no: for {key} the comparison does not hold')
        pa, pb = ('rec', fa), ('rec', fb)
    elif kind == 'num_const':
        C = round(L * R.uniform(0.4, 1.6) / 10) * 10 + R.choice([0, 0, 0.5, 0.99]); C = round(max(C, 20), 2)
        d = R.choice([0.01, 1, 5, 10, 40]) if not HARD else R.choice([0.01, 0.1]); A[fa] = round(C - d, 2); B[fa] = round(C + d, 2)
        if R.random() < 0.2: A[fa] = C
        cs = famt(R, C, R.choice(['dollar', 'plain', 'comma']))
        tpl = R.choice([('Is the {f1} of {ent} {key} above {c}?', 'gt'), ('Is {key} showing a {f1} under {c}?', 'lt'),
                        ('Does the {f1} for {key} reach {c} or more?', 'ge')])
        q = tpl[0].format(f1=F1(fa), ent=ent, key=key, c=cs)
        lab = lambda r: {'gt': r[fa] > C, 'lt': r[fa] < C, 'ge': r[fa] >= C}[tpl[1]]
        ques = noul(q, 'yes, the threshold condition holds', 'no, it does not')
        pa, pb = ('rec', fa), ('q', cs)
    elif kind == 'date_cmp':
        k = R.randint(1, 40) if not HARD else R.choice([1, 2])
        A[da] = fields[db] - k; B[da] = fields[db] + k
        if R.random() < 0.15: A[da] = fields[db]
        tpl = R.choice([('Did the {d1} of {ent} {key} come before its {d2}?', 'lt'), ("Is {key}'s {d2} later than its {d1}?", 'lt'),
                        ('Was the {d1} on or after the {d2} for {key}?', 'ge'), ('For {key}, does the {d1} fall strictly after the {d2}?', 'gt')])
        q = tpl[0].format(d1=F1(da), d2=F1(db), ent=ent, key=key)
        lab = lambda r: {'lt': r[da] < r[db], 'ge': r[da] >= r[db], 'gt': r[da] > r[db]}[tpl[1]]
        ques = noul(q, 'yes, the dates are in that order', 'no, they are not')
        pa, pb = ('rec', da), ('rec', db)
    elif kind == 'date_window':
        N = R.choice([7, 14, 30, 60, 90]); k = R.randint(1, max(1, min(10, N // 2))) if not HARD else 1
        A[db] = fields[da] + N - (0 if R.random() < 0.25 else k); B[db] = fields[da] + N + k
        tpl = R.choice([('Is the {d2} of {key} no more than {n} days after its {d1}?', 'le'),
                        ('Does more than {n} days separate the {d1} and the {d2} of {ent} {key}?', 'gt'),
                        ('Is {key}\'s {d2} within {n} days of the {d1}?', 'le')])
        q = tpl[0].format(d1=F1(da), d2=F1(db), ent=ent, key=key, n=N)
        lab = lambda r: (r[db] - r[da] <= N) if tpl[1] == 'le' else (r[db] - r[da] > N)
        ques = noul(q, 'yes', 'no')
        pa, pb = ('rec', da), ('rec', db)
    elif kind == 'date_const':
        k = R.randint(1, 30); Co = fields[da]
        cs = fdate(R, Co)
        A[da] = Co - k; B[da] = Co + k
        tpl = R.choice([("Is {key}'s {d1} before {c}?", 'lt'), ('Was the {d1} on {ent} {key} later than {c}?', 'gt')])
        q = tpl[0].format(d1=F1(da), key=key, ent=ent, c=cs)
        lab = lambda r: r[da] < Co if tpl[1] == 'lt' else r[da] > Co
        ques = noul(q, 'yes', 'no')
        pa, pb = ('rec', da), ('q', cs)
    elif kind == 'status_eq':
        sa = st; sb = R.choice([x for x in STAT if x != sa]); A['status'] = sa; B['status'] = sb
        quote = R.random() < 0.6
        sq = f"'{sa}'" if quote else sa
        tpl = R.choice(['Is {ent} {key} currently in {s} status?', 'Does the {tool} record for {key} show the state {s}?',
                        'Is the status of {key} {s}?'])
        q = tpl.format(ent=ent, key=key, tool=tool, s=sq)
        lab = lambda r: r['status'] == sa
        ques = noul(q, f'the status shown for {key} is {sa}', f'the status shown for {key} is different')
        pa, pb = ('q', sa), ('rec', 'status')
    elif kind == 'id_eq':
        v = A['holder_zip']; nv = list(v); i = R.randrange(len(v)); nv[i] = str((int(nv[i]) + R.randint(1, 9)) % 10); nv = ''.join(nv)
        B['holder_zip'] = nv
        q = R.choice(['Is the holder_zip on file for {key} exactly {v}?', 'Does {ent} {key} list {v} as the holder zip?']).format(key=key, ent=ent, v=v)
        lab = lambda r: r['holder_zip'] == v
        ques = noul(q, f'the zip on file is {v}', 'the zip on file differs')
        pa, pb = ('q', v), ('rec', 'holder_zip')
    else:
        raise ValueError(kind)
    # render values
    def rend(r):
        out = {}
        for f in fields:
            v = r[f]
            if f in (da, db): out[f] = fdate(R, v, dfmt)
            elif f in (fa, fb): out[f] = famt(R, v, afmt) if afmt != 'plain' else round(float(v), 2)
            else: out[f] = v
        return out
    fmt = R.choices(['json', 'kv', 'jsonml', 'prose'], [4, 3, 2, 1])[0]
    pos = place(R, ms)
    if pos is None: return None
    distract = HARD or R.random() < 0.5
    nd = 3 if HARD else 1
    dposs = []
    if distract:
        for _ in range(nd):
            dp = place(R, ms, avoid=pos)
            if dp is None or dp in dposs: return None
            dposs.append(dp)
    dkeys = [mk(R) for _ in dposs]

    def build(r, other):
        doc = Doc(s)
        rr = rend(r)
        t, loc = record_text(R_fixed[0], tool, kf, key, rr, fmt)
        ins = [(pos, 0, t, [(f, (hint, vs)) for f, (hint, vs) in loc.items()])]
        for j, (dp, dk) in enumerate(zip(dposs, dkeys)):
            ro = rend(other)
            t2, _ = record_text(R_fixed[1], tool, kf, dk, ro, fmt)
            ins.append((dp, j + 1, t2, []))
        for p_, _, tt, mks in sorted(ins, key=lambda x: (-x[0], -x[1])):
            doc.insert(p_, tt, mks)
        return doc
    R_fixed = [random.Random(R.random()), random.Random(R.random())]
    s1 = R_fixed[0].getstate(); s2 = R_fixed[1].getstate()
    ea, eb = lab(A), lab(B)
    if ea == eb: return None
    da_ = build(A, B); R_fixed[0].setstate(s1); R_fixed[1].setstate(s2)
    db_ = build(B, A)

    def ptr(doc):
        out = {}
        for nm, (src, f) in (('a', pa), ('b', pb)):
            if src == 'rec':
                mk_ = doc.mark(f)
                if mk_ is None: return None
                out[nm] = ['s', mk_[0], mk_[2]]
            else:
                out[nm] = ['q', None, f]
        return out
    pA, pB = ptr(da_), ptr(db_)
    if pA is None or pB is None: return None
    if R.random() < 0.5:
        return db_.s, eb, da_.s, ea, ques, pB, pA
    return da_.s, ea, db_.s, eb, ques, pA, pB


FIRST = ['Wei', 'Amara', 'Lucas', 'Priya', 'Tomás', 'Hana', 'Omar', 'Greta', 'Kofi', 'Mei']
IDENT = [('phone number', 'linked_phone', lambda R: f'{dg(R, 3)}-{dg(R, 3)}-{dg(R, 4)}'),
         ('date of birth', 'holder_dob', lambda R: f'{R.randint(1, 12):02d}/{R.randint(1, 28):02d}/{R.randint(1950, 2002)}'),
         ('zip code', 'holder_zip', lambda R: dg(R, 5)),
         ('member number', 'member_ref', lambda R: 'MB' + dg(R, 6)),
         ('card ending', 'card_last4', lambda R: dg(R, 4))]
USAY = ['user: My {n} is {v}.', 'user: Sure, {n}: {v}', 'user: you can verify me with my {n}, {v}', 'user: It should be {v} (that is my {n}).']


def perturb(R, v):
    idx = [i for i, ch in enumerate(v) if ch.isdigit()]
    i = R.choice(idx[-5:])
    return v[:i] + str((int(v[i]) + R.randint(1, 9)) % 10) + v[i + 1:]


def gen_ident(R, s, two):
    ms = messages(s)
    us = [m for m in ms if m[2] == 'user']
    if len(ms) < 4 or not us: return None
    lo = ms[0][0]; hi = lo + 0.8 * (ms[-1][1] - lo)
    cu = [m for m in us if m[0] <= hi]
    if not cu: return None
    um = R.choice(cu)
    later = [m for m in ms if m[0] > um[0]]
    if not later: return None
    rpos = R.choice(later)[0]; upos = later[0][0]
    picks = R.sample(IDENT, 2 if two else 1)
    vals = [p[2](R) for p in picks]
    tool, ent, kf, mk = R.choice(ENT[:12]); key = mk(R)
    j = R.randrange(len(picks))
    fmt = R.choice(['json', 'kv', 'jsonml'])

    def build(bad):
        doc = Doc(s)
        stated = list(vals)
        if bad: stated[j] = perturb(R_[0], vals[j])
        if two:
            ut = f'user: My {picks[0][0]} is {stated[0]} and my {picks[1][0]} is {stated[1]}.'
            doc.insert(upos, ut, [('u0', stated[0]), ('u1', (ut.find(stated[0]) + len(stated[0]), stated[1]))])
        else:
            ut = R_[1].choice(USAY).format(n=picks[0][0], v=stated[0])
            doc.insert(upos, ut, [('u0', stated[0])])
        flds = {'name': R_[1].choice(FIRST) + ' ' + R_[1].choice(['Chen', 'Okafor', 'Silva', 'Novak', 'Haddad'])}
        for p, v in zip(picks, vals): flds[p[1]] = v
        t, loc = record_text(R_[1], 'lookup_customer_profile', 'profile_id', 'cp_' + dg(R_[1], 6), flds, fmt)
        doc.insert(rpos + len(ut) + 1, t, [(f, loc[f]) for f in loc])
        return doc
    R_ = [random.Random(R.random()), random.Random(R.random())]
    st = [r.getstate() for r in R_]
    good = build(False)
    for r, x in zip(R_, st): r.setstate(x)
    badd = build(True)
    if two:
        q = R.choice(['Did the customer give both their {a} and their {b}, and do both agree with the lookup_customer_profile result?',
                      'Do the {a} and {b} the customer typed both match the customer profile that was looked up?']).format(a=picks[0][0], b=picks[1][0])
    else:
        q = R.choice(['Does the {a} the customer gave match what the lookup_customer_profile call returned?',
                      'Is the {a} stated by the customer identical to the one in the customer profile?']).format(a=picks[0][0])
    ques = noul(q, 'yes, it matches the record exactly', 'no, it differs from the record (even by one character)')
    jj = j if two else 0

    def ptr(doc):
        u = doc.mark(f'u{jj}'); r_ = doc.mark(picks[jj][1])
        return {'a': ['s', u[0], u[2]], 'b': ['s', r_[0], r_[2]]}
    if R.random() < 0.5: return good.s, 'true', badd.s, 'false', ques, ptr(good), ptr(badd)
    return badd.s, 'false', good.s, 'true', ques, ptr(badd), ptr(good)


def gen_user_amt(R, s):
    ms = messages(s)
    us = [m for m in ms if m[2] == 'user']
    if len(ms) < 4 or not us: return None
    um = R.choice(us[: max(1, int(len(us) * 0.8))])
    later = [m for m in ms if m[0] > um[0]]
    if not later: return None
    rpos = R.choice(later)[0]; upos = later[0][0]
    cap = round(R.choice([200, 300, 500, 750, 1000, 2000, 2500, 5000]) * 1.0, 2)
    d = R.choice([1, 5, 20, 50, 100]); xa = round(cap - min(d, cap / 2), 2); xb = round(cap + d, 2)
    capf = R.choice(['per_day_allowance', 'withdrawal_cap', 'approved_limit', 'max_refundable', 'cycle_cap'])
    tool, ent, kf, mk = R.choice(ENT); key = mk(R)
    fmt = R.choice(['json', 'kv', 'jsonml'])
    R_ = random.Random(R.random()); s0 = R_.getstate()
    tpl = R.choice(['user: I need to move ${x} today.', 'user: Please send ${x} for me.', 'user: Can I take out ${x}?',
                    'user: The amount is ${x}.'])

    def build(x):
        R_.setstate(s0)
        doc = Doc(s)
        xs = f'{x:,.2f}'
        ut = tpl.replace('{x}', xs)
        doc.insert(upos, ut, [('u', '$' + xs)])
        t, loc = record_text(R_, tool, kf, key, {'status': 'active', capf: cap}, fmt)
        doc.insert(rpos + len(ut) + 1, t, [(capf, loc[capf])])
        return doc
    A, B = build(xa), build(xb)
    q = R.choice(['Is the amount the customer asked for within the {c} shown for {ent} {key}?',
                  'Does the customer request stay at or under the {c} of {key}?']).format(c=hum(capf) if R.random() < 0.5 else capf, ent=ent, key=key)
    ques = noul(q, 'yes, the requested amount does not exceed it', 'no, the requested amount is above it')
    p = lambda doc: {'a': ['s', doc.mark('u')[0], doc.mark('u')[2]], 'b': ['s', doc.mark(capf)[0], doc.mark(capf)[2]]}
    if R.random() < 0.5: return A.s, 'true', B.s, 'false', ques, p(A), p(B)
    return B.s, 'false', A.s, 'true', ques, p(B), p(A)


def gen_exists(R, s):
    ms = messages(s)
    us = [m for m in ms if m[2] == 'user' and len(m[3]) > 40]
    if not us: return None
    um = R.choice(us)
    words = um[3][6:].split()
    if len(words) < 6: return None
    lead = ' '.join(words[:6])
    if '"' in lead: return None
    typ = R.choice(['amount', 'date', 'ref'])
    if typ == 'amount':
        lit = f'${R.randint(5, 3000)}.{R.randint(0, 99):02d}'; with_ = R.choice([f'It was {lit}.', f'They charged me {lit}.', f'I paid {lit} for it.'])
        without = R.choice(['It was quite a lot.', 'They charged me twice.', 'I paid for it already.']); nm = 'dollar amount'
    elif typ == 'date':
        o = datetime.date(2025, R.randint(1, 12), R.randint(1, 28)).toordinal(); lit = fdate(R, o)
        with_ = R.choice([f'This happened on {lit}.', f'I noticed it on {lit}.']); without = R.choice(['This happened recently.', 'I noticed it last week.']); nm = 'specific calendar date'
    else:
        lit = R.choice(['REF-', 'CN', 'Q-']) + dg(R, 6); with_ = f'My reference is {lit}.'; without = 'I do not have the reference handy.'; nm = 'reference number'
    end = um[1]
    seg = s[um[0]:end].rstrip('\n'); cut = um[0] + len(seg)

    def build(add):
        doc = Doc(s)
        doc.s = s[:cut] + ' ' + add + s[cut:]
        if add is with_:
            i = cut + 1 + add.find(lit); doc.marks.append(('lit', i, i + len(lit)))
        return doc
    A, B = build(with_), build(without)
    q = f'Does the customer\'s message that begins "{lead}" mention a {nm}?'
    ques = noul(q, f'yes, it includes a {nm}', f'no, it does not include one')
    pa = {'a': ['s', A.mark('lit')[0], A.mark('lit')[2]], 'b': None}
    pb = {'a': ['null', None, None], 'b': None}
    if R.random() < 0.5: return A.s, 'true', B.s, 'false', ques, pa, pb
    return B.s, 'false', A.s, 'true', ques, pb, pa


HARD = False
KINDS = [('num_cmp', 3), ('num_const', 2), ('date_cmp', 3), ('date_window', 2), ('date_const', 1), ('status_eq', 2), ('id_eq', 1),
         ('ident1', 2), ('ident2', 1), ('user_amt', 2), ('exists', 1)]


def main():
    global HARD
    HARD = len(sys.argv) > 4 and sys.argv[4] == 'hard'
    lo_t, hi_t = (3500, 7000) if HARD else (150, 3200)
    out = sys.argv[1]; npairs = int(sys.argv[2]); R = random.Random(20261006 + 12 + (int(sys.argv[3]) if len(sys.argv) > 3 else 0))
    pool = []
    with open(f'{KIT}/train_pool.jsonl') as f:
        for l in f:
            r = json.loads(l)
            if r['task'] in EV: continue
            if lo_t <= r['n_state_tok'] <= hi_t and conv_bounds(r['state']): pool.append(dict(state=r['state'], task=r['task'], domain=r['domain']))
    R.shuffle(pool); print('pool', len(pool), flush=True)
    kinds = [k for k, w in KINDS for _ in range(w)]
    items = []; cnt = collections.Counter(); pid = 0; i = 0; fails = collections.Counter()
    while pid < npairs and i < npairs * 20:
        r = pool[i % len(pool)]; i += 1
        kind = kinds[pid % len(kinds)] if R.random() < 0.7 else R.choice(kinds)
        try:
            if kind in ('ident1', 'ident2'): res = gen_ident(R, r['state'], kind == 'ident2')
            elif kind == 'user_amt': res = gen_user_amt(R, r['state'])
            elif kind == 'exists': res = gen_exists(R, r['state'])
            else: res = gen_record_pair(R, r['state'], kind)
        except AssertionError as e:
            fails[kind] += 1; continue
        if not res: fails[kind] += 1; continue
        sa, ea, sb, eb, ques, pa, pb = res
        ea = 'true' if ea is True else 'false' if ea is False else ea
        eb = 'true' if eb is True else 'false' if eb is False else eb
        assert ea != eb
        for b in BANNED: assert b not in ques['instructions'], b
        for tag, s, e, p in (('a', sa, ea, pa), ('b', sb, eb, pb)):
            items.append(dict(id=f'xr{pid}{tag}', kind=kind, pair=pid, task=r['task'], domain=r['domain'], state=s, questions={'x': ques},
                              expected={'x': e}, ptr=p))
        pid += 1; cnt[kind] += 1
    with open(out, 'w') as f:
        for it in items: f.write(json.dumps(it) + '\n')
    print('pairs', pid, dict(cnt), 'fails', dict(fails), 'items', len(items))


if __name__ == '__main__':
    main()

"""H7: CF-style reading augmentation on TRAIN-split states only (evalkit/train_pool.jsonl minus split.json eval_tasks).
Kinds (pairs a/b with labels by construction):
  probe_*  : F0's CF-probe generator (amount_vs_limit, date_order, status_equal, id_match, +/- distractor), copied from the g4 replica of
             f0/build_cf.py (recovered/g4/g4/prep_ft.py), applied to whole real states (no windowing).
  human_insert : a customer message asking for a human, inserted at a user turn in the first 75% of a conversation with no such request
                 (question = the request's own human question: cc_asked_for_human / asked_for_human / UserAskedForHuman).
  amount_insert: a sentence with a dollar amount inserted after the first sentence of a proposed message with no '$' (states_amount).
  wrapup_replace: proposed message replaced by a mid-work question (false) vs a closing message (true) (WrapsUp).
Templates are written here (not copied from the eval pairs). Output: one item per line {id, kind, pair, state, questions, expected}.
python3 gen_cf.py OUT.jsonl"""
import os, re, sys, json, copy, random, collections

KIT = os.path.expanduser('~/decider2/evalkit') if os.path.exists(os.path.expanduser('~/decider2/evalkit')) else os.path.expanduser('~/work/evalkit')
EV = set(json.load(open(f'{KIT}/split.json'))['eval_tasks'])
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
    if rng.random() < 0.5:   # swap a/b so the first item's label is not fixed per kind
        A, B = B, A; ea, eb = eb, ea
    cand = [m for m in ms[1:] if m[0] < ms[0][0] + 0.6 * (ms[-1][1] - ms[0][0])]
    if not cand: return None
    m = rng.choice(cand)
    tool2, args2, key2, _ = record(rng, dom)
    rng2_idx = [rng.randrange(10 ** 6)]

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
    sa_, sb_ = build(A), build(B)
    if not sa_ or not sb_: return None
    return sa_, ea, sb_, eb, ques


HUMAN_RX = re.compile(r'human|agent|representative|supervisor|real person|manager|transfer|someone else|operator|live person', re.I)
HUMAN_T = ["I'd really rather speak with an actual person about this. Can you connect me to someone?",
           "Please get me a human agent, I don't want to keep going back and forth with a bot.",
           "Is there a supervisor I could talk to instead?",
           "Could you put me through to a live representative, please?",
           "I want to talk to a real human being about my issue.",
           "Can I be transferred to one of your staff members? I'd prefer a person.",
           "Honestly, just connect me with a human. That would be easier.",
           "Let me speak to someone from your team directly, not an AI.",
           "I need a human representative to handle this for me.",
           "Get me a manager or a human agent please."]
AMT_T = ["Your refund of ${a} will be issued to the original payment method.",
         "The total comes to ${a}.",
         "That leaves a price difference of ${a}, which will be charged to your card.",
         "You will receive ${a} back within 5-7 business days.",
         "The new item costs ${a} in total, including tax.",
         "A refund of ${a} has been calculated for the returned items.",
         "The difference of ${a} will be refunded to your gift card."]
MID_T = ["Could you please confirm the order number you want me to look at?",
         "Before I proceed, can you tell me which item you'd like to exchange and the new option you prefer?",
         "I've located your account. Which of these orders would you like to modify?",
         "Thanks. To continue, I'll need your zip code to verify your identity.",
         "I can help with that. Do you want the refund to go to your original payment method or a gift card?",
         "Let me check that for you. Can you share the email address on the account?"]
END_T = ["Everything is taken care of: your request has been completed and you'll receive an email confirmation shortly. Is there anything else I can help you with?",
         "Your return has been processed successfully. Thanks for reaching out, and have a great day!",
         "All set! The exchange is complete. Let me know if there's anything else you need.",
         "Done. The order has been cancelled and the refund is on its way. Is there anything else I can do for you today?",
         "That's all finished on my end. Thank you for your patience, and take care!"]


def prop_bounds(s):
    m = re.search(r"^--- (ASSISTANT'S PROPOSED MESSAGE|PROPOSED MESSAGE[^\n]*) ---\n", s, re.M)
    if not m: return None
    st = m.end(); nx = re.search(r'^--- ', s[st:], re.M)
    return st, (st + nx.start() if nx else len(s))


def main():
    out = sys.argv[1]; rng = random.Random(20261006)
    pool = []
    with open(f'{KIT}/train_pool.jsonl') as f:
        for l in f:
            r = json.loads(l)
            if r['task'] in EV: continue
            pool.append(r)
    rng.shuffle(pool)
    items = []; cnt = collections.Counter(); pid = 0

    def emit(kind, r, sa, ea, sb, eb, qn, qspec):
        nonlocal pid
        for tag, s, e in (('a', sa, ea), ('b', sb, eb)):
            items.append(dict(id=f'h7cf{pid}{tag}', kind=kind, pair=pid, task=r['task'], domain=r['domain'], state=s, questions={qn: qspec}, expected={qn: e}))
        pid += 1; cnt[kind] += 1

    # probes: 4 kinds x distract, 180 pairs each
    short = [r for r in pool if 300 <= r['n_state_tok'] <= 4500 and len(messages(r['state'])) >= 4]
    i = 0
    for kind in ['amount_vs_limit', 'date_order', 'status_equal', 'id_match']:
        for distract in (False, True):
            got = 0
            while got < 180 and i < 10 ** 6:
                r = short[i % len(short)]; i += 1
                res = probe(rng, r['state'], r['domain'], kind, distract)
                if not res: continue
                sa, ea, sb, eb, ques = res
                emit('probe_' + kind + ('_distract' if distract else ''), r, sa, ea, sb, eb, 'probe', ques); got += 1
    # human_insert
    HQ = ('cc_asked_for_human', 'asked_for_human', 'UserAskedForHuman')
    got = 0
    for r in pool:
        if got >= 450: break
        qn = next((q for q in HQ if q in r['questions']), None)
        if not qn or r['n_state_tok'] > 5000: continue
        ms = messages(r['state']); us = [m for m in ms if m[2] == 'user']
        if not us or any(HUMAN_RX.search(m[3]) for m in us): continue
        lo = ms[0][0]; hi = lo + 0.75 * (ms[-1][1] - lo)
        cand = [m for m in us if m[0] <= hi]
        if not cand: continue
        m = rng.choice(cand)
        sb = insert_at(r['state'], m[0], 'user: ' + rng.choice(HUMAN_T))
        emit('human_insert', r, r['state'], 'false', sb, 'true', qn, r['questions'][qn]); got += 1
    # amount_insert / wrapup_replace (proposed-message edits)
    ga = gw = 0
    for r in pool:
        if ga >= 300 and gw >= 250: break
        if r['n_state_tok'] > 5000: continue
        pb = prop_bounds(r['state'])
        if not pb: continue
        st, en = pb; msg = r['state'][st:en]
        if 'states_amount' in r['questions'] and ga < 300 and '$' not in msg and not re.search(r'\d+\.\d\d', msg):
            mm = re.search(r'[.!?](\s)', msg)
            if not mm: continue
            a = f'{rng.randint(5, 900)}.{rng.randint(0, 99):02d}'
            sent = rng.choice(AMT_T).replace('{a}', a)
            nm = msg[:mm.end()] + sent + ' ' + msg[mm.end():]
            sb = r['state'][:st] + nm + r['state'][en:]
            emit('amount_insert', r, r['state'], 'false', sb, 'true', 'states_amount', r['questions']['states_amount']); ga += 1
        elif 'WrapsUp' in r['questions'] and gw < 250:
            tail = '\n' if msg.endswith('\n') else ''
            sa = r['state'][:st] + rng.choice(MID_T) + tail + r['state'][en:]
            sb = r['state'][:st] + rng.choice(END_T) + tail + r['state'][en:]
            emit('wrapup_replace', r, sa, 'false', sb, 'true', 'WrapsUp', r['questions']['WrapsUp']); gw += 1
    with open(out, 'w') as f:
        for it in items: f.write(json.dumps(it) + '\n')
    print('pairs', pid, dict(cnt), 'items', len(items))


if __name__ == '__main__':
    main()

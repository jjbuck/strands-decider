"""J11 synthetic decision corpus: structured states (conversation + tool calls with JSON results) and typed questions with EXACT labels.

Train families (iid test split by seed):
  cmp_num     field comparison: is <field_a> of <entity id> greater than <field_b> (or than a constant)?  distractor records flip the answer
  cmp_date    date order between two date fields of one record (mixed formats)
  id_match    do the details the customer stated match the record on file? (one-digit edits); and which record matches (choice)
  status      what is the status of <entity id>? (choice) / is it <status>? (noul), distractors carry other statuses
  policy2     a 2-condition rule over the record the customer is asking about (amount/date/status)
  multihop2   order -> account -> tier across two tool results and turns
  count       how many records satisfy a predicate (score levels) / how many times the customer asked for X
  argmax      which record has the highest <field> (choice over ids) / is the total above N
Held-out families (never trained; test only):
  H_policy3   3-condition rule mixing identity, date and amount conditions, new wording
  H_multihop3 three hops (order -> account -> branch -> region)
  H_long      cmp_num / status with 3-4x the distractors and filler (2-4k tokens)
  H_schema    cmp_num / status / cmp_date on a domain whose entity, key names and statuses never occur in training
Output rows use the strands_decider Example format (kind/state/instructions/options/label/task/instruction_variants) + 'family','split'.
usage: python synth.py OUTDIR [n_train]
"""
import json, random, sys, os, datetime as dt

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

DOMAINS = {
    "bank": dict(ent="account", tool="get_account_details", idf=lambda r: f"acct_{r.randint(100000, 999999)}",
                 nums=[("balance", "daily_transfer_limit"), ("pending_charges", "available_credit"), ("requested_withdrawal", "balance")],
                 dates=[("opened_on", "last_statement_on"), ("card_issued_on", "card_expires_on")],
                 status=("status", ["active", "frozen", "closed", "pending_review", "dormant"]), tiers=["basic", "silver", "gold", "platinum"]),
    "retail": dict(ent="order", tool="get_order_details", idf=lambda r: f"#W{r.randint(1000000, 9999999)}",
                   nums=[("order_total", "refund_cap"), ("items_returned", "items_ordered"), ("shipping_paid", "shipping_quote")],
                   dates=[("ordered_on", "delivered_on"), ("delivered_on", "return_window_ends")],
                   status=("status", ["processing", "shipped", "delivered", "cancelled", "returned"]), tiers=["standard", "plus", "premier"]),
    "airline": dict(ent="reservation", tool="get_reservation_details", idf=lambda r: "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6)),
                    nums=[("fare_paid", "travel_credit"), ("checked_bags", "bag_allowance"), ("change_fee", "voucher_value")],
                    dates=[("booked_on", "departure_date"), ("departure_date", "return_date")],
                    status=("status", ["confirmed", "cancelled", "checked_in", "waitlisted", "flown"]), tiers=["regular", "silver", "gold", "diamond"]),
    "telecom": dict(ent="line", tool="lookup_line", idf=lambda r: f"LN-{r.randint(10000, 99999)}",
                    nums=[("data_used_gb", "data_cap_gb"), ("amount_due", "autopay_limit"), ("roaming_minutes", "roaming_allowance")],
                    dates=[("bill_due_on", "paid_on"), ("activated_on", "contract_ends_on")],
                    status=("status", ["active", "suspended", "ported_out", "past_due", "cancelled"]), tiers=["prepaid", "unlimited", "business"]),
    "insurance": dict(ent="claim", tool="fetch_claim", idf=lambda r: f"CLM{r.randint(100000, 999999)}",
                      nums=[("claimed_amount", "coverage_limit"), ("deductible_paid", "deductible"), ("repair_estimate", "vehicle_value")],
                      dates=[("incident_date", "filed_on"), ("filed_on", "decision_due")],
                      status=("status", ["open", "approved", "denied", "under_review", "withdrawn"]), tiers=["bronze", "silver", "gold"]),
}
# held-out schema: entity, tool, keys and statuses never used in training
HELD = dict(ent="consignment", tool="trace_consignment", idf=lambda r: f"CN/{r.randint(1000, 9999)}/{r.choice('XYZQ')}",
            nums=[("declared_worth", "indemnity_ceiling"), ("gross_kilos", "permitted_kilos")],
            dates=[("lodged_at", "promised_by"), ("collected_at", "lodged_at")],
            status=("disposition", ["in_bond", "released", "held_at_customs", "abandoned", "relayed"]), tiers=["economy", "express"])

FILL_USER = ["Hi, I need some help please.", "Thanks for checking.", "Can you look into this for me?", "I've been waiting a while.",
             "Okay, that makes sense.", "Sorry, one more thing.", "I'm not sure what happened.", "Please hurry, I'm in a rush.",
             "I already explained this to someone else yesterday.", "Can you double check that?", "Alright.", "Hmm, let me think."]
FILL_ASST = ["Happy to help with that.", "Let me check that for you.", "One moment while I look this up.", "Thanks, I've pulled up the details.",
             "I understand, let's sort this out.", "Could you confirm a few details first?", "I can see the record now.",
             "Here is what I found.", "Is there anything else you need?", "I'll take a look at the history on this."]
NAMES = ["Maria Lopez", "James Chen", "Aisha Khan", "Tom Becker", "Yuki Tanaka", "Priya Nair", "Omar Haddad", "Elena Rossi", "Liam O'Brien",
         "Sofia Petrova", "David Kim", "Grace Okafor", "Noah Fischer", "Hana Novak", "Lucas Silva"]


def fmt_money(r, v, json_ok=True):
    k = r.random()
    if json_ok and k < 0.45: return v  # raw JSON number
    s = f"{v:,.2f}" if r.random() < 0.6 else f"{v:.2f}"
    return r.choice(["$", "USD ", ""]) + s


def fmt_date(r, d, json_ok=True):
    k = r.random()
    if k < 0.45: return d.isoformat()
    if k < 0.7: return f"{MONTHS[d.month - 1]} {d.day}, {d.year}"
    if k < 0.85: return f"{d.month:02d}/{d.day:02d}/{d.year}"
    return f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}"


def rand_date(r, lo=2021, hi=2025):
    a = dt.date(lo, 1, 1).toordinal(); b = dt.date(hi, 12, 28).toordinal()
    return dt.date.fromordinal(r.randint(a, b))


def num_pair(r, want_gt):
    """two values a, b with a > b iff want_gt; often close (hard), sometimes far."""
    base = round(r.choice([r.uniform(10, 200), r.uniform(200, 5000), r.uniform(5000, 90000)]), 2)
    mode = r.random()
    if mode < 0.4: delta = round(r.uniform(0.01, 0.05) * base, 2) or 0.01  # close
    elif mode < 0.6: delta = r.choice([0.01, 1.0, 0.5, 10.0])                 # very close / digit-level
    else: delta = round(r.uniform(0.1, 0.9) * base, 2) or 0.01
    hi_v, lo_v = round(base + delta, 2), base
    return (hi_v, lo_v) if want_gt else (lo_v, hi_v)


def int_pair(r, want_gt):
    b = r.randint(1, 12); a = b + r.randint(1, 3) if want_gt else max(0, b - r.randint(0, 3))
    return a, b


def is_int_field(f): return any(t in f for t in ("items", "bags", "minutes", "kilos", "_gb"))


COMPACT = os.environ.get("J11_COMPACT", "0") == "1"


def conv(r, turns, n_fill=(1, 4)):
    """interleave meaningful turns with filler; returns text. COMPACT: at most one filler turn (denser supervision per token)."""
    if COMPACT: n_fill = (0, 1)
    out = []
    for i, (who, txt) in enumerate(turns):
        for _ in range(r.randint(*n_fill) if i == 0 else r.randint(0, n_fill[1] - 1)):
            if r.random() < 0.5: out.append(("user", r.choice(FILL_USER)))
            else: out.append(("assistant", r.choice(FILL_ASST)))
        out.append((who, txt))
    return out


def render(r, D, turns, calls, newest=None):
    """calls: list of (tool, args dict, result obj)."""
    intro = r.choice([f"A conversation between a customer and a support assistant is below, then the tool calls the assistant has made so far.",
                      f"Below is a support chat followed by the assistant's tool calls and their results.",
                      f"Transcript of a customer service session, with the system lookups made during it."])
    lines = [intro, "", "--- CONVERSATION SO FAR ---"]
    for who, t in turns: lines.append(f"{who}: {t}")
    if newest is not None:
        lines += ["", "--- CUSTOMER'S NEWEST MESSAGE ---", newest]
    lines += ["", "--- TOOL CALLS ---"]
    if not calls: lines.append("(none)")
    for i, (tool, args, res) in enumerate(calls):
        lines.append(f"[{i + 1}] {tool}({json.dumps(args)})")
        lines.append("result: " + fmt_result(r, res))
    return "\n".join(lines)


def fmt_result(r, res):
    """JSON (indented or compact) or the YAML-like record listing real tool results use."""
    k = r.random()
    if k < 0.45: return json.dumps(res, indent=2)
    if k < 0.65: return json.dumps(res)
    recs = res if isinstance(res, list) else [res]
    out = [f"Found {len(recs)} record(s):", ""]
    for j, rec in enumerate(recs):
        items = list(rec.items())
        out.append(f"{j + 1}. Record ID: {items[0][1]}")
        for kk, vv in items[1:]: out.append(f"   {kk}: {vv}")
    return "\n".join(out)


def record(r, D, rid, fields, extra=True):
    rec = {"id": rid} if r.random() < 0.5 else {f"{D['ent']}_id": rid}
    items = list(fields.items())
    if extra:
        noise = [("customer", r.choice(NAMES)), ("currency", "USD"), ("channel", r.choice(["web", "phone", "store", "app"])),
                 ("notes", r.choice(["", "VIP", "flagged for review", "auto-renew on"])), ("region", r.choice(["north", "south", "east", "west"]))]
        items += r.sample(noise, r.randint(1, 3))
    r.shuffle(items)
    rec.update(items)
    return rec


def opts_noul(): return [["false", "the statement does not hold for this state"], ["true", "the statement holds for this state"]]


def ex(kind, state, instr, options, label, fam, variants=()):
    return dict(kind=kind, state=state, instructions=instr, options=options, label=label, task=f"j11_{fam}", family=fam,
                instruction_variants=list(variants))


# ------------------------------------------------------------------ families
def target_turns(r, D, tid, ask):
    who = r.choice(NAMES)
    t = [("user", f"Hi, this is {who}. {ask} My {D['ent']} is {tid}." if r.random() < 0.5 else f"{ask} It's about {D['ent']} {tid}."),
         ("assistant", r.choice(FILL_ASST))]
    return t


def fam_cmp_num(r, D, n_dis=(0, 3), filler=(1, 4)):
    fa, fb = r.choice(D["nums"]); want = r.random() < 0.5
    tid = D["idf"](r); recs = []
    mk = (lambda w: int_pair(r, w)) if is_int_field(fa) else (lambda w: num_pair(r, w))
    a, b = mk(want)
    const_mode = r.random() < 0.3
    if const_mode:
        c = b; b = mk(r.random() < 0.5)[1]
    fmt = (lambda v: v) if is_int_field(fa) else (lambda v: fmt_money(r, v))
    recs.append(record(r, D, tid, {fa: fmt(a), fb: fmt(b)}))
    for _ in range(r.randint(*n_dis)):
        a2, b2 = mk(not want); recs.append(record(r, D, D["idf"](r), {fa: fmt(a2), fb: fmt(b2)}))
    r.shuffle(recs)
    in_q = r.random() < 0.5
    ref = f"{D['ent']} {tid}" if in_q else f"the {D['ent']} the customer is asking about"
    turns = conv(r, target_turns(r, D, tid, "I have a question about my numbers."), filler)
    calls = [(D["tool"], {"ids": [x.get("id", x.get(f"{D['ent']}_id")) for x in recs]}, recs if len(recs) > 1 else recs[0])]
    state = render(r, D, turns, calls)
    if const_mode:
        cs = f"{c:,.2f}" if not is_int_field(fa) else str(c)
        instr = f"In the {D['tool']} result, is the {fa} of {ref} greater than {cs}?"
        label = int(a > c)
    else:
        instr = f"In the {D['tool']} result, is the {fa} of {ref} greater than its {fb}?"
        label = int(want)
    var = [instr.replace("greater than", "larger than"), instr.replace("is the", "does the").replace("greater than", "exceed").replace("?", "?")]
    return ex("noul", state, instr, opts_noul(), label, "cmp_num", var)


def fam_cmp_date(r, D, n_dis=(0, 3), filler=(1, 4)):
    fa, fb = r.choice(D["dates"]); want = r.random() < 0.5
    tid = D["idf"](r)
    def pair(w):
        d1 = rand_date(r); gap = r.choice([1, 2, 3, 7, 30, r.randint(1, 400)])
        d2 = dt.date.fromordinal(d1.toordinal() + gap)
        return (d1, d2) if w else (d2, d1)
    a, b = pair(want)
    recs = [record(r, D, tid, {fa: fmt_date(r, a), fb: fmt_date(r, b)})]
    for _ in range(r.randint(*n_dis)):
        a2, b2 = pair(not want); recs.append(record(r, D, D["idf"](r), {fa: fmt_date(r, a2), fb: fmt_date(r, b2)}))
    r.shuffle(recs)
    in_q = r.random() < 0.5
    ref = f"{D['ent']} {tid}" if in_q else f"the {D['ent']} the customer is asking about"
    turns = conv(r, target_turns(r, D, tid, "I need to check some dates."), filler)
    state = render(r, D, turns, [(D["tool"], {"ids": [x.get("id", x.get(f"{D['ent']}_id")) for x in recs]}, recs)])
    instr = f"For {ref}, is the {fa} date earlier than the {fb} date?"
    return ex("noul", state, instr, opts_noul(), int(want), "cmp_date", [f"For {ref}, does {fa} come before {fb}?"])


def mk_phone(r): return f"{r.randint(200, 989)}{r.randint(200, 999)}{r.randint(1000, 9999)}"
def fmt_phone(r, p): return r.choice([f"{p[:3]}-{p[3:6]}-{p[6:]}", f"({p[:3]}) {p[3:6]}-{p[6:]}", p, f"{p[:3]}.{p[3:6]}.{p[6:]}"])
def edit_digit(r, s):
    i = r.randrange(len(s)); c = r.choice([d for d in "0123456789" if d != s[i]]); return s[:i] + c + s[i + 1:]


def fam_id_match(r, D, filler=(1, 4), held=False):
    name = r.choice(NAMES); phone = mk_phone(r); dob = rand_date(r, 1950, 2003); zipc = f"{r.randint(10000, 99999)}"
    rec_vals = dict(phone=phone, dob=dob, zip=zipc)
    kinds = r.sample(["phone", "dob", "zip"], 2)
    match = r.random() < 0.5
    stated = dict(rec_vals)
    if not match:
        k = r.choice(kinds)
        if k == "dob":
            d = rec_vals["dob"]; stated["dob"] = dt.date.fromordinal(d.toordinal() + r.choice([-1, 1, 10, -10, 365]))
        else: stated[k] = edit_digit(r, rec_vals[k])
    def say(k):
        v = stated[k]
        if k == "phone": return f"my phone number is {fmt_phone(r, v)}"
        if k == "dob": return f"my date of birth is {fmt_date(r, v)}"
        return f"my zip code is {v}"
    tid = D["idf"](r)
    ask = f"I'm {name} and I want to access my {D['ent']}. " + " and ".join(say(k) for k in kinds) + "."
    turns = conv(r, [("assistant", "To verify your identity, please give me two details on file."), ("user", ask)], filler)
    def rec_out(vals):
        o = {"name": name, "phone": fmt_phone(r, vals["phone"]), "date_of_birth": fmt_date(r, vals["dob"]), "zip_code": vals["zip"]}
        o = record(r, D, tid, o, extra=r.random() < 0.5); return o
    choice_mode = r.random() < 0.35
    if not choice_mode:
        state = render(r, D, turns, [("find_customer", {"name": name}, rec_out(rec_vals))])
        instr = "Do the identifying details the customer stated match the customer record on file exactly?"
        return ex("noul", state, instr, opts_noul(), int(match), "id_match",
                  ["Did the customer give details that agree with the record found?", "Is the customer's stated identity consistent with the record?"])
    # choice: which record (by id) matches the stated details; others differ by one digit/day
    n = r.randint(2, 4); recs = []; ids = []
    true_i = r.randrange(n)
    for i in range(n):
        vals = dict(stated)
        if i != true_i:
            k = r.choice(kinds)
            if k == "dob": vals["dob"] = dt.date.fromordinal(stated["dob"].toordinal() + r.choice([-1, 1, 31]))
            else: vals[k] = edit_digit(r, stated[k])
        rid = f"CUST-{r.randint(1000, 9999)}"; ids.append(rid)
        o = {"customer_id": rid, "name": name, "phone": fmt_phone(r, vals["phone"]), "date_of_birth": fmt_date(r, vals["dob"]), "zip_code": vals["zip"]}
        recs.append(o)
    state = render(r, D, turns, [("search_customers", {"name": name}, recs)])
    instr = "Which customer record matches every detail the customer stated?"
    opts = [[i, f"the record with customer_id {i}"] for i in ids]
    return ex("choice", state, instr, opts, true_i, "id_match", ["Which of the found records is this customer's?"])


def fam_status(r, D, n_dis=(1, 4), filler=(1, 4)):
    key, vocab = D["status"]; tid = D["idf"](r)
    st = r.choice(vocab); recs = [record(r, D, tid, {key: st})]
    for _ in range(r.randint(*n_dis)):
        recs.append(record(r, D, D["idf"](r), {key: r.choice([v for v in vocab if v != st])}))
    r.shuffle(recs)
    in_q = r.random() < 0.5
    ref = f"{D['ent']} {tid}" if in_q else f"the {D['ent']} the customer is asking about"
    turns = conv(r, target_turns(r, D, tid, "What's going on with this?"), filler)
    state = render(r, D, turns, [(D["tool"], {"ids": [x.get("id", x.get(f"{D['ent']}_id")) for x in recs]}, recs)])
    if r.random() < 0.5:
        instr = f"According to the tool results, what is the {key} of {ref}?"
        opts = [[v, f"the {key} is {v}"] for v in vocab]
        return ex("choice", state, instr, opts, vocab.index(st), "status")
    ask = st if r.random() < 0.5 else r.choice(vocab)
    instr = f"According to the tool results, is the {key} of {ref} '{ask}'?"
    return ex("noul", state, instr, opts_noul(), int(ask == st), "status")


def fam_policy(r, D, ncond=2, filler=(1, 4), fam="policy2"):
    """rule over the target record; conditions drawn from amount<=cap, date on/after cutoff, status in set (+ identity for ncond 3)."""
    fa, _ = r.choice(D["nums"]); fd, _ = r.choice(D["dates"]); key, vocab = D["status"]
    tid = D["idf"](r)
    cap = round(r.uniform(50, 3000), 2) if not is_int_field(fa) else r.randint(2, 10)
    cutoff = rand_date(r, 2022, 2024)
    okset = r.sample(vocab, 2)
    conds = r.sample(["amt", "date", "status"], 2) if ncond == 2 else ["id", "date", "amt"]
    want = r.random() < 0.5
    # choose per-condition truth: all true if want, else exactly one false (hard) or random
    truth = {c: True for c in conds}
    if not want:
        for c in r.sample(conds, 1 if r.random() < 0.7 else 2): truth[c] = False
    def amt_v(ok):
        if is_int_field(fa): return cap - r.randint(0, 2) if ok else cap + r.randint(1, 3)
        return round(cap - r.choice([0.0, 0.01, r.uniform(0, cap * 0.5)]), 2) if ok else round(cap + r.choice([0.01, 1, r.uniform(1, cap)]), 2)
    def date_v(ok):
        g = r.choice([0, 1, 3, 30, 200]) if ok else -r.choice([1, 2, 30, 200]);  return dt.date.fromordinal(cutoff.toordinal() + g)
    fields = {}
    if "amt" in conds: fields[fa] = amt_v(truth["amt"]) if is_int_field(fa) else fmt_money(r, amt_v(truth["amt"]))
    else: fields[fa] = amt_v(r.random() < 0.5) if is_int_field(fa) else fmt_money(r, amt_v(r.random() < 0.5))
    fields[fd] = fmt_date(r, date_v(truth["date"])) if "date" in conds else fmt_date(r, rand_date(r))
    if "status" in conds: fields[key] = r.choice(okset) if truth["status"] else r.choice([v for v in vocab if v not in okset])
    else: fields[key] = r.choice(vocab)
    recs = [record(r, D, tid, fields)]
    for _ in range(r.randint(0, 2)):
        f2 = {fa: amt_v(r.random() < 0.5) if is_int_field(fa) else fmt_money(r, amt_v(r.random() < 0.5)), fd: fmt_date(r, rand_date(r)), key: r.choice(vocab)}
        recs.append(record(r, D, D["idf"](r), f2))
    r.shuffle(recs)
    capt = str(cap) if is_int_field(fa) else f"{cap:,.2f}"
    parts = {"amt": f"its {fa} is at most {capt}", "date": f"its {fd} is on or after {cutoff.isoformat()}",
             "status": f"its {key} is {' or '.join(okset)}", "id": "the customer's stated phone number matches the one on the record"}
    calls = [(D["tool"], {"ids": [x.get("id", x.get(f"{D['ent']}_id")) for x in recs]}, recs)]
    turns = target_turns(r, D, tid, "I'd like to request an exception.")
    if "id" in conds:
        ph = mk_phone(r); said = ph if truth["id"] else edit_digit(r, ph)
        turns.append(("user", f"You can reach me at {fmt_phone(r, said)}."))
        for x in recs:
            if x.get("id", x.get(f"{D['ent']}_id")) == tid: x["contact_phone"] = fmt_phone(r, ph)
            else: x["contact_phone"] = fmt_phone(r, mk_phone(r))
    state = render(r, D, conv(r, turns, filler), calls)
    if fam == "policy2":
        instr = (f"Policy: an exception may be granted for a {D['ent']} only if " + " and ".join(parts[c] for c in conds) +
                 f". Does the policy allow an exception for the {D['ent']} the customer is asking about?")
    else:  # held-out wording
        instr = (f"Rule check. Grant the request only when ALL of these hold for the customer's {D['ent']}: " +
                 "; ".join(f"({i + 1}) {parts[c]}" for i, c in enumerate(conds)) + ". Should the request be granted?")
    return ex("noul", state, instr, opts_noul(), int(all(truth.values())), fam)


def fam_multihop(r, D, hops=2, filler=(1, 4), fam="multihop2"):
    """order (in conversation) -> account (tool 1) -> tier (tool 2) [-> region (tool 3) for hops=3]."""
    n = r.randint(2, 4)
    orders = [f"#W{r.randint(1000000, 9999999)}" for _ in range(n)]
    accts = [f"acct_{r.randint(100000, 999999)}" for _ in range(n)]
    tiers = D["tiers"]; br = [f"BR-{r.randint(10, 99)}" for _ in range(n)]
    regions = ["north", "south", "east", "west", "central"]
    t_of = [r.choice(tiers) for _ in range(n)]; reg_of = [r.choice(regions) for _ in range(n)]
    ti = r.randrange(n)
    turns = conv(r, [("user", f"I'm calling about my order {orders[ti]}."), ("assistant", "Let me find the account it was placed on.")], filler)
    perm = list(range(n)); r.shuffle(perm)
    calls = [("get_orders", {"order_ids": orders}, [{"order_id": orders[i], "placed_on_account": accts[i]} for i in perm])]
    perm2 = list(range(n)); r.shuffle(perm2)
    if hops == 2:
        calls.append(("get_accounts", {"account_ids": accts}, [{"account_id": accts[i], "tier": t_of[i], "branch": br[i]} for i in perm2]))
        r.shuffle(calls)
        instr = "What tier is the account that the customer's order was placed on?"
        opts = [[t, f"the account tier is {t}"] for t in tiers]
        return ex("choice", render(r, D, turns, calls), instr, opts, tiers.index(t_of[ti]), fam)
    calls.append(("get_accounts", {"account_ids": accts}, [{"account_id": accts[i], "home_branch": br[i]} for i in perm2]))
    perm3 = list(range(n)); r.shuffle(perm3)
    calls.append(("get_branches", {"branch_ids": br}, [{"branch_id": br[i], "region": reg_of[i]} for i in perm3]))
    r.shuffle(calls)
    instr = "In which region is the home branch of the account that the customer's order was placed on?"
    opts = [[g, f"the region is {g}"] for g in regions]
    return ex("choice", render(r, D, turns, calls), instr, opts, regions.index(reg_of[ti]), fam)


def fam_count(r, D, filler=(1, 3)):
    if r.random() < 0.5:
        key, vocab = D["status"]; target = r.choice(vocab); n = r.randint(2, 9)
        recs = [record(r, D, D["idf"](r), {key: r.choice(vocab)}, extra=r.random() < 0.3) for _ in range(n)]
        c = sum(x[key] == target for x in recs)
        turns = conv(r, [("user", f"How many of my {D['ent']}s are {target}?")], filler)
        state = render(r, D, turns, [(D["tool"] + "_list", {}, recs)])
        instr = f"How many {D['ent']}s in the tool result have {key} '{target}'?"
    else:
        target = r.choice(["refund", "a manager", "a callback", "a discount"]); c = r.randint(0, 5)
        ts = []
        for _ in range(c): ts.append(("user", r.choice([f"I want {target}.", f"Please, I need {target}.", f"Again: {target}, please.", f"Can I get {target}?"])))
        for _ in range(r.randint(1, 5)): ts.append(("user", r.choice(FILL_USER)))
        r.shuffle(ts)
        turns = []
        for t in ts: turns += [t, ("assistant", r.choice(FILL_ASST))]
        state = render(r, D, turns, [])
        instr = f"How many separate times did the customer ask for {target}?"
    levels = ["none", "exactly one", "exactly two", "exactly three", "exactly four", "five or more"]
    return ex("score", state, instr, [[str(i), d] for i, d in enumerate(levels)], min(c, 5), "count")


def fam_argmax(r, D, filler=(1, 3)):
    fa, _ = r.choice(D["nums"]); n = r.randint(2, 5)
    ids = [D["idf"](r) for _ in range(n)]
    if is_int_field(fa): vals = r.sample(range(0, 40), n)
    else:
        base = r.uniform(50, 5000); vals = [round(base * r.uniform(0.5, 1.5), 2) for _ in range(n)]
        if len(set(vals)) < n: vals = [v + i * 0.01 for i, v in enumerate(vals)]
    recs = [record(r, D, i, {fa: (v if is_int_field(fa) else fmt_money(r, v))}, extra=r.random() < 0.4) for i, v in zip(ids, vals)]
    turns = conv(r, [("user", f"Which of my {D['ent']}s has the most {fa.replace('_', ' ')}?")], filler)
    state = render(r, D, turns, [(D["tool"] + "_list", {}, recs)])
    if r.random() < 0.6:
        instr = f"Which {D['ent']} in the tool result has the highest {fa}?"
        return ex("choice", state, instr, [[i, f"{D['ent']} {i}"] for i in ids], max(range(n), key=lambda k: vals[k]), "argmax")
    tot = sum(vals); thr = round(tot * r.choice([0.9, 0.97, 1.03, 1.1]), 2 if not is_int_field(fa) else 0)
    instr = f"Is the total {fa} across all {D['ent']}s in the tool result greater than {thr:,}?"
    return ex("noul", state, instr, opts_noul(), int(tot > thr), "argmax")


TRAIN_FAMS = dict(cmp_num=fam_cmp_num, cmp_date=fam_cmp_date, id_match=fam_id_match, status=fam_status,
                  policy2=lambda r, D: fam_policy(r, D, 2), multihop2=lambda r, D: fam_multihop(r, D, 2), count=fam_count, argmax=fam_argmax)


def held_out(r, name):
    D = DOMAINS[r.choice(list(DOMAINS))]
    if name == "H_policy3": return fam_policy(r, D, 3, fam="H_policy3")
    if name == "H_multihop3": return fam_multihop(r, D, 3, fam="H_multihop3")
    if name == "H_long":
        e = fam_cmp_num(r, D, n_dis=(8, 14), filler=(4, 10)) if r.random() < 0.5 else fam_status(r, D, n_dis=(8, 14), filler=(4, 10))
        e = lengthen(r, e, r.randint(20, 50), r.randint(8, 16))
        e["family"] = "H_long"; e["task"] = "j11_H_long"; return e
    if name == "H_schema":
        f = r.choice([fam_cmp_num, fam_status, fam_cmp_date]); e = f(r, HELD); e["family"] = "H_schema"; e["task"] = "j11_H_schema"; return e
    raise KeyError(name)


HELD_FAMS = ["H_policy3", "H_multihop3", "H_long", "H_schema"]


def lengthen(r, e, n_fill, n_calls):
    """neutral padding: extra filler turns at the start of the conversation and unrelated tool calls at the end (other entity types, fresh ids)."""
    st = e["state"]; head, sep, rest = st.partition("--- CONVERSATION SO FAR ---\n")
    fill = "".join(f"{r.choice(['user', 'assistant'])}: {r.choice(FILL_USER + FILL_ASST)}\n" for _ in range(n_fill))
    st = head + sep + fill + rest
    k = st.count("\n[") + 1
    for j in range(n_calls):
        D2 = DOMAINS[r.choice(list(DOMAINS))]; recs = []
        for _ in range(r.randint(1, 3)):
            fa, fb = r.choice(D2["nums"]); fd, _ = r.choice(D2["dates"])
            recs.append(record(r, D2, D2["idf"](r), {fa: fmt_money(r, r.uniform(5, 5000)), fd: fmt_date(r, rand_date(r))}))
        st += f"\n[{k + j}] {D2['tool']}_history({{}})\nresult: " + json.dumps(recs, indent=2)
    e["state"] = st; return e


def main():
    out = sys.argv[1]; n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 240000
    os.makedirs(out, exist_ok=True)
    fams = list(TRAIN_FAMS)
    r = random.Random(11)
    with open(f"{out}/synth_train.jsonl", "w") as f:
        for i in range(n_train):
            fam = fams[i % len(fams)]; e = TRAIN_FAMS[fam](r, DOMAINS[r.choice(list(DOMAINS))]); e["split"] = "train"
            if r.random() < (0.0 if COMPACT else 0.3): e = lengthen(r, e, r.randint(0, 20), r.randint(0, 6))
            f.write(json.dumps(e) + "\n")
    r = random.Random(20261006)
    with open(f"{out}/synth_test.jsonl", "w") as f:
        for fam in fams:
            for i in range(300):
                e = TRAIN_FAMS[fam](r, DOMAINS[r.choice(list(DOMAINS))]); e["split"] = "test_iid"; e["id"] = f"{fam}-{i}"
                if r.random() < (0.0 if COMPACT else 0.3): e = lengthen(r, e, r.randint(0, 20), r.randint(0, 6))
                f.write(json.dumps(e) + "\n")
        for fam in HELD_FAMS:
            for i in range(300):
                e = held_out(r, fam); e["split"] = "test_held"; e["id"] = f"{fam}-{i}"; f.write(json.dumps(e) + "\n")
    print("done", out)


if __name__ == "__main__":
    main()

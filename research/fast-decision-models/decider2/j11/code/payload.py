"""Exact value channels: a deterministic typed parse of rendered text into literals, with per-token payloads.

For every literal (date, phone, email, id/code, number, short string value) we keep:
  type  1 num, 2 date, 3 id/code/phone/email, 4 string value, 5 key name
  hash  64-bit hash of the CANONICAL form (dates -> ISO, phones -> digits, amounts -> repr of the float, ids -> uppercase alnum)
  val   float value (numbers; dates as proleptic ordinal days) or None
Per token we also keep a key hash (the JSON / 'key: value' field the token's value belongs to) and a record hash
(hash of the id of the innermost record -- JSON object or numbered listing item -- that contains the token).
Pure python (used in box-side prep and testable on the laptop).
"""
import re, hashlib, datetime as dt

MON = {m.lower(): i + 1 for i, m in enumerate(["January", "February", "March", "April", "May", "June", "July", "August", "September",
                                                "October", "November", "December"])}
MON3 = {k[:3]: v for k, v in MON.items()}
MONRX = "|".join(sorted(list(MON) + list(MON3), key=len, reverse=True))

R_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?Z?)?\b")
R_US = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")
R_MDY = re.compile(r"\b(" + MONRX + r")\.? (\d{1,2}),? (\d{4})\b", re.I)
R_DMY = re.compile(r"\b(\d{1,2}) (" + MONRX + r")\.? (\d{4})\b", re.I)
R_PHONE = re.compile(r"(?:\+?1[ -.]?)?(?:\(\d{3}\) ?|\b\d{3}[-. ])\d{3}[-. ]\d{4}\b")
R_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
R_ID = re.compile(r"(?<![\w$])#?[A-Za-z][\w/-]*\d[\w/-]*|(?<![\w$.,])\d+[A-Za-z][\w-]*")
R_NUM = re.compile(r"(?<![\w.])(?:USD ?|\$)?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])")
R_JKEY = re.compile(r'"([A-Za-z_][\w ]{0,40})"\s*:\s*')
R_YKEY = re.compile(r"^[ \t]*(?:\d+\.\s+)?([A-Za-z_][\w ]{0,30}):[ \t]+", re.M)
R_JSTR = re.compile(r':\s*"([^"\n]{1,48})"')
R_ITEM = re.compile(r"^[ \t]*\d+\.\s+[^\n]*$", re.M)
ID_KEY = re.compile(r"^(id|.*_id|record id|.*\bid)$", re.I)


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


def _date(y, m, d):
    try: return dt.date(int(y), int(m), int(d))
    except ValueError: return None


def literals(text):
    """-> list of (start, end, type, canonical, val) non-overlapping, by priority."""
    cand = []
    for m in R_ISO.finditer(text):
        d = _date(*m.groups()); d and cand.append((m.start(), m.end(), 2, d.isoformat(), float(d.toordinal()), 0))
    for m in R_US.finditer(text):
        d = _date(m.group(3), m.group(1), m.group(2)); d and cand.append((m.start(), m.end(), 2, d.isoformat(), float(d.toordinal()), 0))
    for m in R_MDY.finditer(text):
        mo = MON.get(m.group(1).lower()) or MON3.get(m.group(1).lower()[:3]); d = _date(m.group(3), mo, m.group(2))
        d and cand.append((m.start(), m.end(), 2, d.isoformat(), float(d.toordinal()), 0))
    for m in R_DMY.finditer(text):
        mo = MON.get(m.group(2).lower()) or MON3.get(m.group(2).lower()[:3]); d = _date(m.group(3), mo, m.group(1))
        d and cand.append((m.start(), m.end(), 2, d.isoformat(), float(d.toordinal()), 0))
    for m in R_PHONE.finditer(text):
        dg = re.sub(r"\D", "", m.group(0))[-10:]; cand.append((m.start(), m.end(), 3, "PH" + dg, None, 1))
    for m in R_EMAIL.finditer(text):
        cand.append((m.start(), m.end(), 3, m.group(0).lower(), None, 1))
    for m in R_ID.finditer(text):
        g = m.group(0).rstrip("-/")
        cand.append((m.start(), m.start() + len(g), 3, re.sub(r"[^A-Z0-9]", "", g.upper()), None, 2))
    for m in R_NUM.finditer(text):
        g = m.group(0); v = float(re.sub(r"[^\d.-]", "", g.replace("USD", "")) or 0)
        cand.append((m.start(), m.end(), 1, repr(v), v, 3))
    for m in R_JSTR.finditer(text):
        cand.append((m.start(1), m.end(1), 4, m.group(1).strip().lower(), None, 4))
    # long 10-digit runs with no separators are phones too
    out = []; taken = []
    cand.sort(key=lambda c: (c[5], c[0]))
    for c in cand:
        s, e = c[0], c[1]
        if e <= s: continue
        if any(not (e <= a or s >= b) for a, b in taken): continue
        typ, can, val = c[2], c[3], c[4]
        if typ == 1 and re.fullmatch(r"\d{10}", text[s:e]): typ, can, val = 3, "PH" + text[s:e], None
        taken.append((s, e)); out.append((s, e, typ, can, val))
    out.sort()
    return out


def keys_and_records(text):
    """-> (key_spans [(s,e,keyname)], record_spans [(s,e,record_canonical_id)])"""
    keys = []
    for m in R_JKEY.finditer(text):
        s = m.end(); e = s
        if s < len(text) and text[s] == '"':
            j = text.find('"', s + 1); e = j + 1 if j > 0 else s
        else:
            while e < len(text) and text[e] not in ",}\n]": e += 1
        keys.append((m.start(1), m.end(1), s, e, m.group(1).lower()))
    for m in R_YKEY.finditer(text):
        s = m.end(); e = text.find("\n", s); e = len(text) if e < 0 else e
        keys.append((m.start(1), m.end(1), s, e, m.group(1).lower()))
    recs = []
    # JSON objects via a brace stack (string-aware)
    st = []; ins = False; esc = False
    for i, ch in enumerate(text):
        if ins:
            if esc: esc = False
            elif ch == "\\": esc = True
            elif ch == '"': ins = False
            continue
        if ch == '"': ins = True
        elif ch == "{": st.append(i)
        elif ch == "}" and st:
            a = st.pop(); recs.append([a, i + 1, None])
    # numbered listing items: from the item line to the next item line / blank line
    items = list(R_ITEM.finditer(text))
    for j, m in enumerate(items):
        e = items[j + 1].start() if j + 1 < len(items) else len(text)
        bl = text.find("\n\n", m.start()); e = min(e, bl) if bl > 0 else e
        recs.append([m.start(), e, None])
    # record id: first id-like key's value inside the span
    for r in recs:
        for ks, ke, vs, ve, name in keys:
            if r[0] <= ks and ve <= r[1] and ID_KEY.match(name):
                v = text[vs:ve].strip().strip('"'); r[2] = re.sub(r"[^A-Z0-9]", "", v.upper()); break
    recs = [tuple(r) for r in recs if r[2]]
    return keys, recs


def token_payloads(text, offsets):
    """offsets: per-token (start, end) char spans into text. -> dict of per-token lists:
       typ (int), lh (int64 literal hash, 0 none), val (float or nan), start (1 if first token of a literal), kh (key hash), rh (record hash)"""
    n = len(offsets)
    typ = [0] * n; lh = [0] * n; val = [float("nan")] * n; start = [0] * n; kh = [0] * n; rh = [0] * n
    lits = literals(text)
    keys, recs = keys_and_records(text)
    # char -> literal index via sweep
    import bisect
    ls = [l[0] for l in lits]
    def lit_at(s, e):
        i = bisect.bisect_right(ls, e - 1) - 1
        if i >= 0 and lits[i][1] > s: return i
        return -1
    seen = set()
    for t, (s, e) in enumerate(offsets):
        if e <= s: continue
        i = lit_at(s, e)
        if i >= 0:
            L = lits[i]; typ[t] = L[2]; lh[t] = h64(L[3]); val[t] = L[4] if L[4] is not None else float("nan")
            if i not in seen: start[t] = 1; seen.add(i)
    # key names themselves (type 5) and key binding of value spans
    kspans = sorted((vs, ve, h64("K:" + name)) for _, _, vs, ve, name in keys)
    knames = sorted((ks, ke, h64(name)) for ks, ke, _, _, name in keys)
    vss = [k[0] for k in kspans]; kns = [k[0] for k in knames]
    rsorted = sorted(recs, key=lambda r: (r[0], -r[1]))
    for t, (s, e) in enumerate(offsets):
        if e <= s: continue
        i = bisect.bisect_right(vss, s) - 1
        if i >= 0 and kspans[i][1] > s: kh[t] = kspans[i][2]
        j = bisect.bisect_right(kns, s) - 1
        if j >= 0 and knames[j][1] > s and typ[t] == 0:
            typ[t] = 5; lh[t] = knames[j][2]
    # record binding: innermost (latest-starting) containing span wins
    tst = [o[0] for o in offsets]
    for a, b, rid in rsorted:
        code = h64("R:" + rid)
        for t in range(bisect.bisect_left(tst, a), bisect.bisect_right(tst, b)):
            s, e = offsets[t]
            if s >= a and e <= b and e > s: rh[t] = code
    return dict(typ=typ, lh=lh, val=val, start=start, kh=kh, rh=rh)


if __name__ == "__main__":
    txt = '''user: My phone is (206) 555-0293 and DOB 07/22/1985, order #W1234567.
result: [{"id": "acct_750650", "balance": "$1,234.56", "daily_transfer_limit": 1500.0, "status": "frozen", "opened_on": "March 5, 2024"}]
Found 1 record(s):

1. Record ID: 6680a37184
   phone_number: 206-555-0293
   date_of_birth: 1985-07-22
'''
    for l in literals(txt): print(l, repr(txt[l[0]:l[1]]))
    k, r = keys_and_records(txt); print(r)
    print([x[4] for x in k])

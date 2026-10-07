"""J7 (b)+(c): exactness-preserving value channels and structure coordinates for rendered decider inputs.  Pure python (regex), no model.

For each character of a rendered text we compute:
  val   : 64-bit hash of the CANONICAL value of the span the char belongs to (0 = not in a value span).  Canonical forms make
          format-variants identical:  04/17/1979 == 1979-04-17 == April 17, 1979;  $1,350.00 == 1350;  (617) 555-0834 == 617-555-0834;
          emails / ids lower-cased; short quoted strings ('delivered').  Equal values -> identical code vectors; a one-digit change -> an
          unrelated code.  This turns "does X match Y" into a vector-identity test that one attention head can do.
  place : for digits of a number, the place value relative to the decimal point (Abacus-style digit coordinate): units 0, tens 1, ...,
          tenths -1.  (0 = not a numeric digit; stored as place+8 in 1..31.)
  key   : hash of the field name governing the char (JSON "key": value, 'key: value' lines, markdown table column header).
  rec   : hash of the record instance (innermost JSON object, an indented 'key: value' block, a markdown table row).
  role  : line role (user / assistant / call / tool / note / other) x section bucket (the '--- SECTION ---' header, hashed to 8).
  turn  : recency of the conversation turn (1 = newest message, up to 31; 0 = not in a turn).
token_feats(text, offsets) reduces these to one value per token (the last char of the token that carries the feature)."""
import re, hashlib, bisect
import numpy as np

MONTHS = {m: i + 1 for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}
ROLE_PREFIX = [('user', re.compile(r'^(user: |customer: )')), ('asst', re.compile(r'^assistant: ')),
               ('call', re.compile(r'^(assistant called |assistant also calls |tool: )')), ('tool', re.compile(r'^(tool result|result: )')),
               ('note', re.compile(r'^(system note|\[\.\.\.)'))]
ROLES = ['other', 'user', 'asst', 'call', 'tool', 'note']
NOT_KEYS = {'user', 'assistant', 'tool', 'result', 'customer', 'http', 'https', 'note', 'content', 'id', 'score'}


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), 'little') or 1


def h_small(s, n):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), 'little') % n


def canon_num(s):
    t = s.replace('$', '').replace(',', '').replace('%', '').strip()
    try:
        v = float(t)
    except ValueError:
        return None
    if abs(v) >= 1e15: return 'num:' + t
    r = ('%.4f' % v).rstrip('0').rstrip('.')
    return 'num:' + (r if r not in ('-0', '') else '0')


RX = [  # (name, regex) in priority order; earlier matches claim their chars
    ('email', re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')),
    ('iso', re.compile(r'\b(\d{4})-(\d{2})-(\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2})?)?\b')),
    ('us', re.compile(r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b')),
    ('long', re.compile(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.? (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})\b')),
    ('dmy', re.compile(r'\b(\d{1,2}) (Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,? (\d{4})\b')),
    ('phone', re.compile(r'(?<![\w-])(?:\+?1[ .-]?)?\(?(\d{3})\)?[ .-]?(\d{3})[ .-](\d{4})(?![\w-])')),
    ('ident', re.compile(r'(?<![\w$])#?[A-Za-z_]*\d[A-Za-z0-9_-]*[A-Za-z_][A-Za-z0-9_-]*|(?<![\w$])#?[A-Za-z_]+[A-Za-z0-9_-]*\d[A-Za-z0-9_-]*|(?<![\w$.,])\d{5,}(?![\w.,])')),
    ('num', re.compile(r'(?<![\w.])-?\$?\d{1,3}(?:,\d{3})+(?:\.\d+)?%?|(?<![\w.])-?\$?\d+(?:\.\d+)?%?')),
    ('qstr', re.compile(r'"([A-Za-z][A-Za-z _-]{0,38})"(?!\s*:)|\'([A-Za-z][A-Za-z _-]{0,38})\'')),
]


def value_spans(text):
    """[(start, end, canonical, kind)] non-overlapping."""
    taken = np.zeros(len(text) + 1, dtype=bool)
    spans = []
    for name, rx in RX:
        for m in rx.finditer(text):
            a, b = m.span()
            if taken[a:b].any(): continue
            g = m.groups()
            c = None
            try:
                if name == 'email': c = 'em:' + m.group(0).lower()
                elif name == 'iso': c = 'dt:%04d-%02d-%02d' % (int(g[0]), int(g[1]), int(g[2]))
                elif name == 'us': c = 'dt:%04d-%02d-%02d' % (int(g[2]), int(g[0]), int(g[1]))
                elif name == 'long': c = 'dt:%04d-%02d-%02d' % (int(g[2]), MONTHS[g[0][:3].lower()], int(g[1]))
                elif name == 'dmy': c = 'dt:%04d-%02d-%02d' % (int(g[2]), MONTHS[g[1][:3].lower()], int(g[0]))
                elif name == 'phone': c = 'ph:' + ''.join(g)
                elif name == 'ident': c = 'id:' + m.group(0).lower().lstrip('#')
                elif name == 'num': c = canon_num(m.group(0))
                elif name == 'qstr':
                    s = (g[0] or g[1]).strip().lower()
                    c = 'str:' + s
                    a, b = m.span(1) if g[0] else m.span(2)
            except (KeyError, ValueError):
                c = None
            if c is None: continue
            taken[a:b] = True
            spans.append((a, b, c, name))
    spans.sort()
    return spans


def annotate(text):
    n = len(text)
    val = np.zeros(n, dtype=np.uint64); place = np.zeros(n, dtype=np.int8); key = np.zeros(n, dtype=np.uint64)
    rec = np.zeros(n, dtype=np.uint64); role = np.zeros(n, dtype=np.int16); turn = np.zeros(n, dtype=np.int8)
    for a, b, c, kind in value_spans(text):
        val[a:b] = h64(c)
        if kind == 'num':
            s = text[a:b]; dot = s.find('.')
            digs = [i for i, ch in enumerate(s) if ch.isdigit()]
            ref = dot if dot >= 0 else (digs[-1] + 1 if digs else 0)
            for i in digs:
                pv = sum(ch.isdigit() for ch in s[i + 1:ref]) if i < ref else -sum(ch.isdigit() for ch in s[ref + 1:i + 1])
                place[a + i] = max(1, min(31, pv + 8))
    # lines: sections, roles, turns, key: value lines, blocks
    lines = []; p = 0
    for ln in text.split('\n'):
        lines.append((p, p + len(ln), ln)); p += len(ln) + 1
    sec = 'top'; cur_role = 'other'; msgs = []   # (start,end) of each conversation message
    blk = None; tab_hdr = None
    for li, (a, b, ln) in enumerate(lines):
        m = re.match(r'^--- (.+?) ---\s*$', ln)
        if m:
            sec = re.sub(r'\d+', '#', m.group(1).strip().upper()); cur_role = 'other'; blk = None; tab_hdr = None
            role[a:b] = ROLES.index('other') + 6 * h_small(sec, 8)
            continue
        for rn, rx in ROLE_PREFIX:
            if rx.match(ln):
                cur_role = rn
                if 'CONVERSATION' in sec or 'MESSAGE' in sec or rn in ('call', 'tool'): msgs.append([a, b])
                break
        else:
            if msgs and msgs[-1][1] == a - 1 and ln.strip(): msgs[-1][1] = b   # continuation line of the current message
        role[a:b] = ROLES.index(cur_role) + 6 * h_small(sec, 8)
        # markdown table rows: record per row, key = column header
        if ln.lstrip().startswith('|') and ln.rstrip().endswith('|'):
            cells = []; q = a + ln.index('|')
            parts = ln.strip().strip('|').split('|')
            off = ln.index('|') + 1
            for c in parts:
                cells.append((a + off, a + off + len(c), c.strip())); off += len(c) + 1
            if tab_hdr is None: tab_hdr = [c[2].lower() for c in cells]; continue
            if all(set(c[2]) <= set('-: ') for c in cells): continue
            r = h64('row@%d' % a)
            for ci, (ca, cb, ct) in enumerate(cells):
                rec[ca:cb] = r
                hk = tab_hdr[ci] if ci < len(tab_hdr) else 'col%d' % ci
                if ci > 0 and len(cells) == 2 and cells[0][2]: hk = cells[0][2].lower()   # 2-col 'Item | Amount' tables: row label is the key
                key[ca:cb] = h64('k:' + re.sub(r'\s+', ' ', hk))
            continue
        tab_hdr = None
        # indented / plain 'key: value' lines (not role prefixes)
        km = re.match(r'^(\s*)(?:\d+\.\s+)?([A-Za-z][\w .()/-]{0,40}?):\s+(\S.*)$', ln)
        if km and km.group(2).split()[0].lower() not in NOT_KEYS and not any(rx.match(ln) for _, rx in ROLE_PREFIX):
            ind = len(km.group(1))
            if ind > 0 or re.match(r'^\s*\d+\.\s', ln):
                if blk is None or ind == 0 or re.match(r'^\s*\d+\.\s', ln): blk = h64('blk@%d' % a)
            else:
                blk = blk if blk is not None else h64('blk@%d' % a)
            va = a + km.start(3)
            key[va:b] = h64('k:' + km.group(2).strip().lower())
            rec[a:b] = blk
        elif not ln.strip():
            blk = None
    # JSON objects: innermost object = record, "key": value -> key over the value chars
    stack = []
    for i, ch in enumerate(text):
        if ch == '{': stack.append(i)
        elif ch == '}' and stack:
            s0 = stack.pop()
            if i - s0 < 4000:
                inner = rec[s0:i + 1]
                r = h64('obj@%d' % s0)
                rec[s0:i + 1] = np.where((inner == 0) | (np.arange(s0, i + 1) < 0), r, inner) if stack else np.where(inner == 0, r, inner)
    for m in re.finditer(r'"([A-Za-z_][\w .-]{0,40})"\s*:\s*', text):
        k = h64('k:' + m.group(1).lower())
        vs = m.end()
        if vs >= n: continue
        if text[vs] == '"':
            ve = text.find('"', vs + 1); ve = n if ve < 0 else ve + 1
        elif text[vs] in '{[':
            continue
        else:
            mm = re.match(r'[^,}\]\n]*', text[vs:]); ve = vs + mm.end()
        key[vs:ve] = k
    # turns: recency index of conversation messages
    for j, (a, b) in enumerate(reversed(msgs)):
        turn[a:b] = min(31, j + 1)
    return dict(val=val, place=place, key=key, rec=rec, role=role, turn=turn)


def token_feats(text, offsets, ann=None):
    """offsets: [(start, end)] char spans per token.  Returns dict of per-token int64 arrays."""
    ann = ann or annotate(text)
    T = len(offsets)
    out = {k: np.zeros(T, dtype=np.int64) for k in ('val', 'place', 'key', 'rec', 'role', 'turn')}
    for t, (a, b) in enumerate(offsets):
        if b <= a:
            if a > 0 and a <= len(text): a, b = a - 1, a
            else: continue
        for k in ('val', 'key', 'rec'):
            seg = ann[k][a:b]
            nz = seg[seg != 0]
            out[k][t] = int(nz[-1] % (1 << 62)) if len(nz) else 0
        seg = ann['place'][a:b]; nz = seg[seg != 0]
        out['place'][t] = int(nz[-1]) if len(nz) else 0
        out['role'][t] = int(ann['role'][b - 1])
        out['turn'][t] = int(ann['turn'][b - 1])
    return out


if __name__ == '__main__':
    import sys, json
    s = ('--- CONVERSATION SO FAR ---\nuser: My email is wei.chen@gmail.com and my phone number is (617) 555-0934, born April 17, 1979.\n'
         'assistant called get_user({"email": "wei.chen@gmail.com"})\n--- TOOL RESULT ---\nresult: Found 1 record(s):\n\n1. Record ID: 76ad9cc60e\n'
         '   name: Wei Chen\n   phone_number: 617-555-0834\n   date_of_birth: 04/17/1979\n'
         'tool result (get_order_details): {"order_id": "#W1234567", "status": "delivered", "total_paid": 1350.00, "refund_limit": 1,250.5}\n'
         '| Item | Amount |\n|---|---|\n| Monthly fee | $22.50 |\n| Minimum balance | $1,350 |\n')
    for a, b, c, k in value_spans(s): print(k, repr(s[a:b]), c)
    an = annotate(s)
    for k in ('key', 'rec', 'turn', 'role', 'place'):
        print(k, ''.join('.' if v == 0 else chr(65 + int(v) % 26) for v in an[k]).replace('\n', ' ')[:400])

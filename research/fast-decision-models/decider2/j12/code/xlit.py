"""J12 XR: deterministic typed-literal extraction (pure python; runs anywhere).

extract(text) -> list of Lit(start, end, eq, grp, val, blk_in, blk_ln)
  eq    : canonical equality key (string). Phones/digit-ids -> digits only; numbers -> normalised float repr; dates -> 'D<ordinal>'; ids/strings -> lower.
  grp   : None | 'num' | 'date'  (order group; a<b / a>b only defined within a group)
  val   : float order value (numbers: value; dates: day ordinal (+ fraction of day for a time))
  blk_in: innermost {...} object id (or line-paragraph id when not inside braces)
  blk_ln: paragraph id (maximal run of non-blank lines, also split at message starts)
"""
import re, datetime, bisect

MONTHS = {m: i + 1 for i, m in enumerate(['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])}
MON = r'(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
PATS = [
    ('email', r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+'),
    ('iso', r'(?<![\w-])(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?(?:\.\d+)?Z?)?(?![\w])'),
    ('us', r'(?<![\w/])(\d{1,2})/(\d{1,2})/(\d{4})(?![\w/])'),
    ('mdy', r'(?<![\w])(' + MON + r')\.? (\d{1,2})(?:st|nd|rd|th)?,? (\d{4})(?![\w])'),
    ('dmy', r'(?<![\w])(\d{1,2})(?:st|nd|rd|th)? (' + MON + r')\.?,? (\d{4})(?![\w])'),
    ('phone', r'(?<![\w-])(?:\+?1[ -])?\(?\d{3}\)?[-. ]\d{3}[-.]\d{4}(?![\w-])'),
    ('id', r'(?<![\w#-])#?[A-Za-z0-9]+(?:[_-][A-Za-z0-9]+)+(?![\w-])|(?<![\w#])#[A-Za-z0-9]+(?![\w])|(?<![\w])(?=[A-Za-z]*\d)(?=\d*[A-Za-z])[A-Za-z0-9]{4,}(?![\w])'),
    ('num', r'(?<![\w.,])(?:[$€£]\s?)?-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w])'),
    ('quote', r'"([^"\n]{1,40})"|\'([^\'\n]{1,40})\''),
    ('kv', r'(?m)^[ \t]*[-*]?[ \t]*[A-Za-z][A-Za-z _]{0,30}:[ \t]+([A-Za-z][A-Za-z_]{1,24})[ \t]*$'),
]
RX = [(n, re.compile(p)) for n, p in PATS]
FIELD = re.compile(r'["\']?([A-Za-z][A-Za-z0-9_ ]{0,40}?)["\']?\s*[:=]\s*["\'$]?\s*$')
WORD = re.compile(r'[a-z][a-z0-9]*')
MSG = re.compile(r'(?m)^(user: |assistant: |assistant called |assistant also calls |tool result|tool: |result: |system note|--- )')


def _ord(y, m, d, hh=0, mm=0, ss=0):
    try:
        return datetime.date(y, m, d).toordinal() + (hh * 3600 + mm * 60 + ss) / 86400.0
    except ValueError:
        return None


def _num(s):
    t = s.replace('$', '').replace('€', '').replace('£', '').replace(',', '').replace('%', '').strip()
    try:
        return float(t)
    except ValueError:
        return None


class Lit:
    __slots__ = ('start', 'end', 'eq', 'grp', 'val', 'kind', 'blk_in', 'blk_ln', 'text', 'field')

    def __init__(self, start, end, eq, grp, val, kind, text):
        self.start, self.end, self.eq, self.grp, self.val, self.kind, self.text = start, end, eq, grp, val, kind, text
        self.blk_in = self.blk_ln = -1; self.field = None

    def __repr__(self):
        return f'Lit({self.kind}:{self.text!r} eq={self.eq} grp={self.grp} val={self.val} blk={self.blk_in}/{self.blk_ln})'


def _mk(kind, m, text):
    s = m.group(0)
    if kind == 'email':
        return s.lower(), None, None
    if kind == 'iso':
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hh = int(m.group(4) or 0); mi = int(m.group(5) or 0); ss = int(m.group(6) or 0)
        o = _ord(y, mo, d, hh, mi, ss)
        if o is None: return None
        return ('D%.5f' % o), 'date', o
    if kind == 'us':
        mo, d, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        o = _ord(y, mo, d)
        if o is None: return None
        return ('D%.5f' % o), 'date', o
    if kind in ('mdy', 'dmy'):
        if kind == 'mdy': mon, d, y = m.group(1), int(m.group(2)), int(m.group(3))
        else: d, mon, y = int(m.group(1)), m.group(2), int(m.group(3))
        o = _ord(y, MONTHS[mon[:3].lower()], d)
        if o is None: return None
        return ('D%.5f' % o), 'date', o
    if kind == 'phone':
        dg = re.sub(r'\D', '', s)
        if len(dg) == 11 and dg[0] == '1': dg = dg[1:]
        return dg, None, None
    if kind == 'id':
        k = s.lower().lstrip('#')
        return k, None, None
    if kind == 'num':
        v = _num(s)
        if v is None: return None
        core = s.replace('$', '').replace('€', '').replace('£', '').replace('%', '').strip().replace(',', '')
        digits_only = core.isdigit()
        if digits_only and len(core) >= 5 and not s.startswith(('$', '€', '£')):
            if core[0] == '0': return core, None, None          # zip-like / leading-zero id: equality only
            return core, 'num', v                                  # long integer: id-like but also orderable
        return ('N%.6g' % v), 'num', v
    if kind == 'quote':
        g = m.group(1) if m.group(1) is not None else m.group(2)
        g = g.strip()
        if not g or len(g.split()) > 4: return None
        return g.lower(), None, None
    if kind == 'kv':
        return m.group(1).lower(), None, None
    return None


def _blocks(text, lits):
    """blk_ln: paragraph id (split at blank lines and message starts). blk_in: innermost brace object (approx; quotes respected) else paragraph."""
    # paragraph boundaries
    bounds = [0]
    for m in re.finditer(r'\n[ \t]*\n', text): bounds.append(m.end())
    for m in MSG.finditer(text): bounds.append(m.start())
    bounds = sorted(set(bounds))
    # brace objects: scan once
    obj_of = {}  # char pos -> innermost object start (only computed for literal starts)
    starts = sorted(set(l.start for l in lits))
    si = 0; stack = []; inq = False; esc = False
    n = len(text)
    want = set(starts)
    i = 0
    # fast path: if no braces at all
    if '{' not in text:
        for l in lits:
            p = bisect.bisect_right(bounds, l.start) - 1
            l.blk_ln = p; l.blk_in = 100000 + p
        return
    res = {}
    for i, ch in enumerate(text):
        if i in want: res[i] = stack[-1] if stack else None
        if inq:
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': inq = False
            elif ch == '\n': inq = False
            continue
        if ch == '"': inq = True
        elif ch == '{': stack.append(i)
        elif ch == '}':
            if stack: stack.pop()
        elif ch == '\n' and stack and len(stack) > 50: stack = []
    for l in lits:
        p = bisect.bisect_right(bounds, l.start) - 1
        l.blk_ln = p
        o = res.get(l.start)
        l.blk_in = (o if o is not None else 100000 + p)


def extract(text):
    taken = bytearray(len(text) + 1)
    lits = []
    for kind, rx in RX:
        for m in rx.finditer(text):
            if kind == 'quote' or kind == 'kv':
                gi = 1 if m.group(1) is not None else 2
                if kind == 'kv': gi = 1
                a, b = m.start(gi), m.end(gi)
            else:
                a, b = m.start(), m.end()
            if a >= b or any(taken[a:b]): continue
            r = _mk(kind, m, text)
            if r is None: continue
            eq, grp, val = r
            lits.append(Lit(a, b, eq, grp, val, kind, text[a:b]))
            for k in range(a, b): taken[k] = 1
    lits.sort(key=lambda l: l.start)
    _blocks(text, lits)
    for l in lits:     # field name: the key immediately before the value on the same line ('key: v', '"key": v', 'key=v')
        ls = text.rfind('\n', 0, l.start) + 1
        pre = text[max(ls, l.start - 60):l.start]
        m = FIELD.search(pre)
        if m: l.field = m.group(1).strip().lower().replace('_', ' ')
    return lits


def last_token(offsets, a, b, base=0):
    """index of the last token whose char span overlaps [a, b) (offsets sorted list of (s, e)); base added. None if none."""
    starts = [o[0] for o in offsets]
    j = bisect.bisect_left(starts, b) - 1
    while j >= 0 and offsets[j][0] == offsets[j][1]: j -= 1   # skip zero-width special tokens
    if j < 0 or offsets[j][1] <= a: return None
    return base + j


DATE_T = (1, 7, 14, 30, 60, 90, 180, 365)
NCAT = 96


def qnorm(qtext):
    return ' ' + ' '.join(WORD.findall(qtext.lower().replace('_', ' '))) + ' '


def build_slots(st_lits, st_off, q_lits, q_off, q0, q_text=''):
    """-> dict of python lists describing the slot set (state slots then question slots).
    st_lits / q_lits: Lit lists with char offsets into the rendered state text / rendered question text.
    st_off / q_off: tokenizer offset maps for those texts. q0 = number of state tokens.
    """
    q_eq = set(l.eq for l in q_lits)
    qn = qnorm(q_text)
    slots = []
    for l in st_lits:
        p = last_token(st_off, l.start, l.end, 0)
        if p is None or p >= q0: continue
        slots.append((p, l, 0))
    for l in q_lits:
        p = last_token(q_off, l.start, l.end, q0)
        if p is None: continue
        slots.append((p, l, 1))
    # join bits
    blk_in_hit = set(); blk_ln_hit = set()
    for p, l, isq in slots:
        if not isq and l.eq in q_eq:
            blk_in_hit.add(l.blk_in); blk_ln_hit.add(l.blk_ln)
    pos, eq, grp, val, cat, blk = [], [], [], [], [], []
    for p, l, isq in slots:
        m1 = int((not isq) and l.eq in q_eq)
        m2a = int((not isq) and l.blk_in in blk_in_hit)
        m2b = int((not isq) and l.blk_ln in blk_ln_hit)
        m3 = int((not isq) and l.field is not None and (' ' + ' '.join(WORD.findall(l.field)) + ' ') in qn)
        g = 0 if l.grp is None else (1 if l.grp == 'num' else 2)
        pos.append(p); eq.append(l.eq); grp.append(g); val.append(l.val if l.val is not None else 0.0)
        cat.append(((((isq * 2 + m1) * 2 + m2a) * 2 + m2b) * 2 + m3) * 3 + g)
        blk.append(('q', l.blk_in) if isq else ('s', l.blk_in))
    return dict(pos=pos, eq=eq, grp=grp, val=val, cat=cat, blk=blk, lits=[s[1] for s in slots], isq=[s[2] for s in slots])


def index_arrays(sl):
    """Integer arrays for the GPU readout (all python lists):
       eqg[N]   equality group id; blkg[N] block group id
       lvl[N]   global order level (or -1); lo[N], hi[N] first/last level index of the slot's group (inclusive)
       dgt[t][N], dlt[t][N]: for date slots: first level index with value > v + t (exclusive cut: mass = C[hi] - C[idx-1]) and
                last level index with value < v - t (mass = C[idx] - C[lo-1]); -1 / sentinel for non-date
       NL       number of levels
    """
    N = len(sl['pos'])
    em = {}; eqg = [em.setdefault(e, len(em)) for e in sl['eq']]
    bm = {}; blkg = [bm.setdefault(b, len(bm)) for b in sl['blk']]
    levels = sorted(set((g, v) for g, v in zip(sl['grp'], sl['val']) if g > 0))
    lid = {x: i for i, x in enumerate(levels)}
    glo = {}; ghi = {}
    for i, (g, v) in enumerate(levels):
        glo.setdefault(g, i); ghi[g] = i
    lvl = [lid[(g, v)] if g > 0 else -1 for g, v in zip(sl['grp'], sl['val'])]
    lo = [glo[g] if g > 0 else 0 for g in sl['grp']]
    hi = [ghi[g] if g > 0 else -1 for g in sl['grp']]
    dvals = [v for (g, v) in levels if g == 2]
    d0 = glo.get(2, 0)
    dgt, dlt = [], []
    for t in DATE_T:
        a, b = [], []
        for g, v in zip(sl['grp'], sl['val']):
            if g != 2:
                a.append(-1); b.append(-1); continue
            k = bisect.bisect_right(dvals, v + t + 1e-6)       # first index with value > v + t  (strict)
            a.append(d0 + k)
            k2 = bisect.bisect_left(dvals, v - t - 1e-6) - 1    # last index with value < v - t  (strict)
            b.append(d0 + k2)
        dgt.append(a); dlt.append(b)
    return dict(N=N, eqg=eqg, blkg=blkg, lvl=lvl, lo=lo, hi=hi, dgt=dgt, dlt=dlt, NL=len(levels), G=len(em), B=len(bm), cat=sl['cat'], pos=sl['pos'])


def qwords(qtext, q_lits, s_lits):
    """plain question words (>= 3 chars) that exactly equal a state string literal (quote / kv / id kinds) become question literals"""
    keys = set(l.eq for l in s_lits if l.kind in ('quote', 'kv', 'id'))
    taken = [(l.start, l.end) for l in q_lits]
    out = []
    for m in re.finditer(r"[A-Za-z][A-Za-z_]{2,}", qtext):
        w = m.group(0).lower()
        if w in keys and not any(a < m.end() and m.start() < b for a, b in taken):
            l = Lit(m.start(), m.end(), w, None, None, 'qword', m.group(0)); l.blk_in = l.blk_ln = 0
            out.append(l)
    return out

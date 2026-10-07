"""J4: self-supervised DECISION-PRETRAINING corpus ("verification pretraining").

Every question is generated from a state's own text and its answer is exact by construction (string search / arithmetic on values
present in the state). No teacher, no human label, no eval item.  Question families (generic reading skills):
  verify  (noul)   'The state contains this exact text: "<span>"'  -- span copied from the state, or the same span with ONE detail
                   replaced (digit / number / date / same-type word from the same state).  Replaced-span detection asked as a decision.
                   Half of the verify questions come as contrast pairs (true span + its perturbed twin on the same state).
  lookup  (choice) 'Which value comes right after "<anchor>" in the state?'  true value + 2-4 same-type values from the same state (binding).
  order   (choice) 'Which of these appears earlier in the state?'  two unique spans.
  who     (choice) 'Who wrote the line containing "<span>"?'  user/customer, assistant, tool result.
  count   (choice) 'How many times does "<x>" appear in the state?'  0, 1, 2, 3, 4 or more.
  numcmp  (noul)   'The number right after "<anchor>" is greater than V.'
  datecmp (choice) 'Which date is earlier: the one right after "<a1>" or the one right after "<a2>"?'
Deliberately NOT generated (held out as far transfer for the CF suite): requests for a human, stated-vs-record identity matching, procedure
intent, amount-in-proposed-message, wrap-up detection.  CF-probe kinds (amount vs limit, date order, status, id match) are NEAR transfer:
the skills overlap (numcmp/datecmp/verify), the templates and data do not (no inserted records, real state content only).

python gen_dp.py OUT.jsonl --n 9000 --q 24 --seed 1
"""
import os, re, json, random, argparse, collections, sys

NUM = re.compile(r'(?<![\w.])\$?\d[\d,]*(?:\.\d+)?(?![\w])')
DATE = re.compile(r'\b(?:\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})\b')
IDT = re.compile(r'(?<![\w#])#?[A-Za-z_]*\d[A-Za-z0-9_\-]{3,}')
CAPW = re.compile(r'\b[A-Z][a-z]{2,}(?: [A-Z][a-z]{2,})?\b')
UPW = re.compile(r'\b[A-Z]{3,}(?:_[A-Z]+)*\b')
WORD = re.compile(r'[A-Za-z][A-Za-z_\-]{3,}')


def norm_ws(s):
    return ' '.join(s.split())


def occ(text, s):
    """non-overlapping occurrence count"""
    return text.count(s)


def perturb_digits(tok, rng):
    ds = [i for i, c in enumerate(tok) if c.isdigit()]
    if not ds: return None
    i = rng.choice(ds)
    c = tok[i]
    nc = rng.choice([d for d in '0123456789' if d != c and not (i == ds[0] and d == '0' and len(ds) > 1 and i == 0)])
    return tok[:i] + nc + tok[i + 1:]


def parse_num(tok):
    try:
        return float(tok.replace('$', '').replace(',', ''))
    except ValueError:
        return None


def fmt_like(tok, v):
    dol = tok.startswith('$'); com = ',' in tok
    dec = len(tok.split('.')[1]) if '.' in tok else 0
    s = f'{v:,.{dec}f}' if com else f'{v:.{dec}f}'
    return ('$' if dol else '') + s


def perturb_num(tok, rng):
    v = parse_num(tok)
    if v is None: return None
    if rng.random() < 0.5 or v == 0:
        return perturb_digits(tok, rng)
    f = rng.choice([0.5, 0.8, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5, 2.0])
    nt = fmt_like(tok, v * f)
    return nt if nt != tok else perturb_digits(tok, rng)


def perturb_date(tok, rng):
    if '/' in tok:
        a, b, c = tok.split('/')
        k = rng.randrange(3)
        parts = [a, b, c]
        n = int(parts[k]) + rng.choice([-2, -1, 1, 2])
        if k < 2: n = max(1, min(28 if k == 1 else 12, n))
        parts[k] = str(n).zfill(len(parts[k]))
        nt = '/'.join(parts)
    else:
        y, m, d = tok.split('-')
        k = rng.randrange(3); parts = [y, m, d]
        n = int(parts[k]) + rng.choice([-2, -1, 1, 2])
        if k == 1: n = max(1, min(12, n))
        if k == 2: n = max(1, min(28, n))
        parts[k] = str(n).zfill(len(parts[k]))
        nt = '-'.join(parts)
    return nt if nt != tok else None


def date_key(tok):
    try:
        if '/' in tok:
            a, b, c = tok.split('/'); return (int(c), int(a), int(b))
        y, m, d = tok.split('-'); return (int(y), int(m), int(d))
    except ValueError:
        return None


class Gen:
    def __init__(self, text, rng):
        self.t = text; self.rng = rng
        self.dets = []  # (start, end, typ, tok)
        seen = set()
        for typ, rx in (('date', DATE), ('num', NUM), ('id', IDT), ('up', UPW), ('cap', CAPW)):
            for m in rx.finditer(text):
                if any(not (m.end() <= s or m.start() >= e) for s, e in seen): continue
                seen.add((m.start(), m.end()))
                self.dets.append((m.start(), m.end(), typ, m.group(0)))
        self.bytype = collections.defaultdict(set)
        for s, e, typ, tok in self.dets: self.bytype[typ].add(tok)

    # ---- helpers
    def window(self, s, e, lw=None, rw=None):
        """word window around [s,e): lw words left, rw right, single line"""
        t = self.t
        ls = t.rfind('\n', 0, s) + 1; le = t.find('\n', e); le = len(t) if le < 0 else le
        left = t[ls:s].split(' '); right = t[e:le].split(' ')
        lw = self.rng.randint(1, 5) if lw is None else lw; rw = self.rng.randint(0, 5) if rw is None else rw
        L = ' '.join(left[-lw:]) if lw else ''
        if not lw: L = ''
        R = ' '.join(right[:rw + 1]) if rw else (right[0] if right else '')
        a0 = s - len(L); span = t[a0:e] + R if rw else t[a0:e]
        return span.strip(), a0

    def swap_same_type(self, typ, tok):
        c = [x for x in self.bytype[typ] if x != tok]
        if typ == 'num':
            c = [x for x in c if x.startswith('$') == tok.startswith('$')]
        return self.rng.choice(c) if c else None

    def perturb(self, typ, tok):
        r = self.rng.random()
        if typ == 'date': return perturb_date(tok, self.rng) if r < 0.7 else self.swap_same_type(typ, tok)
        if typ == 'num': return perturb_num(tok, self.rng) if r < 0.7 else self.swap_same_type(typ, tok)
        if typ == 'id': return perturb_digits(tok, self.rng) if r < 0.7 else self.swap_same_type(typ, tok)
        return self.swap_same_type(typ, tok)

    def pick(self, types=None):
        if types is None:   # prefer hard details: numbers, ids, dates (60%), status-like caps words (15%), capitalised words (25%)
            u = self.rng.random()
            types = ('num', 'id', 'date') if u < 0.6 else (('up',) if u < 0.75 else ('cap',))
            c = [d for d in self.dets if d[2] in types] or self.dets
        else:
            c = [d for d in self.dets if d[2] in types]
        return self.rng.choice(c) if c else None

    # ---- families
    def verify(self, pair=False):
        out = []
        for _ in range(6):
            d = self.pick()
            if d is None: return out
            s, e, typ, tok = d
            span, a0 = self.window(s, e)
            if len(span) < 8 or len(span) > 140 or '"' in span: continue
            if occ(self.t, span) < 1: continue
            alt = self.perturb(typ, tok)
            if not alt or alt == tok: continue
            off = s - a0
            fspan = span[:off] + alt + span[off + len(tok):]
            if fspan in self.t or norm_ws(fspan) in norm_ws(self.t): continue
            tmpl = self.rng.choice(['The state contains this exact text: "{}"', 'This text appears word for word in the state: "{}"',
                                    'The state includes the exact string "{}"'])
            want_true = self.rng.random() < 0.5
            if pair:
                out.append(self._noul(tmpl.format(span), 'true', 'verify'))
                out.append(self._noul(tmpl.format(fspan), 'false', 'verify'))
            else:
                out.append(self._noul(tmpl.format(span if want_true else fspan), 'true' if want_true else 'false', 'verify'))
            return out
        return out

    def lookup(self):
        for _ in range(6):
            d = self.pick(('num', 'id', 'date', 'up', 'cap'))
            if d is None: return []
            s, e, typ, tok = d
            ls = self.t.rfind('\n', 0, s) + 1
            left = self.t[ls:s]
            words = left.split(' ')
            k = self.rng.randint(2, 6)
            anchor = ' '.join(words[-k:]).strip() if len(words) >= 2 else left.strip()
            if len(anchor) < 6 or len(anchor) > 80 or '"' in anchor: continue
            if occ(self.t, anchor) != 1: continue
            if not self.t[ls:s].endswith(anchor) and not self.t[ls:s].rstrip().endswith(anchor): continue
            pool = [x for x in self.bytype[typ] if x != tok and '"' not in x]
            if len(pool) < 1: continue
            K = min(len(pool), self.rng.randint(2, 4))
            opts = self.rng.sample(sorted(pool), K) + [tok]
            self.rng.shuffle(opts)
            if len(set(opts)) != len(opts): continue
            q = self.rng.choice(['In the state, which value comes right after "{}"?', 'What is written immediately after "{}" in the state?',
                                 'Read the state: which of these follows "{}"?']).format(anchor)
            return [self._choice(q, opts, tok, 'lookup')]
        return []

    def order(self):
        for _ in range(8):
            d1 = self.pick(); d2 = self.pick()
            if d1 is None or d2 is None or d1 == d2: return []
            sp1, a1 = self.window(d1[0], d1[1], lw=self.rng.randint(1, 3), rw=self.rng.randint(0, 3))
            sp2, a2 = self.window(d2[0], d2[1], lw=self.rng.randint(1, 3), rw=self.rng.randint(0, 3))
            if sp1 == sp2 or '"' in sp1 + sp2 or len(sp1) > 70 or len(sp2) > 70 or len(sp1) < 5 or len(sp2) < 5: continue
            if occ(self.t, sp1) != 1 or occ(self.t, sp2) != 1: continue
            p1, p2 = self.t.find(sp1), self.t.find(sp2)
            if abs(p1 - p2) < 40: continue
            first = sp1 if p1 < p2 else sp2
            opts = [sp1, sp2]; self.rng.shuffle(opts)
            return [self._choice('Which of these appears earlier in the state?', opts, first, 'order')]
        return []

    def who(self):
        lines = []
        pos = 0
        for line in self.t.split('\n'):
            role = None
            if line.startswith('user: ') or line.startswith('customer: '): role = 'the user (customer)'
            elif line.startswith('assistant: '): role = 'the assistant'
            elif line.startswith('tool result'): role = 'a tool result'
            if role and len(line) > 40: lines.append((role, line, pos))
            pos += len(line) + 1
        if len(set(r for r, _, _ in lines)) < 2: return []
        for _ in range(6):
            role, line, p0 = self.rng.choice(lines)
            body = line.split(': ', 1)[1] if ': ' in line else line
            ws = body.split(' ')
            if len(ws) < 6: continue
            i = self.rng.randrange(0, max(1, len(ws) - 5)); span = ' '.join(ws[i:i + self.rng.randint(4, 8)]).strip()
            if '"' in span or len(span) < 12 or occ(self.t, span) != 1: continue
            opts = ['the user (customer)', 'the assistant', 'a tool result']
            self.rng.shuffle(opts)
            return [self._choice(f'Who wrote the line that contains "{span}"?', opts, role, 'who')]
        return []

    def count(self):
        ws = collections.Counter(m.group(0) for m in IDT.finditer(self.t))
        ws.update(m.group(0) for m in CAPW.finditer(self.t))
        ws.update(m.group(0) for m in UPW.finditer(self.t))
        cands = [w for w in ws if '"' not in w and len(w) >= 4]
        if not cands: return []
        r = self.rng.random()
        labels = ['0', '1', '2', '3', '4 or more']
        for _ in range(8):
            if r < 0.15:
                w = self.rng.choice(cands); alt = perturb_digits(w, self.rng) if any(c.isdigit() for c in w) else None
                if not alt or alt in self.t: continue
                x, n = alt, 0
            else:
                # prefer counts 2..5
                multi = [w for w in cands if 2 <= occ(self.t, w) <= 6]
                w = self.rng.choice(multi) if multi and self.rng.random() < 0.7 else self.rng.choice(cands)
                x, n = w, occ(self.t, w)
                # count must be exact as a whole-token count
                if len(re.findall(r'(?<![\w#])' + re.escape(w) + r'(?![\w])', self.t)) != n: continue
            lab = labels[min(n, 4)]
            return [self._choice(f'How many times does "{x}" appear in the state?', labels, lab, 'count')]
        return []

    def numcmp(self):
        for _ in range(8):
            d = self.pick(('num',))
            if d is None: return []
            s, e, typ, tok = d
            v = parse_num(tok)
            if v is None or v == 0: continue
            ls = self.t.rfind('\n', 0, s) + 1
            words = self.t[ls:s].split(' ')
            anchor = ' '.join(words[-self.rng.randint(2, 5):]).strip()
            if len(anchor) < 6 or '"' in anchor or occ(self.t, anchor) != 1: continue
            f = self.rng.choice([0.5, 0.8, 0.9, 0.97, 1.03, 1.1, 1.25, 2.0])
            V = fmt_like(tok, v * f)
            Vn = parse_num(V)
            if Vn is None or Vn == v: continue
            lab = 'true' if v > Vn else 'false'
            return [self._noul(f'The number written right after "{anchor}" in the state is greater than {V}.', lab, 'numcmp')]
        return []

    def datecmp(self):
        ds = [d for d in self.dets if d[2] == 'date']
        if len(set(d[3] for d in ds)) < 2: return []
        for _ in range(8):
            d1, d2 = self.rng.sample(ds, 2)
            if d1[3] == d2[3]: continue
            k1, k2 = date_key(d1[3]), date_key(d2[3])
            if k1 is None or k2 is None or k1 == k2: continue
            anchors = []
            for s, e, _, _ in (d1, d2):
                ls = self.t.rfind('\n', 0, s) + 1
                a = ' '.join(self.t[ls:s].split(' ')[-self.rng.randint(2, 4):]).strip()
                anchors.append(a)
            a1, a2 = anchors
            if a1 == a2 or min(len(a1), len(a2)) < 5 or '"' in a1 + a2 or occ(self.t, a1) != 1 or occ(self.t, a2) != 1: continue
            o1, o2 = f'the date after "{a1}"', f'the date after "{a2}"'
            lab = o1 if k1 < k2 else o2
            opts = [o1, o2]; self.rng.shuffle(opts)
            return [self._choice('Which date is earlier in time?', opts, lab, 'datecmp')]
        return []

    # ---- question dicts
    def _noul(self, ins, lab, fam):
        return dict(q=dict(type='noul', instructions=ins), label=lab, fam=fam)

    def _choice(self, ins, opts, lab, fam):
        return dict(q=dict(type='choice', instructions=ins, criteria={o: '' for o in opts}), label=lab, fam=fam)

    def make(self, n):
        fams = [('verify', .16), ('verify_pair', .09), ('lookup', .27), ('order', .11), ('who', .11), ('count', .11), ('numcmp', .10), ('datecmp', .05)]
        out = []; tries = 0
        while len(out) < n and tries < n * 4:
            tries += 1
            u = self.rng.random(); acc = 0
            for f, w in fams:
                acc += w
                if u < acc: break
            if f == 'verify': out += self.verify()
            elif f == 'verify_pair': out += self.verify(pair=True)
            else: out += getattr(self, f)()
        # dedupe instructions
        seen = set(); res = []
        for q in out:
            k = json.dumps(q['q'], sort_keys=True)
            if k in seen: continue
            seen.add(k); res.append(q)
        return res[:n]


def load_states(rng, max_chars):
    ev = set(json.load(open(os.path.expanduser('~/decider2/evalkit/split.json') if os.path.exists(os.path.expanduser('~/decider2/evalkit/split.json'))
                            else os.path.expanduser('~/work/evalkit/split.json')))['eval_tasks'])
    kit = os.path.expanduser('~/decider2/evalkit') if os.path.exists(os.path.expanduser('~/decider2/evalkit')) else os.path.expanduser('~/work/evalkit')
    agent = []
    for l in open(f'{kit}/train_pool.jsonl'):
        r = json.loads(l)
        if r['task'] in ev: continue
        st = r['state'] if isinstance(r['state'], str) else json.dumps(r['state'], indent=2)
        agent.append(('agent:' + r['domain'], r['task'], st))
    docs = []
    sd = os.path.expanduser('~/code/strands-decider/data/synthetic') if os.path.exists(os.path.expanduser('~/code/strands-decider')) else os.path.expanduser('~/work/sd/data/synthetic')
    for fn in ('generated_v16.jsonl', 'generated_v18.jsonl'):
        seen = set()
        for l in open(f'{sd}/{fn}'):
            r = json.loads(l); st = r['state'] if isinstance(r['state'], str) else json.dumps(r['state'], indent=2)
            if st in seen: continue
            seen.add(st); docs.append(('doc', fn, st))
    out = []
    for src, task, st in agent + docs:
        if len(st) > max_chars:      # keep the head and the end (the decision-relevant newest turns), cut at a line boundary
            h = st[:max_chars // 3]; t = st[-(max_chars - len(h)):]
            t = t[t.find('\n') + 1:]
            st = h[:h.rfind('\n')] + '\n...[earlier text omitted]...\n' + t
        out.append((src, task, st))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--n', type=int, default=9000); ap.add_argument('--q', type=int, default=24)
    ap.add_argument('--seed', type=int, default=1); ap.add_argument('--max_chars', type=int, default=9000); ap.add_argument('--p_doc', type=float, default=0.2)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    sts = load_states(rng, a.max_chars)
    ag = [s for s in sts if s[0].startswith('agent')]; dc = [s for s in sts if s[0] == 'doc']
    rng.shuffle(ag); rng.shuffle(dc)
    print('agent states', len(ag), 'docs', len(dc), file=sys.stderr)
    fam = collections.Counter(); lab = collections.Counter(); n = 0; ia = idc = 0
    with open(a.out, 'w') as f:
        while n < a.n:
            if rng.random() < a.p_doc and idc < len(dc): src, task, st = dc[idc]; idc += 1
            elif ia < len(ag): src, task, st = ag[ia]; ia += 1
            else: break
            qs = Gen(st, rng).make(a.q)
            if len(qs) < 6: continue
            for q in qs: fam[q['fam']] += 1; lab[(q['fam'], q['label'] if q['q']['type'] == 'noul' or q['fam'] == 'count' else 'x')] += 1
            f.write(json.dumps(dict(src=src, task=task, state=st, qs=qs)) + '\n'); n += 1
    print('states', n, 'questions', sum(fam.values()), dict(fam), file=sys.stderr)
    print(sorted(lab.items()), file=sys.stderr)

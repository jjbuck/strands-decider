"""M2 segmenter (pure python; text level + token mapping through a tokenizer's offsets).

A rendered state is '<state>\n' + content + '\n</state>\n'.  Rows (tokens) get:
  U    : the tokens of '<state>\n' (the shared sink / universal compile context; every segment sees them)
  seg  : segment index >= 0 for content rows
  END  : the tokens of '</state>\n' -- moved to the front of every question branch (reader rows; exactly equivalent to hobson)
Granularities:
  nat   : natural segments. A new segment starts at a section header (--- X ---; the header belongs to what follows it), a role line
          (user:, assistant:, assistant called, tool result, system note, [... omitted], Not shown, Search results, ...), a KB
          document entry (numbered title line followed by '   ID: doc_...'), a hook note. Free-text states (no headers, no role lines)
          split at paragraphs (blank lines); JSON states at top-level keys. Blank lines attach to the preceding segment.
  sec   : sections. One segment per header section (non-contiguous when documents / notes are cut out of it); each KB document and
          each hook note is its own segment. Free-text / JSON states are one segment.
  const : one-sided. Only KB documents and hook notes are isolated segments (each sees U + itself); every other row is a READER
          (sees everything before it, native). This is J9's in-place compile layout ('S') expressed as a mask.
  blkN  : fixed N-token blocks over the content rows (structure ignored).
"""
import re, bisect, json

HDR = re.compile(r'^--- .* ---\s*$')
ROLE = re.compile(r'^(user:|assistant:|assistant called|tool result|tool:|system note|\[\.\.\. |\[earlier text omitted\]|Not shown|Knowledge-base documents|Search results|result:|customer:|agent:)')
NUMDOC = re.compile(r'^\s*\d+\. \S')
IDL = re.compile(r'^\s+ID: doc_')
NOTE = re.compile(r'^(system note from the steering hook|tool result: (GUIDANCE|DENIED|CONFIRMATION_FAILED):|.*\[Note from the supervising system)')
JKEY = re.compile(r'^  "[^"]+": ')
U_TXT = '<state>\n'
E_TXT = '</state>\n'


def split_lines(text):
    return re.findall(r'[^\n]*\n|[^\n]+$', text)


def role_kind(ln):
    s = ln.lstrip()
    for k, p in (('user', 'user:'), ('asst', 'assistant:'), ('call', 'assistant called'), ('tool', 'tool result'), ('tool', 'tool:'),
                 ('note', 'system note'), ('omit', '[... '), ('omit', '[earlier text omitted]'), ('notshown', 'Not shown'),
                 ('search', 'Search results'), ('search', 'Knowledge-base documents'), ('tool', 'result:'), ('user', 'customer:'), ('asst', 'agent:')):
        if s.startswith(p): return k
    return 'other'


def seg_spans(rendered, gran='nat'):
    """-> (c_u, c_e, spans) ; c_u = end char of U, c_e = start char of END; spans = list of (c0, c1, key, kind) over [c_u, c_e) in order.
    key identifies the segment (spans sharing a key form one non-contiguous segment); kind in frame/hdr:<NAME>/user/asst/call/tool/note/
    doc/omit/notshown/search/para/json/other. For gran 'const', non-isolated spans get key None (reader rows)."""
    assert rendered.startswith(U_TXT) and rendered.endswith(E_TXT), rendered[:20]
    c_u = len(U_TXT); c_e = len(rendered) - len(E_TXT)
    body = rendered[c_u:c_e]
    lines = split_lines(body)
    n = len(lines)
    starts = []; c = c_u
    for ln in lines: starts.append(c); c += len(ln)
    structured = any(HDR.match(ln) for ln in lines) or sum(1 for ln in lines if ROLE.match(ln)) >= 2 or \
        any(NUMDOC.match(lines[t]) and IDL.match(lines[t + 1]) for t in range(n - 1))
    is_json = False
    if not structured and body.lstrip()[:1] in ('{', '['):
        try: json.loads(body); is_json = True
        except Exception: pass
    blank = [not ln.strip() for ln in lines]
    # per line: trigger kind (None = continues current segment)
    trig = [None] * n
    for t, ln in enumerate(lines):
        if blank[t]: continue
        if structured:
            if HDR.match(ln): trig[t] = 'hdr:' + ln.strip().strip('-').strip().split(' (')[0][:40]
            elif NUMDOC.match(ln) and t + 1 < n and IDL.match(lines[t + 1]): trig[t] = 'doc'
            elif NOTE.match(ln): trig[t] = 'note'
            elif ROLE.match(ln): trig[t] = role_kind(ln)
        elif is_json:
            if JKEY.match(ln): trig[t] = 'json'
        else:
            if t > 0 and blank[t - 1]: trig[t] = 'para'
    if n and trig[0] is None: trig[0] = 'frame' if structured else ('json' if is_json else 'para')
    spans = []
    if gran == 'sent' and not structured and not is_json:     # stress test: every sentence of a free-text state its own segment
        k = -1
        for mt in re.finditer(r'[^.!?]*[.!?]+|[^.!?]+$', body):
            if not mt.group(0): continue
            k += 1; spans.append([c_u + mt.start(), c_u + mt.end(), k, 'sent'])
        return c_u, c_e, [tuple(x) for x in spans]
    if gran == 'sent': gran = 'nat'
    if gran == 'nat' or gran.startswith('blk'):
        k = -1
        for t in range(n):
            if trig[t] is not None:
                k += 1; spans.append([starts[t], starts[t] + len(lines[t]), k, trig[t]])
            else:
                spans[-1][1] = starts[t] + len(lines[t])
        spans = split_docmeta(spans, rendered)
    elif gran in ('sec', 'const'):
        nk = 0; sec_key = None; cur = None; cur_kind = None
        for t in range(n):
            tg = trig[t]
            if tg is not None:
                if tg.startswith('hdr:') or tg == 'frame' or (not structured and t == 0):
                    sec_key = nk; nk += 1; cur = sec_key; cur_kind = tg
                elif tg in ('doc', 'note'):
                    cur = nk; nk += 1; cur_kind = tg
                elif cur_kind in ('doc', 'note') or cur is None:
                    if sec_key is None: sec_key = nk; nk += 1
                    cur = sec_key; cur_kind = 'sec'
                if spans and spans[-1][2] == cur:
                    spans[-1][1] = starts[t] + len(lines[t])
                else:
                    spans.append([starts[t], starts[t] + len(lines[t]), cur, tg if tg in ('doc', 'note') else 'sec'])
            else:
                spans[-1][1] = starts[t] + len(lines[t])
        spans = split_docmeta(spans, rendered)
        if gran == 'const':
            for sp in spans:
                if sp[3] not in ('doc', 'note'): sp[2] = None
    else:
        raise ValueError(gran)
    return c_u, c_e, [tuple(s) for s in spans]


RANK = re.compile(r'^\s*\d+\.')
SCORE_L = re.compile(r'(?m)^[ \t]+Score: -?[\d.]+[ \t]*\n')


def split_docmeta(spans, text):
    """a KB document entry's volatile parts (its retrieval rank 'N.' and its 'Score: x' line) become their own tiny 'docmeta'
    segments, so the document segment (title, ID, content) has the same tokens in every request -> exactly precomputable"""
    nk = 1 + max([sp[2] for sp in spans if sp[2] is not None] + [0])
    out = []
    for c0, c1, key, kind in spans:
        if kind != 'doc': out.append([c0, c1, key, kind]); continue
        seg = text[c0:c1]; cuts = []
        m = RANK.match(seg)
        if m: cuts.append((0, m.end()))
        mm = SCORE_L.search(seg)
        if mm: cuts.append((mm.start(), mm.end()))
        pos = 0
        for a, b in sorted(cuts):
            if a > pos: out.append([c0 + pos, c0 + a, key, 'doc'])
            out.append([c0 + a, c0 + b, nk, 'docmeta']); nk += 1; pos = b
        if pos < len(seg): out.append([c0 + pos, c1, key, 'doc'])
    return out


def token_segments(offsets, rendered, gran='nat'):
    """offsets: per-token (start, end) char offsets of the whole rendered state (tokenizer offsets mapping).
    -> dict(u = number of U tokens, ne = number of END tokens, seg = list per content token (segment index >= 0, or -2 = reader row),
            kinds = list per segment index, nseg)"""
    c_u, c_e, spans = seg_spans(rendered, 'nat' if gran.startswith('blk') else gran)
    st = [o[0] for o in offsets]
    T = len(st)
    u = sum(1 for x in st if x < c_u)
    ne = sum(1 for x in st if x >= c_e)
    body_tok = list(range(u, T - ne))
    if gran.startswith('blk'):
        N = int(gran[3:])
        seg = [(j // N) for j in range(len(body_tok))]
        kinds = ['blk'] * (max(seg) + 1 if seg else 0)
        return dict(u=u, ne=ne, seg=seg, kinds=kinds, nseg=len(kinds))
    sstarts = [sp[0] for sp in spans]
    remap = {}; kinds = []; seg = []
    for t in body_tok:
        j = bisect.bisect_right(sstarts, st[t]) - 1
        j = max(j, 0)
        key, kind = spans[j][2], spans[j][3]
        if key is None: seg.append(-2); continue
        if key not in remap: remap[key] = len(kinds); kinds.append(kind)
        seg.append(remap[key])
    return dict(u=u, ne=ne, seg=seg, kinds=kinds, nseg=len(kinds))


if __name__ == '__main__':      # laptop smoke test on suite states (text only, a crude whitespace "tokenizer")
    import sys, json, os, collections
    sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
    def render(st): return f"<state>\n{(st if isinstance(st, str) else json.dumps(st, indent=2, ensure_ascii=False)).strip()}\n</state>\n"
    def fake_offsets(text):
        return [(m.start(), m.end()) for m in re.finditer(r'\S+\s*|\s+', text)]
    for suite in sys.argv[1:] or ['REAL-agree', 'LONG', 'CF-probe', 'JB-hard']:
        its = [json.loads(l) for l in open(os.path.expanduser(f'~/decider2/evalkit/suites/{suite}.jsonl'))][:200]
        for gran in ('nat', 'sec', 'const', 'blk256'):
            ns = []; kc = collections.Counter(); rd = 0; tot = 0
            for it in its:
                r = render(it['state']); off = fake_offsets(r)
                ts = token_segments(off, r, gran)
                ns.append(ts['nseg']); tot += len(ts['seg']); rd += sum(1 for x in ts['seg'] if x == -2)
                for x in ts['seg']:
                    if x >= 0: kc[ts['kinds'][x]] += 1
            print(f'{suite:10s} {gran:6s} mean segs {sum(ns)/len(ns):6.1f}  reader share {rd/max(1,tot):.2f}  kinds', dict(kc.most_common(8)))

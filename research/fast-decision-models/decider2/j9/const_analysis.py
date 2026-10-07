"""J9: how much of a real gate state is a deployment constant? (pure Python; no models)

Method (held out by task): a library of constant *lines* is built from the train-split pool (train_pool.jsonl, 12,747 requests of
train-split tau tasks). A line is a deployment constant if the same line occurs in states of >= K distinct train tasks (default 3).
Numbered list prefixes ("4. ") and KB "Score:" lines are normalised (the number is dynamic). A line cut by the renderer
(" ...[truncated]") counts as constant if its text is a prefix of a library line. Every eval line is then typed by structure:
  frame   : the hook's framing sentence(s) before the first section header
  markup  : section headers (--- X ---), omission markers, list scaffolding
  doc     : inside a KB document entry (ID: doc_...), or a document-title list line
  note    : a steering-hook note (system note from the steering hook / GUIDANCE / [Note from the supervising system])
  tool    : a tool schema (Tool unlocked / Description / Parameters lines of an unlocked tool)
  conv    : everything else (conversation, tool results) -- counted as constant only if it is in the library ("recurring")
Constant share = chars (or tokens, with a tokenizer on the box) of constant lines / all chars.
Also reported: the share inside *compilable blocks*: maximal runs of constant lines of one type, each >= MINB tokens.
"""
import json, re, sys, os, collections, hashlib, glob, bisect

K = int(os.environ.get('K', 3))
MINB = int(os.environ.get('MINB', 16))
KIT = os.path.expanduser('~/decider2/evalkit') if os.path.exists(os.path.expanduser('~/decider2/evalkit')) else os.path.expanduser('~/work/evalkit')
TOK = None
if os.environ.get('TOKENIZER'):
    from transformers import AutoTokenizer
    TOK = AutoTokenizer.from_pretrained(os.environ['TOKENIZER'])

NUM = re.compile(r'^(\s*)\d+\. ')
SCORE = re.compile(r'^\s+Score: -?[\d.]+\s*$')
TRUNC = ' ...[truncated]'
HDR = re.compile(r'^--- .* ---$')


def norm(line):
    return NUM.sub(r'\1#. ', line)


def lines_of(state):
    return state.split('\n')


def build_library(pool_path, k=K):
    seen = collections.defaultdict(set)
    for l in open(pool_path):
        d = json.loads(l)
        t = d['task']
        for ln in set(lines_of(d['state'])):
            if len(ln.strip()) < 2:
                continue
            seen[norm(ln)].add(t)
    lib = {ln for ln, ts in seen.items() if len(ts) >= k}
    return lib, sorted(lib)


def is_const(ln, lib, libs):
    n = norm(ln)
    if n in lib:
        return True
    if n.endswith(TRUNC.strip()) or n.endswith(TRUNC):
        p = n[:-len(TRUNC)] if n.endswith(TRUNC) else n[:-len(TRUNC.strip())]
        if len(p) < 8:
            return False
        i = bisect.bisect_left(libs, p)
        return i < len(libs) and libs[i].startswith(p)
    return False


def ntok(text):
    if TOK is None:
        return len(text) / 3.4
    return len(TOK(text, add_special_tokens=False)['input_ids'])


def classify(state, lib, libs):
    """[(type, is_const, text)] per line (the newline counted with its line)."""
    out = []
    lines = lines_of(state)
    in_doc = False; in_tool = False; seen_hdr = False; in_note = False
    for ln in lines:
        s = ln.strip()
        if HDR.match(s) or s == '[earlier text omitted]':
            seen_hdr = True; in_doc = False; in_tool = ('TOOL JUST UNLOCKED' in s); in_note = False
            out.append(('markup', True, ln)); continue
        if not seen_hdr and not state.startswith('[earlier text omitted]'):
            out.append(('frame', is_const(ln, lib, libs) or not s, ln)); continue
        if re.match(r'^(user|assistant|tool result|tool|system note|assistant called|\[\.\.\. )', s):
            in_doc = False; in_tool = False
            in_note = s.startswith('system note from the steering hook') or 'GUIDANCE:' in s[:40] or 'DENIED:' in s[:40]
        if re.match(r'^\s*ID: doc_', ln) or SCORE.match(ln):
            in_doc = True
        if s.startswith('Tool unlocked:') or (in_tool and s):
            t = 'tool'
        elif SCORE.match(ln):
            out.append(('doc', False, ln)); continue
        elif in_doc or re.search(r'\(doc_[a-z0-9_()\-]+\)$', s):
            t = 'doc'
        elif in_note:
            t = 'note'
        elif s.startswith('[... ') or s in ('(no conversation yet)', 'none yet', '(none)', 'none'):
            t = 'markup'
        else:
            t = 'conv'
        c = is_const(ln, lib, libs)
        # a doc header line "N. Title" right before "ID: doc_" belongs to the doc
        if t == 'conv' and NUM.match(ln) and c:
            t = 'doc?'
        out.append((t, c, ln))
    # retro-fix: a numbered line followed by an ID: doc_ line is a doc header
    for i in range(len(out) - 1):
        if out[i][0] in ('conv', 'doc?', 'note') and re.match(r'^\s*ID: doc_', out[i + 1][2]):
            out[i] = ('doc', out[i][1], out[i][2])
    out = [(('doc' if t == 'doc?' else t), c, ln) for t, c, ln in out]
    return out


def account(state, lib, libs):
    cl = classify(state, lib, libs)
    tot = collections.Counter(); blocks = 0; block_tok = 0; nblk = 0
    run_t, run_tok = None, 0.0
    def flush():
        nonlocal blocks, block_tok, nblk, run_t, run_tok
        if run_t is not None and run_tok >= MINB:
            block_tok += run_tok; nblk += 1
        run_t, run_tok = None, 0.0
    for t, c, ln in cl:
        n = ntok(ln + '\n')
        key = (t if c else ('dyn' if t == 'conv' else t + '_dyn'))
        if t == 'conv' and c:
            key = 'recurring'
        tot[key] += n
        comp = c and t in ('frame', 'markup', 'doc', 'note', 'tool')
        if comp:
            if run_t != t and not (run_t in ('doc', 'note') and t in ('doc', 'note')):
                flush(); run_t = t
            run_tok += n
        else:
            flush()
    flush()
    tot['_blocks'] = block_tok; tot['_nblk'] = nblk
    return tot


def summarize(rows, label):
    keys = ['frame', 'markup', 'doc', 'note', 'tool', 'recurring', 'dyn', 'doc_dyn', 'note_dyn', 'tool_dyn', 'markup_dyn', 'frame_dyn']
    agg = collections.defaultdict(lambda: collections.Counter())
    for g, tot in rows:
        agg[g].update(tot); agg[g]['_n'] += 1
        agg['ALL'].update(tot); agg['ALL']['_n'] += 1
    res = {}
    for g in sorted(agg, key=lambda x: (x != 'ALL', str(x))):
        a = agg[g]; T = sum(a[k] for k in keys)
        const = sum(a[k] for k in ('frame', 'markup', 'doc', 'note', 'tool'))
        res[str(g)] = {'n': a['_n'], 'mean_tok': T / a['_n'], 'const_share': const / T, 'block_share': a['_blocks'] / T,
                       'blocks_per_req': a['_nblk'] / a['_n'], **{k: a[k] / T for k in keys}}
    print(f'== {label}  (K={K}, MINB={MINB}, {"tokens" if TOK else "chars/3.4"})')
    print(f'{"group":<58} {"n":>5} {"tok":>6} {"const":>6} {"blocks":>6} {"frame":>6} {"markup":>6} {"doc":>6} {"note":>6} {"tool":>6} {"recur":>6} {"dyn":>6}')
    for g, r in res.items():
        print(f'{g[:58]:<58} {r["n"]:>5} {r["mean_tok"]:>6.0f} {r["const_share"]:>6.3f} {r["block_share"]:>6.3f} {r["frame"]:>6.3f} {r["markup"]:>6.3f} {r["doc"]:>6.3f} {r["note"]:>6.3f} {r["tool"]:>6.3f} {r["recurring"]:>6.3f} {r["dyn"]+r["doc_dyn"]+r["note_dyn"]+r["tool_dyn"]+r["markup_dyn"]:>6.3f}')
    return res


def main():
    lib, libs = build_library(f'{KIT}/train_pool.jsonl')
    print('library lines', len(lib), file=sys.stderr)
    out = {}
    for suite in ['REAL-agree', 'LONG', 'CF', 'CF-probe']:
        rows = []
        seen = set()
        for l in open(f'{KIT}/suites/{suite}.jsonl'):
            it = json.loads(l)
            h = hashlib.sha1(it['state'].encode()).hexdigest()
            if h in seen: continue
            seen.add(h)
            g = (it.get('domain'), it.get('hook'), os.path.basename(str(it.get('spec'))))
            rows.append((g, account(it['state'], lib, libs)))
        out[suite] = summarize(rows, suite)
    # raw gate logs (all records, deduplicated by state); library = train pool (eval-task states included here: report only)
    if os.environ.get('RAW'):
        rows = []; seen = set()
        for f in glob.glob(os.environ['RAW']):
            dom = 'banking' if 'banking' in f else ('retail' if 'retail' in f else 'other')
            for l in open(f):
                try: r = json.loads(l)
                except Exception: continue
                s = r.get('state')
                if not s: continue
                h = hashlib.sha1(s.encode()).hexdigest()
                if h in seen: continue
                seen.add(h)
                rows.append(((dom, r.get('lifecycle')), account(s, lib, libs)))
        out['RAW'] = summarize(rows, 'raw gate logs')
    json.dump(out, open(os.environ.get('OUT', 'const_share.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

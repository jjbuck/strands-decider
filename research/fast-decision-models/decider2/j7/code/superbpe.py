"""J7 (a): a domain super-token vocabulary layered on hobson's Qwen3.5 tokenizer (box only).

A super-token is a sequence of >= 2 Qwen token ids learned by BPE over Qwen ids (not over bytes), so every new token is, by construction,
a concatenation of tokens hobson already reads, and an original position maps to exactly one new position.  SuperBPE-style: merges may
cross whitespace and punctuation (multi-word tokens, JSON key+punctuation runs), but never cross
  * a digit token or any token containing a digit   (values stay exact: digits are never merged),
  * a token containing a newline                     (lines stay aligned: option-end rows and '<answer>' keep their own end positions).
Training data: train_pool states (train split only, eval tasks excluded) + the deployment's question texts (deployment constants).

  python superbpe.py train N OUT.json      -> learn N merges; OUT.json = {'tokens': [[qid, qid, ...], ...]}
  python superbpe.py stat OUT.json         -> token counts of every evalkit suite (state and question) with Qwen vs super-tokens
"""
import os, sys, json, re, time, collections
sys.path[:0] = [os.path.expanduser('~/work/evalkit')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

KIT = os.path.expanduser('~/work/evalkit')
BASE_CP = 0x10000          # mapped chars live in the supplementary planes (never whitespace, never normalised)


def qwen_tok():
    from transformers import AutoTokenizer
    import glob
    ck = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*'))[0]
    return AutoTokenizer.from_pretrained(ck)


class Boundary:
    """which Qwen ids are hard boundaries (never part of a merge)."""

    def __init__(self, tok):
        V = len(tok)
        self.V = V
        strs = tok.convert_ids_to_tokens(list(range(V)))
        b = np.zeros(V, dtype=bool)
        for i, s in enumerate(strs):
            if s is None: b[i] = True; continue
            t = tok.convert_tokens_to_string([s])
            if any(ch.isdigit() for ch in t) or '\n' in t or '\r' in t: b[i] = True
        for i in tok.all_special_ids: b[i] = True
        self.b = b


def segments(ids, bnd):
    """split an id list into runs that may merge; boundary ids are singleton segments."""
    out = []; cur = []
    for t in ids:
        if bnd[t]:
            if cur: out.append(cur); cur = []
            out.append([t])
        else:
            cur.append(t)
    if cur: out.append(cur)
    return out


def corpus_ids(tok, texts, bs=256):
    res = []
    for i in range(0, len(texts), bs):
        res += tok(texts[i:i + bs], add_special_tokens=False)['input_ids']
    return res


def train(n_merges, out, max_len=24, min_freq=8):
    tok = qwen_tok(); B = Boundary(tok)
    EV = set(json.load(open(f'{KIT}/split.json'))['eval_tasks'])
    texts = []; qtexts = set()
    from strands_decider.prompting import render_question, render_state
    from pydantic import TypeAdapter
    import strands_decider.schema as SC
    ta = TypeAdapter(SC.Question)
    with open(f'{KIT}/train_pool.jsonl') as f:
        for l in f:
            r = json.loads(l)
            if r['task'] in EV: continue
            texts.append(render_state(r['state']))
            for qn, qd in r['questions'].items():
                qtexts.add(render_question(ta.validate_python(qd)).text)
    texts += sorted(qtexts)
    print('texts', len(texts), 'question texts', len(qtexts), flush=True)
    t0 = time.time()
    ids = corpus_ids(tok, texts)
    print('tokenised %.0fs, %d tokens' % (time.time() - t0, sum(map(len, ids))), flush=True)
    # word counts over mergeable segments (length >= 2); words are strings of mapped chars
    wc = collections.Counter()
    for s in ids:
        for seg in segments(s, B.b):
            if len(seg) >= 2: wc[''.join(chr(BASE_CP + t) for t in seg)] += 1
    alphabet = sorted({c for w in wc for c in w})
    print('distinct segments', len(wc), 'alphabet', len(alphabet), flush=True)
    model = models.BPE()
    tk = Tokenizer(model)
    tk.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    trainer = trainers.BpeTrainer(vocab_size=len(alphabet) + n_merges, min_frequency=min_freq, show_progress=False,
                                  initial_alphabet=alphabet, limit_alphabet=len(alphabet) + 10, max_token_length=max_len, special_tokens=[])

    def it():
        for w, c in wc.items():
            for _ in range(c):                 # true frequencies: boilerplate that recurs in the deployment should get merges
                yield w
    t0 = time.time()
    tk.train_from_iterator(it(), trainer=trainer, length=None)
    print('bpe trained %.0fs' % (time.time() - t0), flush=True)
    vocab = tk.get_vocab()
    toks = [[ord(c) - BASE_CP for c in s] for s in sorted(vocab, key=lambda s: vocab[s]) if len(s) >= 2]
    tk.save(out + '.tok.json')
    json.dump({'tokens': toks, 'n_merges': n_merges, 'max_len': max_len, 'min_freq': min_freq}, open(out, 'w'))
    print('saved', len(toks), 'super-tokens', flush=True)


class Super:
    """encode text -> (ids in the extended vocab, span of original Qwen positions per new token).
    New ids are V + k for super-token k."""

    def __init__(self, path, tok=None):
        self.tok = tok or qwen_tok(); self.B = Boundary(self.tok); self.V = self.B.V
        d = json.load(open(path))
        self.toks = [tuple(t) for t in d['tokens']]
        self.idx = {t: self.V + k for k, t in enumerate(self.toks)}
        self.tk = Tokenizer.from_file(path + '.tok.json')
        self.alpha = set(self.tk.get_vocab())

    def merge_ids(self, ids, protect=None):
        """Qwen ids -> (new ids, ends) where ends[k] = index (in ids) of the last Qwen token covered by new token k.
        protect: optional bool per Qwen token; protected tokens are never merged (kept at Qwen granularity)."""
        out = []; ends = []; pos = 0
        if protect is not None:
            b = self.B.b
            segs = []; cur = []
            for t, pr in zip(ids, protect):
                if b[t] or pr:
                    if cur: segs.append(cur); cur = []
                    segs.append([t])
                else: cur.append(t)
            if cur: segs.append(cur)
        else:
            segs = segments(ids, self.B.b)
        words = []
        for seg in segs:
            if len(seg) >= 2 and all(chr(BASE_CP + t) in self.alpha for t in seg): words.append(seg)
            else: words.append(None)
        enc = self.tk.encode_batch([''.join(chr(BASE_CP + t) for t in seg) for seg, w in zip(segs, words) if w is not None], add_special_tokens=False) if any(w is not None for w in words) else []
        ei = 0
        for seg, w in zip(segs, words):
            if w is None:
                for t in seg:
                    out.append(t); ends.append(pos); pos += 1
                continue
            e = enc[ei]; ei += 1
            for piece in e.tokens:
                tt = tuple(ord(c) - BASE_CP for c in piece)
                if len(tt) == 1: out.append(tt[0])
                else: out.append(self.idx[tt])
                pos += len(tt); ends.append(pos - 1)
        assert pos == len(ids), (pos, len(ids))
        return out, ends

    def constituents(self, nid):
        return (nid,) if nid < self.V else self.toks[nid - self.V]


def stat(path):
    import evalkit as EK
    from pydantic import TypeAdapter
    import strands_decider.schema as SC
    from strands_decider.prompting import render_question, render_state
    ta = TypeAdapter(SC.Question)
    S = Super(path); tok = S.tok
    res = {}
    for suite in ['JB-all', 'REAL-agree', 'LONG', 'CF', 'CF-probe']:
        its = EK.load_suite(suite)
        st0 = st1 = q0 = q1 = 0; per = []
        qseen = {}
        for it in its:
            s = tok(render_state(it['state']), add_special_tokens=True)['input_ids']
            m, _ = S.merge_ids(s)
            st0 += len(s); st1 += len(m); per.append(len(s) / max(1, len(m)))
            for qn, qd in it['questions'].items():
                txt = render_question(ta.validate_python(qd)).text
                if txt not in qseen:
                    qi = tok(txt, add_special_tokens=False)['input_ids']; qseen[txt] = (len(qi), len(S.merge_ids(qi)[0]))
                a, b = qseen[txt]; q0 += a; q1 += b
        res[suite] = dict(n=len(its), state_qwen=st0, state_super=st1, state_ratio=round(st0 / st1, 3), state_ratio_median=round(float(np.median(per)), 3),
                          q_qwen=q0, q_super=q1, q_ratio=round(q0 / max(1, q1), 3), total_ratio=round((st0 + q0) / (st1 + q1), 3))
        print(suite, res[suite], flush=True)
    return res


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        train(int(sys.argv[2]), sys.argv[3], max_len=int(os.environ.get('MAXLEN', 24)), min_freq=int(os.environ.get('MINF', 8)))
    elif sys.argv[1] == 'stat':
        r = stat(sys.argv[2]); json.dump(r, open(sys.argv[2] + '.stat.json', 'w'), indent=1)

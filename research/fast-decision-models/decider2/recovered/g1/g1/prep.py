"""F7 prep: render + tokenize every training/eval row ONCE, exactly as the serving engine does
(state tokenized with special tokens and right-truncated to the budget left after the question;
question tokenized separately, front-truncated; pointer positions from the question's offsets).
Option order / instruction variant are drawn once per row (seeded) -- one epoch, one draw, as
the v19 collator does on the fly. Teacher (hobson) and student see identical ids.

usage: python prep.py corpus            -> rows_corpus.pt  (train_v5 [-val] + v19 synthetic docs)
       python prep.py real POOL.jsonl   -> rows_real.pt    (KL-only rows over real states)
"""
import sys, os, json, random, collections
import torch
from transformers import AutoTokenizer
sys.path.insert(0, os.path.expanduser("~/work/sd/src"))
from strands_decider.data.format import Example, split_examples
from strands_decider.prompting import render_question, render_state
from strands_decider.infer import _option_token_index
from pydantic import TypeAdapter
import strands_decider.schema as SC

MAXLEN = int(os.environ.get("MAXLEN", 3072))
TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
W = os.path.expanduser("~/work")


def fit(state_text, qtext, rq, max_len=MAXLEN, mqf=0.75):
    e = TOK([qtext], add_special_tokens=False, return_offsets_mapping=True)
    q = e["input_ids"][0]; off = e["offset_mapping"][0]
    reserve = min(len(q), max(1, int(max_len * mqf)))
    cut = max(0, len(q) - reserve)
    q = q[cut:]; off = off[cut:]
    s = TOK(state_text, add_special_tokens=True, truncation=True, max_length=max(1, max_len - reserve))["input_ids"]
    opt = _option_token_index(off, rq.option_spans, 0)
    return s, q, [len(s) + i for i in opt]


def order_for(ex, rng):
    n = ex.n_options
    if ex.kind == "score":
        return list(reversed(range(n))) if rng.random() < 0.5 else None
    o = list(range(n)); rng.shuffle(o); return o


def dist_for(ex, order, eps=0.1):
    if ex.kind != "score":
        return None
    n = ex.n_options; g = ex.label
    nb = [l for l in (g - 1, g + 1) if 0 <= l < n]
    mass = {g: 1.0 - eps} if nb else {g: 1.0}
    for l in nb: mass[l] = eps / len(nb)
    ol = list(order) if order is not None else list(range(n))
    d = [0.0] * n
    for l, m in mass.items(): d[ol.index(l)] += m
    return d


def enc_example(ex, rng, train=True, src=""):
    order = order_for(ex, rng) if train else None
    pool = ex.all_instructions()
    instr = rng.choice(pool) if (train and len(pool) > 1) else None
    q = ex.to_question(instr)
    rq = render_question(q, option_order=order)
    s, qq, opt = fit(render_state(ex.state), rq.text, rq)
    lab = ex.label if order is None else list(order).index(ex.label)
    return dict(ids=torch.tensor(s + qq, dtype=torch.int32), opt=opt, n=ex.n_options, label=lab, kind=ex.kind,
                dist=dist_for(ex, order) if train else None, w=1.0, src=src, task=ex.task, nstate=len(s))


def load(path):
    return [Example.from_dict(json.loads(l)) for l in open(path)]


def corpus():
    rng = random.Random(0)
    v5 = load(f"{W}/training/data/train_v5.jsonl")
    tr, va = split_examples(v5, val_fraction=0.03, seed=0)  # v19: val_fraction 0.03, stratified by task
    syn = f"{W}/sd/data/synthetic"
    docs = []
    for name in ("generated_v16", "generated_v18", "adequacy_gen"):
        docs += [(name, e) for e in load(f"{syn}/{name}.jsonl")]
    rows = [enc_example(e, rng, True, "v5") for e in tr]
    print("v5 encoded", len(rows), flush=True)
    rows += [enc_example(e, rng, True, n) for n, e in docs]
    print("docs encoded", len(docs), flush=True)
    # learning-curve evals: v5 val (cap 60/task), recipe held-out generated + adequacy evals
    ev = []
    bt = collections.defaultdict(list)
    for e in va: bt[e.task].append(e)
    for t in sorted(bt):
        ev += [enc_example(e, rng, False, "val_v5") for e in bt[t][:60]]
    for name in ("generated_v16_eval", "generated_v18_eval", "adequacy_gen_eval"):
        ev += [enc_example(e, rng, False, name) for e in load(f"{syn}/{name}.jsonl")]
    torch.save(dict(rows=rows, maxlen=MAXLEN), "rows_corpus.pt")
    torch.save(dict(rows=ev, maxlen=MAXLEN), "rows_lceval.pt")
    nt = sum(len(r["ids"]) for r in rows)
    print("corpus rows", len(rows), "tokens", nt, "eval rows", len(ev), collections.Counter(r["src"] for r in ev))


ta = TypeAdapter(SC.Question)


def real(pool_path, per_state=0, maxlen=4096):
    """KL-only rows: (real state, question) with no gold; hobson's distribution is the target."""
    rng = random.Random(1)
    rows = []
    recs = [json.loads(l) for l in open(pool_path)] if pool_path.endswith(".jsonl") else json.load(open(pool_path))
    nst = int(os.environ.get("N_STATES", 0))
    if nst and nst < len(recs):
        recs = random.Random(5).sample(recs, nst)
    for k, r in enumerate(recs):
        qs = list(r["questions"].items())
        if per_state and len(qs) > per_state:
            qs = rng.sample(qs, per_state)
        for name, qd in qs:
            q = ta.validate_python(qd)
            rq0 = render_question(q)
            n = rq0.n_slots
            # canonical render (no shuffling) so it matches serving; the head is order-free anyway
            order = None
            if rq0.kind != "score":
                order = list(range(n)); rng.shuffle(order)
            rq = render_question(q, option_order=order)
            s, qq, opt = fit(render_state(r["state"]), rq.text, rq, max_len=maxlen)
            rows.append(dict(ids=torch.tensor(s + qq, dtype=torch.int32), opt=opt, n=n, label=0, kind=rq.kind, dist=None,
                             w=0.0, src="real", task=str(r.get("task", k)), nstate=len(s), qname=name, rid=r.get("rid")))
    torch.save(dict(rows=rows, maxlen=maxlen), os.environ.get("OUT", "rows_real.pt"))
    nt = sum(len(r["ids"]) for r in rows)
    print("real rows", len(rows), "states", len(recs), "tokens", nt)


if __name__ == "__main__":
    if sys.argv[1] == "corpus":
        corpus()
    else:
        real(sys.argv[2], per_state=int(os.environ.get("PER_STATE", 0)), maxlen=int(os.environ.get("RMAX", 4096)))

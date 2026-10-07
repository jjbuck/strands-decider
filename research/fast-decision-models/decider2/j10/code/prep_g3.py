"""G3 prep: F7's prep.py (same rows, same seeds, same draws) but every row is tokenised twice from the SAME rendered text:
  *_q.pt  Qwen3.5 tokenizer (identical to F7's rows; hobson teacher runs on these)
  *_s.pt  student tokenizer (MoE torso)
usage: STUDENT=Tiiny/SmallThinker-4BA0.6B-Instruct python prep_g3.py corpus
       STUDENT=... N_STATES=6000 PER_STATE=1 RMAX=9216 python prep_g3.py real ~/work/evalkit/train_pool.jsonl
       python prep_g3.py merge rows_X        -> copies t_logits/teacher_temps from rows_X_q.pt into rows_X_s.pt
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
W = os.path.expanduser("~/work")
TQ = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B-Base")
TS = AutoTokenizer.from_pretrained(os.environ.get("STUDENT", "microsoft/bitnet-b1.58-2B-4T-bf16"), trust_remote_code=False)


def fit(tok, state_text, qtext, rq, max_len=MAXLEN, mqf=0.75):
    e = tok([qtext], add_special_tokens=False, return_offsets_mapping=True)
    q = e["input_ids"][0]; off = e["offset_mapping"][0]
    reserve = min(len(q), max(1, int(max_len * mqf)))
    cut = max(0, len(q) - reserve)
    q = q[cut:]; off = off[cut:]
    s = tok(state_text, add_special_tokens=True, truncation=True, max_length=max(1, max_len - reserve))["input_ids"]
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
    st = render_state(ex.state)
    lab = ex.label if order is None else list(order).index(ex.label)
    out = []
    for tok in (TQ, TS):
        s, qq, opt = fit(tok, st, rq.text, rq)
        out.append(dict(ids=torch.tensor(s + qq, dtype=torch.int32), opt=opt, n=ex.n_options, label=lab, kind=ex.kind,
                        dist=dist_for(ex, order) if train else None, w=1.0, src=src, task=ex.task, nstate=len(s)))
    return out


def load(path):
    return [Example.from_dict(json.loads(l)) for l in open(path)]


def save2(rows2, name, maxlen):
    for j, suf in enumerate(("q", "s")):
        torch.save(dict(rows=[r[j] for r in rows2], maxlen=maxlen), f"{name}_{suf}.pt")
    for j, suf in enumerate(("q", "s")):
        print(name, suf, "rows", len(rows2), "tokens", sum(len(r[j]["ids"]) for r in rows2), flush=True)


def corpus():
    rng = random.Random(0)
    v5 = load(f"{W}/training/data/train_v5.jsonl")
    tr, va = split_examples(v5, val_fraction=0.03, seed=0)
    syn = f"{W}/sd/data/synthetic"
    docs = []
    for name in ("generated_v16", "generated_v18", "adequacy_gen"):
        docs += [(name, e) for e in load(f"{syn}/{name}.jsonl")]
    rows = [enc_example(e, rng, True, "v5") for e in tr]
    print("v5 encoded", len(rows), flush=True)
    rows += [enc_example(e, rng, True, n) for n, e in docs]
    print("docs encoded", len(docs), flush=True)
    ev = []
    bt = collections.defaultdict(list)
    for e in va: bt[e.task].append(e)
    for t in sorted(bt):
        ev += [enc_example(e, rng, False, "val_v5") for e in bt[t][:60]]
    for name in ("generated_v16_eval", "generated_v18_eval", "adequacy_gen_eval"):
        ev += [enc_example(e, rng, False, name) for e in load(f"{syn}/{name}.jsonl")]
    save2(rows, "rows_corpus", MAXLEN)
    save2(ev, "rows_lceval", MAXLEN)


ta = TypeAdapter(SC.Question)


def real(pool_path, per_state=0, maxlen=4096):
    rng = random.Random(1)
    rows = []
    recs = [json.loads(l) for l in open(pool_path)]
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
            order = None
            if rq0.kind != "score":
                order = list(range(n)); rng.shuffle(order)
            rq = render_question(q, option_order=order)
            st = render_state(r["state"])
            two = []
            for tok in (TQ, TS):
                s, qq, opt = fit(tok, st, rq.text, rq, max_len=maxlen)
                two.append(dict(ids=torch.tensor(s + qq, dtype=torch.int32), opt=opt, n=n, label=0, kind=rq.kind, dist=None,
                                w=0.0, src="real", task=str(r.get("task", k)), nstate=len(s), qname=name, rid=r.get("rid")))
            rows.append(two)
    save2(rows, "rows_real", maxlen)


def merge(name):
    Q = torch.load(f"{name}_q.pt"); S = torch.load(f"{name}_s.pt")
    assert len(Q["rows"]) == len(S["rows"])
    n = 0
    for a, b in zip(Q["rows"], S["rows"]):
        assert a["n"] == b["n"] and a["label"] == b["label"]
        if "t_logits" in a: b["t_logits"] = a["t_logits"]; n += 1
    if "teacher_temps" in Q: S["teacher_temps"] = Q["teacher_temps"]
    torch.save(S, f"{name}_s.pt"); print("merged", name, n, "teacher rows", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "corpus":
        corpus()
    elif sys.argv[1] == "real":
        real(sys.argv[2], per_state=int(os.environ.get("PER_STATE", 0)), maxlen=int(os.environ.get("RMAX", 4096)))
    else:
        merge(sys.argv[2])

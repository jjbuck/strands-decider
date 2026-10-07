"""J11 box-side prep: every source -> data/prep/<name>.pt = list of rows (numpy arrays inside; torch only for saving).
row keys:
  ids int32 [T] (state then question, hobson layout), nstate, opt (absolute last-token idx per option), n, kind, label, w (gold weight),
  dist (score smoothing or None), t (teacher probs or None), meta
  pl: payload tuple (typ int8, lh/kh/rh int64, val float32, st int8) aligned with ids
  position-free decomposition (slot arms): qp int32 = question stem + tail (no options; last token = <answer>), qp_pl,
  op = list of int32 arrays (one per option, its text WITHOUT the 'k. ' numbering, in rendered order), op_pl
usage: python prep.py synth|corpus|real|real2|lceval|kit|rot
"""
import sys, os, json, random, re
from multiprocessing import Pool
import numpy as np, torch
sys.path.insert(0, os.path.expanduser("~/work/sd/src")); sys.path.insert(0, os.path.expanduser("~/work/evalkit"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
from strands_decider.prompting import render_question, render_state
from strands_decider.infer import _option_token_index
from strands_decider.data.format import Example
from pydantic import TypeAdapter
import strands_decider.schema as SC
import payload as PL

W = os.path.expanduser("~/work/j11/data"); OUT = f"{W}/prep"; os.makedirs(OUT, exist_ok=True)
TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base")
ta = TypeAdapter(SC.Question)
PIECES = [TOK.decode([i]) for i in range(len(TOK))]
NUMRX = re.compile(r"^\d+\. ")


def fit(state_text, qtext, rq, max_len=16384, mqf=0.75):
    e = TOK([qtext], add_special_tokens=False, return_offsets_mapping=True)
    q = e["input_ids"][0]; off = e["offset_mapping"][0]
    reserve = min(len(q), max(1, int(max_len * mqf)))
    cut = max(0, len(q) - reserve); q = q[cut:]; off = off[cut:]
    s = TOK(state_text, add_special_tokens=True, truncation=True, max_length=max(1, max_len - reserve))["input_ids"]
    opt = _option_token_index(off, rq.option_spans, 0)
    return s, q, [len(s) + i for i in opt]


def _pl(ids):
    off = []; c = 0; parts = []
    for i in ids:
        p = PIECES[i]; parts.append(p); off.append((c, c + len(p))); c += len(p)
    d = PL.token_payloads("".join(parts), off)
    return (np.array(d["typ"], np.int8), np.array(d["lh"], np.int64), np.array(d["kh"], np.int64), np.array(d["rh"], np.int64),
            np.array(d["val"], np.float32), np.array(d["start"], np.int8))


def _pf(qids, n):
    """question ids (hobson layout) -> (stem+tail ids, [option ids]) or None."""
    txt = "".join(PIECES[i] for i in qids)
    a = txt.find("<options>\n"); b = txt.rfind("\n</options>")
    if a < 0 or b < 0: return None
    lines = txt[a + len("<options>\n"):b].split("\n")
    if len(lines) != n: return None
    stem = txt[:a + len("<options>\n")] + txt[b + 1:]
    qp = TOK(stem, add_special_tokens=False)["input_ids"]
    op = [TOK(NUMRX.sub("", l, count=1), add_special_tokens=False)["input_ids"] or [TOK("?", add_special_tokens=False)["input_ids"][0]] for l in lines]
    return qp, op


def finish(r):
    """add payloads + position-free decomposition; returns row with numpy arrays."""
    ids = list(map(int, r["ids"]))
    r["ids"] = np.array(ids, np.int32); r["pl"] = _pl(ids)
    pf = _pf(ids[r["nstate"]:], r["n"])
    if pf is None:
        r["qp"] = None
    else:
        qp, op = pf
        r["qp"] = np.array(qp, np.int32); r["qp_pl"] = _pl(qp)
        r["op"] = [np.array(o, np.int32) for o in op]; r["op_pl"] = [_pl(o) for o in op]
    return r


def dist_for(kind, n, label, order, eps=0.1):
    if kind != "score": return None
    nb = [l for l in (label - 1, label + 1) if 0 <= l < n]
    mass = {label: 1.0 - eps} if nb else {label: 1.0}
    for l in nb: mass[l] = eps / len(nb)
    ol = list(order) if order is not None else list(range(n))
    d = [0.0] * n
    for l, m in mass.items(): d[ol.index(l)] += m
    return d


def rot_orders(kind, n):
    """distinct non-identity option orders used by the order-invariance metric: 3 rotations (score: reversal only)."""
    if kind == "score": return [("rev", list(reversed(range(n))))]
    out = []; seen = {tuple(range(n))}
    for k in (1, 2, 3):
        o = [(i + k) % n for i in range(n)]
        if tuple(o) not in seen: seen.add(tuple(o)); out.append((f"rot{k}", o))
    return out


def _synth_one(args):
    i, l, split, rot = args
    j = json.loads(l); rng = random.Random(1000003 * i + (0 if split == "train" else 7))
    ex = Example(kind=j["kind"], state=j["state"], instructions=j["instructions"], options=j["options"], label=j["label"],
                 task=j["task"], instruction_variants=j.get("instruction_variants", []))
    train = split == "train"; n = ex.n_options
    if len(set(o[0] for o in ex.options)) != n: return []
    if train:
        if ex.kind == "score": order = list(reversed(range(n))) if rng.random() < 0.5 else None
        else: order = list(range(n)); rng.shuffle(order)
        pool = ex.all_instructions(); instr = rng.choice(pool) if len(pool) > 1 else None
        variants = [(None, order)]
    else:
        variants = [(None, None)] + (rot_orders(ex.kind, n) if rot else [])
        instr = None
    out = []
    for tag, order in variants:
        rq = render_question(ex.to_question(instr), option_order=order)
        s, q, opt = fit(render_state(ex.state), rq.text, rq)
        lab = ex.label if order is None else list(order).index(ex.label)
        r = dict(ids=s + q, nstate=len(s), opt=opt, n=n, kind=ex.kind, label=lab, w=1.0,
                 dist=dist_for(ex.kind, n, ex.label, order) if train else None, t=None,
                 meta=dict(src="synth", family=j["family"], split=j["split"], id=j.get("id"), labels=list(rq.slot_labels), rot=tag))
        out.append(finish(r))
    return out


def synth():
    """chunked (memory): test -> synth_test.pt; train -> synth_train_{k}.pt of CH rows each."""
    NT = int(os.environ.get("NTRAIN", 200000)); CH = 25000
    for split in ("test", "train"):
        if split == "test" and os.path.exists(f"{OUT}/synth_test.pt"): continue
        buf = []; k = 0
        def flush(buf, k):
            with Pool(7) as pool: res = pool.map(_synth_one, buf, chunksize=128)
            rows = [r for rs in res for r in rs]; del res
            name = f"{OUT}/synth_test.pt" if split == "test" else f"{OUT}/synth_train_{k}.pt"
            torch.save(rows, name); print(split, k, len(rows), "tokens", sum(len(r["ids"]) for r in rows), flush=True)
        for i, l in enumerate(open(f"{W}/synth/synth_{split}.jsonl")):
            if split == "train" and i >= NT: break
            buf.append((i, l, split, split == "test"))
            if split == "train" and len(buf) == CH:
                if not os.path.exists(f"{OUT}/synth_train_{k}.pt"): flush(buf, k)
                buf = []; k += 1
        if buf: flush(buf, k)


def teacher_probs(r, temps):
    if "t_logits" not in r: return None
    T = temps["by_kind"].get(r["kind"], temps["T"]) if temps else 1.0
    return torch.softmax(torch.tensor(r["t_logits"]) / T, -1).tolist()


def from_rows(name, src):
    D = torch.load(f"{W}/{name}_q.pt", weights_only=False); temps = D.get("teacher_temps")
    rows = []
    for r in D["rows"]:
        rows.append(dict(ids=r["ids"].tolist(), nstate=int(r["nstate"]), opt=list(r["opt"]), n=int(r["n"]), kind=r["kind"],
                         label=int(r["label"]), w=float(r["w"]), dist=r.get("dist"), t=teacher_probs(r, temps),
                         meta=dict(src=src, task=r.get("task"), qname=r.get("qname"))))
    del D
    with Pool(8) as pool: rows = pool.map(finish, rows, chunksize=64)
    print(name, len(rows), "teacher", sum(r["t"] is not None for r in rows), "tokens", sum(len(r["ids"]) for r in rows),
          "pf ok", sum(r["qp"] is not None for r in rows), flush=True)
    torch.save(rows, f"{OUT}/{src}.pt")


def _kit_one(args):
    suite, iid, qn, st, spec, rot = args
    q = ta.validate_python(spec); rq0 = render_question(q)
    variants = [(None, None)] + (rot_orders(rq0.kind, rq0.n_slots) if rot else [])
    out = []
    for tag, order in variants:
        rq = render_question(q, option_order=order)
        s, qq, opt = fit(render_state(st), rq.text, rq)
        r = dict(ids=s + qq, nstate=len(s), opt=opt, n=rq.n_slots, kind=rq.kind, label=-1, w=0.0, dist=None, t=None,
                 meta=dict(src="kit", suite=suite, iid=iid, q=qn, labels=list(rq.slot_labels), rot=tag))
        out.append(finish(r))
    return out


def kit():
    import evalkit as EK
    items = [(s, i, q, st, sp, True) for s, i, q, st, sp in EK.all_question_items()]
    with Pool(8) as pool: res = pool.map(_kit_one, items, chunksize=8)
    rows = [r for rs in res for r in rs]
    print("kit", len(rows), "tokens", sum(len(r["ids"]) for r in rows), "max", max(len(r["ids"]) for r in rows),
          "pf ok", sum(r["qp"] is not None for r in rows), flush=True)
    torch.save(rows, f"{OUT}/kit.pt")


if __name__ == "__main__":
    a = sys.argv[1]
    if a == "synth": synth()
    elif a == "corpus": from_rows("rows_corpus", "corpus")
    elif a == "real": from_rows("rows_real", "real")
    elif a == "real2": from_rows("rows_real2", "real2")
    elif a == "lceval": from_rows("rows_lceval", "lceval")
    elif a == "kit": kit()

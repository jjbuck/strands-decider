"""extra real train-split rows for KL: train_pool states NOT in g3's seed-5 6000 sample, 1 random question each (seed 2), maxlen 4096,
same rendering/fit as g3's prep (Qwen tokenizer). -> rows_real2_q.pt (then teacher.py adds hobson t_logits)."""
import sys, os, json, random, torch
sys.path.insert(0, os.path.expanduser("~/work/sd/src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoTokenizer
from strands_decider.prompting import render_question, render_state
from strands_decider.infer import _option_token_index
from pydantic import TypeAdapter
import strands_decider.schema as SC
TQ = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B-Base")
ta = TypeAdapter(SC.Question)

def fit(tok, state_text, qtext, rq, max_len=4096, mqf=0.75):
    e = tok([qtext], add_special_tokens=False, return_offsets_mapping=True)
    q = e["input_ids"][0]; off = e["offset_mapping"][0]
    reserve = min(len(q), max(1, int(max_len * mqf)))
    cut = max(0, len(q) - reserve); q = q[cut:]; off = off[cut:]
    s = tok(state_text, add_special_tokens=True, truncation=True, max_length=max(1, max_len - reserve))["input_ids"]
    opt = _option_token_index(off, rq.option_spans, 0)
    return s, q, [len(s) + i for i in opt]

recs = [json.loads(l) for l in open(os.path.expanduser("~/work/evalkit/train_pool.jsonl"))]
split = json.load(open(os.path.expanduser("~/work/evalkit/split.json")))
evt = set(split["eval_tasks"]) if isinstance(split["eval_tasks"], list) else set(t for v in split["eval_tasks"].values() for t in v)
used = set(id(r) for r in random.Random(5).sample(recs, 6000))
rest = [r for r in recs if id(r) not in used and r["task"] not in evt]
print("pool", len(recs), "rest", len(rest), flush=True)
N = int(os.environ.get("N", 6000)); rng = random.Random(2)
rest = rng.sample(rest, min(N, len(rest)))
rows = []
for k, r in enumerate(rest):
    name, qd = rng.choice(list(r["questions"].items()))
    q = ta.validate_python(qd); rq0 = render_question(q); n = rq0.n_slots; order = None
    if rq0.kind != "score": order = list(range(n)); rng.shuffle(order)
    rq = render_question(q, option_order=order)
    s, qq, opt = fit(TQ, render_state(r["state"]), rq.text, rq)
    rows.append(dict(ids=torch.tensor(s + qq, dtype=torch.int32), opt=opt, n=n, label=0, kind=rq.kind, dist=None, w=0.0, src="real2",
                     task=str(r.get("task", k)), nstate=len(s), qname=name, rid=r.get("rid")))
torch.save(dict(rows=rows, maxlen=4096), "rows_real2_q.pt")
print("rows", len(rows), "tokens", sum(len(r["ids"]) for r in rows), flush=True)

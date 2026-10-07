"""Build the fixed pilot mixes (identical for every torso). Run on the GPU box in ~/work/training."""
import json, random, collections, sys
from transformers import AutoTokenizer
rnd = random.Random(0)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
def ntok(r):
    return len(tok(str(r["state"]))["input_ids"]) + 60 + 25*len(r["options"])
def rd(p): return [json.loads(l) for l in open(p)]
PER_TASK = int(sys.argv[1]) if len(sys.argv) > 1 else 250
short = rd("data/train_v5.jsonl")
bytask = collections.defaultdict(list)
for r in short: bytask[r["task"]].append(r)
mix = []
for t, rows in sorted(bytask.items()):
    rnd.shuffle(rows); mix += rows[:PER_TASK]
def take(name, n, maxtok=1800):
    rows = rd(f"data/synthetic/{name}.jsonl"); rnd.shuffle(rows)
    out = []
    for r in rows:
        if len(out) >= n: break
        if len(str(r["state"])) > maxtok*4.2: continue
        out.append(r)
    return out
docs = take("generated_v16", 90, 1500) + take("generated_v18", 60, 1500)
adeq = take("adequacy_gen", 220) + take("flips_v20", 100)
mix += docs + adeq
rnd.shuffle(mix)
with open("data/pilot_train.jsonl", "w") as f:
    for r in mix: f.write(json.dumps(r) + "\n")
tot = sum(ntok(r) for r in mix)
print("rows", len(mix), "approx tokens", tot, "docs", len(docs), "adeq", len(adeq))
# eval sets (disjoint from train by construction: eval files / holdout tasks)
ev = []
hold = rd("data/train_v5.holdout.jsonl")
ht = collections.defaultdict(list)
for r in hold: ht[r["task"]].append(r)
for t, rows in sorted(ht.items()):
    if t.startswith("ruletaker"): continue
    rnd.shuffle(rows); ev += [dict(r, task="HO:"+t) for r in rows[:150]]
for name, n in [("generated_v16_eval", 120), ("generated_v18_eval", 100), ("adequacy_gen_eval", 150), ("flips_v20_eval", 80)]:
    rows = take(name, n, maxtok=2400)
    ev += [dict(r, task="EV:"+name) for r in rows]
with open("data/pilot_eval.jsonl", "w") as f:
    for r in ev: f.write(json.dumps(r) + "\n")
print("eval rows", len(ev), collections.Counter(r["task"] for r in ev))

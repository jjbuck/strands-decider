"""Evaluate a checkpoint on JevBench public (231) + pilot held-out sets. Usage: jbeval.py CKPT TAG [--trunc N]"""
import sys, json, time, torch, collections, math
from strands_decider.modeling import StrandsDeciderModel
from strands_decider.infer import SystemOneEngine, EngineConfig
from strands_decider.schema import NoulQuestion, ChoiceQuestion, ScoreQuestion
from strands_decider.data.format import Example
from strands_decider.evaluate import collect_logits, predictions_from_logits
ck, tag = sys.argv[1], sys.argv[2]
JB = "/home/ubuntu/work/training/jevbench_public"
model = StrandsDeciderModel.load(ck)
if hasattr(model.torso, "merge_and_unload"): model.torso = model.torso.merge_and_unload()
model.config.max_length = 3072
eng = SystemOneEngine(model, EngineConfig(device="cuda", use_prefix_cache=False))
def q_of(t):
    q = t["question"]
    if q["type"] == "noul": return NoulQuestion(instructions=q["instructions"], criteria=q.get("criteria"))
    if q["type"] == "choice": return ChoiceQuestion(instructions=q["instructions"], criteria=q["criteria"])
    return ScoreQuestion(instructions=q["instructions"], criteria=q["criteria"])
tasks = []
for f in ["easy", "original", "hard"]:
    tasks += [json.loads(l) for l in open(f"{JB}/{f}.jsonl")]
res = []; t0 = time.time()
tier = {}
for f in ["easy", "original", "hard"]:
    for l in open(f"{JB}/{f}.jsonl"): tier[json.loads(l)["id"]] = f
for t in tasks:
    typ = t["question"]["type"]
    try:
        r = eng.ask(t["state"], {"q": q_of(t)})
    except Exception as e:
        res.append(dict(id=t["id"], fam=t["family"], tier=tier[t["id"]], ok=0, conf=0.5, err=str(e)[:80])); continue
    a = r.answers["q"]
    if typ == "noul":
        p = a.noul; ok = int((p >= 0.5) == (t["expected"] == "yes")); conf = max(p, 1-p); pt = p if t["expected"]=="yes" else 1-p
    elif typ == "choice":
        ok = int(a.choice == t["expected"]); conf = a.confidence; pt = a.probabilities.get(t["expected"], 0.0)
    else:
        probs = a.probabilities; keys = list(probs.keys())
        best = max(keys, key=lambda k: probs[k])
        ok = int(str(best) == str(t["expected"])); conf = max(probs.values()); pt = probs.get(str(t["expected"]), 0.0)
        ok = ok
    res.append(dict(id=t["id"], fam=t["family"], tier=tier[t["id"]], ok=ok, conf=conf, pt=pt))
n = len(res); acc = sum(r["ok"] for r in res) / n
bt = collections.defaultdict(list)
for r in res: bt[r["tier"]].append(r["ok"])
fam = collections.defaultdict(list)
for r in res: fam[r["fam"]].append(r["ok"])
nll = -sum(math.log(max(r.get("pt", 1e-6), 1e-6)) for r in res) / n
print(f"[{tag}] JEVBENCH acc={acc:.3f} ({sum(r['ok'] for r in res)}/{n}) " + " ".join(f"{k}={sum(v)/len(v):.3f}" for k, v in sorted(bt.items())) + f" nll={nll:.3f} secs={time.time()-t0:.0f}")
print(f"[{tag}] FAMILIES " + " ".join(f"{k}={sum(v)}/{len(v)}" for k, v in sorted(fam.items())))
# held-out short + generated sets
exs = [Example.from_dict(json.loads(l)) for l in open("data/pilot_eval.jsonl")]
order = sorted(range(len(exs)), key=lambda i: len(str(exs[i].state)))
exs_sorted = [exs[i] for i in order]
model.cuda().eval()
lg, lb, ns, _ = collect_logits(model, exs_sorted, device="cuda", batch_size=8, max_length=3072)
preds = predictions_from_logits(lg, lb, ns, exs_sorted, temperature=1.0)
by = collections.defaultdict(list)
for e, p in zip(exs_sorted, preds): by[e.task].append(int(p.pred == p.label))
print(f"[{tag}] HELDOUT " + " ".join(f"{k}={sum(v)/len(v):.3f}(n={len(v)})" for k, v in sorted(by.items())))
allok = [x for v in by.values() for x in v]
print(f"[{tag}] HELDOUT_MEAN {sum(allok)/len(allok):.3f}")
json.dump(dict(jb=res, held={k: v for k, v in by.items()}), open(f"eval_{tag}.json", "w"))

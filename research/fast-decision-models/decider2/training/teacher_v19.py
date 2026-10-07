import sys, json, torch, glob, os
from strands_decider.modeling import StrandsDeciderModel, MASK_VALUE
from strands_decider.data.format import Example
from strands_decider.evaluate import collect_logits
from strands_decider.modeling import masked_log_softmax
ck = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*"))[0]
model = StrandsDeciderModel.load(ck)
model.torso = model.torso.merge_and_unload()
exs = [Example.from_dict(json.loads(l)) for l in open("data/pilot_train.jsonl")]
order = sorted(range(len(exs)), key=lambda i: len(str(exs[i].state)))
srt = [exs[i] for i in order]
lg, lb, ns, _ = collect_logits(model, srt, device="cuda", batch_size=8, max_length=2048)
probs = masked_log_softmax(lg, ns).exp()
out = [None]*len(exs)
agree = 0
for r, i in enumerate(order):
    k = int(ns[r]); p = probs[r,:k].tolist(); out[i] = p
    agree += int(max(range(k), key=lambda j: p[j]) == exs[i].label)
with open("data/pilot_teacher_v19.jsonl","w") as f:
    for i, p in enumerate(out): f.write(json.dumps({"i": i, "probs": p})+"\n")
print("teacher argmax==gold", agree/len(exs), len(exs))

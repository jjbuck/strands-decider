"""Rebuild the exact train/val item sequence that extract.py used (same seeds), so teacher logits in feats_*.pt align."""
from lib import *

def load_syn(name, n, seed):
    rows = [json.loads(l) for l in open(f"{SD}/data/synthetic/{name}.jsonl")]
    rnd2 = random.Random(seed); rnd2.shuffle(rows)
    out = []
    for r in rows:
        if len(out) >= n: break
        if isinstance(r["options"], str): r["options"] = json.loads(r["options"])
        r["label"] = int(r["label"]); r["weight"] = float(r.get("weight", 1))
        if isinstance(r.get("instruction_variants"), str): r["instruction_variants"] = json.loads(r["instruction_variants"])
        if len(str(r["state"])) > 11000: continue
        out.append(Example.from_dict(r))
    return out

def build(which):
    rnd = random.Random(1234); items = []
    spec = {"train": [("generated_v16", 800, True), ("generated_v18", 700, True), ("adequacy_gen", 1000, True), ("flips_v20", 600, True)],
            "val": [("generated_v16_eval", 130, False), ("generated_v18_eval", 110, False), ("adequacy_gen_eval", 130, False), ("flips_v20_eval", 60, False)]}[which]
    train = which == "train"
    for k, (name, n, _) in enumerate(spec):
        for i, ex in enumerate(load_syn(name, n, 100 + k + (0 if train else 7))):
            order = None
            if train and ex.kind != "score":
                order = list(range(ex.n_options)); rnd.shuffle(order)
            elif train and rnd.random() < .5:
                order = list(reversed(range(ex.n_options)))
            instr = rnd.choice(ex.all_instructions()) if train else None
            q = ex.to_question(instr)
            lab = ex.label if order is None else order.index(ex.label)
            items.append(dict(id=f"{name}-{i}", kind=ex.kind, state=ex.state, q=q, order=order, label=lab))
    rnd.shuffle(items)
    return items

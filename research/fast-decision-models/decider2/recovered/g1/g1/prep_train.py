"""G1 training rows: real train-split states (KL-only, w=0) + train_v5 gold rows (w=1), rendered serving-style (F7's prep.fit),
hobson-v19 (merged) teacher logits (raw, T=1) on every row.  -> rows_train.pt, rows_dev.pt (held-out real rows for learning curves)
usage: python prep_train.py N_REAL N_GOLD"""
import os, sys, json, random, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1lib as G
from prep import enc_example, load
from torch.nn.utils.rnn import pad_sequence

n_real, n_gold = int(sys.argv[1]), int(sys.argv[2])
t0 = time.time()
real_items = G.pool_items(n_real + 200, seed=5, max_tok=4000, min_tok=200)
# exclude the covariance / model-study states (seeds 11, 23) is unnecessary for training, but keep dev disjoint from train
dev_items, tr_items = real_items[:200], real_items[200:]
def real_rows(items, seed):
    rows = G.fit_rows(items, maxlen=4096, per_item=1, seed=seed)
    for r in rows: r.update(w=0.0, label=0, kind=r["rq"].kind, src="real")
    return rows
rows = real_rows(tr_items, 3); dev = real_rows(dev_items, 4)
rng = random.Random(0)
v5 = load(os.path.expanduser("~/work/training/data/train_v5.jsonl"))
rng.shuffle(v5)
for e in v5[:n_gold]:
    r = enc_example(e, rng, True, "v5")
    r["ids"] = r["ids"].long(); rows.append(r)
print("rows", len(rows), "dev", len(dev), "tokens", sum(len(r["ids"]) for r in rows), round(time.time() - t0), "s", flush=True)

m = G.load_hobson()
TT = dict(m.config.temperature_by_kind); T0 = m.config.temperature
pad = m.tokenizer.pad_token_id


@torch.inference_mode()
def teach(rows, budget=24576):
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]["ids"])); k = 0
    while k < len(order):
        L = len(rows[order[k]]["ids"]); nb = max(1, min(32, budget // L))
        ch = [rows[i] for i in order[k:k + nb]]; k += len(ch)
        ids = pad_sequence([r["ids"].long() for r in ch], batch_first=True, padding_value=pad).cuda()
        am = pad_sequence([torch.ones(len(r["ids"]), dtype=torch.long) for r in ch], batch_first=True).cuda()
        W = max(r["n"] for r in ch)
        opt = torch.tensor([r["opt"] + [-1] * (W - r["n"]) for r in ch]).cuda()
        ns = torch.tensor([r["n"] for r in ch]).cuda()
        lg = m(input_ids=ids, attention_mask=am, n_slots=ns, opt_idx=opt, temperature=1.0)["logits"].float().cpu()
        for j, r in enumerate(ch): r["t_logits"] = lg[j, :r["n"]].tolist()


teach(rows); teach(dev)
for r in rows + dev:
    r.pop("rq", None)
ag = [int(torch.tensor(r["t_logits"]).argmax()) == r["label"] for r in rows if r["w"] > 0]
print("teacher done", round(time.time() - t0), "s; teacher==gold", sum(ag) / max(1, len(ag)), flush=True)
torch.save(dict(rows=rows, TT=TT, T0=T0), "rows_train.pt")
torch.save(dict(rows=dev, TT=TT, T0=T0), "rows_dev.pt")

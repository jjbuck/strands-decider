"""F7 teacher: frozen hobson-v19 (LoRA merged) pointer logits on pre-tokenized rows (prep.py), T=1 raw.
usage: python teacher.py rows_X.pt [tok_budget]  -> adds r['t_logits'] (fp32 list, rendered order) in place"""
import sys, os, glob, time, json, torch
sys.path.insert(0, os.path.expanduser("~/work/sd/src"))
from strands_decider.modeling import StrandsDeciderModel
from torch.nn.utils.rnn import pad_sequence
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fast
fast.apply(bf16_lora=False)

path = sys.argv[1]; budget = int(sys.argv[2]) if len(sys.argv) > 2 else 24000
CK = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*"))[0]
m = StrandsDeciderModel.load(CK)
m.torso = m.torso.merge_and_unload()
m = m.cuda().eval()
print("hobson temps", m.config.temperature, m.config.temperature_by_kind, flush=True)
D = torch.load(path); rows = D["rows"]
todo = [i for i, r in enumerate(rows) if "t_logits" not in r]
todo.sort(key=lambda i: -len(rows[i]["ids"]))
pad = m.tokenizer.pad_token_id
t0 = time.time(); ntok = 0; k = 0; agree = []; nb = 0
with torch.inference_mode():
    while k < len(todo):
        L = len(rows[todo[k]]["ids"]); nbatch = max(1, min(64, budget // L))
        chunk = todo[k:k + nbatch]; k += len(chunk)
        ids = pad_sequence([rows[i]["ids"].long() for i in chunk], batch_first=True, padding_value=pad).cuda()
        am = pad_sequence([torch.ones(len(rows[i]["ids"]), dtype=torch.long) for i in chunk], batch_first=True).cuda()
        W = max(rows[i]["n"] for i in chunk)
        opt = torch.tensor([rows[i]["opt"] + [-1] * (W - rows[i]["n"]) for i in chunk]).cuda()
        ns = torch.tensor([rows[i]["n"] for i in chunk]).cuda()
        out = m(input_ids=ids, attention_mask=am, n_slots=ns, opt_idx=opt, temperature=1.0)
        lg = out["logits"].float().cpu()
        for j, i in enumerate(chunk):
            r = rows[i]; r["t_logits"] = lg[j, :r["n"]].tolist()
            if r["w"] > 0: agree.append(int(lg[j, :r["n"]].argmax()) == r["label"])
        ntok += int(am.sum()); nb += 1
        if nb % 200 == 0:
            el = time.time() - t0
            print(f"{k}/{len(todo)} rows {ntok} tok {ntok/el:.0f} tok/s {el:.0f}s agree {sum(agree)/max(1,len(agree)):.3f}", flush=True)
el = time.time() - t0
print(f"done {len(todo)} rows {ntok} tok in {el:.0f}s ({ntok/max(el,1e-9):.0f} tok/s); teacher argmax==gold {sum(agree)/max(1,len(agree)):.4f} (n={len(agree)})", flush=True)
D["teacher_temps"] = dict(T=m.config.temperature, by_kind=m.config.temperature_by_kind)
torch.save(D, path)

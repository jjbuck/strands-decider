"""G3 student eval on evalkit suites (F7 ev.py protocol): canonical render, serving-style fit at max_length 16384, raw T=1 probabilities in
canonical label space. usage: python ev_g3.py MODEL CKPT TAG SUITE [SUITE...] -> results/SUITE.TAG.jsonl"""
import sys, os, json, time, torch
sys.path.insert(0, os.path.expanduser("~/work/sd/src"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.modeling import StrandsDeciderConfig, build_head, masked_log_softmax, gather_options, pool_last_token
from strands_decider.prompting import render_question, render_state
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moe as M
from prep_g3 import fit
ta = TypeAdapter(SC.Question)
KIT = os.path.expanduser("~/work/evalkit")


def load_student(model, ck):
    g = M.load_cfg(model); W = M.load_weights(g)
    t = M.MoETorso(g, W, r=16, alpha=32).cuda()
    t.lora.load_state_dict(torch.load(os.path.join(ck, "lora.pt")))
    t.W = t.merged(inplace=True); t.use_lora = False; t.eval(); t.tg = True
    c = StrandsDeciderConfig.from_json(os.path.join(ck, "strands_decider_config.json"))
    h = build_head(c, g.d); h.load_state_dict(torch.load(os.path.join(ck, "slot_head.pt"))); h = h.cuda().eval()
    tok = AutoTokenizer.from_pretrained(M.MODELS[model])
    return t, h, tok


def items_rows(tok, suite, maxlen):
    its = [json.loads(l) for l in open(f"{KIT}/suites/{suite}.jsonl")]
    rows = []
    for it in its:
        for qn, qd in it["questions"].items():
            rq = render_question(ta.validate_python(qd))
            s, q, opt = fit(tok, render_state(it["state"]), rq.text, rq, max_len=maxlen)
            rows.append(dict(id=it["id"], qn=qn, rq=rq, ids=torch.tensor(s + q), opt=opt, n=rq.n_slots))
    return rows


@torch.inference_mode()
def run(t, h, pad, rows, tokb=24576):
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]["ids"]))
    k = 0
    while k < len(order):
        L = len(rows[order[k]]["ids"]); nb = max(1, min(32, tokb // L))
        ch = [rows[i] for i in order[k:k + nb]]; k += len(ch)
        ids = pad_sequence([r["ids"] for r in ch], batch_first=True, padding_value=pad).cuda()
        am = pad_sequence([torch.ones(len(r["ids"]), dtype=torch.long) for r in ch], batch_first=True).cuda()
        Wd = max(r["n"] for r in ch)
        opt = torch.tensor([r["opt"] + [-1] * (Wd - r["n"]) for r in ch]).cuda()
        ns = torch.tensor([r["n"] for r in ch]).cuda()
        hid = t(ids, am).last_hidden_state
        lp = masked_log_softmax(h(pool_last_token(hid, am).float(), gather_options(hid, opt).float()), ns).float().cpu()
        for j, r in enumerate(ch):
            r["p"] = {lab: float(lp[j, i].exp()) for i, lab in enumerate(r["rq"].slot_labels)}


if __name__ == "__main__":
    model, ck, tag = sys.argv[1:4]; suites = sys.argv[4:]
    t, h, tok = load_student(model, ck)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    os.makedirs("results", exist_ok=True)
    for su in suites:
        t0 = time.time()
        rows = items_rows(tok, su, int(os.environ.get("EVMAX", 16384)))
        run(t, h, pad, rows)
        out = {}
        for r in rows: out.setdefault(r["id"], {})[r["qn"]] = r["p"]
        with open(f"results/{su}.{tag}.jsonl", "w") as f:
            for i, q in out.items(): f.write(json.dumps(dict(id=i, q=q)) + "\n")
        print(su, tag, len(rows), "questions", f"{time.time()-t0:.0f}s", flush=True)

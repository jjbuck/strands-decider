"""G1 evalkit runner. Canonical render, serving-style fit at max_length 16384 (as the refs), hobson's per-kind calibrated temperatures.
  python ev_g1.py TAG MODEL SUBSET
    MODEL = hobson | proj:which:kind:b:frac:method | ckpt:PATH.pt
    SUBSET = dev (REAL-agree 1/4, CF 1/3 pairs, CF-probe 1/3 pairs, JB-all) | full (JB-all, REAL-agree, LONG, CF, CF-probe) | comma list SUITE[/every]
Structured models are evaluated through their dense reconstruction W = BTT(R, L) (same function; bf16 rounding of W)."""
import os, sys, json, time, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g1lib as G, slib as S

tag, spec, subset = sys.argv[1:4]
t0 = time.time()
m = G.load_hobson()
TT = dict(m.config.temperature_by_kind); T0 = m.config.temperature


def apply_spec(m, spec):
    if spec == "hobson": return {}
    if spec.startswith("proj:"):
        _, which, kind, b, frac, meth = spec.split(":")
        covs = torch.load("covs.pt"); errs = []
        for (i, an, par, attr, lin) in G.targets(m, which):
            W = lin.weight.data.float(); C = covs[(i, G.GROUP[an])].cuda()
            What, fac, mac = G.project_one(W, C, kind, float(frac), int(b), meth, refine_steps=150)
            errs.append(S.out_err(W, What, C)); lin.weight.data = What.to(lin.weight.dtype).contiguous()
        return dict(mean_out_err=sum(errs) / len(errs))
    if spec.startswith("ckpt:"):
        ck = torch.load(spec[5:], map_location="cuda"); st = ck["state"]; args = ck["args"]
        mods = dict(m.named_modules())
        done = 0
        for (i, an, par, attr, lin) in G.targets(m, args["which"]):
            pref = [n for n, mm in mods.items() if mm is lin][0]
            if args["kind"] == "btt":
                W = S.btt_from_factors(st[pref + ".R"].float(), st[pref + ".L"].float())
            elif args["kind"] == "lrbd":
                W = S.lrbd_from(st[pref + ".U"].float(), st[pref + ".V"].float(), st[pref + ".D"].float())
            elif args["kind"] == "lora":
                W = lin.weight.data.float() + st[pref + ".B"].float() @ st[pref + ".A"].float()
            lin.weight.data = W.to(lin.weight.dtype).contiguous(); done += 1
        return dict(ckpt=spec[5:], replaced=done, info=ck.get("info"))
    raise ValueError(spec)


info = apply_spec(m, spec)
print("model", spec, json.dumps(info, default=str), round(time.time() - t0), "s", flush=True)

_orig = G.run_rows
@torch.inference_mode()
def run_rows_T(m, rows, tokb=24576):
    from torch.nn.utils.rnn import pad_sequence
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]["ids"]))
    pad = m.tokenizer.pad_token_id; k = 0
    while k < len(order):
        L = len(rows[order[k]]["ids"]); nb = max(1, min(32, tokb // L))
        ch = [rows[i] for i in order[k:k + nb]]; k += len(ch)
        ids = pad_sequence([r["ids"] for r in ch], batch_first=True, padding_value=pad).cuda()
        am = pad_sequence([torch.ones(len(r["ids"]), dtype=torch.long) for r in ch], batch_first=True).cuda()
        W = max(r["n"] for r in ch)
        opt = torch.tensor([r["opt"] + [-1] * (W - r["n"]) for r in ch]).cuda()
        ns = torch.tensor([r["n"] for r in ch]).cuda()
        T = torch.tensor([TT.get(r["rq"].kind, T0) for r in ch], dtype=torch.float32).cuda()
        lp = m(input_ids=ids, attention_mask=am, n_slots=ns, opt_idx=opt, temperature=T)["log_probs"].float().cpu()
        for j, r in enumerate(ch):
            r["p"] = {lab: float(lp[j, i].exp()) for i, lab in enumerate(r["rq"].slot_labels)}


if subset == "dev": suites = [("JB-all", 1), ("REAL-agree", 4), ("CF", 3), ("CF-probe", 3)]
elif subset == "full": suites = [("JB-all", 1), ("REAL-agree", 1), ("CF", 1), ("CF-probe", 1), ("LONG", 1)]
else: suites = [(s.split("/")[0], int(s.split("/")[1]) if "/" in s else 1) for s in subset.split(",")]
os.makedirs("res", exist_ok=True)
for su, every in suites:
    t1 = time.time()
    its = G.suite_items(su, every)
    rows = G.fit_rows(its, maxlen=16384)
    run_rows_T(m, rows)
    G.write_preds(rows, f"res/{su}.{tag}.jsonl")
    print(su, tag, len(rows), "questions", f"{time.time() - t1:.0f}s", flush=True)
print("done", round(time.time() - t0), flush=True)

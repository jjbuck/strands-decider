"""J11 final table: per run synthetic (iid / held / per family), evalkit suites, ECE, invariance, inference GFLOPs + A10G latency at T=1000.
  python3 table.py  (reads ~/decider2/j11/runs/*/score.json, lat_*.json) -> results.json + markdown on stdout"""
import os, json, glob, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
J = os.path.expanduser("~/decider2/j11")
lat = {}
for f in glob.glob(f"{J}/lat_[SM].json"):
    for r in json.load(open(f)): lat[(r["arm"], r["d"], r["L"], r["Q"], r["T"])] = r
rows = []; res = {}
for d in sorted(glob.glob(f"{J}/runs/*/score.json")):
    name = os.path.basename(os.path.dirname(d)); o = json.load(open(d)); meta = json.load(open(os.path.join(os.path.dirname(d), "meta.json")))
    e = o.get("eval", {}); S = o["suites"]; G = o["gt"]
    arm = meta["arm"]; la = lat.get(("vslot" if arm == "vslot" else arm, meta["d"], meta["L"], 1, 1000)) or {}
    la4 = lat.get((arm, meta["d"], meta["L"], 1, 4000)) or {}
    r = dict(run=name, arm=arm, d=meta["d"], L=meta["L"], params_nonemb_M=round(meta["params_nonemb"] / 1e6, 1), params_M=round(meta["params"] / 1e6, 1),
             train_pf=meta["budget_pf"], steps=meta["n_steps"], synth_iid=e.get("synth_iid"), synth_held=e.get("synth_held"), synth_ece=e.get("synth_ece"),
             synth=e.get("synth"), JB_all=S["JB-all"].get("acc"), JB_hard=S["JB-hard"].get("acc"), REAL_agree_sd=S["REAL-agree"].get("agree_sd"),
             LONG_agree_sd=S["LONG"].get("agree_sd"), CF_acc=S["CF"].get("acc"), CF_flip=S["CF"].get("flip"), CFp_acc=S["CF-probe"].get("acc"),
             CFp_flip=S["CF-probe"].get("flip"), REAL_label=S["REAL-label"].get("acc"), ECE_CF=G["CF"]["pred"]["ece"], ECE_JB=G["JB-all"]["pred"]["ece"],
             mcnemar_JB_all=G["JB-all"]["mcnemar"], mcnemar_JB_hard=G["JB-hard"]["mcnemar"], inv=o.get("kit_inv"),
             gflops_1000=la.get("gflops"), ms_1000=la.get("ms"), ms_4000=la4.get("ms"), lceval_agree=e.get("lceval_agree_hobson"))
    rows.append(r); res[name] = r
json.dump(res, open(f"{J}/results.json", "w"), indent=1)
f = lambda x, n=3: "-" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
print("| run | non-emb M | train PF | synth iid | held | JB-all | JB-hard | REAL agree_sd | LONG agree_sd | CF acc/flip | CF-probe acc/flip | REAL-label | ECE CF | |dp| rot | GF@1000 | ms@1000 | ms@4000 |")
print("|" + "---|" * 18)
print("| hobson-v19 (2B, reference) | ~1500 | - | - | - | 0.723 | 0.523 | 1 | 1 | 0.594/0.268 | 0.653/0.328 | 0.785 | 0.114 | - | ~2740 (GEMM) | 52.7 (fused) | 200.5 |")
order = {"XS": 0, "S": 1, "M": 2}; arms = ["dec", "decv", "slot", "belief", "vslot", "vslot2"]
rows.sort(key=lambda r: (order[r["run"].split("_")[0]], arms.index(r["run"].split("_", 1)[1]) if r["run"].split("_", 1)[1] in arms else 9))
for r in rows:
    print(f"| {r['run']} | {r['params_nonemb_M']} | {r['train_pf']} | {f(r['synth_iid'])} | {f(r['synth_held'])} | {f(r['JB_all'])} | {f(r['JB_hard'])} | "
          f"{f(r['REAL_agree_sd'])} | {f(r['LONG_agree_sd'])} | {f(r['CF_acc'])}/{f(r['CF_flip'])} | {f(r['CFp_acc'])}/{f(r['CFp_flip'])} | {f(r['REAL_label'])} | "
          f"{f(r['ECE_CF'])} | {f((r['inv'] or {}).get('mean_abs_dp'), 4)} | {f(r['gflops_1000'], 1)} | {f(r['ms_1000'], 2)} | {f(r['ms_4000'], 2)} |")

print()
fams = ["argmax", "multihop2", "id_match", "cmp_num", "cmp_date", "status", "count", "policy2", "H_policy3", "H_multihop3", "H_long", "H_schema"]
print("| run | " + " | ".join(fams) + " |"); print("|" + "---|" * (len(fams) + 1))
for r in rows:
    print(f"| {r['run']} | " + " | ".join(f"{r['synth'][k]:.2f}" for k in fams) + " |")

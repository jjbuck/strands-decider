import sys, os, json, torch, time, statistics as st, subprocess
sys.path.insert(0, os.path.expanduser("~/work/systems/g")); sys.path.insert(0, os.path.expanduser("~/work/d1"))
from prof_d1 import load_torso, capture
from common import CKPT
from lean import Lean
from lean2 import Lean2
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(CKPT); rt = tok._tokenizer
reqs = json.load(open(os.path.expanduser("~/work/tokens/real_reqs.json")))
T = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
real = []
for r in reqs:
    ids = rt.encode(r["state"], add_special_tokens=False).ids
    if len(ids) >= T: real.append(torch.tensor(ids[:T]).view(1, T))
    if len(real) >= 20: break
torso = load_torso()
def smi():
    return subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
for name, ln in (("base", Lean(torso)), ("fold", Lean2(torso, fuse="gnorm,prep,conv,fold"))):
    ids = torch.zeros(1, T, dtype=torch.long, device="cuda")
    g, out = capture(lambda: ln.forward(ids))
    pools = {"random": [torch.randint(1000, 100000, (1, T)) for _ in range(20)], "real": real,
             "pad": [torch.full((1, T), tok.pad_token_id or 0) for _ in range(20)], "same_token": [torch.full((1, T), 1234) for _ in range(20)]}
    for pn, pool in pools.items():
        for _ in range(3): ids.copy_(pool[0]); g.replay()
        torch.cuda.synchronize(); ts = []
        for x in pool:
            ids.copy_(x); torch.cuda.synchronize(); a = time.perf_counter(); g.replay(); torch.cuda.synchronize(); ts.append((time.perf_counter() - a) * 1e3)
        print(name, pn, T, "median", round(st.median(ts), 2), "p95", round(sorted(ts)[18], 2), "|", smi(), flush=True)
    del g, out; torch.cuda.empty_cache()

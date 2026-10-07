"""hobson-v19 bf16 latency on THIS box with the doc's fused runtime (d1 lean2, all fusions incl. fold), same protocol as bench_j10:
CUDA graph, fresh state ids per call (pinned H2D) + replay + D2H, median / p95 of 30 after 5 warm.  1 question = state + the real question;
4 questions = state + the 4 real questions as ONE causal pass (cost proxy for hobson's packed path: same GEMM rows, slightly more attention).
python hob_lat.py TS -> res_hob_lat.jsonl"""
import os, sys, json, time, statistics as st
sys.path[:0] = [os.path.expanduser('~/work/d1'), os.path.expanduser('~/work/systems/g'), os.path.expanduser('~/work/sd/src'), os.path.dirname(os.path.abspath(__file__))]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch
from strands_decider.modeling import StrandsDeciderModel
from lean2 import Lean2
from common import CKPT
from transformers import AutoTokenizer
import bench_j10 as BB

Ts = [int(x) for x in sys.argv[1].split(',')]
m = StrandsDeciderModel.load(CKPT)
torso = m.torso.merge_and_unload().eval().cuda().to(torch.bfloat16)
ln = Lean2(torso, fuse='addrms,gnorm,silu,prep,conv,gemm_swiglu,fold'); del torso; torch.cuda.empty_cache()
tok = AutoTokenizer.from_pretrained(CKPT)
reqs = [r for r in BB.real_questions(tok, 400) if len(r['qs']) >= 4]
qs4 = reqs[0]['qs'][:4]
print('hobson question tokens', [len(q['ids']) for q in qs4], flush=True)
out_path = 'res_hob_lat.jsonl'
for T in Ts:
    for nq in (1, 4):
        q = [t for qq in qs4[:nq] for t in qq['ids']]
        ids = torch.tensor([[5] * T + q], device='cuda')
        def run():
            return ln.forward(ids)[0, -1].float()
        g, out = BB.capture(run)
        class R_: pass
        rq = R_(); rq.T = T; rq.ids = ids[0]
        w = BB.wall(rq, g, out)
        rec = dict(model='hobson-bf16-lean2', T=T, nq=nq, qtok=len(q), wall=w)
        print(json.dumps(rec), flush=True)
        with open(out_path, 'a') as f: f.write(json.dumps(rec) + '\n')
        del g, out; torch.cuda.empty_cache()

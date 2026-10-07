import os, sys, json, time
sys.path[:0] = [os.path.expanduser('~/work/j9'), os.path.expanduser('~/work/evalkit')]
import torch
import j9lib as J
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
m = J.J9(); m.free_hf(); eng = m.p.eng; tok = eng.tok; m.setup_u(tok); m.head = m.head0
LL = J.LineLib(os.path.expanduser('~/work/evalkit/train_pool.jsonl'))
its = [json.loads(l) for l in open(os.path.expanduser('~/work/evalkit/suites/REAL-agree.jsonl'))]
it = [x for x in its if x['domain'] == 'banking_knowledge' and x['hook'] == 'after_tool_call'][3]
st = render_state(it['state']); req = J.tokenize_pieces(tok, J.pieces(st, LL))
qn = list(it['questions'])[0]; qd = it['questions'][qn]
rq = render_question(ta.validate_python(qd)); s, qs = eng._fit(st, [rq.text]); opt = eng._option_idx([rq], 0)[0].tolist()
with torch.inference_mode():
    for lay in ('native', 'R', 'S'):
        for r in range(3):
            torch.cuda.synchronize(); t0 = time.time()
            if lay == 'native': m.native_logits(s, qs[0], opt, rq.kind)
            else:
                t1 = time.time(); P = J.build_plan(m, req, qs[0], lay, m.dev); tp = time.time() - t1
                h = m.fwd_c(P, comp='affine')
            torch.cuda.synchronize()
            print(lay, f'{(time.time()-t0)*1000:.0f} ms', '' if lay == 'native' else f'plan {tp*1000:.0f} ms')
    from torch.profiler import profile, ProfilerActivity
    P = J.build_plan(m, req, qs[0], 'R', m.dev)
    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        h = m.fwd_c(P, comp='affine'); torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=18))

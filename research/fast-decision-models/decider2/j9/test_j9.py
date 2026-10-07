import os, sys, json, time
sys.path[:0] = [os.path.expanduser('~/work/j9'), os.path.expanduser('~/work/evalkit')]
import torch
import j9lib as J
from j9lib import chunk_gated_delta_rule
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
torch.manual_seed(0)
# (b) affine identity of the gated delta rule
T = 300; dev = 'cuda'
q = torch.randn(1, T, 16, 128, device=dev, dtype=torch.bfloat16); k = torch.randn_like(q); v = torch.randn_like(q)
g = -torch.rand(1, T, 16, device=dev) * 0.1; beta = torch.rand(1, T, 16, device=dev).to(torch.bfloat16)
S0 = torch.randn(1, 16, 128, 128, device=dev) * 0.1; Sin = torch.randn(1, 16, 128, 128, device=dev) * 0.1
_, E = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=S0, output_final_state=True, use_qk_l2norm_in_kernel=True)
I0 = torch.eye(128, device=dev)[None, None].expand(1, 16, 128, 128).contiguous()
_, A = chunk_gated_delta_rule(q, k, torch.zeros_like(v), g, beta, initial_state=I0, output_final_state=True, use_qk_l2norm_in_kernel=True)
_, Sd = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=Sin, output_final_state=True, use_qk_l2norm_in_kernel=True)
Sc = torch.matmul(A, Sin - S0) + E
print('affine identity rel err', float((Sc - Sd).norm() / Sd.norm()), 'A norm', float(A.norm() / I0.norm()))
m = J.J9(); m.free_hf(); eng = m.p.eng; tok = eng.tok; m.setup_u(tok); m.head = m.head0
print('U', m.U, tok.convert_ids_to_tokens(m.U))
LL = J.LineLib(os.path.expanduser('~/work/evalkit/train_pool.jsonl'))
its = [json.loads(l) for l in open(os.path.expanduser('~/work/evalkit/suites/REAL-agree.jsonl'))]
it = [x for x in its if x['domain'] == 'banking_knowledge' and x['hook'] == 'after_tool_call'][3]
st = render_state(it['state'])
req = J.tokenize_pieces(tok, J.pieces(st, LL))
print([(k, len(ids)) for k, ids, _ in req])
qn = list(it['questions'])[0]; qd = it['questions'][qn]
rq = render_question(ta.validate_python(qd)); s, qs = eng._fit(st, [rq.text]); opt = eng._option_idx([rq], 0)[0].tolist()
print('tokens equal', [t for _, ids, _ in req for t in ids] == s, len(s), len(qs[0]))
with torch.inference_mode():
    ref = torch.softmax(m.native_logits(s, qs[0], opt, rq.kind)[:rq.n_slots].float(), -1)
    print('native', ref.tolist())
    # (a) all-dyn plan == native
    req_d = [('frame' if k == 'frame' else 'dyn', ids, key) for k, ids, key in req]
    for lay in ('R', 'S'):
        P = J.build_plan(m, req_d, qs[0], lay, m.dev); h = m.fwd_c(P)
        print('all-dyn', lay, torch.softmax(m.logits_c(P, h, opt, rq.kind)[:rq.n_slots].float(), -1).tolist(), 'live', P.n_live)
    for lay in ('R', 'S'):
        for comp in ('affine', 'last', 'skip'):
            torch.cuda.synchronize(); t0 = time.time()
            P = J.build_plan(m, req, qs[0], lay, m.dev); h = m.fwd_c(P, comp=comp)
            pr = torch.softmax(m.logits_c(P, h, opt, rq.kind)[:rq.n_slots].float(), -1)
            torch.cuda.synchronize()
            print(lay, comp, pr.tolist(), 'live', P.n_live, 'T', P.T, f'{(time.time()-t0)*1000:.0f} ms')
print('mem', torch.cuda.max_memory_allocated() / 1e9)

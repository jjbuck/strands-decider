"""Option-order sensitivity of hobson-v19 in the fused TTL runtime (d1 lean2; verified vs the reference at cos >= 0.99987), one sequence per
(state, question rendering), max length 16384 (no truncation): REAL-agree + JB-all questions in the original order and under up to 3 cyclic
option rotations (rotlib; score questions: reversal only). Calibrated per-kind temperatures as deployed. -> rot_hob.json {iid: {q: {'r0': {label: p}, ..}}}"""
import os, sys, json, time
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path[:0] = [os.path.expanduser('~/work/h4'), os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens'), os.path.expanduser('~/work/j1')]
import torch
from tt_lean import TTL
import evalkit as EK
from kitrun import load_P, ta
from strands_decider.prompting import render_question, render_state
from rotlib import orders_for

dev = 'cuda'
Pm = load_P(); eng = Pm.eng; head = Pm.model.head.eval()
rt = TTL(Pm.tm, list(range(10)))
out = {}; t0 = time.time(); n = 0
with torch.inference_mode():
    for k, (s, iid, q, st, spec) in enumerate(EK.all_question_items(['REAL-agree', 'JB-all'])):
        qq = ta.validate_python(spec); base = render_question(qq)
        stt = render_state(st)
        for j, o in enumerate([None] + orders_for(base.kind, base.n_slots)):
            rq = render_question(qq, option_order=o)
            sids, qids = eng._fit(stt, [rq.text])
            opt = eng._option_idx([rq], len(sids))[0].tolist()
            ids = torch.tensor([sids + qids[0]], device=dev)
            rt.set_fuse(ids.shape[1])
            h, _ = rt.fwd(ids)
            lg = head(h[-1:].float(), h[torch.tensor(opt, device=dev)][None].float())[0] / Pm.temp_for(rq.kind)
            p = torch.softmax(lg.float(), -1).tolist()
            out.setdefault(iid, {}).setdefault(q, {})[f'r{j}'] = {lab: p[i] for i, lab in enumerate(rq.slot_labels)}
            n += 1
        if k % 200 == 0: print(k, n, '%.0fs' % (time.time() - t0), flush=True)
json.dump(out, open(os.path.expanduser('~/work/j1/rot_hob.json'), 'w'))
print('done', n, '%.0fs' % (time.time() - t0), flush=True)

import os, sys, json, numpy as np
sys.path[:0] = [os.path.expanduser('~/work/j7'), os.path.expanduser('~/work/evalkit')]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
from superbpe import Super, qwen_tok
from strands_decider.infer import _option_token_index
import evalkit as EK
from strands_decider.prompting import render_state
tok = qwen_tok()
class Eng:
    def __init__(s): s.tok = tok
    def _fit(s, st, qts):
        enc = tok(qts, add_special_tokens=False, return_offsets_mapping=True)
        s._last_offsets = enc['offset_mapping']
        return tok(st, add_special_tokens=True)['input_ids'], enc['input_ids']
    def _option_idx(s, rendered, base):
        import torch
        return torch.tensor([[base + i for i in _option_token_index(o, rq.option_spans, 0)] for rq, o in zip(rendered, s._last_offsets)])
import j7lib
S64 = Super(os.path.expanduser('~/work/j7/tok/sb64k.json'), tok=tok)
P = j7lib.Prep(Eng(), {'64k': S64})
its = EK.load_suite('CF-probe')[:3] + EK.load_suite('REAL-agree')[:3]
for it in its:
    st = render_state(it['state'])
    for qn, qd in it['questions'].items():
        pr = P.base(st, qd)
        b0 = P.build(pr, with_feats=True); b1 = P.build(pr, level='64k', with_feats=True)
        ids = pr['s'] + pr['q']
        # option rows map to the same original token
        assert [b1['ends'][o] for o in b1['opt']] == [b0['opt'][k] for k in range(len(b0['opt']))], 'opt map'
        assert b1['ends'][-1] == len(ids) - 1
        # constituents reproduce the original ids
        rec = []
        for t in b1['ids']: rec += list(S64.constituents(t))
        assert rec == ids
        nzv0 = int((b0['feats']['val'] != 0).sum()); nzv1 = int((b1['feats']['val'] != 0).sum())
        print(it['id'], qn, 'orig', len(ids), 'super', len(b1['ids']), 'val tokens', nzv0, nzv1, 'distinct vals', len(set(b0['feats']['val'][b0['feats']['val'] != 0].tolist())), len(set(b1['feats']['val'][b1['feats']['val'] != 0].tolist())))
print('ok')

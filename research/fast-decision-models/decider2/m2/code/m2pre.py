"""M2 precompilation check: every KB-document / hook-note segment of a request is computed STANDALONE ([U + segment], local positions,
its own forward) and its rows' hidden states are spliced into the request at every layer. Under segment isolation (pos=local) this
must not change the decision; under hobson's own layout the same splice is A6's out-of-context compile (J9).
python m2pre.py OUT.json --cfg SPEC [--n 80] [--ckpt CK]"""
import os, sys, json, argparse, collections
sys.path[:0] = [os.path.expanduser('~/work/m2'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from m2lib import M2, Cfg
from strands_decider.prompting import render_state
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--cfg', required=True); ap.add_argument('--n', type=int, default=80)
ap.add_argument('--ckpt', default='')
a = ap.parse_args()
m = M2(); m.free_hf(); m.setup()
if a.ckpt:
    import m2train_util as TU; TU.load_student(m, a.ckpt)
else: m.head = m.head0
cfg = Cfg(a.cfg)
its = [x for x in EK.load_suite('REAL-agree') if x['domain'] == 'banking_knowledge'][:a.n]
rows = []
with torch.inference_mode():
    for it in its:
        st = render_state(it['state']); names = list(it['questions'])
        prs = [m.prep_q(st, it['questions'][q]) for q in names]; s = prs[0]['s']
        ts = m.segs_for(st, s, cfg.gran if cfg.gran in ('nat', 'sec', 'const') else 'nat')
        I = m.build(s, ts, prs, cfg)
        comp = [j for j in range(I.nseg) if I.kinds[j] in ('doc', 'note')]
        if not comp: continue
        h_live = m.fwd_m2(I, keep=tuple(range(24)), keep_rows=torch.arange(I.Tm, device=m.dev))
        kept_live = dict(m.kept); lg_live = m.logits_m2(I, h_live)
        ov_rows = []; ov_vals = collections.defaultdict(list)
        for j in comp:
            rr = (I.segt == j).nonzero()[:, 0]
            ids = [I.ids_state[t] for t in range(I.u)] + [I.ids_state[int(t)] for t in rr]
            ts1 = dict(u=I.u, ne=0, seg=[0] * len(rr), kinds=[I.kinds[j]], nseg=1)
            J1 = m.build(ids, ts1, [], cfg)
            m.fwd_m2(J1, keep=tuple(range(24)), keep_rows=torch.arange(I.u, J1.Tm, device=m.dev))
            ov_rows.append(rr)
            for i in range(24): ov_vals[i].append(m.kept[i])
        rows_t = torch.cat(ov_rows)
        I.override = {i: (rows_t, torch.cat(ov_vals[i])) for i in range(24)}
        hd = max(float((kept_live[i][rows_t].float() - I.override[i][1].float()).abs().max()) for i in (0, 11, 23))
        rel = float((kept_live[23][rows_t].float() - I.override[23][1].float()).norm() / kept_live[23][rows_t].float().norm())
        h_c = m.fwd_m2(I); lg_c = m.logits_m2(I, h_c)
        for qn, x, y in zip(names, lg_live, lg_c):
            px, py = torch.softmax(x.float(), -1), torch.softmax(y.float(), -1)
            rows.append(dict(id=it['id'], q=qn, agree=int(px.argmax() == py.argmax()), dp=float((px - py).abs().max()), comp_rows=int(rows_t.numel()),
                             state_rows=I.Tm, max_abs_hidden_diff=hd, rel_hidden_diff_l23=rel))
out = dict(cfg=cfg.spec, n=len(rows), agree=sum(r['agree'] for r in rows), dp_max=max(r['dp'] for r in rows), dp_median=sorted(r['dp'] for r in rows)[len(rows) // 2],
           comp_share=sum(r['comp_rows'] for r in rows) / sum(r['state_rows'] for r in rows),
           rel_hidden_diff_l23_max=max(r['rel_hidden_diff_l23'] for r in rows), rows=rows)
print({k: v for k, v in out.items() if k != 'rows'}, flush=True)
json.dump(out, open(os.path.expanduser(a.out), 'w'), indent=1)

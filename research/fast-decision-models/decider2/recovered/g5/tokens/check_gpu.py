import json, time, torch, sys
sys.path.insert(0,'.')
from plib import *
p = P()
tm = p.tm
la = tm.layers[0].linear_attn
print('linear attn class', type(la).__name__, 'chunk fn', getattr(la,'chunk_gated_delta_rule',None).__module__ if hasattr(la,'chunk_gated_delta_rule') else None, 'causal_conv1d_fn', getattr(la,'causal_conv1d_fn',None))
print(tm.config.layer_types[:8], tm.config._attn_implementation, next(tm.parameters()).dtype)
rows=[]
for fn in ['easy','original','hard']:
    for l in open(f'data/{fn}.jsonl'):
        r=json.loads(l); r['tier']=fn; rows.append(r)
print(len(rows))
long=[r for r in rows if len(r['state'])>900][:6]+rows[:3]
for r in long:
    pr=p.prep(r['state'], r['question'])
    pb=p.base(pr)
    pe=p.eng._slot_probs_batched(render_state(r['state']), [pr['rq'].text], [pr['rq'].n_slots], [pr['rq'].kind], rendered=[pr['rq']])[0][0,:pr['rq'].n_slots]
    pp=p.pruned(pr,[(11,0.125)])
    pz=p.pruned(pr,[(11,0.0)],sink=0)
    print(r['id'], pr['q0'], pr['L']-pr['q0'], 'max|base-engine|', float((pb-pe).abs().max()), 'pruned k11 r.125 dKL', float((pb*(pb.clamp_min(1e-6)/pp.clamp_min(1e-6)).log()).sum()), 'r0 sink0', float((pb*(pb.clamp_min(1e-6)/pz.clamp_min(1e-6)).log()).sum()))

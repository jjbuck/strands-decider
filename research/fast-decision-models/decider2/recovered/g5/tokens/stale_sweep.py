import json, time, sys, torch
sys.path.insert(0,'.')
from plib import *
p=P()
items=[]
for fn in ['easy','original','hard']:
    for l in open(f'data/{fn}.jsonl'):
        r=json.loads(l)
        if len(r['state'])>900: items.append(r)
out=open('stale_sweep.jsonl','w')
CFG=[(22,None),(3,None),(7,None),(11,None),(15,None),(7,15),(11,15)]
for r in items:
    pr=p.prep(r['state'],r['question']); pb=p.base(pr)
    rec=dict(id=r['id'],family=r['family'],kind=pr['rq'].kind,labels=list(pr['rq'].slot_labels),expected=r['expected'],q0=pr['q0'],nq=pr['L']-pr['q0'],K=pr['rq'].n_slots,base=pb.tolist(),cfg={})
    for k,ql in CFG:
        rec['cfg'][f'stale|k{k}|qro{ql}']=p.pruned_stale(pr,k,qro_layer=ql).tolist()
    # sanity: k=23 -> must equal base
    out.write(json.dumps(rec)+'\n'); out.flush()
print('done')

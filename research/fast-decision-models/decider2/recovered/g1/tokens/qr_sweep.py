import json, time, sys, torch
sys.path.insert(0,'.')
from plib import *
p=P()
items=[]
for fn in ['easy','original','hard']:
    for l in open(f'data/{fn}.jsonl'):
        r=json.loads(l)
        if len(r['state'])>900: items.append(r)
CFG={}
for k in (11,15,19):
    CFG[f'qr|state dead@{k}|question readout-only@{k}']=(k,k)
    CFG[f'qr|state dead@{k}|question readout-only@{k+4 if k<19 else 23}']=(k,k+4 if k<19 else 23)
CFG['qr|state dead@7|question readout-only@11']=(7,11)
CFG['qr|state keep.125@7|question readout-only@11']=(7,11,.125)
out=open('qr_sweep.jsonl','w'); t0=time.time()
for r in items:
    pr=p.prep(r['state'],r['question']); pb=p.base(pr)
    rec=dict(id=r['id'],family=r['family'],kind=pr['rq'].kind,labels=list(pr['rq'].slot_labels),expected=r['expected'],q0=pr['q0'],nq=pr['L']-pr['q0'],K=pr['rq'].n_slots,base=pb.tolist(),cfg={})
    for name,c in CFG.items():
        k,ql=c[0],c[1]; sk=c[2] if len(c)>2 else 0.0
        rec['cfg'][name]=p.pruned_qr(pr,k,ql,state_keep=sk,sink=0 if sk==0 else 4).tolist()
    out.write(json.dumps(rec)+'\n'); out.flush()
print('done',time.time()-t0)

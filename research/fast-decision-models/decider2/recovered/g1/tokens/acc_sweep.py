import json, time, sys, random, torch, argparse
sys.path.insert(0,'.')
from plib import *
ap=argparse.ArgumentParser(); ap.add_argument('--set',default='jev'); ap.add_argument('--out',required=True); ap.add_argument('--maxq',type=int,default=3); ap.add_argument('--nreq',type=int,default=8)
a=ap.parse_args()
p=P()
CFG={}
def add(name,sched,scorer='qattn',sink=4): CFG[name]=(sched,scorer,sink)
for k in (3,7,11,15,19):
    add(f'q|k{k}|r.25',[(k,.25)]); add(f'q|k{k}|r.0625',[(k,.0625)]); add(f'q|k{k}|r0',[(k,0.0)])
for k in (3,7,11): add(f'rand|k{k}|r.0625',[(k,.0625)],'random'); add(f'tail|k{k}|r.0625',[(k,.0625)],'tail')
for k in (7,11,15): add(f'q|k{k}|dead',[(k,0.0)],'qattn',0)
add('q|k7|r.5',[(7,.5)]); add('q|k11|r.5',[(11,.5)])
add('M1 3:.5,7:.25,11:0',[(3,.5),(7,.25),(11,0.0)])
add('M2 3:.25,7:.0625,11:0',[(3,.25),(7,.0625),(11,0.0)])
add('M3 7:.25,11:.0625,15:0',[(7,.25),(11,.0625),(15,0.0)])
add('M4 3:.125,7:0',[(3,.125),(7,0.0)])
add('M5 3:.5,7:.125',[(3,.5),(7,.125)])
items=[]
if a.set=='jev':
    for fn in ['easy','original','hard']:
        for l in open(f'data/{fn}.jsonl'):
            r=json.loads(l); r['tier']=fn; items.append(dict(id=r['id'],family=r['family'],state=r['state'],q=r['question'],expected=r['expected']))
else:
    reqs=json.load(open('real_reqs.json')); rng=random.Random(1)
    by={}
    for i,r in enumerate(reqs): by.setdefault(r['hook'],[]).append((i,r))
    for hook,rs in by.items():
        for i,r in rs[:a.nreq]:
            names=list(r['questions']); rng.shuffle(names)
            for n in names[:a.maxq]:
                items.append(dict(id=f'{hook}#{i}#{n}',family=hook,state=r['state'],q=r['questions'][n],expected=None))
print('items',len(items),flush=True)
out=open(a.out,'w'); t0=time.time(); gen=torch.Generator(device='cuda'); gen.manual_seed(0)
for it in items:
    pr=p.prep(it['state'],it['q'])
    pb=p.base(pr)
    rec=dict(id=it['id'],family=it['family'],kind=pr['rq'].kind,labels=list(pr['rq'].slot_labels),expected=it['expected'],q0=pr['q0'],nq=pr['L']-pr['q0'],base=pb.tolist(),cfg={})
    if pr['q0']>=256:
        for name,(sched,sc,sink) in CFG.items():
            pp=p.pruned(pr,sched,sc,sink,rng=gen)
            rec['cfg'][name]=pp.tolist()
    out.write(json.dumps(rec)+'\n'); out.flush()
    del pr
print('done',time.time()-t0,flush=True)

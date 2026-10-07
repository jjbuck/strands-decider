"""Falsification experiment: can a short LoRA self-distillation 'heal' early state-token eviction in hobson-v19?
Teacher = unpruned hobson (adapters off). Student = same torso + fresh LoRA, run with state tokens evicted (random schedule per step).
Train: real banking hook requests (even request idx), teacher-labelled (no gold). Eval: held-out real requests (odd idx) + JevBench long tasks (gold)."""
import json, time, sys, random, math, torch, argparse
sys.path.insert(0,'.')
from plib import *
from torch.utils.checkpoint import checkpoint
from peft import LoraConfig, get_peft_model
ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=400); ap.add_argument('--lr',type=float,default=2e-4); ap.add_argument('--budget_min',type=float,default=18)
ap.add_argument('--out',default='heal_out.json'); ap.add_argument('--rank',type=int,default=16)
a=ap.parse_args()
p=P(); tm=p.tm; NL=p.NL; dev='cuda'
with torch.no_grad():
    for m in p.model.modules():
        for k,prm in list(m._parameters.items()):
            if prm is not None: m._parameters[k]=torch.nn.Parameter(prm.detach().clone(),requires_grad=False)
        for k,b in list(m._buffers.items()):
            if b is not None: m._buffers[k]=b.clone()
        for k,v in list(vars(m).items()):
            if isinstance(v,torch.Tensor) and v.is_inference(): setattr(m,k,v.clone())
torch.cuda.empty_cache()
for prm in p.model.parameters(): prm.requires_grad_(False)
cfg=LoraConfig(r=a.rank,lora_alpha=2*a.rank,lora_dropout=0.0,target_modules=r"(.*\.)?layers\.\d+\.(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_\w+|out_proj)|mlp\.(gate|up|down)_proj)")
pm=get_peft_model(tm,cfg)
ntrain=sum(x.numel() for x in pm.parameters() if x.requires_grad); print('trainable',ntrain,flush=True)
SCHED={'k7dead':([(7,0.0)],0),'k11dead':([(11,0.0)],0),'k3q25_k7dead':([(3,.25),(7,0.0)],0),'k3q50_k7q12_k11dead':([(3,.5),(7,.125),(11,0.0)],0)}

def layer(i,h,cos,sin): return tm.layers[i](h,position_embeddings=(cos,sin),attention_mask=None,position_ids=None,past_key_values=None)
def fwd(pr,sched,sink,grad):
    q0,L=pr['q0'],pr['L']; nq=L-q0
    ids=torch.tensor([pr['s']+pr['q']],device=dev)
    with torch.no_grad():
        h=tm.embed_tokens(ids); pos=torch.arange(L,device=dev).view(1,1,-1).expand(3,1,-1); cos,sin=tm.rotary_emb(h,pos)
    stage=dict(sched); n_alive=q0
    for i in range(NL):
        if i in stage:
            m=stage[i]; tgt=int(round(m*q0)) if m<=1 else int(m); tgt=max(tgt,sink); tgt=min(tgt,n_alive)
            if tgt<n_alive and tgt>0:
                with torch.no_grad(): sc=p.attn_scores(i,h.detach(),cos,sin,nq).clone()
                if sink>0: sc[:sink]=float('inf')
                top=torch.topk(sc,tgt).indices.sort().values
            elif tgt<n_alive:
                top=torch.zeros(0,dtype=torch.long,device=dev)
            do_evict=tgt<n_alive
        else: do_evict=False
        if grad: h=checkpoint(layer,i,h,cos,sin,use_reentrant=False)
        else: h=layer(i,h,cos,sin)
        if do_evict:
            rows=torch.cat([top,torch.arange(n_alive,n_alive+nq,device=dev)]); h=h[:,rows]; cos=cos[:,rows]; sin=sin[:,rows]; n_alive=tgt
    h=tm.norm(h); pooled=h[:,-1].float(); opt=h[:,[n_alive+o for o in pr['opt']]].float()
    lg=p.model.head(pooled,opt)/p.temp_for(pr['rq'].kind)
    return masked_log_softmax(lg,torch.tensor([pr['rq'].n_slots],device=dev))[0,:pr['rq'].n_slots]

reqs=json.load(open('real_reqs.json')); rng=random.Random(7)
def items_from(idx_filter,maxq):
    out=[]
    for i,r in enumerate(reqs):
        if not idx_filter(i): continue
        names=list(r['questions']); rng.shuffle(names)
        for n in names[:maxq]: out.append((f"{r['hook']}#{i}#{n}",r['state'],r['questions'][n]))
    return out
train=items_from(lambda i:i%2==0,4); hold=items_from(lambda i:i%2==1,2); rng.shuffle(train); rng.shuffle(hold); hold=hold[:30]
import os
Q=int(os.environ.get('QUICK','0'))
if Q: hold=hold[:Q]
jev=[]
for fn in ['easy','original','hard']:
    for l in open(f'data/{fn}.jsonl'):
        r=json.loads(l)
        if len(r['state'])>900: jev.append((r['id'],r['state'],r['question'],r['expected']))
if Q: jev=jev[:Q]
print('train',len(train),'hold',len(hold),'jev long',len(jev),flush=True)

@torch.no_grad()
def teacher(pr):
    with pm.disable_adapter(): return fwd(pr,[(NL+5,1.0)],0,False)
def ok(rq,lp,expected):
    pr_=lp.exp().tolist()
    if rq.kind=='noul': return ('yes' if pr_[rq.slot_labels.index('true')]>=.5 else 'no')==expected
    if rq.kind=='choice': return rq.slot_labels[max(range(len(pr_)),key=lambda t:pr_[t])]==expected
    return round(sum(int(l)*q for l,q in zip(rq.slot_labels,pr_)))==expected
@torch.no_grad()
def evaluate(tag):
    res={}
    def ag(rq,t,s):
        if rq.kind=='score':
            ar=torch.arange(len(t),device=dev).float()
            return int(round(float((t.exp()*ar).sum()))==round(float((s.exp()*ar).sum())))
        return int(t.argmax()==s.argmax())
    acc={}
    for setname,data in (('real',[(n,st,q,None) for n,st,q in hold]),('jev',jev)):
        for sname in SCHED: acc[(setname,sname)]=dict(agree=[],kl=[],acc=[],base_acc=[])
        for name,state,q,exp in data:
            pr=p.prep(state,q); t=teacher(pr)
            for sname,(sch,sink) in SCHED.items():
                s=fwd(pr,sch,sink,False); a=acc[(setname,sname)]
                a['agree'].append(ag(pr['rq'],t,s)); a['kl'].append(float((t.exp()*(t-s)).sum()))
                if exp is not None: a['acc'].append(ok(pr['rq'],s,exp)); a['base_acc'].append(ok(pr['rq'],t,exp))
    for (setname,sname),a in acc.items():
        res[f'{setname}|{sname}']={k:(sum(v)/len(v) if v else None) for k,v in a.items()}
    print('EVAL',tag,json.dumps(res),flush=True)
    return res
out={}
# untrained baseline for the same schedules (adapter B=0 so identical to no adapter)
t0=time.time(); out['before']=evaluate('before'); print('eval secs',time.time()-t0,flush=True)
opt=torch.optim.AdamW([x for x in pm.parameters() if x.requires_grad],lr=a.lr,weight_decay=0.0)
names=list(SCHED); t0=time.time(); losses=[]; step=0; accum=4
pm.train()
while step<a.steps and (time.time()-t0)<a.budget_min*60:
    name,state,q=train[step%len(train)]
    pr=p.prep(state,q)
    t=teacher(pr)
    sname=names[rng.randrange(len(names))]; sch,sink=SCHED[sname]
    s=fwd(pr,sch,sink,True)
    loss=(t.exp()*(t-s)).sum()/accum
    loss.backward(); losses.append(float(loss)*accum)
    step+=1
    if step%accum==0:
        lr=a.lr*min(1,step/40)*(0.5*(1+math.cos(math.pi*min(1,step/a.steps))))
        for g in opt.param_groups: g['lr']=lr
        torch.nn.utils.clip_grad_norm_([x for x in pm.parameters() if x.requires_grad],1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
    if step%25==0: print(f'step {step} loss(last25) {sum(losses[-25:])/25:.4f} t {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB',flush=True)
pm.eval()
out['after']=evaluate('after'); out['steps']=step; out['train_secs']=time.time()-t0; out['loss_curve']=[sum(losses[i:i+25])/25 for i in range(0,len(losses),25)]
json.dump(out,open(a.out,'w'),indent=1)

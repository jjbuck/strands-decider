import json, time, sys, statistics as st, torch
sys.path.insert(0,'.')
from plib import *
p=P(); tm=p.tm; NL=p.NL
reqs=json.load(open('real_reqs.json'))
# a real ~100-token noul question
qd=[r for r in reqs if r['hook']=='before_tool_call'][0]; qname=list(qd['questions'])[0]; Q=qd['questions'][qname]
texts=[r['state'] for r in reqs if r['n_state']>3200][:40]
print('n texts',len(texts),flush=True)
def mk(text,N):
    ids=p.tok(text,add_special_tokens=False)['input_ids'][:N]
    return p.tok.decode(ids)
def ms(f,reps=12,warm=2,inputs=None):
    for w in range(warm): f(inputs[w])
    ts=[]
    for r in range(reps):
        x=inputs[warm+r]
        torch.cuda.synchronize(); t=time.perf_counter(); f(x); torch.cuda.synchronize(); ts.append((time.perf_counter()-t)*1000)
    ts.sort(); return dict(med=round(st.median(ts),1),p95=round(ts[int(.95*(len(ts)-1))],1),min=round(ts[0],1))

@torch.inference_mode()
def live(pr, sched, scorer='qattn', sink=4, upto=None):
    """pruned forward with no stored activations (what a server would run). sched like plib.pruned. upto: run only first `upto` layers (profiling)."""
    q0,L=pr['q0'],pr['L']; nq=L-q0
    ids=torch.tensor([pr['s']+pr['q']],device='cuda')
    h=tm.embed_tokens(ids)
    pos=torch.arange(L,device='cuda').view(1,1,-1).expand(3,1,-1)
    cos,sin=tm.rotary_emb(h,pos)
    stage=dict(sched); n_alive=q0
    for i in range(NL if upto is None else upto):
        if i in stage:
            m=stage[i]; tgt=int(round(m*q0)) if m<=1 else int(m); tgt=max(tgt,sink); tgt=min(tgt,n_alive)
            if tgt<n_alive and scorer=='qattn': sc=p.attn_scores(i,h,cos,sin,nq); 
            h=p.layer(i,h,cos,sin)
            if tgt<n_alive:
                if scorer=='qattn':
                    if sink>0: sc=sc.clone(); sc[:sink]=float('inf')
                    top=torch.topk(sc,tgt).indices.sort().values
                else:  # 'none': dead = keep only first `sink` tokens
                    top=torch.arange(tgt,device='cuda')
                rows=torch.cat([top,torch.arange(n_alive,n_alive+nq,device='cuda')])
                h=h[:,rows]; cos=cos[:,rows]; sin=sin[:,rows]; n_alive=tgt
        else:
            h=p.layer(i,h,cos,sin)
    if upto is not None: return h
    return p.finish(h,[n_alive+o for o in pr['opt']],pr['rq'])

@torch.inference_mode()
def stock(pr):
    return p.eng._slot_probs_batched('x',[],[],[]) if False else None

res={}
Ns=[int(a) for a in sys.argv[1].split(',')] if len(sys.argv)>1 else [128,512,1000,2000,3000]
for N in Ns:
    prs=[p.prep(mk(texts[i%len(texts)]+f' salt{i}',N),Q) for i in range(14)]
    row={}
    @torch.inference_mode()
    def f_stock(pr):
        ids,mask=p._pad([pr['s']+pr['q']])
        out=p.model(input_ids=ids,attention_mask=mask,n_slots=torch.tensor([pr['rq'].n_slots],device='cuda'),temperature=p._temperatures([pr['rq'].kind]) if hasattr(p,'_temperatures') else p.eng._temperatures([pr['rq'].kind]),opt_idx=p.eng._option_idx([pr['rq']],pr['q0']) if False else torch.tensor([[pr['q0']+o for o in pr['opt']]],device='cuda'))
        return out
    row['stock_hf_forward']=ms(f_stock,inputs=prs)
    row['full_myloop']=ms(lambda pr: live(pr,[(NL+5,1.0)]),inputs=prs)
    row['k15_dead']=ms(lambda pr: live(pr,[(15,0.0)],'none',0),inputs=prs)
    row['k11_dead']=ms(lambda pr: live(pr,[(11,0.0)],'none',0),inputs=prs)
    row['k11_r.0625_qattn']=ms(lambda pr: live(pr,[(11,.0625)]),inputs=prs)
    row['k15_r.0625_qattn']=ms(lambda pr: live(pr,[(15,.0625)]),inputs=prs)
    row['k7_r.125_qattn']=ms(lambda pr: live(pr,[(7,.125)]),inputs=prs)
    row['M3_7:.25,11:.0625,15:0']=ms(lambda pr: live(pr,[(7,.25),(11,.0625),(15,0.0)]),inputs=prs)
    for u in (4,8,12,16):
        row[f'first{u}_layers_only']=ms(lambda pr: live(pr,[(NL+5,1.0)],upto=u),inputs=prs)
    res[N]=row
    print('N',N,'tokens(actual q0 %d, nq %d)'%(prs[0]['q0'],prs[0]['L']-prs[0]['q0']),flush=True)
    for k,v in row.items(): print('   %-28s med %7.1f  p95 %7.1f  min %7.1f'%(k,v['med'],v['p95'],v['min']),flush=True)
json.dump(res,open('lat_prune.json','w'),indent=1)

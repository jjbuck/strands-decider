import time, statistics as st, sys, torch, json
sys.path.insert(0,'.')
from plib import *
p=P(); tm=p.tm
def t(f,reps=10,warm=2):
    for _ in range(warm): f()
    ts=[]
    for _ in range(reps):
        torch.cuda.synchronize(); a=time.perf_counter(); f(); torch.cuda.synchronize(); ts.append((time.perf_counter()-a)*1000)
    ts.sort(); return round(st.median(ts),3)
res={}
with torch.inference_mode():
    for N in [128,256,512,1024,2048,4096,8192,16384]:
        h=torch.randn(1,N,2048,device='cuda',dtype=torch.bfloat16)*0.5
        pos=torch.arange(N,device='cuda').view(1,1,-1).expand(3,1,-1); cos,sin=tm.rotary_emb(h,pos)
        r={}
        r['linear_layer(L0)']=t(lambda: p.layer(0,h,cos,sin))
        r['full_attn_layer(L3)']=t(lambda: p.layer(3,h,cos,sin))
        r['mlp_only']=t(lambda: tm.layers[0].mlp(h))
        r['linear_attn_only']=t(lambda: tm.layers[0].linear_attn(h))
        r['full_attn_only']=t(lambda: tm.layers[3].self_attn(h,position_embeddings=(cos,sin),attention_mask=None))
        res[N]=r; print(N,r,flush=True)
        del h
json.dump(res,open('layer_scaling.json','w'),indent=1)

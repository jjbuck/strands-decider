"""Input-level selection ('scout then read'): keep B state tokens worth of sentences chosen by a scorer, re-encode from scratch with hobson, compare with the full-state answer.
Scorers: bm25 (zero-cost lexical), attn7 (oracle-ish: hobson's own layer-7 question->state attention summed per segment, i.e. what a learned scout would imitate), tail, headtail, random."""
import json, re, math, sys, random, collections, time, torch
sys.path.insert(0,'.')
from plib import *
p=P(); tok=p.tok
jev=[]
for fn in ['easy','original','hard']:
    for l in open(f'data/{fn}.jsonl'):
        r=json.loads(l)
        if len(r['state'])>900: jev.append(r)
print('long tasks',len(jev),flush=True)
def segs(text):
    # split into sentence/line segments keeping separators
    parts=re.findall(r'[^\n]*?(?:[.!?](?=\s)|\n|$)\s*',text)
    return [x for x in parts if x.strip()]
def words(s): return re.findall(r"[a-z0-9$%.]+",s.lower())
def bm25(seg_list,query,k1=1.2,b=.75):
    docs=[words(s) for s in seg_list]; N=len(docs); avg=sum(len(d) for d in docs)/max(1,N)
    df=collections.Counter(w for d in docs for w in set(d)); qs=set(words(query))
    out=[]
    for d in docs:
        tf=collections.Counter(d); sc=0
        for w in qs:
            if w in tf:
                idf=math.log(1+(N-df[w]+.5)/(df[w]+.5)); sc+=idf*tf[w]*(k1+1)/(tf[w]+k1*(1-b+b*len(d)/avg))
        out.append(sc)
    return out
def run(state,qd,ids_only=False):
    pr=p.prep(state,qd)
    with torch.inference_mode():
        pp=p.eng._slot_probs_batched(render_state(state),[pr['rq'].text],[pr['rq'].n_slots],[pr['rq'].kind],rendered=[pr['rq']])[0][0,:pr['rq'].n_slots]
    return pp,pr
def ok(rq,pp,expected):
    l=pp.tolist()
    if rq.kind=='noul': return ('yes' if l[rq.slot_labels.index('true')]>=.5 else 'no')==expected
    if rq.kind=='choice': return rq.slot_labels[max(range(len(l)),key=lambda t:l[t])]==expected
    return round(sum(int(a)*q for a,q in zip(rq.slot_labels,l)))==expected
def pred(rq,pp):
    l=pp.tolist()
    if rq.kind=='score': return round(sum(int(a)*q for a,q in zip(rq.slot_labels,l)))
    return max(range(len(l)),key=lambda t:l[t])
BUD=[128,256,512]; SC=['bm25','attn7','tail','headtail','random']
res=collections.defaultdict(list); rng=random.Random(0)
for r in jev:
    text=r['state'].strip(); S=segs(text); ntok=[len(tok(s,add_special_tokens=False)['input_ids']) for s in S]
    pb,pr=run(r['state'],r['question']); full_pred=pred(pr['rq'],pb); fo=ok(pr['rq'],pb,r['expected'])
    res['full'].append((fo,1,r['family'],sum(ntok)))
    # attn7 segment scores
    with torch.inference_mode():
        p.base(pr)
        sc_tok=p.attn_scores(7,pr['hs'][7],pr['cos'],pr['sin'],pr['L']-pr['q0']).float().cpu()   # over state tokens (incl. wrapper tokens)
    # map segments to token positions: tokenise rendered state w/ offsets
    rs=render_state(r['state']); enc=p.tok(rs,add_special_tokens=False,return_offsets_mapping=True)
    starts=[]; pos=rs.find(text[:30]) if text else 0
    # char start of each segment in rs
    cur=rs.find('\n')+1; segspan=[]
    for s in S:
        i=rs.find(s[:40],cur) if s else cur
        if i<0: i=cur
        segspan.append((i,i+len(s))); cur=i+len(s)
    seg_attn=[]
    for (a,b) in segspan:
        v=[float(sc_tok[t]) for t,(x,y) in enumerate(enc['offset_mapping']) if t<len(sc_tok) and x>=a and x<b]
        seg_attn.append(sum(v))   # total attention mass received (sum)
    q_text=pr['rq'].text
    scores={'bm25':bm25(S,q_text),'attn7':seg_attn,'tail':list(range(len(S))),'headtail':[(len(S)-i if i>=len(S)//2 else len(S)//2-i+len(S)) for i in range(len(S))],'random':[rng.random() for _ in S]}
    for B in BUD:
        for name in SC:
            order=sorted(range(len(S)),key=lambda i:-scores[name][i]); keep=set(); used=0
            for i in order:
                if used+ntok[i]<=B or not keep:
                    keep.add(i); used+=ntok[i]
            new=''.join(S[i] for i in sorted(keep))
            pp,pr2=run(new,r['question'])
            res[(name,B)].append((ok(pr2['rq'],pp,r['expected']),int(pred(pr2['rq'],pp)==full_pred),r['family'],used))
    del pr
print('n',len(jev),'full acc %.3f'%(sum(x[0] for x in res['full'])/len(jev)))
for B in BUD:
    for name in SC:
        v=res[(name,B)]
        print('B=%3d %-9s acc %.3f  agree-with-full %.3f  mean kept %.0f'%(B,name,sum(x[0] for x in v)/len(v),sum(x[1] for x in v)/len(v),sum(x[3] for x in v)/len(v)))
fam=collections.defaultdict(list)
for x in res[('attn7',256)]: fam[x[2]].append(x)
json.dump({str(k):v for k,v in res.items()},open('select_res.json','w'))
for B in (256,):
    for name in ('bm25','attn7','tail'):
        f=collections.defaultdict(list)
        for x,fo in zip(res[(name,B)],res['full']): f[x[2]].append((x[0],x[1],fo[0]))
        print(name,B,{k:(len(v),round(sum(a for a,_,_ in v)/len(v),2),round(sum(c for _,_,c in v)/len(v),2),round(sum(b for _,b,_ in v)/len(v),2)) for k,v in sorted(f.items())},'(n, acc, fullacc, agree)')

import sys, torch
sys.path.insert(0,'.')
from plib import *
p=P()
bad=[n for n,x in p.model.named_parameters() if x.is_inference()]
print('inference params',len(bad),bad[:5])
badb=[(n,tuple(b.shape)) for n,b in p.model.named_buffers() if b.is_inference()]
print('inference buffers',badb[:10])
for n,m in p.model.named_modules():
    for k,v in vars(m).items():
        if isinstance(v,torch.Tensor) and v.is_inference(): print('attr',n,k)

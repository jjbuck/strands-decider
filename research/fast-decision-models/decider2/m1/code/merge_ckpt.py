"""checkpoint arithmetic (no training): converted-mixer parameters from one checkpoint + LoRA from another (disjoint parameter sets, same transfer init).
python merge_ckpt.py C_FROM LORA_FROM OUT"""
import sys, torch
a = torch.load(sys.argv[1], map_location='cpu'); b = torch.load(sys.argv[2], map_location='cpu')
out = {k: v for k, v in a.items() if k.startswith('C.')}
out.update({k: v for k, v in b.items() if k.startswith('lora.')})
out['_meta'] = dict(a['_meta'])
torch.save(out, sys.argv[3]); print('merged', len(out), 'tensors ->', sys.argv[3])

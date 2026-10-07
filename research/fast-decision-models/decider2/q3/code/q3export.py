"""export deployed-kernel codes {(i,k): (q int8, s fp32)} from Q3 checkpoints without loading the model (CPU).
python q3export.py IN.pt OUT.pt [IN OUT ...]   IN may be 'gptq' (= ~/work/q3/gptq4.pt)"""
import os, sys, torch
args = sys.argv[1:]
for src, dst in zip(args[0::2], args[1::2]):
    if src == 'gptq':
        d = torch.load(os.path.expanduser('~/work/q3/gptq4.pt'), map_location='cpu'); out = {k: (e['q'].to(torch.int8), e['s'].float()) for k, e in d.items()}
    else:
        sd = torch.load(os.path.expanduser(src), map_location='cpu'); out = {}
        for k, e in sd['gemms'].items():
            if e['mode'] == 'lat': out[k] = (torch.round(e['A'].float() / e['s'][:, None]).clamp(-7, 7).to(torch.int8), e['s'].float())
            elif e['mode'] == 'code': out[k] = (e['q'].to(torch.int8), e['s'].float())
            else: raise ValueError(e['mode'])
    assert len(out) == 96, len(out)
    torch.save(out, os.path.expanduser(dst)); print('exported', src, '->', dst, flush=True)

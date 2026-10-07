import os, sys, json, time
sys.path.insert(0, os.path.expanduser('~/work/h3'))
import torch, h3lib as H, evalkit as EK
m = H.H3(); m.free_hf()
print('mem after load %.1fG' % (torch.cuda.memory_allocated() / 1e9), flush=True)
its = EK.load_suite('REAL-agree')[:4]
refs = EK.load_refs('REAL-agree')
fmts = [H.DENSE, H.Q('int4'), H.Q('int4', fold=True), H.Q('int4', rot=True), H.Q('nvfp4'), H.Q('mxfp4'), H.Q('int8'), H.Q('int4', 'bf16'), H.Q('bf16', 'int4')]
rows = [(it, qn) for it in its for qn in it['questions']]
outs = {}
with torch.inference_mode():
    for ci, qc in enumerate(fmts):
        m.wcache = {}
        for it, qn in rows:
            pr = m.prep(it['state'], it['questions'][qn]); ids = pr['s'] + pr['q']
            torch.cuda.synchronize(); t0 = time.time()
            p = m.probs(m.forward(ids, qc), pr).tolist()
            torch.cuda.synchronize(); outs.setdefault((it['id'], qn), []).append((round(time.time() - t0, 3), [round(x, 3) for x in p], len(ids)))
for it, qn in rows:
    ref = refs[it['id']]['hobson'][qn]
    print(qn, 'ref', {k: round(v, 3) for k, v in ref.items()}, flush=True)
    for qc, o in zip(fmts, outs[(it['id'], qn)]): print('   ', qc.rules[:1], qc.fold, qc.rot, o)
print('maxmem %.1fG' % (torch.cuda.max_memory_allocated() / 1e9))
x = torch.randn(5, 6144, device='cuda'); print('rot check', (H.rot_rows(x) - x @ H.rot_mat(6144, 'cuda')).abs().max().item())
x = torch.randn(5, 2048, device='cuda'); print('rot check', (H.rot_rows(x) - x @ H.rot_mat(2048, 'cuda')).abs().max().item())

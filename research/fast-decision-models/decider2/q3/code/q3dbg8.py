"""debug w4q8 training-mode forward: FAST (training) vs eager, grad vs no-grad, several requests incl. a short train_v5 row"""
import os, sys
sys.path[:0] = [os.path.expanduser('~/work/q3')]
import torch, q3lib as QL
m = QL.Q3()
pool, devset, v5 = QL.load_data(5000, 5, 0)
init = torch.load(os.path.expanduser('~/work/q3/gptq4.pt'), map_location='cpu')
src = {k: dict(q=e['q'], s=e['s'], Wc=(e['q'].float() * e['s'][:, None]).to(torch.bfloat16)) for k, e in init.items()}; del init
q8 = torch.load(os.path.expanduser('~/work/q3/gptq8.pt'), map_location='cpu')
QL.FAST = True
params = m.set_student('code', src, q8src=q8)
rqs = [m.prep(r['state'], [r['questions'][n] for n in sorted(r['questions'])[:3]]) for r in [r for r in pool[:80] if len(r['questions']) >= 2 and r['n'] < 2000][:3]]
rqs.insert(1, m.prep(v5[0]['state'], [QL.v5_q(v5[0])]))
for j, rq in enumerate(rqs):
    with torch.no_grad(): (lt, kt), _ = m.run(rq, student=False)
    for fast in (True, False):
        QL.FAST = fast
        for grad in (True, False):
            with torch.set_grad_enabled(grad):
                (ls, ks), lay = m.run(rq, student=True, ckpt=grad)
                loss, kl, hid = QL.loss_fn(lt, kt, ls, ks, 0.5, None)
            print('req', j, 'T', lay['T'], 'Ls', lay['Ls'], 'fast', fast, 'grad', grad, 'kl %.4f hid %.4f' % (kl, hid), flush=True)
    QL.FAST = True

import sys, time, torch, numpy as np
import train as T, models as Mo
rows = [r for r in T.load_rows('synth_test') if r['meta']['rot'] is None]
vmap, V = T.build_vocab([rows])
class A: pass
arms = sys.argv[1].split(','); d = int(sys.argv[2]); L = int(sys.argv[3]); mt = int(sys.argv[4])
for arm in arms:
    a = A(); a.d = d; a.L = L; a.R = 3
    m = Mo.build(arm, V, d, L).cuda(); T.init_weights(m, 24)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, fused=True)
    mbs = T.micro_batches(rows[:2400], arm, mt)
    torch.cuda.synchronize(); t = time.time(); tok = 0; fl = 0
    for i, mb in enumerate(mbs):
        b = T.collate(arm, mb, vmap, 'cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16): outs = m(b)
        l = T.loss_fn(outs, b).mean(); l.backward(); opt.step(); opt.zero_grad()
        tok += sum(len(r['ids']) for r in mb); fl += 3 * sum(T.row_flops(arm, a, r) for r in mb)
        if i == 2: torch.cuda.synchronize(); t = time.time(); tok = 0; fl = 0
    torch.cuda.synchronize(); el = time.time() - t
    print(arm, d, L, mt, f'tok/s {tok/el:.0f} TFLOPS(analytic) {fl/el/1e12:.1f}', 'mem GB %.1f' % (torch.cuda.max_memory_allocated()/1e9), flush=True)
    del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

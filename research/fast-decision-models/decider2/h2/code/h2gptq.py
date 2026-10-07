"""H1-format GPTQ codes on g1 (same calibration recipe as H1: h1calib.py, 64 train-split states, then h1lib.gptq act-order on R^T H R).
python h2gptq.py calib | codes w8|w4"""
import os, sys, time, torch
sys.path[:0] = [os.path.expanduser('~/work/h1'), os.path.expanduser('~/work/g2')]
import h1lib as HL
cmd = sys.argv[1]
if cmd == 'calib':
    sys.argv = ['h1calib.py', '--n', '64']
    exec(open(os.path.expanduser('~/work/h1/h1calib.py')).read())
else:
    wb = sys.argv[2]
    g = HL.H1(wq='gptq', wq8='gptq')
    prec = 'w8a8' if wb == 'w8' else 'w4a4'
    out = {}; t0 = time.time()
    for i in range(24):
        for k in HL.GEMMS:
            q, s = g.qweight(i, k, prec); out[(i, k)] = (q.cpu(), s.cpu()); g.Qw = {}
        print('layer', i, f'{time.time()-t0:.0f}s', flush=True)
    torch.save(out, os.path.expanduser(f'~/work/h2/codes_gptq_{wb}.pt'))
    print('saved', flush=True)

import torch, time
x = torch.empty(512 << 20, dtype=torch.uint8, device='cuda'); y = torch.empty_like(x)
xf = x.view(torch.float32)
def t(fn, n=10):
    fn(); torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n): fn()
    e1.record(); e1.synchronize(); return e0.elapsed_time(e1) / n / 1000
tc = t(lambda: y.copy_(x)); ts = t(lambda: xf.sum())
print('copy GB/s (read+write)', round(2 * x.numel() / tc / 1e9), 'read-only sum GB/s', round(x.numel() / ts / 1e9))
# small read: 25 MB, as one gate_up int8 weight, cycled over 8 buffers
bufs = [torch.empty(25 << 20, dtype=torch.uint8, device='cuda').view(torch.float32) for _ in range(8)]
def rd():
    for b in bufs: b.sum()
tr = t(rd); print('25MB x8 sum GB/s', round(8 * (25 << 20) / tr / 1e9), 'per-call us', round(tr / 8 * 1e6, 1))

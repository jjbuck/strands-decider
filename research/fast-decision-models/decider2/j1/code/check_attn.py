import torch, time, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
dev='cuda'
def bench(fn, n=5):
    for _ in range(2): fn()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.time()-t)/n*1000
for T in (512, 3072, 9216):
    q = torch.randn(1, 8, T, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 4, T, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 4, T, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
    mask = torch.ones(1, 1, T, T, device=dev, dtype=torch.bool); mask[..., :, T-100:] = False; mask[..., T-100:, :] = True
    for name, be, m in (('flash', SDPBackend.FLASH_ATTENTION, None), ('eff', SDPBackend.EFFICIENT_ATTENTION, None), ('eff_mask', SDPBackend.EFFICIENT_ATTENTION, mask), ('math_mask', SDPBackend.MATH, mask), ('cudnn', SDPBackend.CUDNN_ATTENTION, None)):
        def fwd():
            with sdpa_kernel([be]):
                return F.scaled_dot_product_attention(q, k, v, attn_mask=m, enable_gqa=True)
        def fb():
            o = fwd(); o.sum().backward()
        try:
            tf = bench(lambda: fwd()); tb = bench(fb)
            fl = 4 * T * T * 2048
            print(T, name, 'fwd %.2f ms (%.1f TFLOPS) fwd+bwd %.2f ms (%.1f TFLOPS eq 3.5x)' % (tf, fl / tf / 1e9, tb, 3.5 * fl / tb / 1e9), flush=True)
        except Exception as e:
            print(T, name, 'FAIL', str(e).split('\n')[0][:150], flush=True)
    del q, k, v, mask; torch.cuda.empty_cache()
# flex attention
try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    fx = torch.compile(flex_attention, dynamic=False)
    for T in (3072,):
        q = torch.randn(1, 8, T, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, 8, T, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, 8, T, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
        q0 = T - 100
        def mm(b, h, qi, ki): return (ki < q0) | (qi >= q0)
        bm = create_block_mask(mm, 1, 1, T, T, device=dev)
        for ko in ({}, {'BLOCK_M': 64, 'BLOCK_N': 32}, {'BLOCK_M': 32, 'BLOCK_N': 32}):
            try:
                def fwd(): return fx(q, k, v, block_mask=bm, kernel_options=ko or None)
                def fb():
                    o = fwd(); o.sum().backward()
                tf = bench(fwd); tb = bench(fb); fl = 4 * T * T * 2048
                print(T, 'flex', ko, 'fwd %.2f ms (%.1f TF) fwd+bwd %.2f ms (%.1f TF eq)' % (tf, fl / tf / 1e9, tb, 3.5 * fl / tb / 1e9), flush=True)
            except Exception as e:
                print(T, 'flex', ko, 'FAIL', str(e).split('\n')[0][:200], flush=True)
except Exception as e:
    print('flex import fail', e)

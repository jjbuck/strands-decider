"""Q4 B14: cost of the persistent kernel's 95 grid barriers alone (96 empty GEMM phases, all CTAs resident), us."""
import os, sys, json, ctypes, torch
sys.path[:0] = [os.path.expanduser('~/work/q4')]
import bench_pk as B
for lib_name in ('libq4pk.so', 'libq4pk2.so'):
    lib = ctypes.CDLL(os.path.expanduser(f'~/work/q4/{lib_name}'))
    lib.q4pk_run.restype = ctypes.c_int
    lib.q4pk_run.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    shapes = B.layer_shapes()
    A = torch.zeros(1, 6144, device='cuda', dtype=torch.int8); W = torch.zeros(1, 6144, device='cuda', dtype=torch.int8); O = torch.zeros(1, 12288, device='cuda', dtype=torch.int32)
    prog = (B.GDesc * len(shapes))(); dprog = torch.empty(ctypes.sizeof(B.GDesc) * len(shapes), device='cuda', dtype=torch.uint8)
    bar = torch.zeros(2, device='cuda', dtype=torch.int32)
    for g_, (N, K) in enumerate(shapes):
        d = prog[g_]; d.A = A.data_ptr(); d.W = W.data_ptr(); d.out = O.data_ptr(); d.M = 0; d.N = N; d.K = K; d.lda = K; d.ldw = K; d.ldo = N; d.alpha = 1.0; d.sk = 1
    for cfg in (0, 3):
        def run():
            rc = lib.q4pk_run(cfg, ctypes.addressof(prog), dprog.data_ptr(), len(shapes), bar.data_ptr(), 0, 0, torch.cuda.current_stream().cuda_stream)
            if rc: raise RuntimeError(rc)
        run(); torch.cuda.synchronize()
        print(json.dumps(dict(lib=lib_name, cfg=cfg, barriers=len(shapes) - 1, us=B.tm_(run))), flush=True)

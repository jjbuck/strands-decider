import torch
def kmeans(X, k, it=20, seed=0):
    gen = torch.Generator(device=X.device); gen.manual_seed(seed)
    C = X[torch.randperm(X.shape[0], generator=gen, device=X.device)[:k]].clone()
    for _ in range(it):
        d = (X * X).sum(1, keepdim=True) - 2 * X @ C.t() + (C * C).sum(1)[None, :]
        a = d.argmin(1)
        Cn = torch.zeros_like(C).index_add_(0, a, X); cnt = torch.bincount(a, minlength=k).float()
        m = cnt > 0; C[m] = Cn[m] / cnt[m, None]
    return C

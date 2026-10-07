"""H1 rotations (copied from h1lib.Rot, H1 FORMAT v0): randomized Kronecker Hadamard on the last dim."""
import math, torch
def paley12(dev):
    q = 11; sq = {(x * x) % q for x in range(1, q)}
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in sq else -1)
    H = torch.eye(12)
    H[0, 1:] += 1; H[1:, 0] -= 1
    for i in range(q):
        for j in range(q): H[1 + i, 1 + j] += chi(j - i)
    return (H / math.sqrt(12)).to(dev)
def hadamard(n, dev):
    H = torch.ones(1, 1, device=dev, dtype=torch.float32)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)
class Rot:
    def __init__(self, K, seed, dev, block=None):
        g = torch.Generator(device='cpu'); g.manual_seed(seed)
        self.K = K; self.block = block
        self.sign = (torch.randint(0, 2, (K,), generator=g) * 2 - 1).float().to(dev)
        if block:
            self.facs = [K // block, block]; self.mats = [None, hadamard(block, dev)]
        else:
            self.facs = [32, 64] if K == 2048 else [12, 16, 32]
            self.mats = [paley12(dev) if n == 12 else hadamard(n, dev) for n in self.facs]
    def __call__(self, x):
        sh = x.shape
        y = (x.float() * self.sign).reshape(-1, *self.facs)
        for ax, Hm in enumerate(self.mats):
            if Hm is None: continue
            y = torch.movedim(torch.tensordot(y, Hm, dims=([ax + 1], [0])), -1, ax + 1)
        return y.reshape(sh)
    def inv(self, x):
        sh = x.shape
        y = x.float().reshape(-1, *self.facs)
        for ax, Hm in enumerate(self.mats):
            if Hm is None: continue
            y = torch.movedim(torch.tensordot(y, Hm.t(), dims=([ax + 1], [0])), -1, ax + 1)
        return (y.reshape(sh) * self.sign)

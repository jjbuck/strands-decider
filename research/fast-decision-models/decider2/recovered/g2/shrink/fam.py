import torch, numpy as np, collections
store = torch.load("fit_store2.pt"); jb = torch.load("feats_jb.pt")["data"]
fam = np.array([f["fam"] for f in jb]); Tok = np.array([f["T"] for f in jb])
Ls = [10, 12, 13, 15, 16, 24]
P = {L: torch.softmax(store[L]["lgj"] / store[L]["T"], -1).argmax(-1).numpy() for L in Ls}; Y = store[24]["yj"].numpy()
print("family n " + " ".join(f"L{L:<3d}" for L in Ls))
for f in sorted(set(fam)):
    m = fam == f
    print(f"{f:18s} {m.sum():3d} " + " ".join(f"{(P[L][m]==Y[m]).mean():.2f}" for L in Ls))
for lo, hi in [(0, 300), (300, 1000), (1000, 100000)]:
    m = (Tok >= lo) & (Tok < hi)
    print(f"tokens [{lo},{hi}) n={m.sum()} " + " ".join(f"L{L}:{(P[L][m]==Y[m]).mean():.2f}" for L in Ls))
# paired disagreement L15 vs L24
d = (P[15] != P[24]).sum(); print("items where L15 pred != L24 pred:", d, "of", len(Y), "; L15 right/L24 wrong:", ((P[15]==Y)&(P[24]!=Y)).sum(), " L15 wrong/L24 right:", ((P[15]!=Y)&(P[24]==Y)).sum())

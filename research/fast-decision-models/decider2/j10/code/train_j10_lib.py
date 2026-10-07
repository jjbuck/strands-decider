"""Shared batch / loss helpers (identical to train_j10.py's F7 recipe)."""
import torch, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def batch_tensors(rows, pad, TT):
    ids = pad_sequence([r['ids'].long() for r in rows], batch_first=True, padding_value=pad)
    am = pad_sequence([torch.ones(len(r['ids']), dtype=torch.long) for r in rows], batch_first=True)
    Wd = max(r['n'] for r in rows)
    opt = torch.tensor([r['opt'] + [-1] * (Wd - r['n']) for r in rows])
    ns = torch.tensor([r['n'] for r in rows]); lab = torch.tensor([r['label'] for r in rows])
    tl = torch.zeros((len(rows), Wd)); has_t = torch.zeros(len(rows), dtype=torch.bool)
    dist = torch.zeros(len(rows), Wd); has_d = torch.zeros(len(rows), dtype=torch.bool)
    for j, r in enumerate(rows):
        if r.get('t_logits') is not None:
            tl[j, :r['n']] = torch.tensor(r['t_logits']) / TT.get(r['kind'], 1.0); has_t[j] = True
        if r.get('dist') is not None:
            dist[j, :r['n']] = torch.tensor(r['dist']); has_d[j] = True
    w = torch.tensor([r['w'] for r in rows])
    return ids, am, opt, ns, lab, tl, has_t, dist, has_d, w


def loss_rows(lp, lab, dist, has_d, tl, has_t, ns, w, tw=1.0, rw=1.0):
    valid = torch.arange(lp.shape[-1], device=lp.device)[None, :] < ns[:, None]
    safe = lp.masked_fill(~valid, 0.0)
    ce = torch.where(has_d, -(dist * safe).sum(-1), -safe.gather(1, lab[:, None]).squeeze(1))
    t_lp = F.log_softmax(tl.masked_fill(~valid, -1e4), -1)
    kl = ((t_lp.exp() * valid) * (t_lp - safe)).sum(-1)
    lab_rows = (w > 0).float(); kl_only = (w == 0).float()
    return lab_rows * ce + has_t.float() * kl * (lab_rows * tw + kl_only * rw)

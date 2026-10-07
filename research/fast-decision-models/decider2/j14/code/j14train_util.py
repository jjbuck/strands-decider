"""J14 training utilities: checkpoint IO, bf16 stochastic-rounding Adam with a decoupled L2-SP anchor."""
import os, math, torch
from j14lib import LoRA, NAMES


def save_ckpt(m, path, meta=None):
    sd = {'_meta': dict(meta or {})}
    if m.clora is not None:
        sd['_meta']['clora_r'] = m.clora.r; sd['_meta']['clora_scale'] = m.clora.scale; sd['_meta']['clora_layers'] = m.clora.layers
        for k, v in m.clora.state_dict().items(): sd['clora.' + k] = v.detach().cpu()
    if m.llora is not None:
        sd['_meta']['llora_r'] = m.llora.r; sd['_meta']['llora_scale'] = m.llora.scale; sd['_meta']['llora_layers'] = m.llora.layers
        for k, v in m.llora.state_dict().items(): sd['llora.' + k] = v.detach().cpu()
    if m.Lc is not None:
        for i in range(24):
            for nm in NAMES: sd[f'Lc.{i}.{nm}'] = m.Lc[i][nm].detach().cpu()
    if m.chead is not None:
        for k, v in m.chead.state_dict().items(): sd['chead.' + k] = v.detach().cpu()
    tmp = path + '.tmp'; torch.save(sd, tmp); os.replace(tmp, path)


def load_ckpt(m, path):
    sd = torch.load(path, map_location='cpu')
    meta = sd['_meta']; dev = m.dev
    if 'clora_r' in meta:
        r = meta['clora_r']; m.clora = LoRA(m.L, r, meta['clora_scale'] * r, dev, layers=meta.get('clora_layers', range(24)))
        m.clora.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('clora.')})
        m.clora.requires_grad_(False)
    if 'llora_r' in meta:
        r = meta['llora_r']; m.llora = LoRA(m.L, r, meta['llora_scale'] * r, dev, layers=meta.get('llora_layers', range(24)))
        m.llora.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('llora.')})
        m.llora.requires_grad_(False)
    if any(k.startswith('Lc.') for k in sd):
        m.Lc = [{nm: sd[f'Lc.{i}.{nm}'].to(dev) for nm in NAMES} for i in range(24)]
    m.live_ft = bool(meta.get('live_ft', False))
    if any(k.startswith('chead.') for k in sd):
        import h3lib as H
        m.chead = H.StdHead(m.head0 if not isinstance(m.head0, H.StdHead) else m.head0).to(dev)
        m.chead.load_state_dict({k[6:]: v.to(dev) for k, v in sd.items() if k.startswith('chead.')})
        m.chead.requires_grad_(False)
    return meta


class AdamSR(torch.optim.Optimizer):
    """Adam on bf16 parameters: bf16 moments, fp32 math, stochastic rounding of the weight update,
    decoupled L2-SP pull toward an anchor tensor (W <- W - lr * l2sp * (W - W0))."""

    def __init__(self, params, anchors, lr=1e-5, betas=(0.9, 0.999), eps=1e-8, l2sp=0.0, factored=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, l2sp=l2sp)); self.factored = factored
        self.anchor = {id(p): a for p, a in zip(self.param_groups[0]['params'], anchors)}

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g['betas']; lr = g['lr']
            for p in g['params']:
                if p.grad is None: continue
                st = self.state[p]
                fac = self.factored and p.dim() == 2
                if not st:
                    st['t'] = 0; st['m'] = torch.zeros_like(p, dtype=torch.bfloat16)
                    if fac:
                        st['vr'] = torch.zeros(p.shape[0], device=p.device); st['vc'] = torch.zeros(p.shape[1], device=p.device)
                    else:
                        st['v'] = torch.zeros_like(p, dtype=torch.bfloat16)
                st['t'] += 1; t = st['t']
                gr = p.grad.float()
                m_ = st['m'].float().mul_(b1).add_(gr, alpha=1 - b1); st['m'].copy_(m_)
                if fac:
                    g2 = gr * gr + 1e-30
                    st['vr'].mul_(b2).add_(g2.mean(1), alpha=1 - b2); st['vc'].mul_(b2).add_(g2.mean(0), alpha=1 - b2)
                    v_ = (st['vr'][:, None] * st['vc'][None, :]) / st['vr'].mean().clamp_min(1e-30); del g2
                else:
                    v_ = st['v'].float().mul_(b2).addcmul_(gr, gr, value=1 - b2); st['v'].copy_(v_)
                upd = (m_ / (1 - b1 ** t)) / ((v_ / (1 - b2 ** t)).sqrt_().add_(g['eps']))
                w = p.float()
                if g['l2sp'] > 0:
                    w.sub_((w - self.anchor[id(p)].float()).mul_(lr * g['l2sp']))
                w.add_(upd, alpha=-lr)
                # stochastic rounding fp32 -> bf16
                wi = w.view(torch.int32)
                noise = torch.randint_like(wi, 0, 1 << 16)
                wi.add_(noise).bitwise_and_(-65536)
                p.copy_(wi.view(torch.float32).to(torch.bfloat16))
                del gr, m_, v_, upd, w, wi, noise


def rel_drift(m):
    num = 0.0; den = 0.0
    for i in range(24):
        for nm in NAMES:
            a = m.L[i][nm].float(); b = m.Lc[i][nm].detach().float()
            num += float((b - a).pow(2).sum()); den += float(a.pow(2).sum())
    return math.sqrt(num / den)

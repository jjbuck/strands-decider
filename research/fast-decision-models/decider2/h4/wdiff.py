"""H4: weight lineage of this-that-1.0 (CPU only). Relative Frobenius distances between this-that, decider-2b (v10, v8), Qwen3.5-2B-Base and
hobson-v19 (base + its LoRA), per tensor family, plus the spectrum of the this-that adaptation delta."""
import os, glob, json, re, collections, torch
from safetensors import safe_open
H = os.path.expanduser('~/.cache/huggingface/hub')


def snap(repo, f='model.safetensors', rev=None):
    c = sorted(glob.glob(f'{H}/models--{repo.replace("/", "--")}/snapshots/*/{f}'))
    return c


def norm_key(k):
    k = re.sub(r'^(model\.language_model\.|model\.|language_model\.model\.)', '', k)
    return k


class ST:
    def __init__(self, paths):
        self.f = [safe_open(p, 'pt') for p in paths]
        self.idx = {}
        for i, h in enumerate(self.f):
            for k in h.keys(): self.idx[norm_key(k)] = (i, k)

    def get(self, k):
        i, kk = self.idx[k]
        return self.f[i].get_tensor(kk).float()


def fam(k):
    m = re.search(r'layers\.\d+\.(.*)\.weight$', k)
    if not m: return k.replace('.weight', '')
    return m.group(1)


def main():
    tt = ST(snap('flock-io/this-that-model-1.0'))
    base = ST(snap('Qwen/Qwen3.5-2B-Base', 'model.safetensors-00001-of-00001.safetensors'))
    dsn = {}
    for p in glob.glob(f'{H}/models--Mapika--decider-2b/snapshots/*/model.safetensors'):
        dsn[p.split('/')[-2][:10]] = ST([p])
    hob = glob.glob(f'{H}/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*/lora/adapter_model.safetensors')[0]
    hcfg = json.load(open(os.path.dirname(hob) + '/adapter_config.json'))
    scale = hcfg['lora_alpha'] / hcfg['r']
    lo = safe_open(hob, 'pt'); lk = {}
    for k in lo.keys():
        m = re.search(r'layers\.(\d+)\.(.*)\.lora_([AB])\.weight$', k)
        if m: lk.setdefault(f'layers.{m.group(1)}.{m.group(2)}.weight', {})[m.group(3)] = k
    keys = [k for k in tt.idx if k.endswith('.weight') and ('layers.' in k or 'embed' in k) and 'mtp' not in k]
    print('tensors', len(keys), 'decider snapshots', list(dsn), 'hobson lora targets', len(lk), 'scale', scale)
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    spec = {}
    for k in keys:
        if k not in base.idx: continue
        T = tt.get(k); B = base.get(k)
        if T.shape != B.shape: continue
        f = fam(k); nb = B.norm().item() + 1e-12
        rows[f]['tt_vs_base'].append((T - B).norm().item() / nb)
        for nm, D in dsn.items():
            if k in D.idx:
                X = D.get(k)
                rows[f][f'tt_vs_dec{nm}'].append((T - X).norm().item() / (X.norm().item() + 1e-12))
                rows[f][f'dec{nm}_vs_base'].append((X - B).norm().item() / nb)
        if k in lk:
            dl = scale * (lo.get_tensor(lk[k]['B']).float() @ lo.get_tensor(lk[k]['A']).float())
            rows[f]['hob_vs_base'].append(dl.norm().item() / nb)
            Hm = B + dl
            rows[f]['tt_vs_hob'].append((T - Hm).norm().item() / (Hm.norm().item() + 1e-12))
        if T.dim() == 2 and re.search(r'layers\.(3|12)\.', k) and min(T.shape) >= 512:
            # spectrum of the adaptation delta (vs the closest decider snapshot if present, else base)
            ref = None
            if dsn:
                best = min(dsn, key=lambda n: (T - dsn[n].get(k)).norm().item() if k in dsn[n].idx else 1e9)
                ref = dsn[best].get(k); rn = 'dec' + best
            else:
                ref = B; rn = 'base'
            for nm, R in ((rn, ref), ('base', B)):
                s = torch.linalg.svdvals(T - R); e = (s ** 2).cumsum(0) / (s ** 2).sum()
                spec[f'{k}|{nm}'] = dict(rel=round((T - R).norm().item() / R.norm().item(), 5), e16=round(e[15].item(), 3), e64=round(e[63].item(), 3), e256=round(e[255].item(), 3))
    out = {f: {m: round(float(sum(v) / len(v)), 5) for m, v in d.items()} for f, d in rows.items()}
    for f, d in out.items(): print(f'{f:32s}', d)
    for k, v in spec.items(): print('spec', k, v)
    json.dump(dict(fam=out, spec=spec), open(os.path.expanduser('~/work/h4/wdiff.json'), 'w'), indent=1)


if __name__ == '__main__':
    torch.set_num_threads(4)
    main()

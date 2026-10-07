"""M2 training utilities: new-module hooks, checkpoint io, cached dev references."""
import os, sys, json, torch
sys.path[:0] = [os.path.expanduser('~/work/m2')]
NAMES = ('Win', 'Wo', 'Wgu', 'Wd')


def add_extra(m, a, cfg):
    """new mixer modules for the layout (full rank). None in the base M2/M3 layouts (masks and depth only)."""
    m.extra = None
    return []


def state_dict(m, meta):
    sd = m.trainable_state()
    if getattr(m, 'mem', None) is not None:
        for k, md in m.mem.items(): sd[f'mem.{k}.A'] = md.A.detach().cpu(); sd[f'mem.{k}.B'] = md.B.detach().cpu()
        meta = dict(meta, mem_scale=m.mem_scale)
    if getattr(m, 'extra', None) is not None:
        for k, v in m.extra.state_dict().items(): sd[f'extra.{k}'] = v.detach().cpu()
    sd['_meta'] = dict(sd['_meta'], **meta)
    return sd


def load_state(m, sd):
    with torch.no_grad():
        for i in range(24):
            for k in NAMES:
                m.lora[i][k].A.copy_(sd[f'lora.{i}.{k}.A']); m.lora[i][k].B.copy_(sd[f'lora.{i}.{k}.B'])
        m.head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith('head.')})
        if getattr(m, 'mem', None) is not None:
            for k, md in m.mem.items(): md.A.copy_(sd[f'mem.{k}.A']); md.B.copy_(sd[f'mem.{k}.B'])
        if getattr(m, 'extra', None) is not None:
            m.extra.load_state_dict({k[6:]: v for k, v in sd.items() if k.startswith('extra.')})


def load_student(m, path):
    """inference: LoRA kept unmerged (as J3/J9), head loaded; returns meta"""
    meta = m.load_trainable(os.path.expanduser(path))
    sd = torch.load(os.path.expanduser(path), map_location=m.dev)
    mk = sorted({k.split('.')[1] for k in sd if k.startswith('mem.')})
    if mk:
        import torch.nn as nn
        m.mem = nn.ModuleDict(); m.mem_scale = sd['_meta'].get('mem_scale', 1.0)
        for k in mk:
            md = nn.Module(); md.A = nn.Parameter(sd[f'mem.{k}.A'].to(m.dev), requires_grad=False); md.B = nn.Parameter(sd[f'mem.{k}.B'].to(m.dev), requires_grad=False)
            m.mem[k] = md
    return sd['_meta']


class DevRef:
    """teacher decisions on the fixed dev set + hobson's decision with an empty state (for agree_sd), cached to disk"""

    def __init__(self, m, dev, path, Teacher):
        self.m = m; self.dev = dev; self.path = path; self.T = Teacher; self.ref = {}

    def build(self):
        if os.path.exists(self.path):
            self.ref = torch.load(self.path); print('devref loaded', len(self.ref), flush=True); return
        from strands_decider.prompting import render_state
        m = self.m
        with torch.no_grad(), self.T():
            for d in self.dev:
                st = render_state(d['state']); names = list(d['questions'])
                prs = [m.prep_q(st, d['questions'][n]) for n in names]
                s = prs[0]['s']
                if len(s) > 12000: continue
                lt = m.statefirst_logits(s, [p['q'] for p in prs], prs)
                st0 = render_state('')
                for n, p, l in zip(names, prs, lt):
                    p0 = m.prep_q(st0, d['questions'][n])
                    l0 = m.statefirst_logits(p0['s'], [p0['q']], [p0])[0]
                    self.ref[(d['task'], d.get('rid', ''), n)] = (torch.softmax(l.float(), -1).cpu(), int(l0.argmax()))
        torch.save(self.ref, self.path); print('devref built', len(self.ref), flush=True)

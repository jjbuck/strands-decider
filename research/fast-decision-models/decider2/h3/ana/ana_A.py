import json, sys, numpy as np
f = sys.argv[1]
d = json.load(open(f)); R = d['res']
FA = (3, 7, 11, 15, 19, 23)
POS = ['layer_in', 'attn_in', 'in_proj', 'core', 'o_in', 'attn_out', 'A+R', 'mlp_in', 'mlp_hid', 'mlp_out', 'layer_out']
print(f, d['fmt'], 'n', d['n'])
print('ql  type  inj_rel(layer_out@ql)  rel@23/inj  readout_rel  dec_tv  dec_flip')
for ql in map(str, range(24)):
    o = R[ql]; inj = o[ql]['layer_out']['rel']
    print(f"{ql:>2} {'ATT' if int(ql) in FA else 'GDN'}  {inj:.4f}  {o['23']['layer_out']['rel']/inj:6.2f}  {o['readout_rel']:.4f}  {o['dec']['tv']:.3f}  {o['dec']['flip']:.3f}")
# error at end of each layer, normalised to injected, for a few ql
print('\nlayer_out rel error / injected, by downstream layer (rows: quantised layer)')
print('ql  ' + ' '.join(f'{i:>5}' for i in range(24)))
for ql in [0, 1, 2, 3, 5, 7, 10, 12, 15, 18, 20, 22]:
    o = R[str(ql)]; inj = o[str(ql)]['layer_out']['rel']
    print(f'{ql:>2}  ' + ' '.join(f"{o[str(i)]['layer_out']['rel']/inj:5.2f}" if str(i) in o else '    .' for i in range(24)))
# within-layer: relative error at each position normalised to the relative error at layer_in (downstream layers only), averaged over ql<i
print('\nwithin-layer amplification (paper metric): rel(pos) / rel(layer_in), mean over (ql < i); GDN vs ATT layers')
for typ in ('GDN', 'ATT'):
    acc = {p: [] for p in POS}
    for ql in range(24):
        o = R[str(ql)]
        for i in range(ql + 1, 24):
            if (i in FA) != (typ == 'ATT'): continue
            e = o[str(i)]; base = e['layer_in']['rel']
            for p in POS:
                if p in e: acc[p].append(e[p]['rel'] / base)
    print(typ, '  '.join(f'{p}={np.median(v):.2f}' for p, v in acc.items() if v))
# absolute gains (Jacobian picture): ||d submodule out|| / ||d residual in||
print('\nabsolute error gains (RMS per token), median over ql < i: B = d(attn_out)/d(layer_in), M = d(mlp_out)/d(A+R), resid growth = d(layer_out)/d(layer_in)')
for i in range(24):
    B = []; M = []; G = []; rI = []; rA = []
    for ql in range(i):
        e = R[str(ql)][str(i)]
        B.append(e['attn_out']['abs'] / e['layer_in']['abs']); M.append(e['mlp_out']['abs'] / e['A+R']['abs']); G.append(e['layer_out']['abs'] / e['layer_in']['abs'])
        rI.append(e['attn_out']['ref'] / e['layer_in']['ref']); rA.append(e['mlp_out']['ref'] / e['A+R']['ref'])
    if B: print(f"{i:>2} {'ATT' if i in FA else 'GDN'}  B={np.median(B):.3f} (signal ratio {np.median(rI):.3f})  M={np.median(M):.3f} (signal ratio {np.median(rA):.3f})  growth={np.median(G):.3f}  resid_norm={R['0'][str(i)]['layer_in']['ref']:.1f}")
# self-error: at the quantised layer itself, which sub-position carries most relative error
print('\nat the quantised layer: rel error per position')
for ql in [0, 1, 2, 3, 6, 7, 12, 15, 20, 23]:
    e = R[str(ql)][str(ql)]
    print(f"{ql:>2} " + ' '.join(f"{p}={e[p]['rel']:.3f}" for p in POS if p in e) + (f" beta={e['beta']['rel']:.3f} g={e['g']['rel']:.3f}" if 'beta' in e else ''))

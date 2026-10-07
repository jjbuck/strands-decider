import sys, json, collections
sys.path.insert(0, '/tmp/decider2/evalkit'); import evalkit as EK
d = json.load(open(sys.argv[1])); P = d['preds']
refs = EK.load_refs('REAL-agree'); its = {it['id']: it for it in EK.load_suite('REAL-agree')}
am = lambda p: max(p, key=p.get)
print('keep', d['keep'], 'layers', d['layers'], 'overlap with global', {k[-30:]: round(v, 3) for k, v in d['overlap'].items()})
tot = collections.defaultdict(lambda: [0, 0, 0, 0])
print(f'{"spec":28s} {"n":>4s} ' + ' '.join(f'{c:>22s}' for c in ('own', 'global', 'other')) + '   (agree vs my dense / agree_sd vs hobson)')
for sp in d['specs']:
    q = sp[2]; ids = [i for i in P['dense'] if q in P['dense'][i] and (its[i]['domain'], its[i]['hook']) == tuple(sp[:2])]
    line = f'{q:28s} {len(ids):4d} '
    for c in ('own', 'global', 'other'):
        ag = [am(P[c][i][q]) == am(P['dense'][i][q]) for i in ids]
        sd = [am(P[c][i][q]) == am(EK._norm(refs[i]['hobson'][q])) for i in ids
              if am(EK._norm(refs[i]['cfg']['nostate'][q])) != am(EK._norm(refs[i]['hobson'][q]))]
        line += f'   {sum(ag)/len(ag):.3f} / {sum(sd)/max(1,len(sd)):.3f} ({len(sd):3d})'
        t = tot[c]; t[0] += sum(ag); t[1] += len(ag); t[2] += sum(sd); t[3] += len(sd)
    print(line)
print(f'{"ALL":28s} {tot["own"][1]:4d} ' + ' '.join(f'   {t[0]/t[1]:.3f} / {t[2]/t[3]:.3f} ({t[3]:3d})' for c, t in tot.items()))

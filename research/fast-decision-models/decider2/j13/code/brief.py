"""compact table from score.py JSON: python brief.py SCORE.json"""
import sys, json
S = json.load(open(sys.argv[1]))
f = lambda d, k: ('%.3f' % d[k]) if d and isinstance(d.get(k), (int, float)) and d.get(k) is not None else '  -  '
print('%-26s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s  %s' % ('config', 'FLOPs', 'REALsd', 'LONGsd', 'CFfgh', 'PRfgh', 'CFacc', 'PRacc', 'CFflip', 'PRflip', 'RLab', 'JB-hard (lost/gained p)'))
for k, s in S.items():
    jb = s.get('JB-hard') or {}
    print('%-26s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s  %s (%s/%s p %s)' % (k, f(s['flops'], 'all'), f(s['REAL'], 'agree_sd'), f(s['LONG'], 'agree_sd'), f(s['CF'], 'flip_given_hobson'),
          f(s['CF-probe'], 'flip_given_hobson'), f(s['CF'], 'acc'), f(s['CF-probe'], 'acc'), f(s['CF'], 'flip'), f(s['CF-probe'], 'flip'), f(s.get('REAL-label'), 'acc'),
          f(jb, 'acc'), jb.get('lost', '-'), jb.get('gained', '-'), f(jb, 'mcnemar_p')))

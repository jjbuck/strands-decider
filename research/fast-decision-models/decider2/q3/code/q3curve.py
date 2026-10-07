"""laptop: score curve preds files (CF-probe + REAL-agree only). python3 q3curve.py FILE [FILE...]"""
import sys, os, json
sys.path.insert(0, os.path.expanduser('~/decider2/q3/code')); sys.path.insert(0, os.path.expanduser('~/decider2/evalkit'))
import q3score as S, evalkit as EK
R = S.load(S.BF16)
print('| file | REAL flips vs hobson | REAL TV | REAL agree_sd | McNemar vs bf16 runtime (lost/gained, p) | CF-probe fgh | CF-probe flip |')
print('|---|---|---|---|---|---|---|')
out = {}
for f in sys.argv[1:]:
    P = S.load(f)
    ra = EK.score('REAL-agree', P, baselines=False)['model']; cp = EK.score('CF-probe', P, baselines=False)['model']
    mc = S.paired_vs(P, R, ('REAL-agree',))
    out[os.path.basename(f)] = dict(real_flips=1 - ra['agree'], real_tv=ra['tv'], agree_sd=ra['agree_sd'], mcnemar=mc, cfp_fgh=cp['flip_given_hobson'], cfp_flip=cp['flip'],
                                   n_real=ra.get('n'), cov=ra.get('coverage'))
    print(f"| {os.path.basename(f)} | {1 - ra['agree']:.2%} | {ra['tv']:.4f} | {ra['agree_sd']:.3f} | {mc['lost']}/{mc['gained']} (p {mc['p']:.3g}) | {cp['flip_given_hobson']:.3f} | {cp['flip']:.3f} |")
json.dump(out, open(os.path.expanduser('~/decider2/q3/curve_scores.json'), 'w'), indent=1)

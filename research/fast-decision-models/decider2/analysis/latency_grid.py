"""latency_grid.py: every measured A10G latency curve, keyed by configuration.

    from latency_grid import CURVES, curve_at
    CURVES['j15/b8'] -> {'pts': {T: ms}, 'src': ..., 'harness': ..., 'ref': 'j15/bf16', 'q': 1}

All values are medians in ms, measured with exclusive use of the GPU, at exact state lengths T (state tokens, Qwen
tokenisation), plus one real question unless 'q' says otherwise. Each curve names the hobson-bf16 curve measured in
the same harness ('ref'), so speedups are same-harness ratios. J15's grid is read from its raw file; the rest are
transcribed from the agents' DRAFT_REPORT.md tables (file noted in 'src').
"""
import json, math, os
from collections import defaultdict

R = os.path.expanduser('~/decider2')


def _j15():
    d = defaultdict(dict)
    for line in open(f'{R}/j15/res/res_bench.jsonl'):
        r = json.loads(line)
        d[(r['prec'], r['mode'])][r['T']] = r['wall']['median']
    return d


_J = _j15()
H = 'j15/bf16'  # the reference curve: 14 lengths, 16-9000 tokens, 125-token question

CURVES = {
    # J15 harness (H2 QRT runtime, rows = T + 125)
    'j15/bf16': dict(pts=_J[('bf16', 'full')], src='j15/res/res_bench.jsonl', label='hobson bf16'),
    'j15/b8': dict(pts=_J[('b8', 'full')], src='j15/res/res_bench.jsonl', label='W8A8-b8'),
    'j15/w4a4': dict(pts=_J[('w4a4', 'full')], src='j15/res/res_bench.jsonl', label='W4A4'),
    'j15/k48': dict(pts=_J[('k48', 'full')], src='j15/res/res_bench.jsonl', label='W4A4, 48 GEMMs W8A8'),
    'j15/k64': dict(pts=_J[('k64', 'full')], src='j15/res/res_bench.jsonl', label='W4A4, 64 GEMMs W8A8'),
    'j15/exit16': dict(pts=_J[('b8', 'pre:16')], src='j15/res/res_bench.jsonl', label='W8A8-b8, exit at layer 16',
                       note='every decision exits; the {16} cascade measured 0.688x b8 end to end on real requests'),
    'j15/exit8': dict(pts=_J[('b8', 'pre:8')], src='j15/res/res_bench.jsonl', label='W8A8-b8, exit at layer 8'),
    # J3 harness (dt_lat.py, 93-token question)
    'j3/hobson': dict(pts={64: 15.3, 128: 15.6, 256: 22.9, 400: 28.2, 1000: 57.1, 4000: 200.8}, src='j3/DRAFT_REPORT.md'),
    'j3/DT-A4': dict(pts={64: 16.4, 128: 16.4, 256: 17.3, 400: 20.2, 1000: 24.2, 4000: 52.5}, src='j3/DRAFT_REPORT.md', ref='j3/hobson'),
    'j3/DT-A8': dict(pts={64: 18.2, 128: 18.3, 256: 20.0, 400: 25.5, 1000: 33.0, 4000: 85.3}, src='j3/DRAFT_REPORT.md', ref='j3/hobson', label='depth-split DT-A8'),
    'j3/DT-A12': dict(pts={64: 20.0, 128: 20.2, 256: 22.7, 400: 30.9, 1000: 41.5, 4000: 117.9}, src='j3/DRAFT_REPORT.md', ref='j3/hobson'),
    'j3/DT-set12': dict(pts={64: 21.2, 128: 21.4, 256: 23.8, 400: 32.0, 1000: 42.7, 4000: 119.1}, src='j3/DRAFT_REPORT.md', ref='j3/hobson', label='depth-split DT-set12'),
    'j3/DT-A16': dict(pts={64: 21.9, 128: 22.1, 256: 25.3, 400: 36.2, 1000: 50.2, 4000: 150.6}, src='j3/DRAFT_REPORT.md', ref='j3/hobson'),
    'j3/hobson_q4': dict(pts={1000: 95.1, 4000: 245.5}, src='j3/DRAFT_REPORT.md', q=4),
    'j3/DT-A8_q4': dict(pts={1000: 60.4, 4000: 114.5}, src='j3/DRAFT_REPORT.md', ref='j3/hobson_q4', q=4),
    # J7 harness (lean2 + TTL); super-token rows = Qwen rows / REAL median compression, x axis in Qwen tokens
    'j7/hobson': dict(pts={64: 15.24, 128: 15.56, 256: 22.84, 400: 28.14, 1000: 57.06, 4000: 201.17}, src='j7/DRAFT_REPORT.md'),
    'j7/super16k': dict(pts={64: 9.33, 128: 14.70, 256: 15.51, 400: 22.33, 1000: 37.36, 4000: 108.46}, src='j7/DRAFT_REPORT.md', ref='j7/hobson', label='super-token vocabulary 16k'),
    'j7/super64k': dict(pts={64: 9.53, 128: 9.74, 256: 15.24, 400: 15.53, 1000: 27.82, 4000: 92.50}, src='j7/DRAFT_REPORT.md', ref='j7/hobson', label='super-token vocabulary 64k'),
    'j7/hobson_q4': dict(pts={64: 48.60, 128: 48.94, 256: 55.64, 400: 67.31, 1000: 93.25, 4000: 243.30}, src='j7/DRAFT_REPORT.md', q=4),
    'j7/super64k_q4': dict(pts={64: 44.80, 128: 45.22, 256: 45.55, 400: 51.63, 1000: 64.24, 4000: 126.37}, src='j7/DRAFT_REPORT.md', ref='j7/hobson_q4', q=4),
    # J9 harness (j9lat.py, 125-token question); c = compiled share of the state
    'j9/hobson': dict(pts={64: 15.2, 128: 15.5, 256: 22.8, 400: 28.1, 1000: 57.1, 4000: 201.4}, src='j9/DRAFT_REPORT.md'),
    'j9/c37': dict(pts={64: 15.0, 128: 16.3, 256: 16.4, 400: 23.7, 1000: 45.0, 4000: 140.0}, src='j9/DRAFT_REPORT.md', ref='j9/hobson', label='compiled documents, 37% of state'),
    'j9/c55': dict(pts={64: 9.7, 128: 15.8, 256: 17.3, 400: 24.2, 1000: 39.2, 4000: 102.8}, src='j9/DRAFT_REPORT.md', ref='j9/hobson', label='compiled documents, 55% of state'),
    'j9/hobson_q4': dict(pts={64: 48.5, 128: 48.9, 256: 55.5, 400: 67.2, 1000: 93.3, 4000: 243.5}, src='j9/DRAFT_REPORT.md', q=4),
    'j9/c55_q4': dict(pts={64: 48.3, 128: 49.1, 256: 50.9, 400: 57.8, 1000: 70.9, 4000: 144.5}, src='j9/DRAFT_REPORT.md', ref='j9/hobson_q4', q=4),
    # J6 harness: deployed question sets; ctx = best in-context layout
    'j6/q4_ctx': dict(pts={64: 101.0, 128: 101.4, 256: 106.4, 400: 113.4, 1000: 146.3, 4000: 294.7}, src='j6/DRAFT_REPORT.md', q=4),
    'j6/q4_late': dict(pts={64: 18.8, 128: 23.0, 256: 32.6, 400: 40.3, 1000: 64.8, 4000: 211.1}, src='j6/DRAFT_REPORT.md', ref='j6/q4_ctx', q=4, label='questions in weights (late), 4 q'),
    'j6/q15_ctx': dict(pts={64: 97.6, 128: 98.0, 256: 102.6, 400: 107.6, 1000: 132.8, 4000: 292.3}, src='j6/DRAFT_REPORT.md', q=15),
    'j6/q15_late': dict(pts={64: 19.8, 128: 24.0, 256: 33.5, 400: 41.2, 1000: 65.7, 4000: 212.1}, src='j6/DRAFT_REPORT.md', ref='j6/q15_ctx', q=15, label='questions in weights (late), 15 q'),
    # J1 harness (encrt.py vs lean2/TTL with the same tile autotuning; 92-token question)
    'j1/hobson': dict(pts={64: 13.2, 128: 14.7, 256: 20.8, 400: 28.1, 1000: 56.7, 2000: 105.2, 4000: 199.2}, src='j1/DRAFT_REPORT.md'),
    'j1/encoder': dict(pts={64: 15.5, 128: 19.0, 256: 27.8, 400: 36.7, 1000: 85.8, 2000: 159.8, 4000: 330.2}, src='j1/DRAFT_REPORT.md', ref='j1/hobson', label='T5Gemma 2B encoder (e1b)'),
    # J2 harness (j2lat.py, 100 question tokens)
    'j2/hobson': dict(pts={64: 15.6, 128: 15.9, 256: 25.4, 400: 32.9, 1000: 57.1, 4000: 205.2}, src='j2/DRAFT_REPORT.md'),
    'j2/bi_all': dict(pts={64: 17.1, 128: 17.9, 256: 28.0, 400: 36.4, 1000: 64.8, 4000: 243.6}, src='j2/DRAFT_REPORT.md', ref='j2/hobson', label='bidirectional GDN'),
    'j2/bi_early': dict(pts={64: 16.4, 128: 17.0, 256: 26.9, 400: 34.9, 1000: 61.8, 4000: 232.7}, src='j2/DRAFT_REPORT.md', ref='j2/hobson'),
    # J10 harness (rt_j10.py, 85-token question)
    'j10/hobson': dict(pts={64: 15.5, 128: 15.9, 256: 25.3, 400: 32.8, 1000: 57.1, 4000: 201.6}, src='j10/DRAFT_REPORT.md'),
    'j10/ternary_a8': dict(pts={64: 10.6, 128: 12.0, 256: 18.1, 400: 21.1, 1000: 48.3, 4000: 206.1}, src='j10/DRAFT_REPORT.md', ref='j10/hobson', label='ternary BitNet 2B, W1.58A8'),
    # J12 harness (lean2, 93-token question)
    'j12/hobson': dict(pts={64: 15.51, 128: 15.88, 256: 25.40, 400: 32.92, 1000: 57.47, 4000: 202.4}, src='j12/DRAFT_REPORT.md'),
    'j12/xr': dict(pts={64: 16.08, 128: 16.45, 256: 25.99, 400: 33.54, 1000: 58.33, 4000: 204.8}, src='j12/DRAFT_REPORT.md', ref='j12/hobson', label='exact-relation module'),
    # J13 harness (lean2, 125-token question)
    'j13/hobson': dict(pts={256: 25.56, 1000: 57.02, 4000: 204.16}, src='j13/DRAFT_REPORT.md'),
    'j13/cut13': dict(pts={256: 20.46, 1000: 40.66, 4000: 135.67}, src='j13/DRAFT_REPORT.md', ref='j13/hobson', label='thin from layer 13 (r64)'),
    # J11 harness: 38M / 114M from-scratch models (not 2B: a different size class)
    'j11/S_dec': dict(pts={64: 2.91, 256: 3.59, 1000: 6.94, 4000: 23.9, 8000: 50.4}, src='j11/DRAFT_REPORT.md'),
    'j11/S_slot': dict(pts={64: 1.47, 256: 1.82, 1000: 3.17, 4000: 11.4, 8000: 25.4}, src='j11/DRAFT_REPORT.md', ref='j11/S_dec', label='slot model, 38M from scratch'),
    'j11/M_dec': dict(pts={64: 4.64, 256: 5.81, 1000: 13.3, 4000: 48.9, 8000: 103}, src='j11/DRAFT_REPORT.md'),
    'j11/M_slot': dict(pts={64: 2.22, 256: 2.77, 1000: 5.53, 4000: 20.4, 8000: 46.4}, src='j11/DRAFT_REPORT.md', ref='j11/M_dec', label='slot model, 114M from scratch'),
    # F7 controls (smaller or shallower: controls, not answers)
    'f7/hob12': dict(pts={256: 8.14, 1000: 26.59, 4000: 101.04}, src='F7_REPORT.md', ref=H, label="hobson's first 12 layers (control)"),
    'f7/q08': dict(pts={256: 9.16, 1000: 24.13, 4000: 95.85}, src='F7_REPORT.md', ref=H, label='Qwen3.5-0.8B decider (control)'),
    # J8: other silicon, the exact hobson function
    'j8/inf2': dict(pts={64: 30.9, 256: 43.1, 1000: 152.0, 4000: 537.5}, src='j8/DRAFT_REPORT.md', ref=H, label='Inferentia2, one NeuronCore'),
    'j8/spr16': dict(pts={64: 95.7, 256: 214.5, 1000: 577, 4000: 2199}, src='j8/DRAFT_REPORT.md', ref=H, label='Sapphire Rapids, 16 cores'),
    # J14 harness: W8A8-b8, T = 32-1000 (J14-oa: 14-17 live rows per question; fails accuracy)
    'j14/b8_q15': dict(pts={32: 68.98, 64: 71.12, 128: 72.63, 256: 78.39, 400: 84.04, 1000: 107.65}, src='j14/DRAFT_REPORT.md', q=15),
    'j14/F_q15': dict(pts={32: 58.42, 64: 59.08, 128: 60.99, 256: 66.89, 400: 73.30, 1000: 95.50}, src='j14/DRAFT_REPORT.md', ref='j14/b8_q15', q=15, label='compiled question headers only (F), 15 q'),
    'j14/oa_q15': dict(pts={32: 20.27, 64: 22.02, 128: 24.75, 256: 28.39, 400: 32.66, 1000: 53.70}, src='j14/DRAFT_REPORT.md', ref='j14/b8_q15', q=15, label='compiled question rows (oa), 15 q'),
    # H6: W4A4 with 64 GEMMs at W8A8 and bf16 question rows (passes fidelity alone; H6 measurement)
    'h6/k64qb': dict(pts={1000: 40.4, 4000: 128.9}, src='docs/FAST_DECISION_MODEL.md section 5', ref=H, label='W4A4 k64 + bf16 question rows'),
    # J5 harness: short-M kernels, T = 32-400, 108-token question
    'j5/bf16s': dict(pts={32: 10.7, 64: 12.1, 128: 14.7, 256: 21.2, 400: 27.7}, src='j5/DRAFT_REPORT.md', label='bf16, short-M GEMMs'),
    'j5/w4a16': dict(pts={32: 11.6, 64: 14.1, 128: 17.6, 256: 25.3, 400: 33.1}, src='j5/DRAFT_REPORT.md', ref='j5/bf16s', label='W4A16 weight-only'),
    'j5/b8': dict(pts={32: 7.38, 64: 8.01, 128: 9.34, 256: 13.3, 400: 17.0, 1000: 36.11}, src='j5/DRAFT_REPORT.md', ref='j5/bf16s', label='W8A8-b8, J5 kernels'),
    'j14/ob_q15': dict(pts={32: 41.24, 64: 43.06, 128: 44.84, 256: 50.08, 400: 53.95, 1000: 77.54}, src='j14/DRAFT_REPORT.md', ref='j14/b8_q15', q=15, label='compiled question rows (ob), 15 q'),
    # Q2 (QRT2C runtime, W8A8-b8 baseline in the same harness; 125-token question)
    'q2/b8': dict(pts={64: 8.57, 256: 14.39, 1000: 36.36, 4000: 142.55}, src='q2/DRAFT_REPORT.md'),
    'q2/k64rr': dict(pts={64: 8.50, 256: 13.42, 1000: 31.89, 4000: 119.81}, src='q2/DRAFT_REPORT.md', ref=H, label='k64rr: 64 GEMMs W8A8, 32 row-role'),
    'q2/rowrole': dict(pts={64: 8.83, 256: 11.61, 1000: 26.34, 4000: 88.75}, src='q2/DRAFT_REPORT.md', ref=H, label='row role: state W4A4, question W8A8'),
    # M1 (A10G fused, bf16) and M2 (A10G fused, bf16, compiled share 0) and N1 (Inferentia2, one NeuronCore)
    'm1/allattn': dict(pts={64: 15.08, 256: 25.26, 1000: 56.54, 4000: 210.99}, src='m1/DRAFT_REPORT.md', ref=H, label='all-attention hobson'),
    'm2/N_k12_c0': dict(pts={64: 18.0, 256: 22.1, 1000: 49.4, 4000: 156.3}, src='m2/NOTES.md (lat_grid_qf.json, fused question pass)', ref=H, label='segment-isolated, state depth 12, nothing compiled'),
    'm2/N_k8_c0': dict(pts={64: 16.4, 256: 19.7, 1000: 41.6, 4000: 128.2}, src='m2/NOTES.md (lat_grid_qf.json, fused question pass)', ref=H, label='segment-isolated, state depth 8, nothing compiled'),
    'm2/C_k12_c0': dict(pts={64: 17.3, 256: 21.1, 1000: 46.3, 4000: 148.4}, src='m2/NOTES.md (lat_grid_C_qf.json)', ref=H, label='constants isolated, state depth 12, nothing compiled'),
    'm2/N_k12_c55': dict(pts={1000: 34.8, 4000: 89.9}, src='m2/NOTES.md (lat_grid_qf.json)', ref=H, label='segment-isolated, state depth 12, 55% compiled'),
    'n1/inf2': dict(pts={64: 32.6, 256: 42.0, 1000: 132.1, 4000: 653.6}, src='n1/DRAFT_REPORT.md', ref=H, label='Inferentia2, pipelined GDN kernel'),
}
QREF = {'q2': 125, 'm1': 93, 'm2': 129, 'n1': 100, 'j15': 125, 'j3': 93, 'j7': 106, 'j9': 125, 'j6': 0, 'j1': 92, 'j2': 100, 'j10': 85, 'j12': 93, 'j13': 125,
        'j11': 100, 'f7': 125, 'j8': 100, 'j14': 0, 'h6': 125, 'j5': 108}
BASELINES = {k for k, c in CURVES.items() if k == H or 'ref' not in c and (k.endswith(('/hobson', '_q4', '_ctx', '_dec')) or c.get('q', 1) > 1)}
for k, c in CURVES.items():
    c.setdefault('ref', None if k in BASELINES else H)
    c.setdefault('q', 1)
    c.setdefault('label', k)
    c.setdefault('qref', QREF[k.split('/')[0]])
    c['pts'] = {int(t): float(v) for t, v in sorted(c['pts'].items())}


def curve_at(key, T):
    """log-log interpolation; linear-in-log extrapolation from the two end points (flagged by the caller)"""
    pts = sorted(CURVES[key]['pts'].items())
    T = max(T, 1)
    if T <= pts[0][0]:
        return pts[0][1]  # the floor: shorter states cost the same (question rows dominate)
    for (t0, v0), (t1, v1) in zip(pts, pts[1:]):
        if T <= t1:
            f = (math.log(T) - math.log(t0)) / (math.log(t1) - math.log(t0))
            return math.exp(math.log(v0) + f * (math.log(v1) - math.log(v0)))
    (t0, v0), (t1, v1) = pts[-2], pts[-1]
    return math.exp(math.log(v1) + (math.log(T) - math.log(t1)) * (math.log(v1) - math.log(v0)) / (math.log(t1) - math.log(t0)))


def speedup(key, T):
    ref = CURVES[key]['ref']
    return curve_at(ref, T) / curve_at(key, T) if ref else None


if __name__ == '__main__':
    for k, c in CURVES.items():
        s = speedup(k, 1000)
        print(f"{k:16} q{c['q']:<2} ref={str(c['ref']):14} @1000 {curve_at(k, 1000):7.1f} ms" + (f"  {s:.2f}x" if s else ''))

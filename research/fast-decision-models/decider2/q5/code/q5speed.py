"""ARITHMETIC (not a measurement): GEMM time of a structure config relative to W8A8-b8 at a given state length and question rows, from MAC
counts and nominal tensor-core rates on Ampere (per MAC, int8 dense = 1): bf16 2, int8 2:4 0.5, int4 dense 0.5, int4 2:4 0.25.
Q4 measured the real kernels (q4/NOTES.md): e.g. int4 2:4 state rows L12-22 = 0.828x of b8 at 1000+125 rows; nominal below says ~0.70x.
python3 q5speed.py"""
ATT = (3, 7, 11, 15, 19, 23)
B8BF = {(23, 'Wd'), (0, 'Wo'), (7, 'Wo'), (11, 'Wo'), (10, 'Wo'), (23, 'Wo'), (12, 'Wo'), (9, 'Wo')}


def macs(i, k):
    return {'Win': (5120 if i in ATT else 8224) * 2048, 'Wo': 2048 * 2048, 'Wgu': 12288 * 2048, 'Wd': 2048 * 6144}[k]


RATE = {'bf16': 2.0, 'int8': 1.0, 'int8_24': 0.5, 'int4': 0.5, 'int4_24': 0.25, 'skip': 0.0}


def gemm_time(state_rows, q_rows, sparse=None, removed=None, skip23=False, table0=False):
    """sparse: {(i,k): (prec_state, prec_q)}; removed: {i: fraction of MLP neurons removed}; returns time units (MACs x rate) summed over rows."""
    t = 0.0
    for i in range(24):
        for k in ('Win', 'Wo', 'Wgu', 'Wd'):
            m = macs(i, k)
            if removed and i in removed and k in ('Wgu', 'Wd'): m *= (1 - removed[i])
            base = 'bf16' if (i, k) in B8BF else 'int8'
            ps, pq = (sparse or {}).get((i, k), (base, base))
            if skip23 and i == 23:
                ms = m * (1024 / 5120) if k == 'Win' else 0.0       # state rows of layer 23: K/V only
            else: ms = m
            if table0 and i == 0 and k == 'Win': ms = 0.0; mq = 0.0
            else: mq = m
            t += state_rows * ms * RATE[ps] + q_rows * mq * RATE[pq]
    return t


if __name__ == '__main__':
    S, Q = 1000, 125
    ref = gemm_time(S, Q)
    L1222 = [(i, k) for i in range(12, 23) for k in ('Win', 'Wo', 'Wgu', 'Wd') if (i, k) not in B8BF]
    L1223 = [(i, k) for i in range(12, 24) for k in ('Win', 'Wo', 'Wgu', 'Wd') if (i, k) not in B8BF]
    cases = {
        'b8': {},
        'int8 2:4 state L12-22': dict(sparse={key: ('int8_24', 'int8') for key in L1222}),
        'int4 2:4 state L12-22': dict(sparse={key: ('int4_24', 'int8') for key in L1222}),
        'int4 2:4 state L12-23 + B13': dict(sparse={key: ('int4_24', 'int8') for key in L1223}, skip23=True, table0=True),
        'int8 2:4 all rows L16-23 + state L12-15': dict(sparse={**{key: ('int8_24', 'int8') for key in L1222 if key[0] < 16}, **{key: ('int8_24', 'int8_24') for key in L1223 if key[0] >= 16}}),
        'B12 50% neurons L12-23 (b8)': dict(removed={i: 0.5 for i in range(12, 24)}),
        'B12 25% neurons L12-23 (b8)': dict(removed={i: 0.25 for i in range(12, 24)}),
    }
    for nm, kw in cases.items():
        print(f'{nm:45s} {gemm_time(S, Q, **kw) / ref:.3f}x of b8 GEMM time (nominal rates, {S}+{Q} rows)')

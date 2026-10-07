def orders_for(kind, K):
    """up to 3 non-identity option orders: cyclic shifts by 1,2,3 (mod K, identity and duplicates skipped); score questions: reversal only"""
    if kind == 'score':
        return [list(reversed(range(K)))]
    outs = []
    for sh in (1, 2, 3):
        o = [(i + sh) % K for i in range(K)]
        if o != list(range(K)) and o not in outs:
            outs.append(o)
    return outs

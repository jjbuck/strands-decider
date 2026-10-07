"""J2 checks (box only):
 1. gates 0 -> every mode equals hobson (H3 plain causal forward) and the evalkit hobson references, single-row and packed;
 2. nonzero gates: packed == single (no cross-row leakage); 'qag' state rows independent of the question (masked state cache); 'qa' not;
 3. train-step throughput (LoRA r16 + gates, checkpointing) for causal / qag."""
import os, sys, json, time, random
sys.path[:0] = [os.path.expanduser('~/work/j2'), os.path.expanduser('~/work/evalkit')]
import torch
import evalkit as EK
from j2lib import J2, Pack, GDN_LAYERS
from j2data import prep_item

m = J2(); m.free_hf(); m.head = m.head0; eng = m.p.eng
items = []
refs = {}
for s in ('REAL-agree', 'CF', 'JB-all'):
    for l in open(os.path.expanduser(f'~/work/evalkit/refs/{s}.hobson.jsonl')):
        r = json.loads(l); refs[r['id']] = r['hobson']
for s in ('REAL-agree', 'CF', 'JB-all'):
    k = 0
    for iid, qn, st, spec in EK.iter_questions(s):
        if iid not in refs or qn not in refs[iid]: continue
        it = prep_item(eng, st, spec); it['id'] = iid; it['qn'] = qn
        if len(it['s']) + len(it['q']) > 3000: continue
        items.append(it); k += 1
        if k >= 4: break
print('items', len(items), [len(i['s']) + len(i['q']) for i in items], flush=True)

with torch.inference_mode():
    # reference: H3 plain forward
    ref = []
    for it in items:
        ids = list(it['s']) + list(it['q'])
        h = m.forward(ids)
        pr = dict(q0=len(it['s']), opt=it['opt'], rq=it['rq'])
        ref.append(m.logits(h, pr).float())
    for it, lg in zip(items, ref):
        p = torch.softmax(lg, -1).tolist(); r = refs[it['id']][it['qn']]
        dp = max(abs(p[j] - r[lab]) for j, lab in enumerate(it['rq'].slot_labels))
        it['dp_ref'] = dp
    print('H3 vs evalkit hobson refs: max |dp| per item', [round(it['dp_ref'], 4) for it in items], flush=True)
    for mode in ('causal', 'qag', 'qa'):
        m.setup_bidir(mode)
        single = [m.decide([it])[0].float() for it in items]
        packed = m.decide(items)
        d1 = max(float((a - b).abs().max()) for a, b in zip(single, ref))
        d2 = max(float((a.float() - b).abs().max()) for a, b in zip(packed, ref))
        print(f'{mode} gates0: single vs H3 max|dlogit| {d1:.4f}  packed vs H3 {d2:.4f}', flush=True)
    # nonzero gates
    for mode in ('qag', 'qa'):
        m.setup_bidir(mode)
        with torch.no_grad():
            for p_ in m.gam.values(): p_.fill_(0.7)
            for p_ in m.lam.values(): p_.fill_(0.6)
        single = [m.decide([it])[0].float() for it in items]
        packed = m.decide(items)
        d2 = max(float((a - b.float()).abs().max()) for a, b in zip(single, packed))
        dref = max(float((a - b).abs().max()) for a, b in zip(single, ref))
        # masked cache: same state, two different questions -> state rows identical?
        it0 = items[0]; it1 = items[1] if items[1]['s'] != it0['s'] else items[2]
        rows = [(list(it0['s']) + list(it0['q']), len(it0['s'])), (list(it0['s']) + list(it1['q']), len(it0['s']))]
        pk = Pack(rows, m.dev, mode)
        hh = m.fwd_pk(pk)
        q0 = len(it0['s']); a = hh[:q0].float(); b = hh[pk.cu_l[1]:pk.cu_l[1] + q0].float()
        print(f'{mode} gates on: packed vs single max|dlogit| {d2:.4f}; vs hobson {dref:.3f}; state rows under 2 questions: max|dh| {float((a - b).abs().max()):.4f} '
              f'(rel {float((a - b).norm() / a.norm()):.5f})', flush=True)

# throughput
m.add_lora(r=16, alpha=32, seed=0)
from h3lib import StdHead
m.head = StdHead(m.head0).to(m.dev)
for mode in ('causal', 'qag'):
    gp = m.setup_bidir(mode)
    params = list(m.lora.parameters()) + list(m.head.parameters()) + gp
    for T_pack, L in ((8192, 160), (8192, 2048), (6144, 6144)):
        n = T_pack // L
        rows = []
        for r in range(n):
            ids = [random.randrange(1000, 50000) for _ in range(L)]
            rows.append(dict(s=ids[:L - 40], q=ids[L - 40:], opt=[5, 10], kind='noul', n_slots=2))
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        for rep in range(3):
            t0 = time.time()
            out = m.decide(rows, ckpt=True)
            loss = sum(o.float().logsumexp(-1) for o in out)
            loss.backward()
            torch.cuda.synchronize()
            dt = time.time() - t0
        for p_ in params: p_.grad = None
        print(f'train {mode} rows {n}x{L}: {dt:.2f}s/pack {n * L / dt:.0f} tok/s peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB', flush=True)

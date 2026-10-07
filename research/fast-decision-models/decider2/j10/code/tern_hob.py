"""J10 BitDistill-lite: distil hobson-v19 into a TERNARY-weight version of itself (W1.58A8, BitNet's format) on the decision task.
Every nn.Linear in hobson's 24 layers (LoRA merged) becomes ternary(W0 + s B A) per tensor (absmean) with per-token int8 activations; LoRA-in-latent
QAT exactly as bitnet_j10 (low-rank STE backward); norms / conv / embeddings stay full precision. Loss = F7's: CE(gold) + KL(hobson) + KL on real rows,
on F7's own Qwen-tokenised rows (f7/rows_*_q.pt, hobson teacher logits). Head initialised from hobson's.
  python tern_hob.py train --out ck_th --max_steps N       python tern_hob.py eval CK TAG SUITE...      python tern_hob.py eval0 TAG SUITE... (no training)"""
import os, sys, json, math, time, random, argparse, glob
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.expanduser('~/work/sd/src')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import torch, torch.nn as nn, torch.nn.functional as F
import bitnet_j10 as BJ
from strands_decider.modeling import StrandsDeciderModel, StrandsDeciderConfig, build_head, masked_log_softmax, gather_options, pool_last_token
HOB = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*'))[0]


class TernLinear(nn.Module):
    def __init__(self, lin, r=16, alpha=32, seed=0):
        super().__init__()
        self.W0 = lin.weight.detach().to(torch.bfloat16)
        dout, din = self.W0.shape
        g = torch.Generator().manual_seed(seed)
        A = torch.empty(r, din); nn.init.kaiming_uniform_(A, a=math.sqrt(5), generator=g)
        self.A = nn.Parameter(A.to(self.W0.device)); self.B = nn.Parameter(torch.zeros(dout, r, device=self.W0.device)); self.s = alpha / r
        self.refresh()

    @torch.no_grad()
    def refresh(self):
        self.Wq, _, _ = BJ.tern_lat(self.W0, self.B, self.A, self.s)

    def forward(self, x):
        xq = BJ.AQ(x, 8, 0)
        if self.training and torch.is_grad_enabled():
            return BJ.TL.apply(xq, self.Wq, self.A, self.B, self.s)
        return xq @ self.Wq.t()


def build(seed=0):
    sys.path.insert(0, os.path.expanduser('~/work/j10/f7code'))
    try:
        import fast; fast.apply(bf16_lora=False)
    except Exception as e:
        print('fast patches unavailable', e, flush=True)
    hob = StrandsDeciderModel.load(HOB)
    torso = hob.torso.merge_and_unload().cuda().to(torch.bfloat16)
    for p in torso.parameters(): p.requires_grad_(False)
    tls = []
    for name, mod in list(torso.layers.named_modules()):
        for cn, ch in list(mod.named_children()):
            if isinstance(ch, nn.Linear):
                t = TernLinear(ch, seed=seed + len(tls)); setattr(mod, cn, t); tls.append(t)
    head = hob.head.cuda().float()
    print('ternary linears', len(tls), 'latent params', sum(t.W0.numel() for t in tls), flush=True)
    return torso, head, tls, hob.tokenizer


def train(a):
    torso, head, tls, tok = build(a.seed)
    lora = [p for t in tls for p in (t.A, t.B)]
    torso.train(); head.train()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = []; TT = {}
    for p in ('f7/rows_corpus_q.pt', 'f7/rows_real_q.pt'):
        D = torch.load(p, weights_only=False); rr = D['rows']; TT = dict(D['teacher_temps']['by_kind'])
        if rr and rr[0].get('src') == 'real': random.Random(4).shuffle(rr); rr = rr[:int(len(rr) * a.frac_real)]
        rows += rr
    rng = random.Random(a.seed); idx = list(range(len(rows))); rng.shuffle(idx)
    MB = 50 * a.bs; steps = []
    for s in range(0, len(idx), MB):
        mb = sorted(idx[s:s + MB], key=lambda i: len(rows[i]['ids']))
        steps += [mb[j:j + a.bs] for j in range(0, len(mb) - a.bs + 1, a.bs)]
    rng.shuffle(steps); TOT = dict(n=a.max_steps); steps = steps[:a.max_steps]; warm = max(1, int(0.03 * a.max_steps))
    opt = torch.optim.AdamW([dict(params=list(head.parameters()), lr=a.head_lr), dict(params=lora, lr=a.lr)], betas=(0.9, 0.95), weight_decay=0.0)
    lam = lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, TOT['n'] - warm))))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    os.makedirs(a.out, exist_ok=True); logf = open(os.path.join(a.out, 'train_log.jsonl'), 'a')
    import train_j10_lib as TLB
    t0 = time.time(); ntok = 0; acc = dict(loss=0., n=0, agree=0, na=0)
    def save_ck(tag):
        ck = os.path.join(a.out, tag); os.makedirs(ck, exist_ok=True)
        torch.save({f'{i}.A': t.A.detach().cpu() for i, t in enumerate(tls)} | {f'{i}.B': t.B.detach().cpu() for i, t in enumerate(tls)}, os.path.join(ck, 'lora.pt'))
        torch.save(head.state_dict(), os.path.join(ck, 'slot_head.pt'))
        json.dump(dict(step=step, total=TOT['n'], tokens=ntok), open(os.path.join(ck, 'meta.json'), 'w'))
    t30 = None; step = 0
    while step < TOT['n']:
        step += 1
        if step == 6: t30 = time.time()
        if step == 36 and a.max_minutes:   # fit the schedule to the time budget, measured on steps 6-35
            per = (time.time() - t30) / 30; left = a.max_minutes * 60 - (time.time() - t0)
            TOT['n'] = min(TOT['n'], step + max(0, int(left / per)))
            print(f'step time {per:.2f}s -> total steps {TOT["n"]}', flush=True)
        total = TOT['n']
        srows = [rows[i] for i in steps[step - 1]]
        micro, cur = [], []
        for r in srows:
            if cur and (len(cur) + 1) * max(len(x['ids']) for x in cur + [r]) > a.tokb: micro.append(cur); cur = []
            cur.append(r)
        micro.append(cur)
        for mb in micro:
            ids, am, optx, ns, lab, tl, has_t, dist, has_d, w = [t.cuda() for t in TLB.batch_tensors(mb, pad, TT)]
            torso.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False}) if ids.numel() > a.ckpt_above else None
            hid = torso(input_ids=ids, attention_mask=am).last_hidden_state
            lp = masked_log_softmax(head(pool_last_token(hid, am).float(), gather_options(hid, optx).float()), ns)
            per = TLB.loss_rows(lp, lab, dist, has_d, tl, has_t, ns, w)
            (per.sum() / len(srows)).backward()
            with torch.no_grad():
                acc['loss'] += float(per.sum()); acc['n'] += len(mb)
                tam = tl.masked_fill(torch.arange(tl.shape[-1], device=tl.device)[None, :] >= ns[:, None], -1e4).argmax(-1)
                acc['agree'] += int(((lp.argmax(-1) == tam) & has_t).sum()); acc['na'] += int(has_t.sum())
            ntok += int(am.sum())
            if ids.numel() > a.ckpt_above: torso.gradient_checkpointing_disable()
        torch.nn.utils.clip_grad_norm_(list(head.parameters()) + lora, 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        for t in tls: t.refresh()
        if step % 20 == 0 or step in (1, 5, total):
            el = time.time() - t0
            e = dict(step=step, loss=acc['loss'] / max(1, acc['n']), agree=acc['agree'] / max(1, acc['na']), el=round(el), tok_s=round(ntok / el), eta_min=round(el / step * (total - step) / 60, 1),
                     mem=round(torch.cuda.max_memory_allocated() / 2**30, 1))
            print(json.dumps(e), flush=True); logf.write(json.dumps(e) + '\n'); logf.flush(); acc = dict(loss=0., n=0, agree=0, na=0)
        if step % 100 == 0: save_ck('last')
    save_ck('final'); print('saved final at step', step, flush=True)


def evaluate(ck, tag, suites):
    import ev_j10 as EV
    from transformers import AutoTokenizer
    torso, head, tls, tok = build()
    if ck:
        sd = torch.load(os.path.join(ck, 'lora.pt'))
        with torch.no_grad():
            for i, t in enumerate(tls): t.A.copy_(sd[f'{i}.A']); t.B.copy_(sd[f'{i}.B']); t.refresh()
        head.load_state_dict(torch.load(os.path.join(ck, 'slot_head.pt')))
    torso.eval(); head.eval()
    T = lambda ids, am: torso(input_ids=ids, attention_mask=am).last_hidden_state
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    os.makedirs('results', exist_ok=True)
    for su in suites:
        t0 = time.time(); rows = EV.items_rows(tok, su, 16384)
        EV.run(T, head, pad, rows, tokb=12288)
        out = {}
        for r in rows: out.setdefault(r['id'], {})[r['qn']] = r['p']
        with open(f'results/{su}.{tag}.jsonl', 'w') as f:
            for i, q in out.items(): f.write(json.dumps(dict(id=i, q=q)) + '\n')
        print(su, tag, len(rows), f'{time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'train':
        ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('--out', required=True); ap.add_argument('--max_steps', type=int, default=600)
        ap.add_argument('--lr', type=float, default=2e-4); ap.add_argument('--head_lr', type=float, default=2e-4); ap.add_argument('--bs', type=int, default=32)
        ap.add_argument('--tokb', type=int, default=4096); ap.add_argument('--ckpt_above', type=int, default=4096); ap.add_argument('--frac_real', type=float, default=0.6)
        ap.add_argument('--seed', type=int, default=0); ap.add_argument('--max_minutes', type=float, default=0)
        train(ap.parse_args())
    elif sys.argv[1] == 'eval':
        evaluate(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif sys.argv[1] == 'eval0':
        evaluate('', sys.argv[2], sys.argv[3:])

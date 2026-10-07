"""J10 student eval on evalkit suites (F7/G3 ev protocol): canonical render, serving-style fit at max_length 16384, one question per sequence,
raw T=1 probabilities in canonical label space.
usage: python ev_j10.py CKPT TAG SUITE [SUITE...]   env AQ="qkv:4:0,gu:4:16" (activation-quant override, emulated), EVMAX
-> results/SUITE.TAG.jsonl"""
import sys, os, json, time, torch
sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.expanduser('~/work/sd/src'), os.path.expanduser('~/work/j10')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.modeling import StrandsDeciderConfig, build_head, masked_log_softmax, gather_options, pool_last_token
from strands_decider.prompting import render_question, render_state
import bitnet_j10 as BJ
from prep_g3 import fit
ta = TypeAdapter(SC.Question)
KIT = os.path.expanduser('~/work/evalkit')


def load_student(ck):
    cfg, W = BJ.load()
    t = BJ.BitNetTorso(cfg, W, r=16, alpha=32, lora=True).cuda()
    del W
    t.load_trainable(torch.load(os.path.join(ck, 'torso_trainable.pt')))
    meta = json.load(open(os.path.join(ck, 'j10_meta.json')))
    t.aq = {k: tuple(v) for k, v in meta.get('aq', t.aq).items()}
    t.eval()
    c = StrandsDeciderConfig.from_json(os.path.join(ck, 'strands_decider_config.json'))
    h = build_head(c, t.d); h.load_state_dict(torch.load(os.path.join(ck, 'slot_head.pt'))); h = h.cuda().eval()
    tok = AutoTokenizer.from_pretrained(BJ.snap())
    return t, h, tok


def items_rows(tok, suite, maxlen):
    its = [json.loads(l) for l in open(f'{KIT}/suites/{suite}.jsonl')]
    rows = []
    for it in its:
        for qn, qd in it['questions'].items():
            rq = render_question(ta.validate_python(qd))
            s, q, opt = fit(tok, render_state(it['state']), rq.text, rq, max_len=maxlen)
            rows.append(dict(id=it['id'], qn=qn, rq=rq, ids=torch.tensor(s + q), opt=opt, n=rq.n_slots))
    return rows


@torch.inference_mode()
def run(t, h, pad, rows, tokb=16384):
    order = sorted(range(len(rows)), key=lambda i: -len(rows[i]['ids']))
    k = 0
    while k < len(order):
        L = len(rows[order[k]]['ids']); nb = max(1, min(32, tokb // L))
        ch = [rows[i] for i in order[k:k + nb]]; k += len(ch)
        ids = pad_sequence([r['ids'] for r in ch], batch_first=True, padding_value=pad).cuda()
        am = pad_sequence([torch.ones(len(r['ids']), dtype=torch.long) for r in ch], batch_first=True).cuda()
        Wd = max(r['n'] for r in ch)
        opt = torch.tensor([r['opt'] + [-1] * (Wd - r['n']) for r in ch]).cuda()
        ns = torch.tensor([r['n'] for r in ch]).cuda()
        hid = t(ids, am)
        lp = masked_log_softmax(h(pool_last_token(hid, am).float(), gather_options(hid, opt).float()), ns).float().cpu()
        for j, r in enumerate(ch):
            r['p'] = {lab: float(lp[j, i].exp()) for i, lab in enumerate(r['rq'].slot_labels)}


if __name__ == '__main__':
    ck, tag = sys.argv[1:3]; suites = sys.argv[3:]
    t, h, tok = load_student(ck)
    if os.environ.get('AQ'):
        for spec in os.environ['AQ'].split(','):
            c, b, blk = spec.split(':'); t.aq[c] = (int(b), int(blk))
    print('aq', t.aq, flush=True)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    os.makedirs('results', exist_ok=True)
    for su in suites:
        t0 = time.time()
        rows = items_rows(tok, su, int(os.environ.get('EVMAX', 16384)))
        run(t, h, pad, rows)
        out = {}
        for r in rows: out.setdefault(r['id'], {})[r['qn']] = r['p']
        with open(f'results/{su}.{tag}.jsonl', 'w') as f:
            for i, q in out.items(): f.write(json.dumps(dict(id=i, q=q)) + '\n')
        print(su, tag, len(rows), 'questions', f'{time.time()-t0:.0f}s', flush=True)

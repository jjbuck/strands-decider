"""full Qwen3.5-id token counts over the fine-tune / eval corpora (states, generated probes, SQuAD), so the compact vocab keeps every
token those tasks use (JSON keys, tool names, ...).  writes ~/work/g4/data/ft_counts.npy"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prep_ft as PF
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

NV = PF.TOK.get_vocab_size()
cnt = np.zeros(NV, np.int64)
full = lambda s: np.array(PF.TOK.encode(s, add_special_tokens=False).ids, dtype=np.int64)
PF.enc = lambda s: full(s).astype(np.int64)          # generator in full-id mode (lengths ~ same)
PF.MAXT = 10 ** 9
D = PF.D
tr = [json.loads(l) for l in open(f'{D}/train_states.jsonl')]; ev = [json.loads(l) for l in open(f'{D}/eval_states.jsonl')]
for st in tr[:3000] + ev[:1500]:
    cnt += np.bincount(full(st['state']), minlength=NV)
for e in PF.gen_probes(tr, 1500, 11, 220, 560, balanced=False) + PF.gen_probes(ev, 400, 12, 540, 560, balanced=True):
    cnt += np.bincount(e['q'], minlength=NV) + np.bincount(e['s'], minlength=NV)
for split in ('train', 'validation'):
    rows = pq.read_table(hf_hub_download('rajpurkar/squad', f'plain_text/{split}-00000-of-00001.parquet', repo_type='dataset')).to_pylist()
    seen = set()
    for r in rows[:30000] if split == 'train' else rows:
        if r['context'] not in seen: seen.add(r['context']); cnt += np.bincount(full(r['context']), minlength=NV)
        cnt += np.bincount(full('Question: ' + r['question'] + '\nOptions: A) ' + r['answers']['text'][0] + '\nB) x\nAnswer:'), minlength=NV)
np.save(f'{D}/ft_counts.npy', cnt)
print('distinct FT tokens', int((cnt > 0).sum()), 'with count>=2', int((cnt >= 2).sum()), 'total', int(cnt.sum()))

"""Download one FineWeb-edu (sample-10BT) parquet shard, tokenize with the Qwen3.5 tokenizer, build the compact vocab
(top 32767 ids by count + OOV), write ~/work/g4/data/{train,val}.bin (uint16 compact ids) + vmap.npy.
usage: python prep_lm.py NTOK_MILLIONS [--vmap existing_vmap.npy]"""
import os, sys, glob, time, json, numpy as np
os.environ['TOKENIZERS_PARALLELISM'] = 'true'
from tokenizers import Tokenizer
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

OUT = os.path.expanduser('~/work/g4/data'); os.makedirs(OUT, exist_ok=True)
NT = int(float(sys.argv[1]) * 1e6)
VM = sys.argv[sys.argv.index('--vmap') + 1] if '--vmap' in sys.argv else None
FTC = sys.argv[sys.argv.index('--ftcounts') + 1] if '--ftcounts' in sys.argv else None
V = 32768


def qwen_tok():
    p = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/tokenizer.json'))
    if not p: p = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B-Base/snapshots/*/tokenizer.json'))
    return Tokenizer.from_file(p[0])


t0 = time.time()
tok = qwen_tok(); EOS = tok.token_to_id('<|endoftext|>'); NV = tok.get_vocab_size()
print('qwen3.5 vocab', NV, 'eos', EOS, flush=True)
files = sorted(f for f in HfApi().list_repo_files('HuggingFaceFW/fineweb-edu', repo_type='dataset') if f.startswith('sample/10BT/'))
print('shards', len(files), files[:3], flush=True)
ids = np.empty(NT + 2_000_000, dtype=np.uint32); n = 0
for f in files:
    path = hf_hub_download('HuggingFaceFW/fineweb-edu', f, repo_type='dataset')
    print('downloaded', f, f'{time.time()-t0:.0f}s', flush=True)
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        texts = pf.read_row_group(rg, columns=['text']).column('text').to_pylist()
        for i in range(0, len(texts), 2000):
            for e in tok.encode_batch(texts[i:i + 2000], add_special_tokens=False):
                a = e.ids; m = len(a) + 1
                if n + m > ids.shape[0]: break
                ids[n:n + len(a)] = a; ids[n + len(a)] = EOS; n += m
            if n >= NT: break
        print(f'rg {rg} tokens {n/1e6:.1f}M {time.time()-t0:.0f}s', flush=True)
        if n >= NT: break
    if n >= NT: break
ids = ids[:n]
cnt = np.bincount(ids, minlength=NV)
if VM:
    vmap = np.load(VM)
else:
    np.save(f'{OUT}/fw_counts.npy', cnt)
    order = np.argsort(-cnt, kind='stable')
    if FTC:   # equal-weight mix of FineWeb and fine-tune/eval-corpus token frequencies (keeps JSON keys, tool names, ...)
        ftc = np.load(FTC)
        score = cnt / cnt.sum() + ftc / ftc.sum(); score[EOS] = 1.0
        keep = np.argsort(-score, kind='stable')[:V - 1]
        print('FT-corpus coverage', 1 - ftc[np.setdiff1d(np.arange(NV), keep)].sum() / ftc.sum(), flush=True)
    else:
        keep = order[:V - 1]
        if EOS not in set(keep.tolist()): keep[-1] = EOS
    assert len(keep) == V - 1 and len(set(keep.tolist())) == V - 1
    vmap = np.full(NV, V - 1, dtype=np.int64); vmap[keep] = np.arange(V - 1)
    np.save(f'{OUT}/vmap.npy', vmap)
cov = 1 - cnt[vmap == V - 1].sum() / cnt.sum()
c = vmap[ids].astype(np.uint16)
nval = 2_000_000
c[:-nval].tofile(f'{OUT}/train.bin'); c[-nval:].tofile(f'{OUT}/val.bin')
json.dump(dict(ntok=int(n), train=int(n - nval), val=nval, coverage=float(cov), eos=int(EOS), eos_c=int(vmap[EOS]), V=V), open(f'{OUT}/meta.json', 'w'))
print(f'done {n/1e6:.1f}M tokens, compact-vocab coverage {cov:.5f}, {time.time()-t0:.0f}s', flush=True)

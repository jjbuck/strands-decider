"""details_match question (125 tokens, = H2's 1q bundle): exact hobson token ids + option offsets -> q1.pt"""
import sys, os, json, torch
sys.path[:0] = [os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from kitrun import load_P
P = load_P()
it = next(json.loads(l) for l in open(os.path.expanduser('~/work/evalkit/suites/REAL-agree.jsonl')) if 'details_match' in json.loads(l)['questions'])
pr = P.prep('x', it['questions']['details_match'])
torch.save(dict(ids=pr['q'], opt=pr['opt'], temp=P.temp_for(pr['rq'].kind), kind=pr['rq'].kind), os.path.expanduser('~/work/j15/q1.pt'))
print('q1', len(pr['q']), pr['opt'], pr['rq'].kind)

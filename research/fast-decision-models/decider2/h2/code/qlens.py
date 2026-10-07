import sys, os, json, collections
sys.path[:0] = [os.path.expanduser('~/work/evalkit'), os.path.expanduser('~/work/tokens')]
os.environ.setdefault('HF_HUB_OFFLINE', '1')
import glob
from transformers import AutoTokenizer
from pydantic import TypeAdapter
import strands_decider.schema as SC
from strands_decider.prompting import render_question, render_state
ta = TypeAdapter(SC.Question)
ck = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--StrandsAgents--strands-decider-2B-hobson-v19/snapshots/*'))[0]
tok = AutoTokenizer.from_pretrained(ck)
cnt = collections.Counter(); spec = {}; L = {}; per_req = collections.Counter(); dom = {}
for suite in ('REAL-agree', 'LONG'):
    for l in open(os.path.expanduser(f'~/work/evalkit/suites/{suite}.jsonl')):
        it = json.loads(l)
        per_req[len(it['questions'])] += 1
        for qn, qd in it['questions'].items():
            key = qn + '|' + json.dumps(qd, sort_keys=True)[:4000]
            cnt[key] += 1
            if key not in spec:
                rq = render_question(ta.validate_python(qd))
                spec[key] = (qn, qd); L[key] = len(tok(rq.text, add_special_tokens=False)['input_ids']); dom[key] = it.get('domain', '?')
print('questions per request', dict(per_req))
print('distinct specs', len(spec))
import statistics as st
ls = [L[k] for k in cnt for _ in range(cnt[k])]
print('token len over question instances: median', st.median(ls), 'mean', round(st.mean(ls), 1), 'p90', sorted(ls)[int(.9 * len(ls))], 'max', max(ls))
top = cnt.most_common(40)
for k, c in top: print(c, L[k], dom[k], spec[k][0])
json.dump([dict(name=spec[k][0], spec=spec[k][1], n=c, ntok=L[k], domain=dom[k]) for k, c in cnt.most_common()], open(os.path.expanduser('~/work/h2/qspecs.json'), 'w'))
it0 = json.loads(open(os.path.expanduser('~/work/evalkit/suites/REAL-agree.jsonl')).readline())
print(list(it0.keys()))

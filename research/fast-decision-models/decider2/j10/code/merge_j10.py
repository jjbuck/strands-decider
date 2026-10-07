"""Check that the regenerated Qwen-tokenised twin equals F7's teacher-labelled rows id-for-id, then copy F7's hobson teacher logits
(and calibrated temperatures) into the BitNet-tokenised rows. usage: python merge_j10.py rows_corpus rows_real rows_lceval"""
import sys, torch
for name in sys.argv[1:]:
    R = torch.load(f'{name}_q.pt', weights_only=False); F7 = torch.load(f'f7/{name}_q.pt', weights_only=False); S = torch.load(f'{name}_s.pt', weights_only=False)
    assert len(R['rows']) == len(F7['rows']) == len(S['rows']), (len(R['rows']), len(F7['rows']), len(S['rows']))
    bad = sum(int(not torch.equal(a['ids'], b['ids']) or a['opt'] != b['opt'] or a['label'] != b['label']) for a, b in zip(R['rows'], F7['rows']))
    n = 0
    for a, s in zip(F7['rows'], S['rows']):
        assert a['n'] == s['n'] and a['label'] == s['label']
        if 't_logits' in a: s['t_logits'] = a['t_logits']; n += 1
    S['teacher_temps'] = F7['teacher_temps']
    torch.save(S, f'{name}_s.pt')
    print(name, 'rows', len(S['rows']), 'id/opt/label mismatches vs F7', bad, 'teacher rows', n,
          'student tokens', sum(len(r['ids']) for r in S['rows']), 'qwen tokens', sum(len(r['ids']) for r in R['rows']), flush=True)

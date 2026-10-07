cd ~/work/q2 && source ~/venv/bin/activate
while ! grep -q EVAL1_DONE run_eval1.log 2>/dev/null; do sleep 20; done
M=~/work/q2
Q2SIDE=1 python q2bench.py cmap:$M/q2map_k24rr.json,cmap:$M/q2map_k48rr.json,cmap:$M/q2map_k64rr.json 1000,4000,256 1q 20 > e2e2.log 2>&1
python q2bench.py rrm0.05,rrm0.1,rrm0.25 1000,4000 1q 20 >> e2e2.log 2>&1
Q2SIDE=1 Q2E2E=~/work/q2/res_e2e_side.jsonl python q2bench.py w4q8c 64,1000,4000,256 1q,15q 20 >> e2e2.log 2>&1
echo BENCH2_DONE
python q2eval.py q2_ck48rr c:map:$M/q2map_k48rr.json > eval_ck48.log 2>&1
python q2eval.py q2_ck64rr c:map:$M/q2map_k64rr.json > eval_ck64.log 2>&1
echo EVAL2_DONE

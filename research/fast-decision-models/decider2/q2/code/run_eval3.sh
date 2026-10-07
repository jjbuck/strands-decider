cd ~/work/q2 && source ~/venv/bin/activate
while ! grep -q BENCH5_DONE run_bench5.log 2>/dev/null; do sleep 20; done
M=~/work/q2
python q2eval.py q2_ck56rr c:map:$M/q2map_k56rr.json > eval_ck56.log 2>&1
echo EVAL3_DONE
Q2SIDE=1 Q2E2E=~/work/q2/res_e2e_k64.jsonl python q2bench.py cmap:$M/q2map_k56rr.json,cmap:$M/q2map_k40rr.json 64,256,1000,4000 1q,15q 20 > e2e6.log 2>&1
echo BENCH6_DONE

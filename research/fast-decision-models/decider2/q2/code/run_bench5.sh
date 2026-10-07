cd ~/work/q2 && source ~/venv/bin/activate
while ! grep -q BENCH4_DONE run_bench4.log 2>/dev/null; do sleep 20; done
M=~/work/q2
Q2SIDE=1 Q2E2E=~/work/q2/res_e2e_k64.jsonl python q2bench.py cmap:$M/q2map_k64rr.json,cmap:$M/q2map_k48rr.json 64,256,1000,4000 1q,15q 20 > e2e5.log 2>&1
echo BENCH5_DONE

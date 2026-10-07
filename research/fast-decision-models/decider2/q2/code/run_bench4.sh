cd ~/work/q2 && source ~/venv/bin/activate
while ! grep -q BENCH3_DONE run_bench3.log 2>/dev/null; do sleep 20; done
Q2E2E=~/work/q2/res_e2e_rrm.jsonl python q2bench.py rrm0.05,rrm0.1,rrm0.25,rrm0.5,w4q8 1000,4000 1q 20 > e2e4.log 2>&1
echo BENCH4_DONE

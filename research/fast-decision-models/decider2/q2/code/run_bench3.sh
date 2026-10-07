cd ~/work/q2 && source ~/venv/bin/activate
while ! grep -q EVAL2_DONE run_bench2.log 2>/dev/null; do sleep 20; done
Q2E2E=~/work/q2/res_e2e_kern.jsonl python q2bench.py w4q8c,b8,w4a4 64,256,1000,4000 1q,15q 20 > e2e3.log 2>&1
python bench_qz.py > bench_qz.log 2>&1
echo BENCH3_DONE

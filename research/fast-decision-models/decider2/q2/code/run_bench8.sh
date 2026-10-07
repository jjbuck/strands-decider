cd ~/work/q2 && source ~/venv/bin/activate
while ! grep -q EVAL4_DONE run_eval4.log 2>/dev/null; do sleep 20; done
python bench_rr2.py 1000,4000,256 125 > bench_rr2.log 2>&1
echo BENCH8_DONE

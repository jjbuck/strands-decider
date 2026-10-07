cd ~/work/q2 && source ~/venv/bin/activate
while ! grep -q BENCH7_DONE run_bench7.log 2>/dev/null; do sleep 20; done
M=~/work/q2
python q2eval.py q2_dk48rr d:map:$M/q2map_k48rr.json > eval_dk48.log 2>&1
python q2eval.py q2_dw4q8 d:w4q8 > eval_dw4q8.log 2>&1
echo EVAL4_DONE

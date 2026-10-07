cd ~/work/q2 && source ~/venv/bin/activate
while ! grep -q BENCH8_DONE run_bench8.log 2>/dev/null; do sleep 20; done
M=~/work/q2
python q2eval.py q2_k48rr_qrt2 map:$M/q2map_k48rr.json > eval_k48q.log 2>&1
echo EVAL5_DONE

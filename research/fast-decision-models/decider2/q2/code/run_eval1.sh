cd ~/work/q2 && source ~/venv/bin/activate
python q2eval.py q2_w4q8c c:w4q8 > eval_w4q8c.log 2>&1
python q2eval.py h2_b8 h2:map:~/work/h1/precmap_w8a8_b8.json:w8a8 > eval_b8.log 2>&1
python q2eval.py h2_bf16 h2:bf16 > eval_bf16.log 2>&1
echo EVAL1_DONE

cd ~/work/q2 && source ~/venv/bin/activate
while ! grep -q EVAL5_DONE run_eval5.log 2>/dev/null; do sleep 20; done
python basis_cost.py > basis_cost.log 2>&1
echo BENCH9_DONE

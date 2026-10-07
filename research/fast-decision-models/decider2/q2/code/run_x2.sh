# latency (exclusive GPU) after the reference evals: grid of exact lengths, then J15's 120 real requests
cd ~/work/q2 && source ~/venv/bin/activate
while ! grep -q X1_DONE run_x1.log 2>/dev/null; do sleep 20; done
python q2xbench.py grid k64,b8 1000,64,256,4000 1q,15q > xgrid.log 2>&1
echo GRID_DONE
python q2xbench.py real $(cat tau.txt) > xreal.log 2>&1
echo X2_DONE $(date +%T)

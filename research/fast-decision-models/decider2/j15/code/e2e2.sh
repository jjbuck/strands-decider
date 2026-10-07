source ~/venv/bin/activate
cd ~/work/j15
echo "=== $(date +%T) dcasc {16}"
python j15bench.py dcasc 16:0.069556 120 _16 2>&1 | grep -v Loading | tail -2
echo "=== $(date +%T) dcasc {8,16}"
python j15bench.py dcasc 8:0.62195,16:0.069556 120 _816 2>&1 | grep -v Loading | tail -2
echo E2E2_DONE $(date +%T)

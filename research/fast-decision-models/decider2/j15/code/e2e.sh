# end-to-end cascades on real requests at their exact lengths (exclusive GPU, after chain2)
source ~/venv/bin/activate
cd ~/work/j15
echo "=== $(date +%T) dcasc {8,16}"
python j15bench.py dcasc 8:0.62195,16:0.069556 200 _816 2>&1 | grep -v Loading | tail -2
echo "=== $(date +%T) dcasc {16}"
python j15bench.py dcasc 16:0.069556 100 _16 2>&1 | grep -v Loading | tail -2
echo "=== $(date +%T) precision casc W4A4->b8 tau .559"
python j15bench.py casc w4a4 b8 0.55935 120 _w4 2>&1 | grep -v Loading | tail -2
echo E2E_DONE $(date +%T)

#!/bin/bash
source ~/venv/bin/activate; cd ~/work/g3; export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python bench_lat.py gr1b 1000 4000 --check 2>&1 | grep '^{' > results/lat_gr1b.log; cut -c1-300 results/lat_gr1b.log
python bench_lat.py euro 1000 4000 --check 2>&1 | grep '^{' > results/lat_euro.log; cut -c1-300 results/lat_euro.log
python synth.py --T 1000 4000 2>&1 | grep '^{' > results/synth.log; cut -c1-400 results/synth.log
for r in results/route_st4b_1000.pt results/route_st4b_4000.pt results/route_gr3b_1000.pt results/route_gr3b_4000.pt; do
  python vllm_bench.py $r mine 2>&1 | grep '^{'
  [ -x ~/vvenv/bin/python ] && ~/vvenv/bin/python vllm_bench.py $r vllm 2>&1 | grep -E '^\{|Error' | tail -n 2
done > results/experts_cmp.log; cat results/experts_cmp.log
echo timing_B_done

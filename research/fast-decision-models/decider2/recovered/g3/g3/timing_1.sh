#!/bin/bash
source ~/venv/bin/activate; cd ~/work/g3; export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p results
nvidia-smi --query-gpu=name,clocks.sm,memory.used,utilization.gpu --format=csv,noheader
date
python synth.py --T 1000 4000 > results/synth.log 2>&1; grep '^{' results/synth.log | cut -c1-330; grep -iE 'error|Traceback' results/synth.log | head -3
date
python bench_lat.py st4b 1000 4000 --check --extra 2>&1 | grep -E '^\{|Error' > results/lat_st4b.log; cut -c1-330 results/lat_st4b.log
date
for r in results/route_st4b_1000.pt results/route_st4b_4000.pt; do
  python vllm_bench.py $r mine 2>&1 | grep '^{'
  ~/vvenv/bin/python vllm_bench.py $r vllm 2>&1 | grep -E '^\{|Error' | tail -n 2
done > results/experts_cmp.log; cat results/experts_cmp.log
date
echo timing_1_done

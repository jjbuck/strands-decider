#!/bin/bash
# exclusive (box.sh --timing): dense baselines + pretrained MoEs
source ~/venv/bin/activate; cd ~/work/g3; export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
mkdir -p results
nvidia-smi --query-gpu=name,clocks.sm,memory.used,utilization.gpu --format=csv,noheader
pgrep -fa python | grep -v pgrep | cut -c1-100
(cd f7code && python lat.py q08 - 0 2>&1 | grep '^{'; python lat.py hob - 0 2>&1 | grep '^{') > results/lat_dense.jsonl
cat results/lat_dense.jsonl | cut -c1-400
python bench_lat.py st4b 1000 4000 --check --extra 2>&1 | grep '^{' > results/lat_st4b.log; cut -c1-300 results/lat_st4b.log
python bench_lat.py gr3b 1000 4000 --check --extra 2>&1 | grep '^{' > results/lat_gr3b.log; cut -c1-300 results/lat_gr3b.log
echo timing_A_done

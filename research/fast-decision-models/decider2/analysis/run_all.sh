#!/bin/bash
# Re-run the whole post-processing pipeline: catalog every agent's predictions, score them, draw the figures.
#   bash ~/decider2/analysis/run_all.sh
# Outputs: catalog.csv, norm/, results/metrics.{csv,json}, figures/*.{svg,csv}, figures/png/*.png.
# The docs get the PNGs only (~/code/jit-eval/docs/figures/) plus the plotted numbers (docs/figures/data/*.csv).
set -euo pipefail
cd "$(dirname "$0")"
python3 catalog.py
python3 metrics.py
python3 latency_grid.py > results/latency_at_1000.txt
python3 figures.py
mkdir -p ~/code/jit-eval/docs/figures/data
cp figures/png/*.png ~/code/jit-eval/docs/figures/
cp figures/*.csv ~/code/jit-eval/docs/figures/data/
echo "figures copied to ~/code/jit-eval/docs/figures"

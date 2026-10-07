#!/bin/bash
# box-side driver for v0 references
cd ~/work/evalkit && source ~/venv/bin/activate
export PYTHONUNBUFFERED=1
mkdir -p logs refs
( python kitrun.py fidelity > logs/fid.log 2>&1; python kitrun.py refs JB-all > logs/refs_jb.log 2>&1; python kitrun.py refs REAL-agree > logs/refs_real.log 2>&1; python kitrun.py refs LONG > logs/refs_long.log 2>&1; echo A_DONE > logs/A_DONE ) &
( python kitrun.py base JB-all > logs/base_jb.log 2>&1; python kitrun.py base REAL-agree > logs/base_real.log 2>&1; python kitrun.py base LONG > logs/base_long.log 2>&1; echo B_DONE > logs/B_DONE ) &
wait

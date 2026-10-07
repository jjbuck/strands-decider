#!/bin/bash
cd ~/work/evalkit && source ~/venv/bin/activate
export PYTHONUNBUFFERED=1
( while [ ! -f logs/A_DONE ]; do sleep 20; done
  python kitrun.py refs CF > logs/refs_cf.log 2>&1; python kitrun.py refs CF-probe > logs/refs_probe.log 2>&1; echo done > logs/C_DONE ) &
( while [ ! -f logs/base_REAL-agree.DONE ]; do sleep 20; done
  ./drive_b.sh CF 3; ./drive_b.sh CF-probe 3; ./drive_b.sh LONG 3; echo done > logs/D_DONE ) &
wait

#!/bin/bash
cd ~/work/evalkit && source ~/venv/bin/activate
export PYTHONUNBUFFERED=1
while [ ! -f logs/C_DONE ]; do sleep 15; done
./drive_b.sh CF 3
BASE_GRID=lite ./drive_b.sh CF-probe 3
BASE_GRID=lite ./drive_b.sh LONG 3
echo done > logs/D2_DONE

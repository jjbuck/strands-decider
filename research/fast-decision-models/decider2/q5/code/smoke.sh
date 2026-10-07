#!/bin/bash
source ~/venv/bin/activate; cd ~/work/q5
python q5cal.py --dom bank --n 4 --layers 22 --neur --fn --frs > smoke_cal_bank.log 2>&1
python q5cal.py --dom ret --n 4 --layers 22 --neur --fn --frs > smoke_cal_ret.log 2>&1
python q5smoke.py > smoke.log 2>&1

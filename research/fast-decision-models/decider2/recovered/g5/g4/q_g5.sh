#!/bin/bash
# g5 queue: S60 arms (+ S125 MUDD at the end)
cd ~/work/g4; source ~/venv/bin/activate; export PYTHONUNBUFFERED=1
python prep_ft.py > logs/prep_ft2.log 2>&1 &
python pretrain.py S60 dense --pflops 83 > logs/pt_S60_dense.log 2>&1
python pretrain.py S60 slot --pflops 83 > logs/pt_S60_slot.log 2>&1
wait
python finetune.py S60 dense > logs/ft_S60_dense.log 2>&1
python finetune.py S60 slot > logs/ft_S60_slot.log 2>&1
python pretrain.py S60 mudd --pflops 83 > logs/pt_S60_mudd.log 2>&1
python pretrain.py S60 dense3 --pflops 83 > logs/pt_S60_dense3.log 2>&1
python finetune.py S60 mudd > logs/ft_S60_mudd.log 2>&1
python finetune.py S60 dense3 > logs/ft_S60_dense3.log 2>&1
python pretrain.py S125 mudd --pflops 180 > logs/pt_S125_mudd.log 2>&1
python finetune.py S125 mudd > logs/ft_S125_mudd.log 2>&1
echo QUEUE_DONE > logs/q_done

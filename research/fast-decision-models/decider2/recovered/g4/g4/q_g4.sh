#!/bin/bash
# g4 queue: S125 arms
cd ~/work/g4; source ~/venv/bin/activate; export PYTHONUNBUFFERED=1
python pretrain.py S125 dense --pflops 180 > logs/pt_S125_dense.log 2>&1
python pretrain.py S125 slot --pflops 180 > logs/pt_S125_slot.log 2>&1
while [ ! -f data/ft.pkl ]; do sleep 30; done
python finetune.py S125 dense > logs/ft_S125_dense.log 2>&1
python finetune.py S125 slot > logs/ft_S125_slot.log 2>&1
python pretrain.py S125 dense3 --pflops 180 > logs/pt_S125_dense3.log 2>&1
python finetune.py S125 dense3 > logs/ft_S125_dense3.log 2>&1
echo QUEUE_DONE > logs/q_done

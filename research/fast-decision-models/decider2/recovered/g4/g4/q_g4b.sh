#!/bin/bash
# g4 queue part 2 (after q_g4.sh): fine-grained MoE at S60
cd ~/work/g4; source ~/venv/bin/activate; export PYTHONUNBUFFERED=1
while [ ! -f logs/q_done ]; do sleep 30; done
python pretrain.py S60 moe --pflops 83 > logs/pt_S60_moe.log 2>&1
python finetune.py S60 moe > logs/ft_S60_moe.log 2>&1
echo QUEUE_DONE > logs/q_done_b

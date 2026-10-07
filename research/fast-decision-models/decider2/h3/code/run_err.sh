#!/bin/bash
source ~/venv/bin/activate; cd ~/work/h3; export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch,h3lib as H; x=torch.randn(5,6144,device='cuda'); print('rot6144', (H.rot_rows(x)-x@H.rot_mat(6144,'cuda')).abs().max().item()); x=torch.randn(5,2048,device='cuda'); print('rot2048', (H.rot_rows(x)-x@H.rot_mat(2048,'cuda')).abs().max().item())" > logs/rotcheck.log 2>&1
python errprop.py A --n 12 --fmt int4 > logs/errA_int4.log 2>&1
python errprop.py A --n 12 --fmt nvfp4 > logs/errA_nvfp4.log 2>&1
python errprop.py B --n 48 --fmt int4 > logs/errB_int4.log 2>&1
python errprop.py A --n 12 --fmt int4 --rot --tag _rot > logs/errA_int4rot.log 2>&1
python errprop.py B --n 48 --fmt nvfp4 > logs/errB_nvfp4.log 2>&1
echo done > logs/err_done

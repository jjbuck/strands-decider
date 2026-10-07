#!/bin/bash
cd ~/work/n1; source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
export HOB_NKI=1 INLINE_W=0 NEURON_RT_VISIBLE_CORES=1
HOB_KMOD=gdn7 python ltest.py n5 4 1152 n5g7_NL4_L1152_noinl 2>&1 | grep -E '^\{|Error|error'
HOB_KMOD=gdn5 python ltest.py n5 4 1152 n5g5_NL4_L1152_noinl 2>&1 | grep -E '^\{|Error|error'

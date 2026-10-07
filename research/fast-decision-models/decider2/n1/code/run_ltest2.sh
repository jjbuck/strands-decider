#!/bin/bash
cd ~/work/n1; source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
export HOB_NKI=1
HOB_KMOD=gdn6 python ltest.py n5 4 1152 n5g6_NL4_L1152 2>&1 | grep -E '^\{|Error|error'
HOB_KMOD=gdn6 HOB_FAKEGDN=2 python ltest.py n5 4 1152 n5g6_NL4_L1152_fake2 2>&1 | grep -E '^\{|Error|error'

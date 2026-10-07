#!/bin/bash
cd ~/work/n1; source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
export HOB_NKI=1 HOB_KMOD=gdn8
INLINE_W=0 python ltest.py n5 4 1152 n5g8_NL4_L1152_noinl 2>&1 | grep -E '^\{|Error|error'
INLINE_W=1 python ltest.py n5 4 1152 n5g8_NL4_L1152 2>&1 | grep -E '^\{|Error|error'
INLINE_W=0 HOB_KER=2 python ltest.py nl 4 1152 nl_NL4_L1152_noinl 2>&1 | grep -E '^\{|Error|error'
HOB_FAKEGDN=2 INLINE_W=1 python ltest.py n5 4 1152 n5g8_NL4_L1152_fake2 2>&1 | grep -E '^\{|Error|error'

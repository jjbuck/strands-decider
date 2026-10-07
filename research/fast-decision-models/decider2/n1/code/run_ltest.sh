#!/bin/bash
# sequential in-situ layer tests (4 layers = 3 GDN + 1 attention, L=1152)
cd ~/work/n1; source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
export HOB_NKI=1 HOB_KER=2 HOB_KMOD=${HOB_KMOD:-gdn5}
for cfg in "n5 4 1152" "nl 4 1152"; do python ltest.py $cfg 2>&1 | grep -E '^\{|Error|error' ; done
for cfg in "n5 4 1152" "nl 4 1152"; do HOB_FAKEGDN=1 python ltest.py $cfg 2>&1 | grep -E '^\{|Error|error' ; done

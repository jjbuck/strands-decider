#!/bin/bash
rm -rf ~/work/j8/neff/wd_L4096_S32_nlk2_C128
mkdir -p ~/work/c4096 && cd ~/work/c4096
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
HOB_KER=2 HOB_NL=1 HOB_NKI=1 HOB_MARK=0 python ~/work/j8/nbuild.py --L 4096 --sel 32 --C 128 --tag L4096_S32_nlk2

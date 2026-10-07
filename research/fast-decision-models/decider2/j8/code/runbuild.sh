#!/bin/bash
# runbuild.sh NAME nbuild-args...   (own cwd per compile; HobNL + NKI k2)
N=$1; shift
mkdir -p ~/work/cwd_$N && cd ~/work/cwd_$N
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
HOB_KER=2 HOB_NL=1 HOB_NKI=1 HOB_MARK=0 python ~/work/j8/nbuild.py "$@"

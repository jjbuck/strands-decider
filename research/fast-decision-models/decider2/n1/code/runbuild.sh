#!/bin/bash
# runbuild.sh NAME KER nbuild-args...   (own cwd per compile; HobNL + NKI kernel KER)
N=$1; K=$2; shift; shift
mkdir -p ~/work/n1/cwd_$N && cd ~/work/n1/cwd_$N
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
HOB_KER=$K HOB_NL=1 HOB_NKI=1 HOB_MARK=0 python ~/work/n1/nbuild.py "$@"

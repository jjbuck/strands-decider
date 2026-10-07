#!/bin/bash
# fullprof.sh WD [NAME]: warm profile of a compiled graph NEFF (3rd execution) -> WD/prof_NAME.json summary
WD=$1; NM=${2:-p}
cd $WD
neuron-profile capture -n graph.neff -s $NM.ntff --num-exec=4 --profile-nth-exec=3 > $NM.capture.log 2>&1
ls $NM*.ntff
F=$(ls -t $NM*.ntff | head -1)
neuron-profile view -n graph.neff -s $F --output-format json --output-file $NM.json > $NM.view.log 2>&1
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
python ~/work/n1/psum.py $NM.json > $NM.summary.json; cat $NM.summary.json

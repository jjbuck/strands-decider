#!/bin/bash
# latgrid.sh CORE TAG:T:M ... : latency of each graph (20 reps), appends to logs/latgrid.jsonl
cd ~/work/n1; source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
C=$1; shift
for a in "$@"; do
  IFS=: read tag T M <<< "$a"
  NEURON_RT_VISIBLE_CORES=$C python nrun.py lat ${tag}_C128 $T $M 20 2>&1 | grep '^{' | python -c "import sys,json; d=json.loads(sys.stdin.read()); d.pop('probs'); print(json.dumps(d))" | tee -a logs/latgrid.jsonl
done

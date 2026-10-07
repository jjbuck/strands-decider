#!/bin/bash
# nprof.sh WD...: warm (3rd exec) profile summary of graph.neff in each dir (small graphs only); prints total_time and engine activity
source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate
for WD in "$@"; do
  cd $WD
  neuron-profile capture -n graph.neff -s w.ntff --num-exec=4 --profile-nth-exec=3 > wcap.log 2>&1
  F=$(ls -t w*.ntff | head -1)
  neuron-profile view -n graph.neff -s $F --output-format summary-json --ignore-instruction-trace --ignore-dma-trace --ignore-event-trace > wsum.json 2> wview.log
  python - <<PY
import json
d = json.load(open('wsum.json')); s = list(d.values())[0]
print(json.dumps(dict(wd='$WD'.split('/')[-1], total_ms=round(s['total_time'] * 1e3, 3), mfu=round(s.get('mfu_estimated_percent', 0), 3),
      model_tflop=round(s.get('model_flops', 0) / 1e12, 4), weight_queue_GB=round(s.get('weight_queue_bytes', 0) / 1e9, 3))))
PY
done

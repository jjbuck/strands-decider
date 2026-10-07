#!/bin/bash
# latency, second batch (EXCLUSIVE GPU; launched by hand after all evals). setspq = bundle compiled in bf16 once, state AND slot rows low-bit.
source ~/venv/bin/activate; cd ~/work/h6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"; K48="map:~/work/h2/precmap_w4a4_k48.json"; K64="map:~/work/h6/precmap_w4a4_k64.json"
BA16=1 python bench6.py $K64 1000,4000 1q,4q,15q plain,sets,setspq k64ba --codes $C >> bench2.log 2>&1
BA16=1 OVL=1 python bench6.py $K64 1000,4000 1q,4q,15q plainqb,setsmix k64baovl --codes $C >> bench2.log 2>&1
BA16=1 python bench6.py $K48 1000,4000 1q,15q setspq k48ba --codes $C >> bench2.log 2>&1
SLOT8=1 BA16=1 OVL=1 python bench6.py $K48 1000,4000 1q,15q plainqb,setsmix k48ba8ovl --codes $C >> bench2.log 2>&1
python bench6.py map:~/work/h2/precmap_w8a8_b8.json:w8a8 1000,4000 1q,4q,15q plain,sets b8 --codes $C >> bench2.log 2>&1
echo BENCH2DONE >> bench2.log

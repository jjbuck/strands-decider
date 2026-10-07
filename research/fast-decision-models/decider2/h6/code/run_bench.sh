#!/bin/bash
# H6 latency matrix (EXCLUSIVE GPU): H2's method (graph replay incl. head + softmax, fresh random state ids per rep via pinned H2D, D2H probs;
# 3 warm + 20 timed; median, p95).  k48 = H1/H2 k48 map, GPTQ codes; BA16 = GDN b/a gates bf16 (H1's best 4-bit point).
source ~/venv/bin/activate; cd ~/work/h6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"; K48="map:~/work/h2/precmap_w4a4_k48.json"
python bench6.py bf16 1000,4000 1q,4q,15q plain,schema,sets bf16 >> bench.log 2>&1
BA16=1 python bench6.py $K48 1000,4000 1q,4q,15q plain,plainqb,schema,schemamix,sets,setsmix k48ba --codes $C >> bench.log 2>&1
BA16=1 OVL=1 python bench6.py $K48 1000,4000 1q,4q,15q plainqb,schemamix,setsmix k48baovl --codes $C >> bench.log 2>&1
python bench6.py $K48 1000,4000 1q,15q plain,schema k48 --codes $C >> bench.log 2>&1
OVL=1 python bench6.py w4a4 1000,4000 1q,15q schema,setsmix w4a4ovl --codes $C >> bench.log 2>&1
echo BENCHDONE >> bench.log

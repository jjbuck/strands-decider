while pgrep -f "[h]1eval.py --sub main" >/dev/null; do sleep 15; done
python h1cmp.py --n 60 --out res/cmp1.json --cfg 'w8a8,gptq8;w8a8,gptq8,map=precmap_w8a8_b8.json;w4a8,gptq;w4a4,gptq;w4a4;w4a4,gptq,ba16;w4a4,gptq,ohead;w4a4,gptq,map=precmap_w4a4_k48.json;w4a8,gptq,map=precmap_w8a8_b8.json;w4a4,gptq,clip=0.85;w4a4,gptq,seed=7;w8a8,gptq8,ba16'

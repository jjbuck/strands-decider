while pgrep -f "[h]1eval.py --cfg dense" >/dev/null; do sleep 10; done
python h1eval.py --cfg 'w4a8,gptq,qb16;w4a4,gptq,ba16,clip=0.85,map=precmap_w4a4_k48.json,qb16' --sub quick --out res/e5_w4var.json

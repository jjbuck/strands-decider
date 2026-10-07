while pgrep -f "[h]1eval.py --cfg w8a8,gptq8,map" >/dev/null; do sleep 10; done
python merge.py res/e3_all.json res/e1_main.json res/e2_main.json
python h1eval.py --cfg 'w8a8,gptq8,qb16;w8a8,gptq8,map=precmap_w8a8_b8.json,qb16;dense' --sub all --out res/e3_all.json

while ! grep -q "config done w8a8,gptq8,qb16" logs/e3.log; do sleep 10; done
pkill -f "[j]obs/e3.sh"; pkill -f "[h]1eval.py --cfg w8a8,gptq8,qb16;"
sleep 5
python merge.py res/e3_dense.json res/e1_main.json
python h1eval.py --cfg 'dense' --sub all --out res/e3_dense.json
python h1eval.py --cfg 'w8a8,gptq8,map=precmap_w8a8_b8.json,qb16' --sub main --out res/e3_b8qb16.json

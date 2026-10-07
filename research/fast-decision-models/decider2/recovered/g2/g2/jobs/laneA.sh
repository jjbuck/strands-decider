python g2map.py --sub qatf --skipdense --cfg 'Q:r4c' --lora lora_qat44_a_s1500.pt --out res/qe1500.json
python g2map.py --sub qsd3 --skipdense --cfg 'Q:r4c' --out res/qe0.json
python g2map.py --sub qsd3 --skipdense --cfg 'Q:r4c' --lora lora_qat44_a_s500.pt --out res/qe500.json
python g2map.py --sub qsd3 --skipdense --cfg 'Q:r4c' --lora lora_qat44_a_s1000.pt --out res/qe1000.json

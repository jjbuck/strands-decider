while ! grep -q "saved 900" logs/qatB.log; do sleep 10; done
pkill -f "[h]1qat.py"; sleep 5
python h1eval.py --cfg 'w4a8,gptq;w4a8,gptq,lrot=lora_qat_w4a8_B_s900.pt;w4a8,gptq,lrot=lora_qat_w4a8_B_s300.pt;w4a8,gptq,lrot=lora_qat_w4a8_B_s600.pt' --sub quick --out res/e4_w4a8curve.json
python h1eval.py --cfg 'w4a4,gptq,ba16,clip=0.85;w4a4,gptq,ba16,clip=0.85,map=precmap_w4a4_k48.json,qb16;w4a8,gptq,qb16' --sub quick --out res/e4_w4var.json

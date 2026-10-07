while pgrep -f "[h]1cmp.py" >/dev/null; do sleep 10; done
python h1qat.py --base w4a8 --gptq --steps 1200 --accum 4 --lr 1e-4 --r 32 --hid 1.0 --ckpts 4 --maxtok 1536 --tag B

# 1) plain-arm deployed-kernel scoring (final.sh); 2) Q1-recommended QAT base w4q8 (state rows W4A4 trainable, question rows W8A8 GPTQ8 fixed):
#    B4.1a scales (1.5M tokens) -> QAD 20M tokens (lr 2e-5, grid-point latents), dev + CF-probe/REAL-agree every 5M.
cd ~/work/q3; source ~/venv/bin/activate
bash final.sh > final.log 2>&1
python q3train.py --mode scale --q8 ~/work/q3/gptq8.pt --lr 1e-3 --warm 10 --tokens 1.5e6 --dev_every 0.75e6 --accum 4 --max_rows 6000 \
  --ck ~/work/q3/ck_scale_w4q8 > ck_scale_w4q8.out 2>&1
python q3train.py --mode qad --q8 ~/work/q3/gptq8.pt --seed_ck ~/work/q3/ck_scale_w4q8/best.pt --lat_init center --lr 2e-5 --warm 30 \
  --tokens 20e6 --dev_every 5e6 --accum 4 --l2sp 100 --w_hid 0.5 --max_rows 6000 \
  --ck ~/work/q3/ck_qad_w4q8 --curve CF-probe,REAL-agree --run qadw4q8 > ck_qad_w4q8.out 2>&1
echo CHAIN5_DONE

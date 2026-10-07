python ev_g1.py denseft ckpt:runs/denseft/step200.pt full > ev_denseft.log 2>&1
python ev_g1.py hobm hobson REAL-agree/4,CF/6 > ev_hobm.log 2>&1
python ev_g1.py mlp_lr_f50 ckpt:runs/mlp_lr_f50/step200.pt full > ev_mlp_lr_f50.log 2>&1

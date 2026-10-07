cd ~/work/q5; source ~/venv/bin/activate
Q5_MEMFRAC=0.50 python q5kit.py --cfgs cfgs_q5.json --tags b8+s24.L13-23.sgptd.int4p.s --out ~/work/q5/preds/b8_s24_L13-23_sgptd_int4p_s.json --start 12
python q5kit.py --cfgs cfgs_q5.json --tags nr.L12-23.dsalc.f0.5 --out ~/work/q5/preds/nr_L12-23_dsalc_f05.json --start 12
python q5kit.py --cfgs cfgs_q5.json --tags s24.L12-22.sgptd.bf16n.s.bank --out ~/work/q5/preds/s24_L12-22_sgptd_bf16n_s_bank.json --start 12

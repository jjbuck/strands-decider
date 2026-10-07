#!/bin/bash
# B1.1 chain: H1 calibration Hessians, spectrum groups 7..0 (set A), held-out pass (set B). Resumable.
source ~/venv/bin/activate; cd ~/work/q1
[ -f ~/work/h1/hess/H_23_Wd.pt ] || (cd ~/work/h1 && python h1calib.py --n 64 2>&1 | grep -v Warning)
for g in 7 6 5 4 3 2 1 0; do python q1spec.py --mode A --group $g 2>&1 | grep -v Warning; done
[ -f ~/work/q1/spec/heldout.json ] || python q1spec.py --mode B 2>&1 | grep -v Warning
echo CHAIN_SPEC_DONE

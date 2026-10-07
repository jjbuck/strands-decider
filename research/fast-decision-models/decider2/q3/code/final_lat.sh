# latency (exclusive GPU) with Q2's q2bench.py (same method as Q2's matrix): b8 (H2), w4a4 (H2; the plain arm's kernel), w4q8c side stream
# (Q2 row role; the w4q8 arm's kernel). Speed does not depend on code values. T = 64/256/1000/4000, 1 and 15 questions, 3 warm + 20 reps.
set -x
cd ~/work/h2; source ~/venv/bin/activate
[ -f bundles.pt ] || python prep_q.py
cd ~/work/q2
Q2SIDE=1 Q2E2E=~/work/q3/res_e2e_q3.jsonl CODES=w8=~/work/q3/codes8.pt,w4=~/work/q3/codes_gptq.pt python q2bench.py b8,w4a4,w4q8c 64,256,1000,4000 1q,15q 20
echo LAT_DONE

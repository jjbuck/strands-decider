# after chain3 (latency). 1) bf16 base with taps (depth cascade on the floor-level base). 2) GPTQ code draws s1/s2 (DEV for selection, eval).
# 3) suffix-refinement verifier quality (W4A4 [0,k) + b8 [k,24)).
source ~/venv/bin/activate
cd ~/work/j15
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"
EV="JB-all,REAL-agree,LONG,CF,CF-probe"
r() { echo "=== $(date +%T) $*"; python j15run.py "$@"; }
r preds/bf16t_dev.jsonl --prec bf16 --suites DEV --taps 8,12,14,16
r preds/bf16t_eval.jsonl --prec bf16 --suites $EV --taps 8,12,14,16
for S in 1 2; do
  [ -f codes_w8_s${S}.pt ] || python j15gptq.py $S
  r preds/b8s${S}_dev.jsonl --prec map:~/work/h1/precmap_w8a8_b8.json:w8a8 --codes ~/work/j15/codes_w8_s${S}.pt --suites DEV --taps 8,12,14,16
  r preds/b8s${S}_eval.jsonl --prec map:~/work/h1/precmap_w8a8_b8.json:w8a8 --codes ~/work/j15/codes_w8_s${S}.pt --suites $EV --taps 8,12,14,16
done
r preds/w4pre12_eval.jsonl --prec map:~/work/j15/pm_w4pre12.json:w4a4 --codes $C --suites $EV
r preds/w4pre18_eval.jsonl --prec map:~/work/j15/pm_w4pre18.json:w4a4 --codes $C --suites $EV
echo CHAIN2_DONE $(date +%T)

# J15 accuracy chain (resumable; each step skips finished questions). nohup bash chain1.sh > chain1.log 2>&1 &
source ~/venv/bin/activate
cd ~/work/j15; mkdir -p preds
C="w8=~/work/h2/codes_gptq_w8.pt,w4=~/work/h2/codes_gptq_w4.pt"
B8="map:~/work/h1/precmap_w8a8_b8.json:w8a8"; K48="map:~/work/h1/precmap_w4a4_k48.json:w4a4"; K64="map:~/work/h6/precmap_w4a4_k64.json:w4a4"
EV="JB-all,REAL-agree,LONG,CF,CF-probe"
TP="4,6,8,10,12,14,16,18,20"
r() { echo "=== $(date +%T) $*"; python j15run.py "$@"; }
r preds/b8_eval.jsonl --prec $B8 --codes $C --suites $EV --taps $TP
r preds/w4a4_eval.jsonl --prec w4a4 --codes $C --suites $EV --taps 24
r preds/b8_dev.jsonl --prec $B8 --codes $C --suites DEV --taps $TP
r preds/w4a4_dev.jsonl --prec w4a4 --codes $C --suites DEV --taps 24
r preds/b8_exit.jsonl --prec $B8 --codes $C --suites EXIT --taps $TP
r preds/w4a4_exit.jsonl --prec w4a4 --codes $C --suites EXIT --taps 24
r preds/k48_eval.jsonl --prec $K48 --codes $C --suites $EV
r preds/k48_dev.jsonl --prec $K48 --codes $C --suites DEV
r preds/bf16_eval.jsonl --prec bf16 --suites $EV
r preds/bf16_dev.jsonl --prec bf16 --suites DEV
echo "=== $(date +%T) k64qb"
cd ~/work/h6 && BA16=1 python h6eval.py ~/work/j15/preds/k64qb_eval.jsonl --prec $K64 --codes $C --layout plainqb --suites $EV
BA16=1 python h6eval.py ~/work/j15/preds/k64qb_dev.jsonl --prec $K64 --codes $C --layout plainqb --suites DEV
cd ~/work/j15
r preds/k48_exit.jsonl --prec $K48 --codes $C --suites EXIT
echo CHAIN1_DONE $(date +%T)

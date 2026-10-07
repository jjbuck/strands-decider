# k64rr (QRT2C) through all eval suites, DEV and EXIT with layer-16 taps; exit head; b8 + bf16 references on this box's codes
cd ~/work/q2 && source ~/venv/bin/activate
F="c:map:$HOME/work/q2/q2map_k64rr.json"
B8="h2:map:$HOME/work/h1/precmap_w8a8_b8.json:w8a8"
r() { echo "=== $(date +%T) $*"; python q2run.py "$@" 2>&1 | grep -v Loading | tail -2; }
r preds/k64_eval.jsonl --fmt $F --taps 16
r preds/k64_dev.jsonl --fmt $F --suites DEV --taps 16
r preds/k64_exit.jsonl --fmt $F --suites EXIT --taps 16
echo "=== $(date +%T) exit head"; python q2exit.py k64 16 2>&1 | grep -v Loading | tail -3
echo X1_EXIT_DONE
r preds/b8_eval.jsonl --fmt $B8
python q2eval.py h2_bf16 h2:bf16 2>&1 | tail -1
echo X1_DONE $(date +%T)

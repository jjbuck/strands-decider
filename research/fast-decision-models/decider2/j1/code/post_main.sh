#!/bin/bash
# after the main (masked) run: [eval A + rot  ||  hobson rotations]  ->  fork B (e1a, full mode from step 1700)  ->  [eval B || Q3 zero-shot window eval]
source ~/venv/bin/activate; cd ~/work/j1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
while pgrep -f "train_j1.py.*ck_main" > /dev/null || [ ! -f ck_main/step3392/lora.pt ]; do sleep 30; done
./final_lat.sh ck_main/step3392
echo "$(date +%T) main done; eval A + hobson rotations" >> chain_post.log
python ev_j1.py ck_main/step3392 preds_A > ev_A.log 2>&1 &
PA=$!
python rot_hob.py > rot_hob.log 2>&1 &
PH=$!
wait $PA; echo "$(date +%T) eval A done" >> chain_post.log
wait $PH; echo "$(date +%T) rot hob done" >> chain_post.log
if [ -f fork1700.pt ] && [ ! -f NO_FORK ]; then
  mkdir -p ck_full; [ -f ck_full/last.pt ] || cp fork1700.pt ck_full/last.pt
  echo "$(date +%T) fork B (full) start" >> chain_post.log
  python train_j1.py --rows rows_corpus_e.pt rows_real_e.pt --out ck_full --mode full --resume >> train_full.log 2>&1
  echo "$(date +%T) fork B done" >> chain_post.log
  python ev_j1.py ck_full/step3392 preds_B --mode full > ev_B.log 2>&1 &
  PB=$!
fi
python ev_j1.py ck_main/step3392 preds_A_loc512 --rot 0 --local "$(cat q3_local.txt)" --window 512 > ev_A_loc.log 2>&1
echo "$(date +%T) Q3 zero-shot eval done" >> chain_post.log
[ -n "$PB" ] && wait $PB && echo "$(date +%T) eval B done" >> chain_post.log
echo "$(date +%T) post chain done" >> chain_post.log

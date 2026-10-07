#!/bin/bash
# launched after the teacher + timing: train SmallThinker decider, then full-suite evals (final first, then 50%, then 75% if time)
source ~/venv/bin/activate; cd ~/work/g3
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
STEPS=${STEPS:-2400}
echo "$(date +%T) train start steps=$STEPS" >> chain_train.log
python train_g3.py --model st4b --rows rows_corpus_s.pt rows_real_s.pt --out ck_st4b --tokb 3072 --ckpt_above 3072 --max_steps $STEPS > train_st4b.log 2>&1
echo "$(date +%T) train done" >> chain_train.log
SUITES="JB-all REAL-agree LONG CF CF-probe"
for f in 1.0 0.5 0.75; do
  s=$(python -c "print(max(1, int(round($f * $STEPS))))")
  [ -f STOP_EVAL ] && break
  python ev_g3.py st4b ck_st4b/step$s st4b_s$s $SUITES >> ev_st4b.log 2>&1
  echo "$(date +%T) eval step$s done" >> chain_train.log
done

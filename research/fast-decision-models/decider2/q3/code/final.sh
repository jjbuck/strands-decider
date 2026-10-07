# Q3 final phase on q3 (exclusive GPU, after QAD): export codes, deployed-kernel scoring (H2 QRT, FORMATS.md F1) of every checkpoint,
# the bf16 runtime reference on this box, then latency (W4A4 vs W8A8-b8, T = 64/256/1000/4000, 1 and 15 questions).
set -x
cd ~/work/q3; source ~/venv/bin/activate
CK=~/work/q3/ck_qad
python q3export.py gptq codes_gptq.pt ck_scale/best.pt codes_b41a.pt
for f in $(ls $CK/t*M.pt | grep -v t0M) $CK/best.pt; do b=$(basename $f .pt); [ -f codes_qad_$b.pt ] || python q3export.py $f codes_qad_$b.pt; done
cd ~/work/h2
python evalrun.py bf16 all _q3 > ~/work/q3/qrt_bf16.log 2>&1
for c in ~/work/q3/codes_gptq.pt ~/work/q3/codes_b41a.pt $(ls ~/work/q3/codes_qad_t*M.pt | sort -V); do
  CODES=$c python evalrun.py w4a4 all _q3 > ~/work/q3/qrt_$(basename $c .pt).log 2>&1
done
ls -la ~/work/h2/preds_*_q3.jsonl
echo FINAL_SCORING_DONE

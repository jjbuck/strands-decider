# w4q8 deployed scoring through Q2's runtime (QRT2C: state rows W4A4 on H2 CUTLASS, question rows W8A8 on q2gemm), then latency.
set -x
cd ~/work/q3; source ~/venv/bin/activate
while pgrep -f "mode qa[d]" >/dev/null; do sleep 20; done
python q3export.py ck_scale_w4q8/best.pt codes_w4q8_s.pt ck_qad_w4q8/best.pt codes_w4q8_qadbest.pt
cd ~/work/q2
for c in w4q8_qadbest gptq w4q8_s; do
  CODES=w8=~/work/q3/codes8.pt,w4=~/work/q3/codes_$c.pt python q2eval.py q3_w4q8_$c c:w4q8 all > ~/work/q3/q2eval_$c.log 2>&1
done
ls -la ~/work/q2/preds/preds_q3_*
bash ~/work/q3/final_lat.sh > ~/work/q3/final_lat.log 2>&1
echo FINAL_W4Q8_DONE

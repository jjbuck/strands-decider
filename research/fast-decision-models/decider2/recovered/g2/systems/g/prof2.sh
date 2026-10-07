source ~/venv/bin/activate
cd ~/work/systems/g
for mode in gemm gemm2 model; do
  for i in 1 2 3 4 5 6; do
    python prof2.py $mode > prof2_$mode.out 2>&1
    if grep -q -E "OutOfMemory|out of memory" prof2_$mode.out; then sleep 20; else break; fi
  done
  echo "=== $mode"; grep -v -i warn prof2_$mode.out | grep -v Loading
done

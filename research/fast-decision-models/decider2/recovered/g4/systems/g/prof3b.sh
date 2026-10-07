source ~/venv/bin/activate
cd ~/work/systems/g
for mode in "2b 64,256,1000,2000" "2b_packed 64,1000" "2b_compile 64,1000"; do
  set -- $mode
  for i in $(seq 1 40); do
    python prof3.py $1 $2 > prof3_$1.out 2>&1
    if grep -q -E "OutOfMemory|out of memory" prof3_$1.out; then sleep 12; else break; fi
  done
  echo "=== $1 (tries $i)"; grep -v -i -E "warn|USDT|Loading" prof3_$1.out | tail -8 | cut -c1-300
done

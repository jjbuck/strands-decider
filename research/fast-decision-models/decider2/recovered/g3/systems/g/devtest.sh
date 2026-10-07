source ~/venv/bin/activate
cd ~/work/systems/g
for i in 1 2 3 4 5 6 7 8 9 10; do
  REPS=2 timeout 300 python bench3.py 2b host,packed4,hfrows4,throughput 128 > devtest.out 2>&1
  if grep -q -E "OutOfMemory|out of memory" devtest.out; then echo "oom retry $i"; sleep 30; else break; fi
done
grep -v -i warn devtest.out | tail -25

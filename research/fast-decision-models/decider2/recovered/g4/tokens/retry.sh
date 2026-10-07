#!/bin/bash
# usage: retry.sh <max_tries> <logfile> <cmd...>   retry on CUDA OOM
source ~/venv/bin/activate
n=$1; log=$2; shift 2
for i in $(seq 1 $n); do
  "$@" > $log 2>&1
  if grep -q "OutOfMemoryError" $log; then echo "attempt $i OOM, sleeping"; sleep 25; else break; fi
done
grep -v -i -E 'warn|Loading|falling' $log | tail -${TAILN:-30}

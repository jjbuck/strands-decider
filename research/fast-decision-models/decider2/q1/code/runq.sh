#!/bin/bash
# sequential GPU job queue: runs lines of ~/work/q1/queue.txt in order; line N done when ~/work/q1/qdone/N exists. Append lines to add jobs.
source ~/venv/bin/activate; cd ~/work/q1; mkdir -p qdone qlogs
while true; do
  n=0; ran=0
  while IFS= read -r line; do
    n=$((n+1)); [ -z "$line" ] && continue; [ -f qdone/$n ] && continue
    echo "$(date +%T) START $n: $line" >> runq.log
    bash -c "$line" > qlogs/$n.log 2>&1; rc=$?
    echo "$(date +%T) END $n rc=$rc" >> runq.log; touch qdone/$n; ran=1; break
  done < queue.txt
  [ $ran -eq 0 ] && sleep 15
done

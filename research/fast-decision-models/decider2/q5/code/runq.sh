#!/bin/bash
# sequential GPU job queue: runs lines of ~/work/q5/$Q.txt in order (Q=queue name, default queue); line N done when ~/work/q5/qdone_$Q/N exists.
# Append lines to add jobs. Lines starting with '#' are skipped.
Q=${1:-queue}
source ~/venv/bin/activate; cd ~/work/q5; mkdir -p qdone_$Q qlogs_$Q
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -f memfrac_$Q ] && export Q5_MEMFRAC=$(cat memfrac_$Q)
while true; do
  n=0; ran=0
  while IFS= read -r line; do
    n=$((n+1)); [ -z "$line" ] && continue; [[ "$line" == \#* ]] && continue; [ -f qdone_$Q/$n ] && continue
    echo "$(date +%T) START $n: $line" >> runq_$Q.log
    bash -c "$line" > qlogs_$Q/$n.log 2>&1; rc=$?
    echo "$(date +%T) END $n rc=$rc" >> runq_$Q.log; touch qdone_$Q/$n; ran=1; break
  done < $Q.txt
  [ $ran -eq 0 ] && sleep 15
done

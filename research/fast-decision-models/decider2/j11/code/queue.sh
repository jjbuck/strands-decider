#!/bin/bash
# sequential GPU job queue: runs each line of jobs.txt not yet in done.txt (re-read after every job); resumable.
source ~/venv/bin/activate; cd ~/work/j11
touch jobs.txt done.txt
while true; do
  job=$(grep -vxF -f done.txt jobs.txt | grep -v '^#' | grep -v '^$' | head -1)
  if [ -z "$job" ]; then sleep 20; continue; fi
  echo "$(date +%T) START $job" >> queue.log
  bash -c "$job"; rc=$?
  echo "$(date +%T) END rc=$rc $job" >> queue.log
  echo "$job" >> done.txt
done

#!/bin/bash
# usage: bg.sh JOBNAME  -> runs ~/work/h1/jobs/JOBNAME.sh detached, log in ~/work/h1/logs/JOBNAME.log
j=$1
cd ~/work/h1
setsid nohup bash -c "exec 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- 9>&-; source ~/venv/bin/activate; export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; cd ~/work/h1; bash jobs/$j.sh" > ~/work/h1/logs/$j.log 2>&1 < /dev/null &
echo "started $j pid $!"

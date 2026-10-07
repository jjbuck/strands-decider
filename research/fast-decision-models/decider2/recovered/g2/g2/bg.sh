#!/bin/bash
# usage: bg.sh JOBNAME   (runs ~/work/g2/jobs/JOBNAME.sh detached from the box.sh lock and from timeout's process group; log in logs/JOBNAME.log)
j=$1
cd ~/work/g2
setsid nohup bash -c "exec 3>&- 4>&- 5>&- 6>&- 7>&- 8>&- 9>&-; source ~/venv/bin/activate; export PYTHONUNBUFFERED=1; cd ~/work/g2; bash jobs/$j.sh" > ~/work/g2/logs/$j.log 2>&1 < /dev/null &
echo "started $j pid $!"

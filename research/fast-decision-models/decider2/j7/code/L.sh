#!/bin/bash
# L.sh NAME cmd... : run "cmd..." under nohup in ~/work/j7 with the venv, log to logs/NAME.log
cd ~/work/j7; source ~/venv/bin/activate; name=$1; shift
nohup "$@" > logs/$name.log 2>&1 < /dev/null &
echo "launched $name pid $!"

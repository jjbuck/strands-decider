#!/bin/bash
# pause my own GPU background jobs (teacher/train/eval python processes started from ~/work/g3) while running "$@", then resume
TP=$(pgrep -f '^python (teacher|train_g3|ev_g3)\.py' | tr '\n' ' ')
trap 'kill -CONT $TP 2>/dev/null' EXIT
[ -n "$TP" ] && kill -STOP $TP
sleep 1
"$@"

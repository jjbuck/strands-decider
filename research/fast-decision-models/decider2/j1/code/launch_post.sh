#!/bin/bash
cd ~/work/j1
setsid nohup ./post_main.sh < /dev/null > post_main.out 2>&1 &

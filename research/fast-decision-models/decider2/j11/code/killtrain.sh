#!/bin/bash
# kill python jobs whose command line STARTS with 'python <script>' (never matches the ssh shell)
for p in $(pgrep -f "^python ${1:-train.py}"); do kill $p; done

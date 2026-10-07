#!/bin/bash
# qbuild.sh: run the compile queue ~/work/n1/queue.txt (lines: ENVVARS -- nbuild args); skip tags with an output .pt or listed in
# failed.txt; one compile at a time (memory). Logs to logs/queue.log.
cd ~/work/n1; source /opt/aws_neuronx_venv_pytorch_2_8_nxd_inference/bin/activate; touch failed.txt
while true; do
  line=$(grep -v '^#' queue.txt | grep -v '^\s*$' | while read -r l; do t=$(echo "$l" | sed 's/.*--tag \([^ ]*\).*/\1/'); ls neff/${t}_C128.pt >/dev/null 2>&1 || grep -qx "$t" failed.txt || { echo "$l"; break; }; done)
  [ -z "$line" ] && { sleep 30; continue; }
  envs=$(echo "$line" | sed 's/ -- .*//'); args=$(echo "$line" | sed 's/.* -- //')
  tag=$(echo "$line" | sed 's/.*--tag \([^ ]*\).*/\1/')
  echo "$(date +%T) START $tag :: $envs :: $args" >> logs/queue.log
  mkdir -p cwd_$tag; ( cd cwd_$tag && env $envs HOB_NKI=1 HOB_MARK=0 python ~/work/n1/nbuild.py $args > ~/work/n1/logs/build_$tag.log 2>&1 )
  ls neff/${tag}_C128.pt >/dev/null 2>&1 || echo "$tag" >> failed.txt
  echo "$(date +%T) END $tag $(tail -1 logs/build_$tag.log | cut -c1-200)" >> logs/queue.log
  rm -rf neff/wd_${tag}*/model 2>/dev/null
done

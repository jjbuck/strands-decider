#!/bin/bash
# M2 training-free sweep, pass 1 (2 shards on one A10G)
cd ~/work/m2 && source ~/venv/bin/activate
C='nat0=|iso_nat=gran=nat;iso=all|iso_sec=gran=sec;iso=all|const=gran=const;iso=all|iso_b256=gran=blk256;iso=all|nat_attn=gran=nat;iso=attn|nat_gdn=gran=nat;iso=gdn|nat_ge8=gran=nat;iso=all;lay=8-23|nat_lt8=gran=nat;iso=all;lay=0-7|nat_sum=gran=nat;iso=all;comp=sum|nat_glob=gran=nat;iso=all;pos=global|f8=freeze=8|f12=freeze=12|nat_f8=gran=nat;iso=all;freeze=8|nat_f12=gran=nat;iso=all;freeze=12|const_f12=gran=const;iso=all;freeze=12|topk64=topk=64|nat_ge16=gran=nat;iso=all;lay=16-23|nat_zero=gran=nat;iso=all;comp=zero'
for s in 0 1; do
  nohup python m2tf.py res/tf1 --configs "$C" --shard $s/2 > res/tf1_shard$s.log 2>&1 &
done
wait
echo SWEEP1_DONE > res/tf1.done

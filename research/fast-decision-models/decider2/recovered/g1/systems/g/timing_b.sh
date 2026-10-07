source ~/venv/bin/activate
cd ~/work/systems/g
export REPS=12
echo "=== 0.8B"; python bench3.py 0.8b single,packed4,host 64,256,1000,2000,4000 2>&1 | grep -v -i warn | tee bench3_08b.out
echo "=== 0.6B"; python bench3.py 0.6b hfsingle 64,256,1000,2000 2>&1 | grep -v -i warn | tee bench3_06b.out
echo "=== throughput"; python bench3.py 2b throughput 256,1000 2>&1 | grep -v -i warn | tee thr_2b.out
python bench3.py 0.8b throughput 256,1000 2>&1 | grep -v -i warn | tee thr_08b.out
echo "=== compile glue"; python bench2.py lean_compile_graph 64,256,1000,2000 1 2>&1 | grep -v -i warn | tee bench2_b.out
echo ALLDONE_B

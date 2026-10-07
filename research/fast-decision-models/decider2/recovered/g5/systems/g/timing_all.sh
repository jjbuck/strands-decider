source ~/venv/bin/activate
cd ~/work/systems/g
export REPS=12
echo "=== prof_stock"; python prof_stock.py 2>&1 | grep -v -i warn | tee prof_stock.out
echo "=== 2B lean steps"; python bench2.py hfeager,hfgraph,lean_eager,lean_graph 64,256,1000,2000 1 2>&1 | grep -v -i warn | tee bench2_a.out
echo "=== 2B single/packed4/host"; python bench3.py 2b single,packed4,host 64,128,256,512,1000,2000,4000 2>&1 | grep -v -i warn | tee bench3_2b.out
echo "=== GEMM/sparse/int8"; python gemm_bench.py 2>&1 | grep -v -i warn | tee gemm_bench.out
echo ALLDONE_A

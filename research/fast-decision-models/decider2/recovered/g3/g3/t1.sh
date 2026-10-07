source ~/venv/bin/activate
C='{"BM":64,"BN1":128,"BK1":64,"BN2":64,"BK2":64,"w1":4,"s1":3,"w2":4,"s2":3,"nf":1}'
CFG=$C SPLIT=1 python vllm_bench.py results/route_st4b_1000.pt mine 2>&1 | grep '^{'
CFG=$C SPLIT=1 BALANCED=1 python vllm_bench.py results/route_st4b_1000.pt mine 2>&1 | grep '^{'
C2='{"BM":64,"BN1":128,"BK1":32,"BN2":64,"BK2":32,"w1":4,"s1":4,"w2":4,"s2":4,"nf":1}'
CFG=$C2 python vllm_bench.py results/route_st4b_1000.pt mine 2>&1 | grep '^{'
C3='{"BM":64,"BN1":128,"BK1":64,"BN2":64,"BK2":64,"w1":8,"s1":2,"w2":4,"s2":4,"nf":1}'
CFG=$C3 python vllm_bench.py results/route_st4b_1000.pt mine 2>&1 | grep '^{'

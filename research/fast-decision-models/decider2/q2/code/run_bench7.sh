cd ~/work/q2 && source ~/venv/bin/activate
while ! grep -q BENCH6_DONE run_eval3.log 2>/dev/null; do sleep 20; done
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode=arch=compute_86,code=sm_86 --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG -shared -Xcompiler -fPIC -Xptxas -v -o libq2gemm_full.so q2gemm.cu > build_full.log 2>&1; echo build rc $?
cp libq2gemm_full.so libq2gemm.so
python test_q2.py 0,1,2,3,4,5,6,7,8,9 > test_full.log 2>&1; grep -E 'FAIL|ALL OK' test_full.log | tail -2
Q2OUT=~/work/q2/res_gemm_var2.jsonl python bench_gemm.py var 140,400,1125,4125 > bench_var2.log 2>&1
Q2OUT=~/work/q2/res_gemm_base2.jsonl python bench_gemm.py base 1125 > bench_base2.log 2>&1
echo BENCH7_DONE

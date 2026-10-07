# full build + tests + GEMM benchmarks (base, var) at hobson shapes; resumable pieces write separate jsonl files
cd ~/work/q2 && source ~/venv/bin/activate
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode=arch=compute_86,code=sm_86 --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG -shared -Xcompiler -fPIC -Xptxas -v -o libq2gemm.so q2gemm.cu > build.log 2>&1; echo build rc $?
python test_q2.py 0,1,2,3,4,5,6,7,8,9 > test.log 2>&1; grep -E 'FAIL|ALL OK' test.log | tail -3
Q2OUT=~/work/q2/res_gemm_base.jsonl python bench_gemm.py base 140,400,1125,4125 > bench_base.log 2>&1; echo base done
Q2OUT=~/work/q2/res_gemm_var.jsonl python bench_gemm.py var 140,400,1125,4125 > bench_var.log 2>&1; echo var done

# box q2b: H2 libs + GPTQ codes (setup_q2.sh), q2gemm full build + tests, J15 DEV/EXIT suites into evalkit, question bundles
set -x
cp -n ~/work/h2/code/* ~/work/h2/ 2>/dev/null
cd ~/work/q2 && bash setup_q2.sh
source ~/venv/bin/activate
cd ~/work/q2
/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode=arch=compute_86,code=sm_86 --expt-relaxed-constexpr --expt-extended-lambda -DNDEBUG -shared -Xcompiler -fPIC -o libq2gemm.so q2gemm.cu > build.log 2>&1; echo q2gemm rc $?
python test_q2.py 0,4 > test.log 2>&1; tail -1 test.log
cp ~/work/j15/suites/DEV.jsonl ~/work/j15/suites/EXIT.jsonl ~/work/evalkit/suites/
cd ~/work/h2 && python prep_q.py > prep_q.log 2>&1; ls -la ~/work/h2/bundles.pt
echo SETUP_Q2B_DONE

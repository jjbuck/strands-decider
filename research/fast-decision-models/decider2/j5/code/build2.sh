cd ~/work/h2
NV="/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode=arch=compute_86,code=sm_86 --expt-relaxed-constexpr -DNDEBUG -shared -Xcompiler -fPIC"
$NV -I ~/work/cutlass/include -I ~/work/cutlass/tools/util/include -o libg2s4x.so ~/work/j5/g2s4x.cu > ~/work/j5/b_g2s4x.log 2>&1; echo rc $?

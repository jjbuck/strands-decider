# box-side: H2 runtime for deployed-kernel scoring/latency (J15's recipe). Run from ~/work with nohup.
set -x
source ~/venv/bin/activate
cd ~/work
cp -rn ~/work/h2/code/* ~/work/h2/
[ -f cutlass/include/cutlass/cutlass.h ] || git clone -q --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git cutlass
[ -f cutlass4/include/cutlass/cutlass.h ] || git clone -q --depth 1 https://github.com/NVIDIA/cutlass.git cutlass4
NV="/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode=arch=compute_86,code=sm_86 --expt-relaxed-constexpr -DNDEBUG -shared -Xcompiler -fPIC"
cd ~/work/h2
[ -f libg2s4.so ] || $NV -I ../cutlass/include -I ../cutlass/tools/util/include -o libg2s4.so g2s4.cu; echo g2s4 rc $?
[ -f libh2mix.so ] || $NV -I ../cutlass4/include -I ../cutlass4/tools/util/include -o libh2mix.so h2mix.cu; echo h2mix rc $?
[ -f libh2evt.so ] || $NV -I ../cutlass4/include -I ../cutlass4/tools/util/include -o libh2evt.so h2evt.cu; echo h2evt rc $?
ls -la ~/work/h2/*.so
echo SETUP_QRT_DONE

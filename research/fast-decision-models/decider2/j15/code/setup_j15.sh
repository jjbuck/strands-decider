# box-side: build CUTLASS libs, H1-recipe GPTQ codes. Run from ~/work/j15 with nohup.
set -x
source ~/venv/bin/activate
cd ~/work
nvidia-smi --query-gpu=name,clocks.max.sm,memory.total --format=csv
python -c "import torch, triton, fla; print(torch.__version__, triton.__version__, fla.__version__)"
[ -f cutlass/include/cutlass/cutlass.h ] || git clone -q --depth 1 --branch v3.5.1 https://github.com/NVIDIA/cutlass.git cutlass
[ -f cutlass4/include/cutlass/cutlass.h ] || git clone -q --depth 1 https://github.com/NVIDIA/cutlass.git cutlass4
(cd cutlass4 && git log -1 --format='%H %cd' && grep -m1 -E "CUTLASS_MAJOR|define CUTLASS_VERSION" include/cutlass/version.h)
NV="/usr/local/cuda/bin/nvcc -O3 -std=c++17 -gencode=arch=compute_86,code=sm_86 --expt-relaxed-constexpr -DNDEBUG -shared -Xcompiler -fPIC"
cd ~/work/h2
[ -f libg2s4.so ] || $NV -I ../cutlass/include -I ../cutlass/tools/util/include -o libg2s4.so g2s4.cu; echo g2s4 rc $?
[ -f libh2mix.so ] || $NV -I ../cutlass4/include -I ../cutlass4/tools/util/include -o libh2mix.so h2mix.cu; echo h2mix rc $?
[ -f libh2evt.so ] || $NV -I ../cutlass4/include -I ../cutlass4/tools/util/include -o libh2evt.so h2evt.cu; echo h2evt rc $?
ls -la ~/work/h2/*.so
cp -n ~/work/h1/precmap_*.json ~/work/h2/ 2>/dev/null
cd ~/work/h2
[ -f ~/work/h1/hess/H_23_Wd.pt ] || python h2gptq.py calib
[ -f codes_gptq_w8.pt ] || python h2gptq.py codes w8
[ -f codes_gptq_w4.pt ] || python h2gptq.py codes w4
ls -la ~/work/h2/codes_gptq_*.pt
cd ~/work/j15 && python prep_q1.py
echo SETUP_DONE

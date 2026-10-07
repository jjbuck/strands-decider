# latency (exclusive GPU): grid of exact lengths, 1 real question (details_match, 125 tok)
source ~/venv/bin/activate
cd ~/work/j15
G="16,32,64,128,256,400,600,1000,1500,2000,3000,4000,6000,9000"
echo "=== $(date +%T) grid"
python j15bench.py grid bf16 $G full 2>&1 | grep -v Loading
python j15bench.py grid b8,w4a4,k48,k64 $G full 2>&1 | grep -v Loading
python j15bench.py grid b8 32,128,400,1000,2000,4000,9000 pre:4,pre:6,pre:8,pre:10,pre:12,pre:14,pre:16,pre:18,pre:20,post:8,post:12 2>&1 | grep -v Loading
echo CHAIN3_DONE $(date +%T)

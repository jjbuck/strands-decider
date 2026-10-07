source ~/venv/bin/activate
cd ~/work/j15
python j15bench.py grid b8 32,128,400,1000,2000,4000,9000 post:14,post:16,post:18 2>&1 | grep -v Loading
python j15bench.py grid b8 16,64,256,600,1500,3000,6000 pre:8,pre:14,pre:16,post:16 2>&1 | grep -v Loading
echo CHAIN3B_DONE $(date +%T)
bash chain2.sh > chain2.log 2>&1

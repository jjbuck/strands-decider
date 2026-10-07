while pgrep -f "jobs/[l]aneA.sh" >/dev/null; do sleep 20; done
python g2map.py --sub realonly --signals qa7 --cfg 'S:r8|P8-25/qa7' --out res/tap8.json
python g2map.py --sub longall --signals qa7 --cfg 'Q:r8' --out res/longr8.json
python g2map.py --sub mid --signals qa7,rand,surp --cfg 'M4-50/(qa7|rand)|M42-25/(qa7|surp)|M2-25/qa7' --out res/nest_mid.json

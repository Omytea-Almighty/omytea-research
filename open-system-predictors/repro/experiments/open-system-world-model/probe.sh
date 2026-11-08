#!/bin/sh
# ROUND 2 optimisation probe on dev seed 101 (simple, tau=20): which setting lets L/K converge?
cd "$(dirname "$0")"; export PYTHONPATH=src OMP_NUM_THREADS=2
mkdir -p runs/probe
for m in L K U2 U1; do
  echo "$m 0.003 2000 identity"; echo "$m 0.01 2000 identity"; echo "$m 0.003 2000 random"; echo "$m 0.003 6000 identity"
done | xargs -P 4 -L 1 sh -c '.venv/bin/python -c "import torch; torch.set_num_threads(2); from osw.run import main; main()" --task simple --tau 20 --model $0 --seed 101 --lr $1 --steps $2 --init $3 --out runs/probe >> runs/probe/probe.log 2>&1'
echo "$(date) probe done" >> runs/probe/probe.log

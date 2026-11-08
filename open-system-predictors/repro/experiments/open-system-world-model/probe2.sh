#!/bin/sh
# ROUND 2 probe extension on dev seed 101 (simple, tau=20): combined setting lr 1e-2 x 6000 steps.
# Runs after probe.sh; same output dir, distinct filenames (tag -st6000-identity).
cd "$(dirname "$0")"; export PYTHONPATH=src OMP_NUM_THREADS=2
while pgrep -f "probe.sh" >/dev/null || pgrep -f "seed 101 --lr" >/dev/null; do sleep 30; done
for m in L K U2 U1; do echo "$m 0.01 6000 identity"; done \
 | xargs -P 4 -L 1 sh -c '.venv/bin/python -c "import torch; torch.set_num_threads(2); from osw.run import main; main()" --task simple --tau 20 --model $0 --seed 101 --lr $1 --steps $2 --init $3 --out runs/probe >> runs/probe/probe.log 2>&1'
echo "$(date) probe2 done" >> runs/probe/probe.log

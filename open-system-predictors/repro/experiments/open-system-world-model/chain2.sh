#!/bin/sh
# v1.1 round 2: development phase on seed 102, then the dev -> confirm gate. Stops there on purpose:
# freezing EXPERIMENT-PREREG-v1.1 and launching --phase confirm are separate, reviewed steps (DEVIATIONS.md D10).
cd "$(dirname "$0")"; export PYTHONPATH=src
log=runs/chain2.log; mkdir -p runs
echo "$(date -u +%FT%TZ) dev2 start $(git rev-parse --short HEAD)" >> $log
.venv/bin/python -m osw.queue --phase dev2 --workers 4 --threads 2 >> runs/dev2-queue.out 2>&1
rc=$?; echo "$(date -u +%FT%TZ) dev2 end rc=$rc" >> $log
.venv/bin/python -m osw.gate_dev >> $log 2>&1
rc=$?; echo "$(date -u +%FT%TZ) gate rc=$rc (0 = PASS); confirm NOT started automatically" >> $log

#!/bin/sh
# v1.1 confirmatory phase (seeds 0-9, one run per configuration), then the frozen analysis.
cd "$(dirname "$0")"; export PYTHONPATH=src
log=runs/chain3.log; mkdir -p runs
echo "$(date -u +%FT%TZ) confirm start $(git rev-parse --short HEAD)" >> $log
.venv/bin/python -m osw.queue --phase confirm --workers 4 --threads 2 >> runs/confirm-queue.out 2>&1
rc=$?; echo "$(date -u +%FT%TZ) confirm end rc=$rc" >> $log
.venv/bin/python -m osw.analyze runs/confirm runs/dev2 > runs/confirm/ANALYZE.out 2>&1
rc=$?; echo "$(date -u +%FT%TZ) analyze rc=$rc" >> $log

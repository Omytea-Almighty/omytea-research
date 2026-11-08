#!/bin/sh
# E2 + E2.1 (frozen beforehand), started at once in parallel with the confirmatory phase, per the E2.1 addendum:
# 2 workers x 1 thread so the confirmatory runs keep most of the machine. Then the pre-registered summary.
cd "$(dirname "$0")"; export PYTHONPATH=src
log=runs/chain5.log; mkdir -p runs
echo "$(date -u +%FT%TZ) E2 start $(git rev-parse --short HEAD)" >> $log
.venv/bin/python -m osw.queue_e2 --workers 2 --threads 1 >> runs/e2-queue.out 2>&1
rc=$?; echo "$(date -u +%FT%TZ) E2 end rc=$rc" >> $log
.venv/bin/python -m osw.queue_e2 --summary > runs/e2-summary.out 2>&1
rc=$?; echo "$(date -u +%FT%TZ) E2 summary rc=$rc" >> $log

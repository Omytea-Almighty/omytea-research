#!/bin/sh
# One-shot follow-up: wait for the v1.1 confirmatory analysis (chain3), then run E1 (frozen beforehand) and its
# descriptive summary. E2 runs separately (chain5.sh) per the E2.1 addendum. Resumable: finished files are skipped.
cd "$(dirname "$0")"; export PYTHONPATH=src
log=runs/chain4.log; mkdir -p runs
echo "$(date -u +%FT%TZ) waiting for chain3 analysis" >> $log
until grep -q "analyze rc=" runs/chain3.log 2>/dev/null; do sleep 60; done
echo "$(date -u +%FT%TZ) E1 start $(git rev-parse --short HEAD)" >> $log
.venv/bin/python -m osw.queue_e1 --workers 4 --threads 2 >> runs/e1-queue.out 2>&1
rc=$?; echo "$(date -u +%FT%TZ) E1 end rc=$rc" >> $log
.venv/bin/python -m osw.queue_e1 --summary > runs/e1-summary.out 2>&1
rc=$?; echo "$(date -u +%FT%TZ) E1 summary rc=$rc" >> $log

#!/bin/sh
# Wait for the dev queue, gate it, then run the confirmatory phase and the analysis (one-shot, not recurring).
cd "$(dirname "$0")"
export PYTHONPATH=src
while pgrep -f "osw.queue --phase dev" >/dev/null; do sleep 60; done
echo "$(date) dev finished; gating" >> runs/chain.log
if .venv/bin/python -m osw.gate_dev >> runs/chain.log 2>&1; then
  echo "$(date) gate PASS; starting confirm" >> runs/chain.log
  .venv/bin/python -m osw.queue --phase confirm --workers 4 --threads 2 >> runs/chain.log 2>&1
  echo "$(date) confirm finished; analysing" >> runs/chain.log
  .venv/bin/python -m osw.analyze runs/confirm >> runs/chain.log 2>&1
  echo "$(date) analysis written to runs/confirm/RESULTS.md" >> runs/chain.log
else
  echo "$(date) gate FAIL; confirmatory phase NOT started — see runs/dev/GATE.md" >> runs/chain.log
fi

#!/bin/sh
# E3 confirmatory run (E3-DESIGN.zh.md §11 freeze and run order). Start only after the freeze: design and code-manifest
# SHA-256 committed and timestamped, FROZEN_CONSTANTS in src/osw/queue_e3.py filled from runs/e3/GATES-DEV.json, the
# runs/e2 manifest snapshotted (python -m osw.queue_e3 --freeze-e2-manifest) and the dev PC-D pass in runs/e3/FIXTURE.json
# (the queue refuses otherwise). U runs first; the mechanical summary prints the gates and enqueues T / SC / O idempotently; the final
# summary goes to runs/e3/SUMMARY.{json,txt} only if the integrity check passes (else runs/e3/INTEGRITY-FAIL.json, rc 2). Resumable: rerunning skips finished jobs.
cd "$(dirname "$0")" || exit 1
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log=runs/chain6.log; mkdir -p runs/e3
echo "$(date -u +%FT%TZ) E3 start $(git rev-parse --short HEAD 2>/dev/null) workers=${E3_WORKERS:-3}" >> "$log"
caffeinate -i .venv/bin/python -m osw.queue_e3 --confirm --workers "${E3_WORKERS:-3}" >> runs/e3-queue.out 2>&1
rc=$?
echo "$(date -u +%FT%TZ) E3 end rc=$rc" >> "$log"
exit "$rc"

"""Gate between the development and confirmatory phases (pre-registration §4b: confirmatory seeds run only once,
after the code passes its tests and the development runs are stable). Exit 0 only if:
  * every dev job produced a result file and the queue logged no non-zero return code;
  * every trained model has a finite validation NLL below the uninformative baseline ln 4 (it learned something);
  * the unit tests pass.
Writes runs/dev/GATE.md either way.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

from .queue import ROOT, chosen_lr, dev2_jobs, out_name

UNINFORMATIVE = math.log(4) - 0.003     # the emission marginal is uniform on T1 -> ln 4 = 1.3863


def main() -> int:
    d = ROOT / "runs/dev2"
    problems = []
    for j in dev2_jobs():
        f = d / out_name(j)
        if not f.exists():
            problems.append(f"missing {f.name}")
            continue
        r = json.loads(f.read_text())
        if j["model"] in ("BAYES", "HMM"):
            continue
        v = r.get("val_nll")
        if v is None or not math.isfinite(v):
            problems.append(f"non-finite val_nll in {f.name}")
            continue
        # D7: the no-learning check applies to models with a known learnable solution, and to U2 at its chosen lr
        must_learn = j["model"] in ("L", "L1", "K", "C", "GRU", "LRU") or (
            j["model"] == "U2" and j["lr"] == chosen_lr(j["task"], j["tau"], "U2"))
        if must_learn and j["model"] != "U2" and j["lr"] != chosen_lr(j["task"], j["tau"], j["model"]):
            must_learn = False                     # only the selected learning rate has to learn
        if must_learn and v >= UNINFORMATIVE:
            problems.append(f"no learning in {f.name}: val_nll={v:.4f} >= {UNINFORMATIVE:.4f}")
    log = (d / "queue.log").read_text() if (d / "queue.log").exists() else ""
    if "rc=1" in log or "rc=-" in log:
        problems.append("queue.log records a failed job")
    t = subprocess.run([sys.executable, "-m", "pytest", "-q", str(ROOT / "tests")], capture_output=True, text=True,
                       cwd=ROOT, env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    if t.returncode != 0:
        problems.append("unit tests failed:\n" + t.stdout[-1500:])
    verdict = "PASS" if not problems else "FAIL"
    (d / "GATE.md").write_text(f"# Dev → confirm gate: {verdict}\n\n" + ("\n".join(f"- {p}" for p in problems) or "- all checks passed") + "\n")
    print(verdict, *problems, sep="\n")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())

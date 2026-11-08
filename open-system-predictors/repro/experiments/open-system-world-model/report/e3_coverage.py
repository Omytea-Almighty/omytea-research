"""Coverage of the E3 BCa intervals at the registered fallback level (post hoc, measured at release, 2026-10-31).

The E3 design (§8) switches the difference rules to 97.5% intervals when the development-gate coverage simulation gives
less than 0.94 at 95%, and asks for the coverage after the fallback to be reported. The development gate stored only the
95% value. This script calls the frozen simulation osw.sim_e3.coverage_sim unchanged, with its own seed, once at each
level, so that both numbers can be reported.

  PYTHONPATH=src .venv/bin/python report/e3_coverage.py      # writes runs/robustness/E3_COVERAGE.json
"""
from __future__ import annotations

import json
from pathlib import Path

from osw import sim_e3, summary_e3

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/robustness/E3_COVERAGE.json"


def main() -> None:
    out = {"code": "osw.sim_e3.coverage_sim (frozen E3 code manifest), called unchanged except for the level argument",
           "seed": sim_e3.SIM_SEED, "n_boot": summary_e3.N_BOOT,
           "levels": {str(lvl): sim_e3.coverage_sim(level=lvl) for lvl in (0.95, 0.975)}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

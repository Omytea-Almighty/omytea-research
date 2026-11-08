"""Exact-enumeration bootstrap intervals for Tables 3 (E1) and 4 (E2) — pre-publication erratum r9 m2 / r10 m5.

The frozen summaries (runs/e1/SUMMARY.json, runs/e2/SUMMARY.json) drew 10,000 Monte Carlo resamples. With 5 folds (E1)
or 3–5 seeds (E2) the full set of ordered resamples is small (5^5 = 3,125 or 3^3 = 27), so this script enumerates every
ordered resample instead and takes the same percentiles (np.percentile / np.nanpercentile, linear interpolation, 2.5 and
97.5). Everything else is the frozen code path: osw.queue_e1.summary's validation-NLL learning-rate choice, fold means
and differences to L; osw.queue_e2's collect(), at_budget() and the shared seed resample of compare(). The registered
labels (E2 "advantage") are not recomputed here; they stay as in the frozen summaries.

  PYTHONPATH=src .venv/bin/python report/exact_ci.py      # writes runs/robustness/EXACT_CI.json
"""
from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/robustness/EXACT_CI.json"


def e1_exact() -> dict:
    """Per model: Δ extrapolation NLL vs L, fold-bootstrap interval over all 5^5 ordered fold resamples."""
    R = [json.loads(f.read_text()) for f in (ROOT / "runs/e1").glob("e1-*.json")]
    best = {}
    for r in R:                                                      # as queue_e1.summary: lr by validation NLL
        k = (r["held_out"], r["seed"], r["model"])
        v = r["val_nll"] if r["val_nll"] is not None else -np.inf
        if k not in best or v < best[k][0]:
            best[k] = (v, r)
    cell = defaultdict(dict)
    for (f, s, m), (_, r) in best.items():
        cell[m][(f, s)] = r
    ref = cell.get("L", {})
    out = {}
    for m, rs in cell.items():
        common = sorted(set(rs) & set(ref))
        d = {key: rs[key]["windows"]["extrap_7_12"]["nll"] - ref[key]["windows"]["extrap_7_12"]["nll"] for key in common}
        folds = sorted({f for f, _ in common})
        fm = np.array([np.mean([v for (f, _), v in d.items() if f == fo]) for fo in folds]) if folds else np.array([])
        if len(fm) > 1:
            means = np.array([fm[list(idx)].mean() for idx in itertools.product(range(len(fm)), repeat=len(fm))])
            lo, hi = np.percentile(means, [2.5, 97.5])
            out[m] = {"delta_vs_L": float(np.mean(list(d.values()))), "ci_exact": [float(lo), float(hi)],
                      "n_resamples": int(len(means)), "n_folds": len(fm)}
    return out


def e2_exact() -> dict:
    """Per arm and budget: HQ − max(GRU, LRU) and HQ − HMM over every ordered seed resample (shared across families)."""
    from osw import queue_e2 as Q
    results = [json.loads(f.read_text()) for f in (ROOT / "runs/e2").glob("e2-*.json")]
    cell = Q.collect(results)
    out = {}
    for arm in ("A1", "A2", "B", "B2", "B3"):
        if arm not in cell:
            continue
        for b in Q.BUDGETS[arm]:
            for comp, a, others in (("HQ_vs_flexible_classical", "HQ", ["GRU", "LRU"]), ("HQ_vs_HMM", "HQ", ["HMM"])):
                ca = cell[arm]
                seeds = sorted({s for fam in [a, *others] for size in ca.get(fam, {}).values() for s in size})
                point = Q.at_budget(ca[a], b) - max(Q.at_budget(ca[o], b) for o in others)
                boots = [Q.at_budget(ca[a], b, list(ss)) - max(Q.at_budget(ca[o], b, list(ss)) for o in others)
                         for ss in itertools.product(seeds, repeat=len(seeds))]
                lo, hi = np.nanpercentile(boots, [2.5, 97.5])
                out[f"{arm}.{comp}@{b}"] = {"delta": float(point), "ci_exact": [float(lo), float(hi)],
                                           "n_resamples": len(boots), "n_seeds": len(seeds),
                                           "n_nan": int(np.isnan(boots).sum())}
    return out


def main():
    res = {"method": "every ordered resample enumerated; percentiles 2.5 / 97.5 (numpy linear interpolation)",
           "e1": e1_exact(), "e2": e2_exact()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps({"e1": {k: v["ci_exact"] for k, v in res["e1"].items()},
                      "e2": {k: v["ci_exact"] for k, v in res["e2"].items()}}, indent=1))


if __name__ == "__main__":
    main()

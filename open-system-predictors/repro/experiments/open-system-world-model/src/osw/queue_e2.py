"""E2 queue and pre-registered summary (design: E2-DESIGN.zh.md). Run only after v1.1 confirm and E1 have finished.

  python -m osw.queue_e2 --workers 4      # every (arm, family, size, seed, lr) once; resumable
  python -m osw.queue_e2 --summary        # writes runs/e2/SUMMARY.md and SUMMARY.json

Learning rate: for each (arm, family, size, seed) the run with the lower validation NLL is kept.
Arm A primary metric: captured fraction CF = 1 - mean_{k<=10} KL(exact || model) / mean_{k<=10} KL(exact || marginal).
Arm B primary metric: RPSS over weeks 3-6 = 1 - RPS(model) / RPS(uniform climatology), test period only.
Each family's metric is interpolated linearly in log(parameters) at fixed budgets; seeds are bootstrapped.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from .models_e2 import SIZES
from .queue import ROOT

OUT = ROOT / "runs/e2"
LRS, STEPS = [3e-3, 1e-2], 6000
SEEDS = {"A1": range(5), "A2": range(5), "B": range(3), "B2": range(3), "B3": range(3)}   # B2/B3: E2.1 addendum
BUDGETS = {"A1": [64, 256], "A2": [64, 256], "B": [150, 500], "B2": [150, 500], "B3": [150, 500]}
TRAINED = ["HQ", "GRU", "LRU"]
ADVANTAGE = 0.10                      # pre-registered margin on the captured fraction / RPSS


def jobs():
    js = []
    for arm, seeds in SEEDS.items():
        for s in seeds:
            for fam in TRAINED:
                for size, lr in itertools.product(SIZES[fam], LRS):
                    js.append(dict(arm=arm, family=fam, size=size, seed=s, lr=lr))
            for size in SIZES["HMM"]:
                js.append(dict(arm=arm, family="HMM", size=size, seed=s, lr=1e-2))
            if arm.startswith("B"):
                js += [dict(arm=arm, family=f, size=0, seed=s, lr=1e-2) for f in ("CLIM", "EMP", "MARKOV1")]
    return js


def name(j):
    return f"e2-{j['arm']}-{j['family']}{j['size'] or ''}-s{j['seed']}-lr{j['lr']:g}.json"


def run(j, threads):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS=str(threads))
    cmd = [sys.executable, "-c", f"import torch; torch.set_num_threads({threads}); from osw.run_e2 import main; main()",
           "--arm", j["arm"], "--family", j["family"], "--size", str(j["size"]), "--seed", str(j["seed"]),
           "--lr", str(j["lr"]), "--steps", str(STEPS), "--out", str(OUT)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    return j, r.returncode, (r.stdout.strip().splitlines() or [""])[-1], r.stderr[-600:]


def metric(r):
    pk = r["per_k"]
    if r["arm"].startswith("B"):
        return 1 - np.mean(pk["rps"][2:6]) / np.mean(pk["rps_clim"][2:6])
    return 1 - np.mean(pk["kl"][:10]) / np.mean(pk["kl_marginal"][:10])


def collect(results):
    best = {}
    for r in results:                                               # keep the lr with the lower validation NLL
        k = (r["arm"], r["family"], r["size"], r["seed"])
        v = r["val_nll"] if r["val_nll"] is not None else -np.inf
        if k not in best or v < best[k][0]:
            best[k] = (v, r)
    cell = defaultdict(lambda: defaultdict(dict))                  # arm -> family -> size -> seed -> (params, metric)
    for (arm, fam, size, seed), (_, r) in best.items():
        cell[arm][fam].setdefault(size, {})[seed] = (r["n_params"], metric(r))
    return cell


def at_budget(sizes: dict, budget, seeds=None):
    """Family curve (one value per size: mean over the given seeds) interpolated in log(params) at the budget."""
    pts = []
    for size, by_seed in sizes.items():
        use = [by_seed[s] for s in (seeds if seeds is not None else by_seed) if s in by_seed]
        if use:
            pts.append((np.log(use[0][0]), float(np.mean([m for _, m in use]))))
    pts.sort()
    x = np.log(budget)
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return np.nan
    return float(np.interp(x, [p for p, _ in pts], [m for _, m in pts]))


def compare(cell_arm, budget, a, others, rng, reps=10_000):
    """Delta = metric_a(budget) - max over `others`; bootstrap over seeds (resampled once per replicate, shared)."""
    seeds = sorted({s for fam in [a, *others] for size in cell_arm.get(fam, {}).values() for s in size})
    point = at_budget(cell_arm[a], budget) - max(at_budget(cell_arm[o], budget) for o in others)
    boots = []
    for _ in range(reps):
        ss = list(rng.choice(seeds, size=len(seeds), replace=True))
        boots.append(at_budget(cell_arm[a], budget, ss) - max(at_budget(cell_arm[o], budget, ss) for o in others))
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return dict(delta=point, ci=[float(lo), float(hi)], advantage=bool(point >= ADVANTAGE and lo > 0))


def summary(results=None):
    results = results if results is not None else [json.loads(f.read_text()) for f in OUT.glob("e2-*.json")]
    cell = collect(results)
    rng = np.random.default_rng(20261022)
    res, L = {}, ["# E2 summary (design E2-DESIGN.zh.md)", ""]
    for arm in ("A1", "A2", "B", "B2", "B3"):
        if arm not in cell:
            continue
        res[arm] = {"curves": {f: {str(s): {"params": v[next(iter(v))][0], "mean": float(np.mean([m for _, m in v.values()])),
                                           "n": len(v)} for s, v in sizes.items()} for f, sizes in cell[arm].items()}}
        for b in BUDGETS[arm]:
            res[arm][f"HQ_vs_flexible_classical@{b}"] = compare(cell[arm], b, "HQ", ["GRU", "LRU"], rng)
            res[arm][f"HQ_vs_HMM@{b}"] = compare(cell[arm], b, "HQ", ["HMM"], rng)
        metric_name = "RPSS steps 3-6" if arm.startswith("B") else "captured fraction (k <= 10)"
        L += [f"## {arm} — {metric_name}", "", "| family | size | params | mean | n |", "|---|---|---|---|---|"]
        for f, sizes in sorted(res[arm]["curves"].items()):
            for s, v in sorted(sizes.items(), key=lambda kv: kv[1]["params"]):
                L.append(f"| {f} | {s} | {v['params']} | {v['mean']:.4f} | {v['n']} |")
        for k, v in res[arm].items():
            if k != "curves":
                L.append(f"- **{k}**: Δ = {v['delta']:+.4f}, 95% CI [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] → "
                         f"{'ADVANTAGE (≥ 0.10, CI > 0)' if v['advantage'] else 'no advantage by the pre-registered rule'}")
        L.append("")
    (OUT / "SUMMARY.json").write_text(json.dumps(res, indent=1))
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.summary:
        return summary()
    todo = [j for j in jobs() if not (OUT / name(j)).exists()]
    with open(OUT / "queue.log", "a") as log, ThreadPoolExecutor(a.workers) as ex:
        log.write(f"start todo={len(todo)}/{len(jobs())}\n"); log.flush()
        for fut in as_completed([ex.submit(run, j, a.threads) for j in todo]):
            j, rc, last, err = fut.result()
            log.write(f"rc={rc} {name(j)} :: {last}\n" + (f"   stderr: {err}\n" if rc else "")); log.flush()
        log.write("end\n")


if __name__ == "__main__":
    main()

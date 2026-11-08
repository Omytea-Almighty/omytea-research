"""E1 (exploratory) job queue and summary, per E1-DESIGN.zh.md. Run only after the v1.1 confirmatory phase has finished.

  python -m osw.queue_e1 --workers 4            # runs every (fold, seed, model, lr) once; resumable
  python -m osw.queue_e1 --summary              # writes runs/e1/SUMMARY.md (descriptive only)

Learning rate: for each (fold, seed, model) the run with the lower validation NLL is kept (validation pedestrians are
disjoint from training pedestrians and from the held-out scene, so the test scene never influences the choice).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .queue import ROOT
from .tasks_ethucy import SCENES

TRAINED = ["U2", "L", "K", "C", "GRU", "LRU"]
FIXED = ["HMM", "MARG", "MARKOV1"]
SEEDS, LRS, STEPS = [0, 1, 2], [3e-3, 1e-2], 6000
OUT = ROOT / "runs/e1"


def jobs():
    js = [dict(held_out=f, model=m, seed=s, lr=lr) for f, s, m, lr in itertools.product(SCENES, SEEDS, TRAINED, LRS)]
    js += [dict(held_out=f, model=m, seed=s, lr=1e-2) for f, s, m in itertools.product(SCENES, SEEDS, FIXED)]
    return js


def name(j):
    return f"e1-{j['held_out']}-{j['model']}-s{j['seed']}-lr{j['lr']:g}-st{STEPS}.json"


def run(j, threads):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS=str(threads))
    cmd = [sys.executable, "-c", f"import torch; torch.set_num_threads({threads}); from osw.run_ethucy import main; main()",
           "--held-out", j["held_out"], "--model", j["model"], "--seed", str(j["seed"]), "--lr", str(j["lr"]),
           "--steps", str(STEPS), "--out", str(OUT)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    return j, r.returncode, (r.stdout.strip().splitlines() or [""])[-1], r.stderr[-600:]


def summary():
    R = [json.loads(f.read_text()) for f in OUT.glob("e1-*.json")]
    best = {}
    for r in R:                                                      # lr selection on validation NLL
        k = (r["held_out"], r["seed"], r["model"])
        v = r["val_nll"] if r["val_nll"] is not None else -np.inf
        if k not in best or v < best[k][0]:
            best[k] = (v, r)
    cell = defaultdict(dict)                                        # model -> (fold, seed) -> extrap NLL
    for (f, s, m), (_, r) in best.items():
        cell[m][(f, s)] = r
    rng = np.random.default_rng(20261021)
    L = ["# E1 (exploratory, descriptive only) — ETH/UCY relative headings, LOSO", "",
         "Design: E1-DESIGN.zh.md (SHA-256 f06879ca...). Not part of H1–H5; no claims.", "",
         "| model | params | extrap NLL [7,12] | in-range NLL [1,6] | RPS moving [7,12] | ECE [7,12] | Δ vs L (extrap NLL) | 95% fold-bootstrap CI | n |",
         "|---|---|---|---|---|---|---|---|---|"]
    ref = cell.get("L", {})
    for m in sorted(cell, key=lambda m: np.mean([r["windows"]["extrap_7_12"]["nll"] for r in cell[m].values()])):
        rs = cell[m]
        g = lambda w, k: np.mean([r["windows"][w][k] for r in rs.values()])
        common = sorted(set(rs) & set(ref))
        d = {key: rs[key]["windows"]["extrap_7_12"]["nll"] - ref[key]["windows"]["extrap_7_12"]["nll"] for key in common}
        folds = sorted({f for f, _ in common})
        fold_mean = np.array([np.mean([v for (f, _), v in d.items() if f == fo]) for fo in folds]) if folds else np.array([])
        ci = (np.percentile(rng.choice(fold_mean, (10_000, len(fold_mean))).mean(1), [2.5, 97.5])
              if len(fold_mean) > 1 else [np.nan, np.nan])
        L.append(f"| {m} | {next(iter(rs.values()))['n_params']} | {g('extrap_7_12', 'nll'):.4f} | {g('in_range_1_6', 'nll'):.4f} | "
                 f"{g('extrap_7_12', 'rps_moving'):.4f} | {g('extrap_7_12', 'ece'):.4f} | "
                 f"{(np.mean(list(d.values())) if d else float('nan')):+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {len(rs)} |")
    (OUT / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


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
        for j, rc, last, err in ex.map(lambda j: run(j, a.threads), todo):
            log.write(f"rc={rc} {name(j)} :: {last}\n" + (f"   stderr: {err}\n" if rc else "")); log.flush()
        log.write("end\n")


if __name__ == "__main__":
    main()

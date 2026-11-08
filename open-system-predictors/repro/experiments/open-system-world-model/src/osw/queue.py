"""Resumable job queue for the experiment phases (dev / confirm). Detached-safe: every finished run is a file,
so a restarted queue skips completed jobs. Writes <out>/STATUS.md after each job for hand-off.

  python -m osw.queue --phase dev --workers 4
  python -m osw.queue --phase confirm --workers 4
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LRS = [1e-3, 3e-3]
CORE = ["U2", "L", "K", "C", "GRU", "LRU"]
SWEEP = ["U1", "U4", "U8"]                       # environment-dimension sweep for H2 (simple task only)
# Round-2 task plan (DEVIATIONS.md D6): H1 on simple/metastable tau=20 (as frozen), H3 on metastable tau_slow=80,
# simple tau=80 and action tau=20 reported as secondary results.
# D9: L1 (one jump operator, parameter-matched to C) on the two H3 configurations.
PLAN2 = [("simple", 20, CORE + SWEEP + ["L1"]), ("metastable", 20, CORE), ("metastable", 80, CORE + ["L1"]),
         ("simple", 80, CORE), ("action", 20, CORE)]
DEV2_SEED = 102


def train_config():
    """Training settings chosen in DEVIATIONS.md D5 (written to runs/TRAIN_CONFIG.json after the seed-101 probe)."""
    f = ROOT / "runs/TRAIN_CONFIG.json"
    return json.loads(f.read_text()) if f.exists() else {"steps": 2000, "init": "identity", "lrs": LRS}


def dev_jobs():
    jobs = []
    for task, tau, models in [("simple", 20, CORE + SWEEP), ("metastable", 20, CORE), ("action", 20, CORE)]:
        for m, lr in itertools.product(models, LRS):
            jobs.append(dict(task=task, tau=tau, model=m, seed=100, lr=lr))
    for task in ["simple", "metastable", "action"]:
        for m in ["BAYES", "HMM"]:
            jobs.append(dict(task=task, tau=20, model=m, seed=100, lr=3e-3))
    return jobs


def dev2_jobs():
    cfg = train_config()
    jobs = []
    for task, tau, models in PLAN2:
        for m, lr in itertools.product(models, cfg["lrs"]):
            jobs.append(dict(task=task, tau=tau, model=m, seed=DEV2_SEED, lr=lr))
        for m in ["BAYES", "HMM"]:
            jobs.append(dict(task=task, tau=tau, model=m, seed=DEV2_SEED, lr=3e-3))
    return jobs


def chosen_lr(task, tau, model):
    """Learning rate selected on the round-2 dev seed's validation NLL (pre-registration §4)."""
    best = None
    for lr in train_config()["lrs"]:
        f = ROOT / "runs/dev2" / out_name(dict(task=task, tau=tau, model=model, seed=DEV2_SEED, lr=lr))
        if f.exists():
            v = json.loads(f.read_text())["val_nll"]
            if v is not None and (best is None or v < best[0]):
                best = (v, lr)
    if best is None:
        raise SystemExit(f"no dev result for {task}-{tau}-{model}; run the dev phase first")
    return best[1]


def confirm_jobs():
    jobs = []
    for seed in range(10):                                   # confirmatory seeds 0-9, one run each
        for task, tau, models in PLAN2:
            for m in models + ["HMM", "BAYES"]:
                lr = 3e-3 if m in ("HMM", "BAYES") else chosen_lr(task, tau, m)
                jobs.append(dict(task=task, tau=tau, model=m, seed=seed, lr=lr))
    return jobs


def out_name(j):
    cfg = train_config()
    # must match run.py's naming exactly: run.py tags every model (HMM and BAYES included) by --steps/--init
    tag = "" if (cfg["steps"] == 2000 and cfg["init"] == "identity") else f"-st{cfg['steps']}-{cfg['init']}"
    return f"{j['task']}-{j['tau']:g}-{j['model']}-s{j['seed']}-lr{j['lr']:g}{tag}.json"


def run_job(j, out, threads):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    cfg = train_config()
    cmd = [sys.executable, "-c", f"import torch; torch.set_num_threads({threads}); from osw.run import main; main()",
           "--task", j["task"], "--tau", str(j["tau"]), "--model", j["model"], "--seed", str(j["seed"]),
           "--lr", str(j["lr"]), "--steps", str(cfg["steps"]), "--init", cfg["init"], "--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    return j, r.returncode, (r.stdout.strip().splitlines() or [""])[-1], r.stderr[-800:]


def write_status(out, jobs, started):
    done = [j for j in jobs if (out / out_name(j)).exists()]
    lines = [f"# Queue status — {out.name}", "",
             f"updated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}  ·  started: {started}",
             f"done: {len(done)} / {len(jobs)}", "", "Remaining:"]
    lines += [f"- {out_name(j)}" for j in jobs if j not in done][:40]
    (out / "STATUS.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["dev", "dev2", "confirm"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    out = ROOT / "runs" / a.phase
    out.mkdir(parents=True, exist_ok=True)
    jobs = {"dev": dev_jobs, "dev2": dev2_jobs, "confirm": confirm_jobs}[a.phase]()
    # slowest first so the long U8 runs do not end up alone at the tail
    jobs.sort(key=lambda j: {"U8": 0, "U4": 1, "L": 2}.get(j["model"], 3))
    todo = [j for j in jobs if not (out / out_name(j)).exists()]
    started = time.strftime("%Y-%m-%d %H:%M:%S %Z")
    write_status(out, jobs, started)
    log = open(out / "queue.log", "a")
    log.write(f"{started} start phase={a.phase} todo={len(todo)}/{len(jobs)} workers={a.workers}\n"); log.flush()
    with ThreadPoolExecutor(a.workers) as ex:
        # as_completed, not map: STATUS.md must refresh whenever any job finishes, not in submission order
        for fut in as_completed([ex.submit(run_job, j, out, a.threads) for j in todo]):
            j, rc, last, err = fut.result()
            log.write(f"{time.strftime('%H:%M:%S')} rc={rc} {out_name(j)} :: {last}\n")
            if rc != 0:
                log.write(f"   stderr: {err}\n")
            log.flush()
            write_status(out, jobs, started)
    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} end phase={a.phase}\n"); log.close()


if __name__ == "__main__":
    main()

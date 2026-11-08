"""E3 job lists, resumable worker pool and gated enqueueing (design: E3-DESIGN.zh.md §5, §8, §11, §13, §16).

  python -m osw.queue_e3 --dry-run            # job counts per block and projected CPU (U 200, T 33, SC 135, O 10, DEV 84)
  python -m osw.queue_e3 --dev --workers 3    # development jobs (seeds 200-204) and REF fixtures; DEV2/DEV3 once the
                                              # G10 learning rates are known (FROZEN_CONSTANTS or GATES-DEV.json)
  python -m osw.queue_e3 --redev              # the single G2 fallback rerun on seeds 205-209
  python -m osw.queue_e3 --dev-gate --tests-ok  # G1-G10 from the dev records -> GATES-DEV.json, FIXTURE.json (dev PC-D)
  python -m osw.queue_e3 --freeze-e2-manifest # at freeze: snapshot the runs/e2 manifest for the integrity check
  python -m osw.queue_e3 --confirm            # U, then the mechanical gates, then T / SC / O as gated; final summary
  python -m osw.queue_e3 --summary            # summary_e3 on the confirmatory records present, no running

Every confirmatory job is built from FROZEN_CONSTANTS, which dev_gate() derives (G2-G10) and which are pasted into this
file before the code manifest is hashed; while any of them is None no confirmatory job list is built. confirm() refuses
to start without the dev PC-D pass in FIXTURE.json and without the freeze-time runs/e2 manifest. Each job runs as its own `python -m osw.run_e3`
subprocess with one thread. A job whose result exists with the current design and code-manifest SHA-256 is skipped; a
result with other SHAs is kept as <name>.stale-<code sha> and the job rerun; a failed job is rerun once (attempt 2) and
then stays failed (summary_e3 imputes it). Gated enqueueing is recomputed from the result files on every invocation,
so it is idempotent; SENS is enqueued after the SC ladder is complete because its rung depends on that ladder.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from fractions import Fraction
from math import ceil

from . import oom_e3, run_e3, sim_e3, summary_e3, tasks_e3

ROOT = run_e3.ROOT
OUT = ROOT / "runs/e3"
CONFIRM, DEV, REDEV = tuple(range(20, 30)), tuple(range(200, 205)), tuple(range(205, 210))
DEV3 = DEV[:3]
DEV_BLOCKS = ("DEV1", "DEV2", "DEV3", "FIX", "REDEV")
C0_LRS = (3e-3, 1e-2)
SIGMAS_A2 = (0.05, 0.1, 0.2, 0.3)
SIGMA_A3 = 0.0375
TAU0_GRID = (0.1, 0.3, 0.6, 0.9)
LR_GRID = (1e-3, 3e-3, 1e-2, 3e-2)
LADDER = {f: [(s, "CPLUS-M" if s <= 8 else "CPLUS") for s in (2, 4, 8, 16, 32)] for f in ("GRU", "LRU")}
DEV_R1, DEV_RS = 3, 1                       # dev C1 / S restarts; R1 and RS are set from their per-restart CPU (G3)
TRIM_ORDER = ("O", "D4", "SENS", "C2_0.2", "A3", "D3", "TOP32", "SC")    # G9 cuts in order; "SC" = terminal rule
G9_TARGET_H, G9_CAP_H = 28.0, 32.0

FROZEN_CONSTANTS = {
    "R1": 6,                                                  # C1 restarts in {3, 4, 5, 6} (G3)
    "RS": 6,                                                  # S restarts (G3)
    "TAU0": 0.1,                                                # S tau0 from TAU0_GRID (G3)
    "CPLUS_LRS": {"GRU": [0.01, 0.03], "LRU": [0.01, 0.03]},                     # two lrs per family from LR_GRID (G10)
    "R_C": {"GRU2": 5, "GRU4": 5, "GRU8": 5, "GRU16": 5, "GRU32": 4, "LRU2": 7, "LRU4": 7, "LRU8": 7, "LRU16": 6, "LRU32": 6},   # replicas per rung (G10)
    "CPU_FLOOR": 602.4304040000001,                                           # A2 HQ3-M dev median cpu_s (G10)
    "FT_STEPS": 3000,                                            # M and C2 fine-tune steps: 3000, or 6000 (G2)
    "MEDIAN_CLAUSE": True,                                       # PC-A median clause kept (G2)
    "G4_PASS": True,                                             # A2w O05 dev median >= 0.50; False removes A2w (G4)
    "G6_PASS": True,                                             # OOM3 dev check; False removes H7 and D2 (G6)
    "H6_REGISTERED": False,                                       # G8 planted simulation passed; False: H6 descriptive
    "CI_LEVEL": 0.975,                                            # 0.95, or 0.975 if BCa coverage < 0.94 (G8)
    "N_TEST_A2W": 10000,                                          # 10,000, or 20,000 if the O05 bootstrap SD > 0.02 (G8)
    "N_TEST_A3": 4000,                                           # 4,000, or 8,000 likewise (G8)
    "TRIMS": ["O", "D4", "SENS", "C2_0.2", "A3", "D3", "TOP32"],                                               # G9 cuts taken from TRIM_ORDER ([] if none)
}
PLACEHOLDER = {"R1": 3, "RS": 3, "TAU0": 0.3, "CPLUS_LRS": {"GRU": [3e-3, 1e-2], "LRU": [3e-3, 1e-2]},
               "R_C": {k: 1 for k in FROZEN_CONSTANTS["R_C"]}, "CPU_FLOOR": 330.0, "FT_STEPS": 3000,
               "MEDIAN_CLAUSE": True, "G4_PASS": True, "G6_PASS": True, "H6_REGISTERED": True, "CI_LEVEL": 0.95,
               "N_TEST_A2W": tasks_e3.PROC["A2w"]["n_test"], "N_TEST_A3": tasks_e3.PROC["A3"]["n_test"], "TRIMS": []}


def missing_constants(C=None) -> list:
    C = FROZEN_CONSTANTS if C is None else C
    miss = []
    for k, v in C.items():
        if isinstance(v, dict):
            miss += [f"{k}.{kk}" for kk, vv in v.items() if vv is None]
        elif v is None:
            miss.append(k)
    return miss


def _check(C):
    miss = missing_constants(C)
    if miss:
        raise ValueError(f"refusing to build confirmatory jobs: frozen constants not set: {', '.join(miss)}")
    bad = [t for t in C["TRIMS"] if t not in TRIM_ORDER]
    if bad:
        raise ValueError(f"unknown trims {bad}")


def J(block, arm, process, family, size, recipe, seed, sigma=None, **kw):
    return dict(block=block, arm=arm, process=process, family=family, size=size, recipe=recipe, seed=seed,
                sigma=sigma, **kw)


def _ntest(js, C):
    """The G8 N_test lock on every A2w / A3 confirmatory job (train and validation draws are unaffected)."""
    nt = {"A2w": C["N_TEST_A2W"], "A3": C["N_TEST_A3"]}
    for j in js:
        if j["process"] in nt:
            j["n_test"] = int(nt[j["process"]])
    return js


# ---------------------------------------------------------------- job lists
def jobs_U(C):
    _check(C)
    ft, cut = C["FT_STEPS"], set(C["TRIMS"])
    js = []
    for s in CONFIRM:
        js.append(J("U", "L-A2-M", "A2", "HQ", 3, "M", s, steps=ft))
        js += [J("U", "L-A2-C0", "A2", "HQ", 3, "C0", s, lr=lr, steps=6000) for lr in C0_LRS]
        js.append(J("U", "L-A2-C1", "A2", "HQ", 3, "C1", s, R=C["R1"]))
        js.append(J("U", "L-A2-S", "A2", "HQ", 3, "S", s, R=C["RS"], tau0=C["TAU0"]))
        js += [J("U", "L-A2-C2", "A2", "HQ", 3, "C2", s, sigma=sg, steps=ft) for sg in SIGMAS_A2
               if not (sg == 0.2 and "C2_0.2" in cut)]
        js.append(J("U", "L-A1-M", "A1", "HQ", 2, "M", s, steps=ft))
        js += [J("U", "L-A1-C0", "A1", "HQ", 2, "C0", s, lr=lr, steps=6000) for lr in C0_LRS]
        js.append(J("U", "L-A1-C2", "A1", "HQ", 2, "C2", s, sigma=0.05, steps=ft))
        js += [J("U", "PC-B", "A1", "HMM", n, "HMM-C", s) for n in (3, 4)]
        js.append(J("U", "PC-O", "A1", "OOM", 3, "OOM-C", s))
        js += [J("U", "REF", p, "REF", 0, "REF", s) for p in ("A1", "A2", "A2w", "A3")]
    return _ntest(js, C)


def jobs_T(C):
    _check(C)
    ft, cut = C["FT_STEPS"], set(C["TRIMS"])
    js = []
    if C["G4_PASS"]:                                            # G4 failure removes A2w before freeze
        for s in CONFIRM:
            js += [J("T", "A2w-M", "A2w", "HQ", 3, "M", s, steps=ft),
                   J("T", "A2w-C2", "A2w", "HQ", 3, "C2", s, sigma=0.05, steps=ft)]
    if "A3" not in cut:
        for s in CONFIRM[:5]:
            js += [J("T", "A3-M", "A3", "HQ", 4, "M", s, steps=ft),
                   J("T", "A3-C2", "A3", "HQ", 4, "C2", s, sigma=SIGMA_A3, steps=ft)]
    if "D4" not in cut:
        js += [J("T", "D4", "A2", "HQ", 4, "M", s, steps=ft) for s in CONFIRM[:3]]
    return _ntest(js, C)


def jobs_SC(C):
    """SC without SENS (whose rung depends on the finished ladder; see jobs_SENS)."""
    _check(C)
    cut, lrs = set(C["TRIMS"]), C["CPLUS_LRS"]
    if "SC" in cut:                                             # G9 terminal rule
        return []
    js = []
    for s in CONFIRM:
        for f in ("GRU", "LRU"):
            js += [J("SC", "A2-decisive", "A2", f, sz, "CPLUS-M", s, R=C["R_C"][f"{f}{sz}"], lrs=list(lrs[f]))
                   for sz in (2, 4, 8)]
        for f in ("GRU", "LRU"):
            js += [J("SC", "A2-upper", "A2", f, sz, "CPLUS", s, lrs=list(lrs[f])) for sz in (16, 32)
                   if not (sz == 32 and "TOP32" in cut)]
        js += [J("SC", "PC-K(b)", "A1", "GRU", 4, "CPLUS", s, lrs=list(lrs["GRU"])),
               J("SC", "PC-K(b)", "A1", "LRU", 16, "CPLUS", s, lrs=list(lrs["LRU"]))]
    if "D3" not in cut:
        js += [J("SC", "D3", "A2w", "LRU", 16, "CPLUS", s, lrs=list(lrs["LRU"]), train_subset=20_000) for s in CONFIRM[:5]]
    return _ntest(js, C)


def lower_bracket(records, family, cut=()) -> tuple:
    """(size, recipe) of the SENS rung (summary_e3.lower_bracket, the same function D7 uses)."""
    return summary_e3.lower_bracket(summary_e3.build_cells(records), family, cut)


def jobs_SENS(C, records=None, placeholder=False):
    _check(C)
    if {"SC", "SENS"} & set(C["TRIMS"]):
        return []
    js = []
    for f in ("GRU", "LRU"):
        rung = (8, "CPLUS-M") if placeholder else lower_bracket(records, f, C["TRIMS"])
        if rung is None:
            continue
        sz, base_recipe = rung
        for s in CONFIRM[:5]:
            base = run_e3.job_name(J("SC", "A2-decisive" if base_recipe == "CPLUS-M" else "A2-upper", "A2", f, sz,
                                     base_recipe, s))
            js.append(J("SC", "SENS", "A2", f, sz, "SENS", s, R=C["R_C"][f"{f}{sz}"], lrs=list(C["CPLUS_LRS"][f]),
                        base=base))
    return js


def jobs_O(C):
    _check(C)
    if not C["G6_PASS"] or "O" in C["TRIMS"]:                  # G6 failure removes D2; G9 cut 1
        return []
    return [J("O", "D2", "A2", "OOM", 9, "OOM-C", s) for s in CONFIRM]


def jobs_dev1():
    """Development jobs that need no frozen constant (75; the last adds HMM-C n = 3 for G1)."""
    js = []
    for s in DEV3:
        js.append(J("DEV1", "G3", "A2", "HQ", 3, "M", s, steps=3000))
        js += [J("DEV1", "G3", "A2", "HQ", 3, "C0", s, lr=lr, steps=6000) for lr in C0_LRS]
        js.append(J("DEV1", "G3", "A2", "HQ", 3, "C1", s, R=DEV_R1))
        js += [J("DEV1", "G3-tau0", "A2", "HQ", 3, "S", s, R=DEV_RS, tau0=t) for t in TAU0_GRID]
        js.append(J("DEV1", "G2/H4", "A2", "HQ", 3, "C2", s, sigma=0.3, steps=3000))
    js += [J("DEV1", "G2", "A2", "HQ", 3, "C2", s, sigma=0.05, steps=3000) for s in DEV]
    for s in DEV3:
        js += [J("DEV1", "G1", "A1", "HQ", 2, "M", s, steps=3000), J("DEV1", "G1", "A1", "HQ", 2, "C2", s, sigma=0.05, steps=3000),
               J("DEV1", "G5", "A1", "HMM", 4, "HMM-C", s), J("DEV1", "G6", "A1", "OOM", 3, "OOM-C", s),
               J("DEV1", "G4", "A2w", "HQ", 3, "M", s, steps=3000), J("DEV1", "G4", "A2w", "HQ", 3, "C2", s, sigma=0.05, steps=3000),
               J("DEV1", "G7", "A3", "HQ", 4, "M", s, steps=3000), J("DEV1", "G7", "A3", "HQ", 4, "C2", s, sigma=SIGMA_A3, steps=3000)]
    for s in DEV[:2]:
        for f in ("GRU", "LRU"):
            js += [J("DEV1", "G10-lr", "A2", f, 4, "LRGRID", s, lr=lr, lr_index=i, steps=6000) for i, lr in enumerate(LR_GRID)]
    js += [J("DEV1", "G9-timing", "A2", "OOM", 9, "OOM-C", DEV[0]), J("DEV1", "G9-timing", "A2", "HQ", 4, "M", DEV[0], steps=3000)]
    js.append(J("DEV1", "G1", "A1", "HMM", 3, "HMM-C", DEV[0]))
    return js


DEV_R_C = 2                                                    # replicas of the G1 CPLUS-M / SENS dev jobs


def _dev_cplus_m(lrs):
    return J("DEV2", "G1", "A2", "GRU", 2, "CPLUS-M", DEV[0], R=DEV_R_C, lrs=list(lrs["GRU"]))


def jobs_dev2(lrs):
    """Development jobs that need the G10 learning rates (12): timing probes, the PC-K(b) dev check, and the G1
    end-to-end runs of CPLUS-M (R = 2) and D3 (A2w LRU16 on the first 20,000 training sequences)."""
    if any(lrs.get(f) is None for f in ("GRU", "LRU")):
        raise ValueError("DEV2 needs CPLUS_LRS from the G10 learning-rate grid")
    js = [J("DEV2", "G10-timing", "A2", f, sz, "CPLUS", DEV[0], lrs=list(lrs[f])) for f in ("GRU", "LRU") for sz in (2, 32)]
    for s in DEV3:
        js += [J("DEV2", "G10-PC-K(b)", "A1", "GRU", 4, "CPLUS", s, lrs=list(lrs["GRU"])),
               J("DEV2", "G10-PC-K(b)", "A1", "LRU", 16, "CPLUS", s, lrs=list(lrs["LRU"]))]
    js += [_dev_cplus_m(lrs), J("DEV2", "G1", "A2w", "LRU", 16, "CPLUS", DEV[0], lrs=list(lrs["LRU"]), train_subset=20_000)]
    return js


def jobs_dev3(lrs):
    """The G1 SENS run (1), after its DEV2 CPLUS-M base record exists."""
    jobs_dev2(lrs)
    return [J("DEV3", "G1", "A2", "GRU", 2, "SENS", DEV[0], R=DEV_R_C, lrs=list(lrs["GRU"]),
              base=run_e3.job_name(_dev_cplus_m(lrs)))]


def jobs_fixture():
    return [J("FIX", "REF", p, "REF", 0, "REF", s) for s in DEV3 for p in ("A1", "A2", "A2w", "A3")]


def jobs_redev():
    """The single G2 fallback: M and C2 (sigma 0.05) fine-tuned for 6,000 steps on seeds 205-209."""
    return [J("REDEV", "G2-rerun", "A2", "HQ", 3, r, s, sigma=0.05 if r == "C2" else None, steps=6000)
            for s in REDEV for r in ("M", "C2")]


# ---------------------------------------------------------------- CPU estimates (§13, low load, CPU-s per job)
def est_cpu(j) -> float:
    r, p, f, sz = j["recipe"], j["process"], j["family"], j["size"]
    if r == "M":
        return {("A2", 3): 330, ("A1", 2): 150, ("A2w", 3): 330, ("A3", 4): 450, ("A2", 4): 400}[(p, sz)]
    if r == "C0":
        return {"A2": 115, "A1": 70}[p]
    if r == "C1":
        return 110 * j["R"]
    if r == "S":
        return 120 * j["R"]
    if r == "C2":
        return {"A2": 62, "A1": 40, "A2w": 65, "A3": 70}[p]
    if r == "HMM-C":
        return 150
    if r == "OOM-C":
        return 150 if sz == 3 else 250
    if r == "REF":
        return 40
    if r in ("CPLUS-M", "SENS"):
        return 330
    if r == "LRGRID":
        return 50
    if p == "A1":
        return 60
    if p == "A2w":
        return 65
    return {2: 40, 4: 45, 8: 50, 16: 65, 32: 113 if f == "GRU" else 92}[sz]


def counts(C=None) -> dict:
    """Job counts and projected CPU-h per block; with C=None the placeholder constants and no trims are used."""
    C = PLACEHOLDER if C is None else C
    blocks = {"U": jobs_U(C), "T": jobs_T(C), "SC": jobs_SC(C) + jobs_SENS(C, placeholder=True), "O": jobs_O(C),
              "DEV": jobs_dev1() + jobs_dev2(C["CPLUS_LRS"]) + jobs_dev3(C["CPLUS_LRS"]), "FIX": jobs_fixture()}
    out = {b: {"jobs": len(js), "cpu_h": round(sum(est_cpu(j) for j in js) / 3600, 2)} for b, js in blocks.items()}
    out["confirm_total"] = sum(out[b]["jobs"] for b in ("U", "T", "SC", "O"))
    out["projected_cpu_h"] = round(sum(v["cpu_h"] for k, v in out.items() if isinstance(v, dict)), 2)
    return out


# ---------------------------------------------------------------- resumable pool
def shas():
    return run_e3.design_sha256(), run_e3.code_manifest_sha256()


def read(out, j):
    f = Path(out) / run_e3.job_name(j)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except json.JSONDecodeError:
        return None


def status(j, out, sh=None) -> str:
    """todo | stale | retry | done (done includes a job that failed twice)."""
    sh = shas() if sh is None else sh
    rec = read(out, j)
    if rec is None:
        return "todo"
    if (rec.get("design_sha256"), rec.get("code_manifest_sha256")) != tuple(sh):
        return "stale"
    if not rec.get("failed"):
        return "done"
    return "done" if rec.get("attempt", 1) >= 2 else "retry"


def _subprocess(j, out, attempt):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), **{v: "1" for v in run_e3.THREAD_VARS})
    cmd = [sys.executable, "-m", "osw.run_e3", "--job", json.dumps(j), "--out", str(out), "--attempt", str(attempt)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    return r.returncode, (r.stdout.strip().splitlines() or [""])[-1], r.stderr[-600:]


def run_one(j, out, sh):
    """Run until the job is done: attempt 1, and on failure attempt 2; a crash without a record leaves a stub."""
    st = status(j, out, sh)
    f = Path(out) / run_e3.job_name(j)
    if st == "stale":
        os.replace(f, f.with_name(f"{f.name}.stale-{read(out, j).get('code_manifest_sha256', 'none')[:8]}"))
    attempt = 2 if st == "retry" else 1
    logs = []
    while True:
        rc, last, err = _subprocess(j, out, attempt)
        logs.append(f"attempt={attempt} rc={rc} {run_e3.job_name(j)} :: {last}" + (f"\n   stderr: {err}" if rc else ""))
        rec = read(out, j)
        ok_rec = rec is not None and rec.get("attempt") == attempt and (rec.get("design_sha256"), rec.get("code_manifest_sha256")) == tuple(sh)
        if ok_rec and not rec.get("failed"):
            return "done", logs
        if attempt >= 2:
            if not ok_rec:
                run_e3.write_record(dict(j, job=dict(j), attempt=attempt, failed=True, cf=None, val_nll=None, nan_flag=False,
                                         error=f"subprocess rc={rc}: {err[-400:]}", design_sha256=sh[0],
                                         code_manifest_sha256=sh[1]), out)
            return "failed", logs
        attempt += 1


def run_jobs(js, out=OUT, workers=3, log=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sh = shas()
    todo = [j for j in js if status(j, out, sh) != "done"]
    logf = open(log or out / "queue.log", "a")
    logf.write(f"{time.strftime('%FT%TZ', time.gmtime())} start todo={len(todo)}/{len(js)} code={sh[1][:12]}\n")
    logf.flush()
    res = {}
    with ThreadPoolExecutor(max(1, workers)) as ex:
        futs = {ex.submit(run_one, j, out, sh): j for j in todo}
        for fut in as_completed(futs):
            state, logs = fut.result()
            res[run_e3.job_name(futs[fut])] = state
            logf.write("\n".join(logs) + "\n")
            logf.flush()
    logf.write(f"{time.strftime('%FT%TZ', time.gmtime())} end failed={sum(v == 'failed' for v in res.values())}\n")
    logf.close()
    return res


def all_done(js, out=OUT) -> bool:
    sh = shas()
    return all(status(j, out, sh) == "done" for j in js)


def records(out=OUT, blocks=("U", "T", "SC", "O")) -> list:
    rs = []
    for f in sorted(Path(out).glob("e3-*.json")):
        r = json.loads(f.read_text())
        if r.get("block") in blocks:
            rs.append(r)
    return rs


# ---------------------------------------------------------------- fixture: PC-D and the label-free §9 checks
FIX_PROCS = ("A1", "A2", "A2w", "A3")


def _fix_row(r):
    c = {"cf_exact": abs(r["cf_exact"] - 1) <= 1e-6, "cf_marginal": abs(r["cf_marginal"]) <= 1e-6,
         "mixture": all(abs(v["cf"] - v["cf_direct"]) <= 1e-6 for v in r["cf_mix"].values()),
         "marginal_vs_stationary": r["marginal_vs_stationary_rel"] <= 0.02,
         "n_params": all(v["ok"] for v in r["n_params_checks"].values()),
         "kraus_vp": bool(r["kraus_vp"]["ok"])}
    return {"checks": c, "ok": all(c.values()), "denominator_k10": r.get("denominator_k10"),
            "share_k3": r.get("share_k3"), "kraus_vp_err": r["kraus_vp"]["max_abs_err"]}


def fixture_checks(recs) -> dict:
    """PC-D (§7) with the label-free §9 checks, from REF records: CF(exact) = 1 and CF(marginal) = 0 (1e-6), mixtures
    equal the direct computation, E2 marginal within 2% of the stationary one, closed-form parameter counts, and the
    true Kraus operators loaded as V P (error < 2e-3, finite gradients). pcd is True only if every check passes on
    every REF record and all four processes are present. The poisoning test is enforced by the test suite (G1)."""
    rows, refs = [], [r for r in recs if r.get("recipe") == "REF"]
    for r in refs:
        try:
            row = {} if r.get("failed") else _fix_row(r)
        except (KeyError, TypeError):
            row = {}
        rows.append(dict(row, process=r["process"], seed=r["seed"], ok=bool(row.get("ok"))))
    missing = sorted(set(FIX_PROCS) - {r["process"] for r in refs})
    return {"rows": rows, "processes_missing": missing, "pcd": bool(rows) and not missing and all(x["ok"] for x in rows),
            "poisoning": "tests/test_e3_integration.py::test_poisoning_leaves_selection_free_fits_bit_identical (G1)"}


def read_fixture(out=OUT):
    f = Path(out) / "FIXTURE.json"
    return json.loads(f.read_text()) if f.exists() else None


def fixture_pcd(out=OUT):
    fx = read_fixture(out)
    return None if fx is None or fx.get("pcd") is None else bool(fx["pcd"])


def write_fixture(out=OUT, dev=None, confirm=None) -> dict:
    """runs/e3/FIXTURE.json: the dev section (G8, before freeze) and the confirmatory label-free section (after U);
    pcd = every present section passes."""
    fx = read_fixture(out) or {}
    for k, v in (("dev", dev), ("confirm", confirm)):
        if v is not None:
            fx[k] = v
    parts = [fx[k]["pcd"] for k in ("dev", "confirm") if k in fx]
    fx["pcd"] = bool(parts) and all(parts)
    Path(out).mkdir(parents=True, exist_ok=True)
    _write_json(Path(out) / "FIXTURE.json", fx)
    return fx


# ---------------------------------------------------------------- development gate G1-G10 (§11)
PENDING, NO_FREEZE, CHANGED, REMOVES = "PENDING", "NO_FREEZE", "PASS_WITH_CHANGE", "FAIL_REMOVES_ARM"
BLOCKING = (PENDING, NO_FREEZE, "REDEV_REQUIRED")


def _pick(recs, **kw):
    """Non-failed records whose fields equal kw (sigma to 1e-9)."""
    def eq(r, k, v):
        return r.get(k) is not None and abs(float(r[k]) - v) < 1e-9 if k == "sigma" else r.get(k) == v
    return [r for r in recs if not r.get("failed") and all(eq(r, k, v) for k, v in kw.items())]


def _med(rs, key="cpu_s"):
    v = [float(r[key]) for r in rs if r.get(key) is not None]
    return float(np.median(v)) if v else None


def _seeds(rs):
    return {int(r["seed"]) for r in rs}


def g1(recs, tests_ok=None) -> dict:
    """Every dev job type ran end to end on seed 200 (no failure, no NaN stop, fields present) and the full test suite
    passed (tests_ok is the operator's record of that run)."""
    dev = [r for r in recs if r.get("block") in ("DEV1", "DEV2", "DEV3", "FIX") and int(r["seed"]) == DEV[0]]
    dummy = {"GRU": [0.0, 0.0], "LRU": [0.0, 0.0]}
    want = {(j["recipe"], j["process"], j["family"], j["size"]) for j in jobs_dev1() + jobs_fixture() + jobs_dev2(dummy)
            + jobs_dev3(dummy) if j["seed"] == DEV[0]}
    have = {(r["recipe"], r["process"], r["family"], r["size"]) for r in dev}
    need = ("cf_exact", "cf_marginal", "kl_marginal")
    bad = [run_e3.job_name(r) for r in dev if r.get("failed") or r.get("nan_flag") or any(
        r.get(k) is None for k in (need if r["recipe"] == "REF" else ("cf", "val_nll", "n_params", "cpu_s", "kl", "kl_marginal")))]
    out = {"missing_types": sorted(map(str, want - have)), "bad_runs": bad, "tests_ok": tests_ok}
    if bad or tests_ok is False:
        return dict(out, status=NO_FREEZE, reason="a seed-200 run failed / hit NaN / lacks fields, or tests failed")
    if want - have or tests_ok is None:
        return dict(out, status=PENDING, reason="job types missing on seed 200, or the test-suite pass not recorded")
    return dict(out, status="PASS")


def _count_median(rs, frac=Fraction(4, 5), med_t=0.95):
    cf = np.array([float(r["cf"]) for r in rs])
    k = int(np.sum(summary_e3.success(cf)))
    return {"k": k, "n": len(cf), "median": float(np.median(cf)), "count_ok": Fraction(k, len(cf)) >= frac,
            "median_ok": bool(summary_e3.ge(np.median(cf), med_t))}


def g2(recs) -> dict:
    """PC-A reachability on A2 O05 (seeds 200-204): count >= 4/5 and median >= 0.95; median unreachable -> clause
    dropped; count failing -> the single 6,000-step rerun on 205-209, else no freeze."""
    c2 = _pick(recs, block="DEV1", process="A2", family="HQ", recipe="C2", sigma=0.05)
    if _seeds(c2) != set(DEV):
        return {"status": PENDING, "reason": "A2 O05 on dev seeds 200-204 incomplete"}
    v = _count_median(c2)
    if v["count_ok"]:
        return dict(v, status="PASS" if v["median_ok"] else CHANGED, FT_STEPS=3000, MEDIAN_CLAUSE=v["median_ok"],
                    reason=None if v["median_ok"] else "median clause unreachable: dropped before freeze (DEVIATIONS)")
    re_ = _pick(recs, block="REDEV", process="A2", family="HQ", recipe="C2", sigma=0.05)
    if _seeds(re_) != set(REDEV):
        return dict(v, status="REDEV_REQUIRED", reason="count < 4/5: run --redev (M and C2, 6,000 steps, 205-209)")
    w = _count_median(re_)
    if not w["count_ok"]:
        return dict(v, redev=w, status=NO_FREEZE, reason="count < 4/5 also after the 6,000-step rerun")
    return dict(v, redev=w, status=CHANGED, FT_STEPS=6000, MEDIAN_CLAUSE=w["median_ok"],
                reason="fine-tune of M and C2 lengthened to 6,000 steps (the one registered G2 change)")


def _m_dev(recs, ft_steps):
    """A2 HQ3-M dev runs at the frozen fine-tune length (the REDEV runs if G2 moved it to 6,000 steps)."""
    block = "REDEV" if ft_steps == 6000 else "DEV1"
    return _pick(recs, block=block, process="A2", family="HQ", size=3, recipe="M")


def _smallest_r(per, target):
    ok = [r for r in (3, 4, 5, 6) if summary_e3.ge(r * per, target, tie=True)]
    return (ok[0], False) if ok else (6, True)


def g3(recs, ft_steps) -> dict:
    """tau0 by mean dev validation NLL (ties: grid order); R1 / RS = smallest of {3, 4, 5, 6} whose dev CPU reaches
    the M dev median CPU (6 if none does, flagged). M itself is never changed."""
    m = _m_dev(recs, ft_steps)
    c1 = _pick(recs, block="DEV1", process="A2", family="HQ", recipe="C1")
    sv = _pick(recs, block="DEV1", process="A2", family="HQ", recipe="S")
    taus = {t: [r["val_nll"] for r in sv if abs(r["tau0"] - t) < 1e-12] for t in TAU0_GRID}
    if not (m and c1 and all(taus.values())) or ft_steps is None:
        return {"status": PENDING, "reason": "A2 M / C1 / S tau0-grid dev runs incomplete (or G2 pending)"}
    m_cpu = _med(m)
    per1 = float(np.median([r["cpu_s"] / r["R"] for r in c1]))
    per_s = float(np.median([r["cpu_s"] / r["R"] for r in sv]))
    (r1, cap1), (rs, cap_s) = _smallest_r(per1, m_cpu), _smallest_r(per_s, m_cpu)
    mean_tau = {t: float(np.mean(v)) for t, v in taus.items()}
    tau0 = min(TAU0_GRID, key=lambda t: (mean_tau[t], TAU0_GRID.index(t)))
    return {"status": "PASS", "R1": r1, "RS": rs, "TAU0": tau0, "m_dev_median_cpu": m_cpu, "c1_cpu_per_restart": per1,
            "s_cpu_per_restart": per_s, "tau0_mean_val_nll": mean_tau, "R1_capped_at_6": cap1, "RS_capped_at_6": cap_s,
            "m_dev_median_cf": _med(m, "cf"), "c1_dev_median_cf": _med(c1, "cf")}


def g4(recs) -> dict:
    c2 = _pick(recs, block="DEV1", process="A2w", family="HQ", recipe="C2", sigma=0.05)
    m = _pick(recs, block="DEV1", process="A2w", family="HQ", recipe="M")
    if len(c2) < len(DEV3):
        return {"status": PENDING, "reason": "A2w O05 dev runs incomplete"}
    med = _med(c2, "cf")
    ok = bool(summary_e3.ge(med, 0.50))
    slopes = {f"{r['recipe']}-s{r['seed']}": r.get("val_slope_last") for r in m + c2}
    return {"status": "PASS" if ok else REMOVES, "G4_PASS": ok, "o05_dev_median": med, "m_dev_median": _med(m, "cf"),
            "val_slope_last500_per_step": slopes,
            "reason": None if ok else "A2w removed; H3b registered as not decidable (pre-freeze)"}


def g5(recs) -> dict:
    h = _pick(recs, block="DEV1", process="A1", family="HMM", size=4, recipe="HMM-C")
    if len(h) < len(DEV3):
        return {"status": PENDING, "reason": "A1 HMM4-C dev runs incomplete"}
    cf = [float(r["cf"]) for r in h]
    ok = all(summary_e3.le(c, 0.88, tie=True) for c in cf)
    return {"status": "PASS" if ok else NO_FREEZE, "cf": cf,
            "reason": None if ok else "CF(HMM4) > 0.88 contradicts the known ceiling: stop and diagnose"}


def exact_hankel_fixture(seed=DEV[0]) -> dict:
    """G6: a rank-3 spectral OOM from the exact A1 word probabilities reproduces the exact predictive (CF = 1)."""
    proc = tasks_e3.process("A1")
    Ks, rho = np.stack(proc.K), proc.stationary()

    def prob(word):
        r = rho
        for x in word:
            r = Ks[x] @ r @ Ks[x].conj().T
        return float(np.trace(r).real)
    w0, A, _ = oom_e3.spectral(prob, proc.O, 3)
    m = oom_e3.OOMWM(proc.O, 3)
    m.load(w0, A)
    _, _, te, exact = tasks_e3.data("A1", seed)
    pred, _ = oom_e3.predict(m, te[0], tasks_e3.K_EVAL)
    cf = tasks_e3.cf_and_kl(pred, exact, te[0][:, tasks_e3.L_CTX:tasks_e3.L_CTX + tasks_e3.K_EVAL])["cf"]
    return {"cf": float(cf), "ok": bool(abs(cf - 1) <= 1e-4)}


def g6(recs, hankel=None) -> dict:
    o = _pick(recs, block="DEV1", process="A1", family="OOM", size=3, recipe="OOM-C")
    if len(o) < len(DEV3):
        return {"status": PENDING, "reason": "A1 OOM3-C dev runs incomplete"}
    k = int(np.sum([summary_e3.ge(float(r["cf"]), 0.95) for r in o]))
    hankel = exact_hankel_fixture() if hankel is None else hankel
    ok = Fraction(k, len(o)) >= Fraction(2, 3) and hankel["ok"]
    return {"status": "PASS" if ok else REMOVES, "G6_PASS": bool(ok), "k": k, "n": len(o), "exact_hankel": hankel,
            "reason": None if ok else "H7 and D2 removed before freeze"}


def g7(recs) -> dict:
    m = _pick(recs, block="DEV1", process="A3", family="HQ", size=4, recipe="M")
    c2 = _pick(recs, block="DEV1", process="A3", family="HQ", size=4, recipe="C2")
    if not m:
        return {"status": PENDING, "reason": "A3 HQ4-M dev runs missing"}
    return {"status": "PASS", "descriptive": True, "m_median_cpu": _med(m), "m_median_cf": _med(m, "cf"),
            "o05_median_cf": _med(c2, "cf"), "n": len(m)}


SD_MAX = {"A1": 0.01, "A2": 0.01, "A2w": 0.02, "A3": 0.02}
O05_SIGMA = {"A1": 0.05, "A2": 0.05, "A2w": 0.05, "A3": SIGMA_A3}
N_TEST_RAISED = {"A2w": 20_000, "A3": 8_000}


def g8(recs, sims=None) -> dict:
    """Fixtures (i)-(iii) on the dev seeds: PC-D rows, distinguishability on A2 (Markov-1..3 <= 0.75, linear
    references < 0.85), O05 test-bootstrap SD (median over dev seeds; A2w / A3 above 0.02 raise N_test, A1 / A2 above
    0.01 block the freeze), 1e-6 floor rescoring (< 0.005) and HQ clamp fraction (< 1e-4); the planted simulation
    (failure demotes H6 to descriptive) and the coverage simulation (< 0.94 switches CI_LEVEL to 0.975)."""
    fx = fixture_checks([r for r in recs if r.get("block") == "FIX"])
    refs2 = [r["ref"] for r in _pick(recs, block="FIX", process="A2", recipe="REF") if r.get("ref")]
    mk = max((x[f"markov{m}"] for x in refs2 for m in (1, 2, 3)), default=None)
    lin = max((x[k] for x in refs2 for k in ("linear_hard", "linear_soft")), default=None)
    dist_ok = mk is not None and summary_e3.le(mk, 0.75, tie=True) and summary_e3.lt(lin, 0.85)
    sd = {p: _med(_pick(recs, block="DEV1", process=p, family="HQ", recipe="C2", sigma=O05_SIGMA[p]), "cf_boot_sd")
          for p in SD_MAX}
    n_test = {p: (N_TEST_RAISED[p] if sd[p] is not None and sd[p] > SD_MAX[p] else tasks_e3.PROC[p]["n_test"])
              for p in N_TEST_RAISED}
    sd_ok = all(sd[p] is not None and sd[p] <= SD_MAX[p] for p in ("A1", "A2"))
    fits = [r for r in recs if r.get("block") in ("DEV1", "DEV2", "DEV3", "REDEV") and not r.get("failed")
            and r.get("cf") is not None and r.get("cf_floor_1e6") is not None]
    floor_bad = [run_e3.job_name(r) for r in fits if not abs(r["cf_floor_1e6"] - r["cf"]) < 0.005]
    clamp_bad = [run_e3.job_name(r) for r in fits if r.get("family") == "HQ" and r.get("clamp_frac_test") is not None
                 and not r["clamp_frac_test"] < 1e-4]
    out = {"fixture": fx, "markov_max_a2": mk, "linear_max_a2": lin, "o05_boot_sd": sd, "floor_rescore_bad": floor_bad,
           "clamp_bad": clamp_bad, "N_TEST_A2W": n_test["A2w"], "N_TEST_A3": n_test["A3"],
           "denominators": {f"{x['process']}-s{x['seed']}": x.get("denominator_k10") for x in fx["rows"]},
           "share_k3": {f"{x['process']}-s{x['seed']}": x.get("share_k3") for x in fx["rows"]}}
    if sims is not None:
        pl, cov = sims["planted"], sims["coverage"]
        out.update(H6_REGISTERED=bool(pl["h6_stays_registered"]), CI_LEVEL=summary_e3.interval_level(cov["coverage"]),
                   planted={k: pl[k] for k in pl if k != "detail"}, coverage=cov["coverage"],
                   detection_underpowered=bool(pl["underpowered"]))
    hard = [k for k, ok in (("PC-D fixture", fx["pcd"]), ("distinguishability", dist_ok), ("O05 SD A1/A2", sd_ok),
                            ("floor rescoring", not floor_bad), ("clamp fraction", not clamp_bad)) if not ok]
    if fx["processes_missing"] or any(sd[p] is None for p in SD_MAX):
        return dict(out, status=PENDING, reason="FIX records or O05 dev runs missing")
    if hard:
        return dict(out, status=NO_FREEZE, reason="failed: " + ", ".join(hard))
    if sims is None:
        return dict(out, status=PENDING, reason="planted and coverage simulations not run")
    changed = not out["H6_REGISTERED"] or out["CI_LEVEL"] != summary_e3.CI_LEVEL or any(
        n_test[p] != tasks_e3.PROC[p]["n_test"] for p in n_test)
    return dict(out, status=CHANGED if changed else "PASS")


def single_cpu(recs, family, size):
    """Single-replica CPLUS CPU on A2 from the G10 timing probes at sizes 2 and 32, linear in size in between."""
    t = {sz: _med(_pick(recs, block="DEV2", process="A2", family=family, size=sz, recipe="CPLUS")) for sz in (2, 32)}
    if None in t.values():
        return None
    return t[2] + (t[32] - t[2]) * (size - 2) / 30


def g10(recs, ft_steps) -> dict:
    """Learning rates (two per family by mean dev validation NLL over seeds 200-201; ties: grid order), CPU_FLOOR (A2
    HQ3-M dev median cpu_s), R_c = ceil(CPU_FLOOR / single-replica CPU), the PC-K(b) dev check (>= 2/3 seeds), r_hat."""
    out, lrs = {}, {}
    for f in ("GRU", "LRU"):
        g = _pick(recs, block="DEV1", process="A2", family=f, size=4, recipe="LRGRID")
        means = {lr: [r["val_nll"] for r in g if abs(r["lr"] - lr) < 1e-15] for lr in LR_GRID}
        if not all(len(v) == 2 for v in means.values()):
            return {"status": PENDING, "reason": f"{f} learning-rate grid incomplete"}
        m = {lr: float(np.mean(v)) for lr, v in means.items()}
        lrs[f] = sorted(sorted(LR_GRID, key=lambda x: (m[x], LR_GRID.index(x)))[:2])
        out[f"{f}_lr_mean_val_nll"] = m
    out["CPLUS_LRS"] = lrs
    floor = _med(_m_dev(recs, ft_steps)) if ft_steps else None
    cpu1 = {f"{f}{s}": single_cpu(recs, f, s) for f in ("GRU", "LRU") for s in (2, 4, 8, 16, 32)}
    if floor is None or None in cpu1.values():
        return dict(out, status=PENDING, reason="DEV2 timing probes or the M dev runs missing")
    out.update(CPU_FLOOR=floor, single_replica_cpu=cpu1, R_C={k: int(ceil(floor / v)) for k, v in cpu1.items()})
    pk = {}
    for f, (sz, t) in summary_e3.PCK_A1.items():
        rs = _pick(recs, block="DEV2", process="A1", family=f, size=sz, recipe="CPLUS")
        if len(rs) < len(DEV3):
            return dict(out, status=PENDING, reason="PC-K(b) dev runs incomplete")
        k = int(np.sum([summary_e3.ge(float(r["cf"]), t) for r in rs]))
        pk[f] = {"k": k, "n": len(rs), "cf": [r["cf"] for r in rs], "ok": Fraction(k, len(rs)) >= Fraction(2, 3)}
    out.update(pck_b_dev=pk, r_hat_a2=[r.get("r_hat") for r in _pick(recs, block="DEV2", process="A2", family="LRU")])
    if not all(v["ok"] for v in pk.values()):
        return dict(out, status=NO_FREEZE, reason="PC-K(b) dev check failed: diagnose the classical pipeline")
    return dict(out, status="PASS")


def measured_cpu(j, recs, C):
    """(CPU-s, source) of one confirmatory job from dev measurements (fine-tune part scaled to the job's steps),
    falling back to the §13 estimate when no dev run of that type exists."""
    r, p, f, sz = j["recipe"], j["process"], j["family"], j["size"]
    dev = [x for x in recs if x.get("block") in DEV_BLOCKS and not x.get("failed") and x.get("cpu_s") is not None]
    v = None
    if r in ("M", "C2"):
        rs = _pick(dev, recipe=r, process=p, family="HQ", size=sz)
        v = None if not rs else float(np.median([(x.get("mom_cpu_s") or 0.0) + (x["cpu_s"] - (x.get("mom_cpu_s") or 0.0))
                                                  * j.get("steps", 3000) / x.get("steps", 3000) for x in rs]))
    elif r == "C0":
        v = _med(_pick(dev, recipe="C0", process=p))
    elif r in ("C1", "S"):
        rs = _pick(dev, recipe=r, process=p)
        v = None if not rs else j["R"] * float(np.median([x["cpu_s"] / x["R"] for x in rs]))
    elif r in ("HMM-C", "OOM-C"):
        v = _med(_pick(dev, recipe=r, size=sz)) or _med(_pick(dev, recipe=r))
    elif r == "REF":
        v = _med(_pick(dev, recipe="REF", process=p))
    elif r in ("CPLUS", "CPLUS-M", "SENS"):
        one = single_cpu(dev, f, sz) if p == "A2" else _med(_pick(dev, recipe="CPLUS", process=p, family=f, size=sz))
        v = None if one is None else one * (j.get("R") or 1)
    return (v, "dev") if v is not None else (float(est_cpu(j)), "s13")


def projection(recs, C, trims) -> dict:
    """Dev measured CPU-h plus every confirmatory job of the design under these cuts (all gated blocks counted)."""
    Ct = dict(C, TRIMS=list(trims))
    js = jobs_U(Ct) + jobs_T(Ct) + jobs_SC(Ct) + jobs_SENS(Ct, placeholder=True) + jobs_O(Ct)
    cp = [measured_cpu(j, recs, Ct) for j in js]
    dev_h = sum(float(x["cpu_s"]) for x in recs if x.get("block") in DEV_BLOCKS and x.get("cpu_s") is not None) / 3600
    return {"trims": list(trims), "dev_h": dev_h, "confirm_h": sum(c for c, _ in cp) / 3600,
            "total_h": dev_h + sum(c for c, _ in cp) / 3600, "n_jobs": len(js), "n_s13_fallback": sum(s == "s13" for _, s in cp)}


def g9(recs, C) -> dict:
    """Cut in the fixed order while the projection exceeds 28 CPU-h; if it still exceeds 32 after all seven cuts, the
    terminal rule adds "SC" (H6 and STOP registered as not run for compute)."""
    trims, steps = [], [projection(recs, C, [])]
    for t in TRIM_ORDER[:-1]:
        if steps[-1]["total_h"] <= G9_TARGET_H:
            break
        trims.append(t)
        steps.append(projection(recs, C, trims))
    if steps[-1]["total_h"] > G9_CAP_H:
        trims.append("SC")
        steps.append(projection(recs, C, trims))
    return {"status": "PASS" if not trims else CHANGED, "TRIMS": trims, "projected_h": steps[-1]["total_h"],
            "steps": steps, "target_h": G9_TARGET_H, "cap_h": G9_CAP_H}


def dev_gate(recs, sims=None, tests_ok=None, hankel=None) -> dict:
    """G1-G10 from the dev records -> every derived constant with its reason, freeze_ok, and the FROZEN_CONSTANTS block."""
    G = {"G1": g1(recs, tests_ok), "G2": g2(recs)}
    ft = G["G2"].get("FT_STEPS")
    G.update(G3=g3(recs, ft), G4=g4(recs), G5=g5(recs), G6=g6(recs, hankel), G7=g7(recs), G8=g8(recs, sims),
             G10=g10(recs, ft))
    C = {"R1": G["G3"].get("R1"), "RS": G["G3"].get("RS"), "TAU0": G["G3"].get("TAU0"),
         "CPLUS_LRS": G["G10"].get("CPLUS_LRS") or {"GRU": None, "LRU": None},
         "R_C": G["G10"].get("R_C") or {k: None for k in FROZEN_CONSTANTS["R_C"]}, "CPU_FLOOR": G["G10"].get("CPU_FLOOR"),
         "FT_STEPS": ft, "MEDIAN_CLAUSE": G["G2"].get("MEDIAN_CLAUSE"), "G4_PASS": G["G4"].get("G4_PASS"),
         "G6_PASS": G["G6"].get("G6_PASS"), "H6_REGISTERED": G["G8"].get("H6_REGISTERED"),
         "CI_LEVEL": G["G8"].get("CI_LEVEL"), "N_TEST_A2W": G["G8"].get("N_TEST_A2W"),
         "N_TEST_A3": G["G8"].get("N_TEST_A3"), "TRIMS": None}
    miss = missing_constants(dict(C, TRIMS=[]))
    G["G9"] = g9(recs, dict(C, TRIMS=[])) if not miss else {"status": PENDING, "reason": "constants missing: " + ", ".join(miss)}
    C["TRIMS"] = G["G9"].get("TRIMS")
    blocking = [k for k, g in G.items() if g["status"] in BLOCKING]
    return {"gates": dict(sorted(G.items(), key=lambda kv: int(kv[0][1:]))), "frozen_constants": C,
            "blocking": blocking, "freeze_ok": not blocking and not missing_constants(C)}


def run_dev_gate(out=OUT, sims=None, tests_ok=None) -> dict:
    g = dev_gate(records(out, DEV_BLOCKS), sims, tests_ok)
    write_fixture(out, dev=g["gates"]["G8"]["fixture"])
    _write_json(Path(out) / "GATES-DEV.json", g)
    return g


def dev_lrs(out=OUT):
    """G10 learning rates for DEV2 / DEV3: FROZEN_CONSTANTS if set, else GATES-DEV.json."""
    lrs = FROZEN_CONSTANTS["CPLUS_LRS"]
    if all(lrs.get(f) is not None for f in ("GRU", "LRU")):
        return lrs
    f = Path(out) / "GATES-DEV.json"
    if f.exists():
        got = json.loads(f.read_text())["frozen_constants"]["CPLUS_LRS"]
        if all(got.get(k) is not None for k in ("GRU", "LRU")):
            return got
    return None


# ---------------------------------------------------------------- confirmatory gates and integrity
def summarize(C, out=OUT, blocks=("U", "T", "SC", "O")) -> dict:
    return summary_e3.summarize(records(out, blocks), level=C["CI_LEVEL"], median_clause=C["MEDIAN_CLAUSE"],
                                g4=C["G4_PASS"], pcd=fixture_pcd(out), trims=C["TRIMS"], g6=C["G6_PASS"],
                                h6_registered=C["H6_REGISTERED"])


def gated_blocks(summary: dict, trims=()) -> list:
    """Blocks to enqueue after U from summary_e3's enqueue flags (T: H1 PASS/PARTIAL; SC: H1 PASS and PC-B PASS;
    O: PC-O PASS), minus G9 cuts."""
    e, cut = summary["enqueue"], set(trims)
    return [b for b in ("T", "SC", "O") if e.get(b) and b not in cut]


def e2_manifest():
    return sorted((str(p.relative_to(ROOT)), run_e3.sha256_file(p)) for p in run_e3.E2_RUNS.rglob("*") if p.is_file())


def freeze_e2_manifest(out=OUT) -> Path:
    """Snapshot the runs/e2 manifest at freeze (refuses to overwrite an existing snapshot)."""
    f = Path(out) / "E2-MANIFEST.json"
    if f.exists():
        raise FileExistsError(f"{f} already exists (taken at freeze)")
    Path(out).mkdir(parents=True, exist_ok=True)
    _write_json(f, e2_manifest())
    return f


def integrity(recs, e2_before=None) -> dict:
    ds = {r.get("design_sha256") for r in recs}
    cs = {r.get("code_manifest_sha256") for r in recs}
    seeds = sorted({int(r["seed"]) for r in recs})
    out = {"n": len(recs), "design_sha256": sorted(ds), "code_manifest_sha256": sorted(cs),
           "one_design": len(ds) == 1, "one_code": len(cs) == 1, "seeds": seeds,
           "seeds_in_confirm_range": all(s in CONFIRM for s in seeds), "n_failed": sum(bool(r.get("failed")) for r in recs)}
    if e2_before is not None:
        out["runs_e2_unchanged"] = e2_manifest() == [tuple(x) for x in e2_before]
    out["ok"] = out["one_design"] and out["one_code"] and out["seeds_in_confirm_range"] and out.get("runs_e2_unchanged", True)
    return out


def _write_json(path, obj):
    tmp = Path(path).with_name(f".{Path(path).name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(summary_e3._py(obj), indent=1))
    os.replace(tmp, path)


def preflight(out=OUT) -> dict:
    """confirm() prerequisites: the freeze-time runs/e2 manifest and the dev PC-D pass (§7: no PC-D pass, no run)."""
    out = Path(out)
    if not (out / "E2-MANIFEST.json").exists():
        raise RuntimeError("runs/e2 manifest not snapshotted at freeze: run --freeze-e2-manifest first")
    fx = read_fixture(out)
    if not fx or not fx.get("dev", {}).get("pcd"):
        raise RuntimeError("PC-D has not passed on the dev fixtures (FIXTURE.json): nothing runs")
    return fx


def confirm(C=None, out=OUT, workers=3) -> dict:
    C = FROZEN_CONSTANTS if C is None else C
    _check(C)
    out = Path(out)
    preflight(out)
    mf = out / "E2-MANIFEST.json"
    U = jobs_U(C)
    run_jobs(U, out, workers)
    if not all_done(U, out):
        raise RuntimeError("U block incomplete; rerun the queue")
    write_fixture(out, confirm=fixture_checks(records(out, ("U",))))
    s = summarize(C, out, ("U",))
    _write_json(out / "GATES-U.json", s)
    (out / "GATES-U.txt").write_text(summary_e3.report(s) + "\n")
    print(summary_e3.report(s))
    blocks = gated_blocks(s, C["TRIMS"])
    _write_json(out / "ENQUEUE.json", {"enqueue": s["enqueue"], "blocks": blocks, "trims": C["TRIMS"]})
    js = (jobs_T(C) if "T" in blocks else []) + (jobs_O(C) if "O" in blocks else []) + (jobs_SC(C) if "SC" in blocks else [])
    run_jobs(js, out, workers)
    if "SC" in blocks and "SENS" not in C["TRIMS"]:
        sens = jobs_SENS(C, records(out, ("SC",)))
        if len(sens) != 10:
            raise RuntimeError("SC ladder incomplete; SENS cannot be enqueued yet")
        run_jobs(sens, out, workers)
    integ = integrity(records(out), json.loads(mf.read_text()))
    if not integ["ok"]:                                         # §11 step 3: no final summary without integrity
        _write_json(out / "INTEGRITY-FAIL.json", {"integrity": integ, "blocks_run": ["U"] + blocks})
        return {"integrity": integ, "blocks_run": ["U"] + blocks}
    final = summarize(C, out)
    final["integrity"] = integ
    final["blocks_run"] = ["U"] + blocks
    _write_json(out / "SUMMARY.json", final)
    (out / "SUMMARY.txt").write_text(summary_e3.report(final) + "\nINTEGRITY " + json.dumps(final["integrity"]) + "\n")
    return final


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dev", action="store_true", help="DEV1 + fixtures, then DEV2 and DEV3 once the G10 lrs are known")
    ap.add_argument("--redev", action="store_true", help="the single G2 fallback rerun on seeds 205-209")
    ap.add_argument("--dev-gate", action="store_true", help="G1-G10 -> GATES-DEV.json and the dev FIXTURE.json")
    ap.add_argument("--tests-ok", action="store_true", help="record that the full test suite passed (G1)")
    ap.add_argument("--no-sims", action="store_true", help="skip the planted / coverage simulations (G8 stays PENDING)")
    ap.add_argument("--freeze-e2-manifest", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.dry_run:
        c = counts()
        for b in ("U", "T", "SC", "O", "DEV", "FIX"):
            print(f"{b:4s} jobs={c[b]['jobs']:4d}  projected CPU-h={c[b]['cpu_h']:.2f}")
        print(f"confirm total={c['confirm_total']}  dev={c['DEV']['jobs']} + fixture {c['FIX']['jobs']}  "
              f"projected CPU-h={c['projected_cpu_h']:.2f} (G9 target {G9_TARGET_H}, cap {G9_CAP_H}; low-load §13 estimates)")
        miss = missing_constants()
        print("frozen constants: " + ("all set" if not miss else f"{len(miss)} unset -> confirmatory build refused"))
        return 0
    if a.dev:
        res = run_jobs(jobs_dev1() + jobs_fixture(), out, a.workers)
        lrs = dev_lrs(out)
        if lrs is not None:
            res.update(run_jobs(jobs_dev2(lrs), out, a.workers))
            if all_done(jobs_dev2(lrs), out):
                res.update(run_jobs(jobs_dev3(lrs), out, a.workers))
        return 1 if any(v == "failed" for v in res.values()) else 0
    if a.redev:
        res = run_jobs(jobs_redev(), out, a.workers)
        return 1 if any(v == "failed" for v in res.values()) else 0
    if a.dev_gate:
        sims = None if a.no_sims else {"planted": sim_e3.planted_gate(), "coverage": sim_e3.coverage_sim()}
        g = run_dev_gate(out, sims, True if a.tests_ok else None)
        for k, v in g["gates"].items():
            print(f"{k}: {v['status']}" + (f"  ({v['reason']})" if v.get("reason") else ""))
        print("FROZEN_CONSTANTS = " + json.dumps(summary_e3._py(g["frozen_constants"])))
        print("freeze_ok=" + str(g["freeze_ok"]) + ("" if g["freeze_ok"] else f"  blocking={g['blocking']}"))
        return 0 if g["freeze_ok"] else 3
    if a.freeze_e2_manifest:
        print(freeze_e2_manifest(out))
        return 0
    if a.summary:
        _check(FROZEN_CONSTANTS)
        s = summarize(FROZEN_CONSTANTS, out)
        print(summary_e3.report(s))
        return 0
    if a.confirm:
        final = confirm(FROZEN_CONSTANTS, out, a.workers)
        return 0 if final["integrity"]["ok"] else 2
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

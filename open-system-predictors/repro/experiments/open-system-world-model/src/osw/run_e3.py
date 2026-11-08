"""E3 runner (design: E3-DESIGN.zh.md §2-§4, §8, §16). One job = one (block, arm, process, family, size, recipe, seed,
sigma, ...) dict -> one result JSON, written atomically (tmp file + rename) under the out dir (default runs/e3).

  python -m osw.run_e3 --job '{"block": "U", "arm": "L-A2-M", "process": "A2", "family": "HQ", "size": 3,
                               "recipe": "M", "seed": 20, "sigma": null, "steps": 3000}' --out runs/e3

Recipes: M C1 S C2 (HQ), C0 (unmodified run_e2.main in a subprocess, raw output in <out>/c0, CPU from
getrusage(RUSAGE_CHILDREN)), CPLUS CPLUS-M SENS LRGRID (GRU/LRU), HMM-C, OOM-C, REF. The four thread variables are
forced to 1 before numpy/torch load and torch.set_num_threads(1) is called per job. Every prediction is checked to sum
to 1 +- 1e-5 before CF = queue_e2.metric(run.metrics KL, run_e2 kl_marginal) is computed. Each record carries the job
fields, cf, per-k kl, kl_marginal, cf_boot_sd (SD of CF over a test-sequence bootstrap, §9 i), val_nll, n_params,
cpu_s (the fit's process_time), job_cpu_s, wall_s, nan_flag,
failed, attempt, design_sha256, code_manifest_sha256 (sorted (path, sha256) of src/osw/*.py) and result_sha256.
A job that raises (not a NaN, which is handled inside the fits) is recorded with failed=True; the queue reruns it once.
"""
from __future__ import annotations

import os

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS")
for _v in THREAD_VARS:
    os.environ[_v] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from . import cplus, hmm, hmm_e3, hq_init, models, models_e2, oom_e3, queue_e2, stiefel, summary_e3, tasks_e2, tasks_e3  # noqa: E402
from .run import predict_torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DESIGN = ROOT / "E3-DESIGN.zh.md"
E2_RUNS = ROOT / "runs/e2"
L_CTX, K = tasks_e3.L_CTX, tasks_e3.K_EVAL
REF_SHUFFLE_CODE = 9                                   # namespace code for the REF shuffle stream (unused by recipes 0-8)
LRGRID_R0 = 900                                        # namespace r of the G10 learning-rate grid init (recipe 4), shared by all lrs
N_SEQ_BOOT = 1000                                      # test-sequence bootstrap replicates for cf_boot_sd
VAL_LAST = 500                                         # A2w M / C2: validation NLL logged over the last 500 steps (G4)
CLOSED_FORM = {("A2", "HQ", 3): 54, ("A2", "HQ", 4): 96, ("A1", "HQ", 2): 16, ("A3", "HQ", 4): 128,
               ("A2", "GRU", 2): 63, ("A2", "GRU", 4): 147, ("A2", "GRU", 8): 387, ("A2", "GRU", 16): 1155,
               ("A2", "GRU", 32): 3843, ("A2", "LRU", 2): 59, ("A2", "LRU", 4): 99, ("A2", "LRU", 8): 259,
               ("A2", "LRU", 16): 771, ("A2", "LRU", 32): 2563, ("A2", "OOM", 9): 242, ("A1", "OOM", 3): 17,
               ("A1", "HMM", 3): 11, ("A1", "HMM", 4): 19, ("A1", "GRU", 4): 130, ("A1", "LRU", 16): 722}


# ---------------------------------------------------------------- provenance
def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def design_sha256() -> str:
    return sha256_file(DESIGN)


def code_manifest(root=ROOT) -> list:
    return sorted((str(p.relative_to(root)), sha256_file(p)) for p in (Path(root) / "src/osw").glob("*.py"))


def code_manifest_sha256(root=ROOT) -> str:
    return hashlib.sha256(json.dumps(code_manifest(root)).encode()).hexdigest()


def job_name(j: dict) -> str:
    """Deterministic file name of a job's result (from the job fields only; a record carries them under "job")."""
    j = j.get("job", j)
    parts = ["e3", j["block"], j["process"], f"{j['family']}{j['size'] or ''}", j["recipe"]]
    if j.get("sigma") is not None:
        parts.append(f"sig{j['sigma']:g}")
    if j.get("lr") is not None:
        parts.append(f"lr{j['lr']:g}")
    if j.get("tau0") is not None and j["recipe"] == "S" and j["block"].startswith("DEV"):
        parts.append(f"tau{j['tau0']:g}")
    return "-".join(parts) + f"-s{j['seed']}.json"


def write_record(rec: dict, out) -> Path:
    """Sanitise to JSON, add result_sha256 (as in run_e2), write atomically."""
    rec = summary_e3._py({k: v for k, v in rec.items() if k != "result_sha256"})
    rec["result_sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / job_name(rec)
    tmp = f.with_name(f".{f.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(rec, indent=1))
    os.replace(tmp, f)
    return f


def _inside(p, base) -> bool:
    p, base = Path(p).resolve(), Path(base).resolve()
    return p == base or base in p.parents


# ---------------------------------------------------------------- data and scoring
def load_data(process, seed, n_test=None):
    return tasks_e3.data(process, seed, n_test)


def _load(j):
    return load_data(j["process"], j["seed"], **({"n_test": j["n_test"]} if j.get("n_test") else {}))


def _y(te):
    return te[0][:, L_CTX:L_CTX + K]


def _renorm(p, floor):
    q = np.clip(np.asarray(p, float), floor, None)
    return q / q.sum(-1, keepdims=True)


def cf_boot_sd(pred, exact, n_boot=N_SEQ_BOOT, seed=summary_e3.BOOT_SEED) -> float:
    """SD of CF over a bootstrap of test sequences (§9 i; same clipping as run.metrics, marginal held at its value)."""
    p, q = np.clip(np.asarray(pred, float), 1e-9, 1), np.clip(exact, 1e-12, 1)
    a = (q * (np.log(q) - np.log(p))).sum(-1)[:, :10].mean(1)
    b = (q * np.log(q / exact[:, -1].mean(0))).sum(-1)[:, :10].mean(1)
    rng, cf = np.random.default_rng(seed), []
    for _ in range(0, n_boot, 100):
        idx = rng.integers(0, len(a), (100, len(a)))
        cf.append(1 - a[idx].mean(1) / b[idx].mean(1))
    return float(np.concatenate(cf).std(ddof=1))


def score(pred, exact, te) -> dict:
    """Normalisation assertion first, then CF / per-k KL / kl_marginal, the 1e-6 floor rescoring (§9 iii) and the
    test-sequence bootstrap SD of CF (§9 i)."""
    pred = np.asarray(pred, float)
    m = tasks_e3.cf_and_kl(pred, exact, _y(te))
    m["cf_floor_1e6"] = tasks_e3.cf_and_kl(_renorm(pred, 1e-6), exact, _y(te))["cf"]
    m["cf_boot_sd"] = cf_boot_sd(pred, exact)
    return m


def _cf(pred, exact, te, check=True):
    return tasks_e3.cf_and_kl(np.asarray(pred, float), exact, _y(te), check=check)["cf"]


def _cf_valid(raw, exact, te):
    """CF from (seq, k) pairs whose raw prediction has no negative entry only (OOM report-only variant)."""
    raw = np.asarray(raw, float)
    ok = (raw >= 0).all(-1)
    q, p = np.clip(exact, 1e-12, 1), np.clip(raw, 1e-9, 1)
    kl_cell = (q * (np.log(q) - np.log(p))).sum(-1)
    kl = np.array([kl_cell[ok[:, k], k].mean() if ok[:, k].any() else np.nan for k in range(raw.shape[1])])
    return float(queue_e2.metric({"arm": "A", "per_k": {"kl": kl, "kl_marginal": tasks_e3.kl_marginal(exact)}}))


def _dims(process):
    p = tasks_e3.process(process)
    return p.O, p.d


def _subset(tr, j):
    n = j.get("train_subset")
    return tr if n is None else (tr[0][:n], tr[1][:n])


# ---------------------------------------------------------------- recipes (each returns a fit dict with "pred")
def _M(j, tr, va, te, exact):
    O, _ = _dims(j["process"])
    f = cplus.fit_M(tr, va, te, d=j["size"], O=O, seed=j["seed"], restarts=j.get("restarts", 48),
                    iters=j.get("mom_iters", 150), steps=j.get("steps", 3000), val_last=_val_last(j))
    f["cf_mom"] = _cf(f.pop("pred_mom"), exact, te)
    return f


def _val_last(j):
    return VAL_LAST if j["process"] == "A2w" else 0


def _C1(j, tr, va, te, exact):
    O, _ = _dims(j["process"])
    return cplus.fit_C1(tr, va, te, d=j["size"], O=O, seed=j["seed"], R1=j["R"], steps=j.get("steps", 6000))


def _S(j, tr, va, te, exact):
    O, _ = _dims(j["process"])
    f = stiefel.fit_S(tr, va, te, None, d=j["size"], O=O, seed=j["seed"], RS=j["R"], tau0=j["tau0"],
                      steps=j.get("steps", 6000))
    f["nan_flag"] = bool(f["nan_flags"][f["selected_restart"]])
    return f


def _C2(j, tr, va, te, exact):
    O, _ = _dims(j["process"])
    V = np.stack(tasks_e3.process(j["process"]).K)
    return cplus.fit_C2(tr, va, te, d=j["size"], O=O, seed=j["seed"], sigma=j["sigma"], V_true=V,
                        steps=j.get("steps", 3000), val_last=_val_last(j))


def _steps(j):
    return dict(steps_probe=j.get("steps_probe", 750), steps_total=j.get("steps_total", 10000))


def _CPLUS(j, tr, va, te, exact):
    O, _ = _dims(j["process"])
    return cplus.cplus_single(j["family"], j["size"], _subset(tr, j), va, te, O, j["seed"], 0, tuple(j["lrs"]),
                              **_steps(j))


def _CPLUS_M(j, tr, va, te, exact):
    O, _ = _dims(j["process"])
    return cplus.cplus_m(j["family"], j["size"], tr, va, te, O, j["seed"], tuple(j["lrs"]), j["R"], **_steps(j))


def _SENS(j, tr, va, te, exact, out):
    """R_c extra replicas (recipe 8), merged with the stored CPLUS-M / CPLUS base record by validation NLL."""
    O, _ = _dims(j["process"])
    base = json.loads((Path(out) / j["base"]).read_text())
    if base.get("failed") or base.get("val_nll") is None:
        raise RuntimeError(f"SENS base record {j['base']} is failed or has no validation NLL")
    f = cplus.sens(j["family"], j["size"], tr, va, te, O, j["seed"], tuple(j["lrs"]), j["R"], base=None, **_steps(j))
    f["replica_cf_extra"] = [score(p, exact, te)["cf"] for p in f.pop("replica_preds")]
    m = score(f.pop("pred"), exact, te)
    use_base = base["val_nll"] <= f["val_nll"]
    keep = base if use_base else m
    res = {k: v for k, v in f.items() if k not in ("cpu_s",)}
    res.update(cf=keep["cf"], kl=keep["kl"], kl_marginal=keep["kl_marginal"], cf_floor_1e6=keep.get("cf_floor_1e6"),
               cf_boot_sd=keep.get("cf_boot_sd"),
               val_nll=base["val_nll"] if use_base else f["val_nll"], sens_source="base" if use_base else "extra",
               cf_extra=m["cf"], val_nll_extra=f["val_nll"], cf_base=base["cf"], val_nll_base=base["val_nll"],
               cpu_s=f["cpu_s"], cpu_s_with_base=f["cpu_s"] + (base.get("cpu_s") or 0.0),
               base_result_sha256=base.get("result_sha256"), nan_flag=base.get("nan_flag") if use_base else f["nan_flag"])
    return res


def _LRGRID(j, tr, va, te, exact):
    """G10 learning-rate grid: one linearly decayed run per lr from an init (and batch stream) shared by all four lrs
    of the same (seed, family), so the lr comparison is not confounded with initialisation."""
    O, _ = _dims(j["process"])
    cplus._setup()
    t0 = time.process_time()
    s = cplus.init_seed(j["seed"], cplus.RECIPE["CPLUS"], LRGRID_R0)
    torch.manual_seed(s)
    m = models_e2.build(j["family"], j["size"], O, L_CTX)
    opt = torch.optim.Adam(m.parameters(), lr=j["lr"])
    st = cplus.new_state(m, opt)
    cplus.run_phase(m, opt, torch.Generator().manual_seed(s), cplus._tensors(tr), j["lr"], j.get("steps", 6000), True, st)
    res = {"pred": cplus.checked_pred(m, te), "val_nll": cplus.val_nll(m, va), "n_params": models.n_params(m),
           "cpu_s": time.process_time() - t0, "init_seed": s}
    res.update(cplus._diag(st, opt))
    return res


def _HMM(j, tr, va, te, exact):
    O, _ = _dims(j["process"])
    f = hmm_e3.fit_HMM(_subset(tr, j), va, te, O, j["size"], j["seed"], restarts=j.get("restarts", 48),
                       em_iters=j.get("em_iters", 100), mom_iters=j.get("mom_iters", 150))
    f["cf_mom_only"] = _cf(f.pop("pred_mom_only"), exact, te)
    f["nan_flag"] = False
    return f


def _OOM(j, tr, va, te, exact):
    O, _ = _dims(j["process"])
    f = oom_e3.fit_OOM(tr, va, te, O, j["size"], j["seed"], restarts=j.get("restarts", 48), steps=j.get("steps", 3000),
                       mom_iters=j.get("mom_iters", 150))
    cand = f.pop("cand_pred")
    f["cand_cf"] = {k: _cf(v, exact, te) for k, v in cand.items()}
    ft = [k for k in ("spectral_ft", "mom_ft") if k in cand]
    best_ft = min(ft, key=lambda k: (f["cand_val_nll"][k], ft.index(k)))
    f.update(cf_spec=f["cand_cf"]["spectral"], cf_ft=f["cand_cf"][best_ft], best_ft=best_ft)
    raw = f.pop("pred_raw")
    f.update(cf_raw_unnormalised=_cf(raw, exact, te, check=False), cf_valid_pairs=_cf_valid(raw, exact, te))
    return f


def _ref(j, tr, va, te, exact):
    """REF predictors plus the label-free fixture checks of §9 (i) and the closed-form parameter counts (iii)."""
    name, seed = j["process"], j["seed"]
    proc = tasks_e3.process(name)
    O, steps = proc.O, j.get("ref_steps", 200)
    t0 = time.process_time()
    ref = {f"markov{m}": _cf(tasks_e3.markov_k(tr[0], te[0], m, K, O), exact, te) for m in (1, 2, 3)}
    ref["linear_hard"] = _cf(tasks_e3.linear_context(tr[0], te[0], K, O, steps=steps), exact, te)
    ns = min(20_000, len(tr[0]))
    soft = tasks_e2.exact_predictive(proc, tr[0][:ns], K)
    ref["linear_soft"] = _cf(tasks_e3.linear_context(tr[0][:ns], te[0], K, O, steps=steps, soft=soft), exact, te)
    rng = np.random.default_rng(cplus.init_seed(seed, REF_SHUFFLE_CODE, 0))
    cf_shuf = _cf(tasks_e3.shuffled_exact(proc, te[0], K, rng), exact, te)
    marg = exact[:, -1].mean(0)
    klm = tasks_e3.kl_marginal(exact)
    mix = {}
    for lam in (0.5, 0.9):
        p = lam * exact + (1 - lam) * marg
        q = np.clip(exact, 1e-12, 1)
        direct = 1 - ((q * (np.log(q) - np.log(np.clip(p, 1e-9, 1)))).sum(-1).mean(0)[:10].mean()) / klm[:10].mean()
        mix[str(lam)] = {"cf": _cf(p, exact, te), "cf_direct": float(direct)}
    stat = tasks_e3.stationary_marginal(proc)
    return {"kraus_vp": kraus_vp_check(proc, te, exact), "ref": ref, "cf_shuffled": cf_shuf, "cf": None, "val_nll": None, "n_params": 0, "nan_flag": False,
            "cf_exact": _cf(exact, exact, te), "cf_marginal": _cf(np.broadcast_to(marg, exact.shape), exact, te),
            "cf_mix": mix, "denominator_k10": float(klm[:10].mean()), "share_k3": float(klm[:3].sum() / klm[:10].sum()),
            "marginal_vs_stationary_rel": float(np.max(np.abs(marg - stat) / stat)), "kl_marginal": klm.tolist(),
            "n_params_checks": n_params_checks(name), "cpu_s": time.process_time() - t0}


def kraus_vp_check(proc, te, exact, n=2000) -> dict:
    """PC-D: the true Kraus operators written as G = V P into HQMMWM reproduce the exact predictive (max abs error
    < 2e-3 from rho0 = I/d) and give finite gradients."""
    m = models_e2.HQMMWM(proc.O, proc.d, L_CTX)
    hq_init.load_warm_start(m, torch.as_tensor(np.stack(proc.K), dtype=torch.complex128).reshape(proc.O * proc.d, proc.d))
    err = float(np.abs(predict_torch(m, te[0][:n], te[1][:n], K) - exact[:n]).max())
    o = torch.as_tensor(te[0][:64])
    p = m(o, torch.as_tensor(te[1][:64]), K)
    (-torch.log(torch.gather(p, 2, o[:, L_CTX:L_CTX + K].unsqueeze(-1)).clamp_min(1e-9)).mean()).backward()
    fin = all(bool(torch.isfinite(t.grad).all()) for t in (m.g_re, m.g_im))
    return {"max_abs_err": err, "grad_finite": fin, "ok": bool(err < 2e-3 and fin)}


def n_params_checks(process) -> dict:
    """Instantiated parameter counts against the closed forms of §9 (iii) for this process."""
    out = {}
    for (p, fam, size), want in CLOSED_FORM.items():
        if p != process:
            continue
        O, _ = _dims(p)
        if fam in ("HQ", "GRU", "LRU"):
            got = models.n_params(models_e2.build(fam, size, O, L_CTX))
        elif fam == "HMM":
            got = models_e2.hmm_params(size, O)
        else:
            got = oom_e3.n_params(O, size)
        out[f"{fam}{size}"] = {"got": int(got), "want": want, "ok": int(got) == want}
    return out


RECIPES = {"M": _M, "C1": _C1, "S": _S, "C2": _C2, "CPLUS": _CPLUS, "CPLUS-M": _CPLUS_M, "LRGRID": _LRGRID,
           "HMM-C": _HMM, "OOM-C": _OOM, "REF": _ref}


# ---------------------------------------------------------------- C0: unmodified run_e2.main in a subprocess
def _C0(j, out):
    assert j["process"] in ("A1", "A2") and j["family"] == "HQ"
    c0dir = Path(out) / "c0"
    if _inside(c0dir, E2_RUNS):
        raise ValueError(f"C0 output {c0dir} would be inside runs/e2")
    c0dir.mkdir(parents=True, exist_ok=True)
    steps = j.get("steps", 6000)
    cmd = [sys.executable, "-c", "import torch; torch.set_num_threads(1); from osw.run_e2 import main; main()",
           "--arm", j["process"], "--family", "HQ", "--size", str(j["size"]), "--seed", str(j["seed"]),
           "--lr", str(j["lr"]), "--steps", str(steps), "--out", str(c0dir)]
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), **{v: "1" for v in THREAD_VARS})
    r0, t0 = resource.getrusage(resource.RUSAGE_CHILDREN), time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=ROOT)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    if p.returncode:
        raise RuntimeError(f"run_e2.main rc={p.returncode}: {p.stderr[-800:]}")
    f = c0dir / f"e2-{j['process']}-HQ{j['size']}-s{j['seed']}-lr{j['lr']:g}.json"
    if not f.exists() or f.stat().st_mtime < t0 - 1:
        raise RuntimeError(f"run_e2.main did not write {f}")
    e2 = json.loads(f.read_text())
    pk = e2["per_k"]
    kl, klm = np.asarray(pk["kl"], float), np.asarray(pk["kl_marginal"], float)
    val = e2["val_nll"]
    return {"cf": float(queue_e2.metric(e2)), "kl": kl.tolist(), "kl_marginal": klm.tolist(), "val_nll": val,
            "init_val_nll": e2["init_val_nll"], "n_params": e2["n_params"], "cpu_s": cpu, "steps": steps,
            "train_curve": e2["train_curve"], "grad_norm_curve": e2["grad_norm_curve"],
            "nan_flag": bool(val is None or not np.isfinite(val) or not np.isfinite(kl).all()),
            "e2_file": str(f.resolve().relative_to(ROOT)) if _inside(f, ROOT) else str(f),
            "e2_result_sha256": e2["result_sha256"], "e2_code_commit": e2.get("code_commit"),
            "normalisation": "HQMMWM.forward renormalises; run_e2 does not return pred, so the sum check is by construction"}


# ---------------------------------------------------------------- one job
def run_job(j: dict, out="runs/e3", attempt: int = 1) -> dict:
    out = Path(out) if Path(out).is_absolute() else ROOT / out
    if _inside(out, E2_RUNS):
        raise ValueError(f"E3 output {out} would be inside runs/e2")
    torch.set_num_threads(1)
    models.L_CTX = hmm.L_CTX = L_CTX                  # this process only (as run_e2.main)
    wall0, cpu0 = time.time(), time.process_time()
    rec = dict(j, job=dict(j), attempt=attempt, design_sha256=design_sha256(), code_manifest_sha256=code_manifest_sha256(),
               started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), threads=torch.get_num_threads())
    try:
        if j["recipe"] == "C0":
            res = _C0(j, out)
        else:
            tr, va, te, exact = _load(j)
            res = _SENS(j, tr, va, te, exact, out) if j["recipe"] == "SENS" else RECIPES[j["recipe"]](j, tr, va, te, exact)
            if "replica_preds" in res:                    # CPLUS-M: every replica's test CF, report-only (D7)
                res["replica_cf"] = [score(p, exact, te)["cf"] for p in res.pop("replica_preds")]
            if "pred" in res:
                res.update(score(res.pop("pred"), exact, te))
        res.setdefault("nan_flag", False)
        exp = CLOSED_FORM.get((j["process"], j["family"], j["size"]))
        res["n_params_closed_form_ok"] = None if exp is None else int(res["n_params"]) == exp
        rec.update({("fit_" + k if k in j else k): v for k, v in res.items()}, failed=False)
    except Exception as e:                                  # LinAlgError and friends; NaN is not a failure (§4)
        rec.update(failed=True, error=f"{type(e).__name__}: {e}"[:2000], cf=None, val_nll=None, nan_flag=False)
    rec.update(wall_s=round(time.time() - wall0, 2), job_cpu_s=time.process_time() - cpu0)
    if j["recipe"] == "C0":
        rec["job_cpu_s"] += rec.get("cpu_s") or 0.0
    f = write_record(rec, out)
    return json.loads(f.read_text())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="job dict as JSON")
    ap.add_argument("--out", default="runs/e3")
    ap.add_argument("--attempt", type=int, default=1)
    a = ap.parse_args(argv)
    r = run_job(json.loads(a.job), a.out, a.attempt)
    print(f"{job_name(r)}: failed={r['failed']} cf={r.get('cf')} val={r.get('val_nll')} cpu={r.get('cpu_s')}"
          + (f" error={r.get('error')}" if r["failed"] else ""))
    return 1 if r["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())

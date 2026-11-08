"""E3 mechanical decision rules (E3-DESIGN §6-§8): label functions, BCa and joint-seed bootstrap, failure imputation, summarize().

Reads run-result dicts only and never trains. Each rule returns exactly one label from an ordered exhaustive table.
Continuous comparisons carry a relative tolerance of 1e-9 and a tie is resolved against the hypothesis the rule tests
(`tie=` below). Counts are exact integers. Expected record keys: block, arm, process ("A1", "A2", "A2w", "A3"), family
("HQ", "GRU", "LRU", "HMM", "OOM", "REF"), size (HQ d, GRU h, LRU n, HMM n, OOM r), recipe ("M", "C0", "C1", "S", "C2",
"CPLUS", "CPLUS-M", "SENS", "HMM-C", "OOM-C", "REF"), seed, sigma (C2 only, else None), cf, val_nll, n_params, cpu_s,
nan_flag, failed. Optional: cf_ft / cf_spec on OOM-C records (H7 relabel), cf_shuffled on REF records (NC),
replica_cf on CPLUS-M records (D7). summarize() also takes the pre-freeze outcomes that change the registered design
(G4 / G6 removals, H6 demotion by the G8 planted simulation, the G9 cuts) and emits their pre-registered labels.
"""
from __future__ import annotations

import json
import math
from fractions import Fraction
from statistics import NormalDist

import numpy as np

REL = 1e-9
SUCCESS = 0.90
BOOT_SEED = 20261025
N_BOOT = 10_000
CI_LEVEL = 0.95
P_HQ = 54
H6_LB_PCT = 0.01
LIN_CEIL_A2 = 0.768
E2_MEANS_A2 = {("GRU", 2): 0.210, ("GRU", 4): 0.288, ("GRU", 8): 0.741, ("GRU", 16): 0.809,
               ("LRU", 2): 0.502, ("LRU", 4): 0.665, ("LRU", 8): 0.717, ("LRU", 16): 0.775}
PCK_A1 = {"GRU": (4, 0.93), "LRU": (16, 0.858)}
ZERO_IMPUTE = {"M", "C2"}                       # tested HQ arm and the oracle reference: a failed seed counts as CF 0
LADDER = {"GRU": [(2, "CPLUS-M"), (4, "CPLUS-M"), (8, "CPLUS-M"), (16, "CPLUS"), (32, "CPLUS")],
          "LRU": [(2, "CPLUS-M"), (4, "CPLUS-M"), (8, "CPLUS-M"), (16, "CPLUS"), (32, "CPLUS")]}
SIGMAS = (0.05, 0.1, 0.2, 0.3)

PASS, FAIL, UNINT, NOT_RUN = "PASS", "FAIL", "UNINTERPRETABLE", "NOT_RUN"
KILL, PARTIAL = "KILL", "PARTIAL"
REACHED, NOT_REACHED = "REACHED", "NOT_REACHED"
LOCAL, NONLOCAL, UNDET = "LOCAL", "NONLOCAL", "UNDETERMINED"
REFUTED, ESTABLISHED = "REFUTED", "ESTABLISHED"
HOLDS, NOT_HOLDS, COMP_HOLDS = "HOLDS", "NOT_HOLDS", "COMPONENT_HOLDS"
NOT_EST, NOT_EST_IMP = "NOT_ESTABLISHED", "NOT_ESTABLISHED_IMPRECISE"
NOT_JUDGED, NOT_JUDGED_COMPUTE = "NOT_JUDGED", "NOT_JUDGED_COMPUTE_MISMATCH"
STOP, NO_STOP = "STOP_TRIGGERED", "NOT_TRIGGERED"
EQUIV, HQ_BETTER, OOM_BETTER, OOM_OPT_FAIL, UNCERTAIN = "EQUIVALENT", "HQ_BETTER", "OOM_BETTER", "OOM_OPT_FAILURE", "UNCERTAIN"
REPRODUCED, NOT_REPRODUCED = "REPRODUCED", "NOT_REPRODUCED"
DESCRIPTIVE, STAT_LIMITED = "DESCRIPTIVE", "STAT_LIMITED_NOT_JUDGED"
NOT_DECIDABLE_PF, REMOVED_PF, NOT_RUN_COMPUTE = "NOT_DECIDABLE_PRE_FREEZE", "REMOVED_PRE_FREEZE", "NOT_RUN_COMPUTE"
PREDICTIONS = {                                  # frozen E3 block of PREDICTIONS-BEFORE-RESULTS.md (author, dev-informed; Brier, §8)
    "H1": {PASS: 0.85, PARTIAL: 0.12, KILL: 0.03}, "H2": {NOT_REACHED: 0.85, PARTIAL: 0.12, REACHED: 0.03},
    "H3a": {PASS: 0.92, PARTIAL: 0.06, KILL: 0.02}, "H3b": {PASS: 0.03, PARTIAL: 0.08, KILL: 0.85, STAT_LIMITED: 0.04},
    "H4": {LOCAL: 0.92, UNDET: 0.06, NONLOCAL: 0.02}, "H5": {ESTABLISHED: 0.92, UNDET: 0.06, REFUTED: 0.02},
    "H6": {HOLDS: 0.75, NOT_HOLDS: 0.25}, "H7": {EQUIV: 0.88, HQ_BETTER: 0.02, OOM_BETTER: 0.02, UNCERTAIN: 0.08},
    "STOP": {STOP: 0.03, NO_STOP: 0.97}, "PC-B": {PASS: 0.95, FAIL: 0.05}, "PC-K": {PASS: 0.85, FAIL: 0.15},
    "PC-C (modifier)": {REPRODUCED: 0.95, PARTIAL: 0.04, NOT_REPRODUCED: 0.01}}


# ---------- tolerant comparisons ----------
def ge(x, t, tie=False):
    x, t = np.asarray(x, float), np.asarray(t, float)
    eq = np.abs(x - t) <= REL * np.maximum(np.abs(x), np.abs(t))
    r = np.where(eq, tie, x > t)
    return bool(r) if r.ndim == 0 else r


def le(x, t, tie=False):
    return ge(-np.asarray(x, float), -np.asarray(t, float), tie)


def lt(x, t, tie=False):
    r = np.logical_not(ge(x, t, not tie))
    return bool(r) if np.ndim(r) == 0 else r


def gt(x, t, tie=False):
    r = np.logical_not(le(x, t, not tie))
    return bool(r) if np.ndim(r) == 0 else r


def success(cf):
    return ge(cf, SUCCESS, tie=False)


# ---------- count bounds and operating characteristics ----------
def _tail_ge(k, n, p):
    return sum(math.comb(n, j) * p ** j * (1 - p) ** (n - j) for j in range(k, n + 1))


def _bisect(f, lo=0.0, hi=1.0):
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) else (lo, mid)
    return (lo + hi) / 2


def cp_lower(k, n, alpha=0.05):
    """One-sided Clopper-Pearson lower bound: P(X >= k | p) = alpha."""
    return 0.0 if k == 0 else _bisect(lambda p: _tail_ge(k, n, p) < alpha)


def cp_upper(k, n, alpha=0.05):
    """One-sided Clopper-Pearson upper bound: P(X <= k | p) = alpha."""
    return 1.0 if k == n else _bisect(lambda p: 1 - _tail_ge(k + 1, n, p) > alpha)


def operating_characteristics(label_of_count, p, n=10):
    out = {}
    for s in range(n + 1):
        lab = label_of_count(s)
        out[lab] = out.get(lab, 0.0) + math.comb(n, s) * p ** s * (1 - p) ** (n - s)
    return out


# ---------- bootstrap ----------
def mean_stat(x):
    return x[..., 0].mean(-1)


def h5_stat(x):
    return x[..., 0].mean(-1) - np.maximum(x[..., 1].mean(-1), x[..., 2].mean(-1))


def boot_indices(n, n_boot=N_BOOT, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n, (n_boot, n))


def bca(x, stat=mean_stat, level=CI_LEVEL, n_boot=N_BOOT, seed=BOOT_SEED):
    """Two-sided BCa interval over seeds (rows); stat maps (..., n, k) -> (...)."""
    x = np.asarray(x, float)
    x = x[:, None] if x.ndim == 1 else x
    n = len(x)
    theta = float(stat(x))
    boot = stat(x[boot_indices(n, n_boot, seed)])
    if np.ptp(boot) == 0:
        return theta, theta
    nd = NormalDist()
    p0 = (np.sum(boot < theta) + 0.5 * np.sum(boot == theta)) / n_boot
    z0 = nd.inv_cdf(min(max(p0, 0.5 / n_boot), 1 - 0.5 / n_boot))
    jack = np.array([stat(np.delete(x, i, 0)) for i in range(n)])
    d = jack.mean() - jack
    den = 6 * np.sum(d ** 2) ** 1.5
    a = float(np.sum(d ** 3) / den) if den > 0 else 0.0
    out = []
    for q in ((1 - level) / 2, (1 + level) / 2):
        w = z0 + nd.inv_cdf(q)
        out.append(float(np.quantile(boot, nd.cdf(z0 + w / max(1 - a * w, 1e-12)))))
    return out[0], out[1]


def interval_level(coverage):
    """§8 registered fallback: BCa coverage below 0.94 in the G8 simulation switches H5/H7/PC-B to 97.5% intervals."""
    return CI_LEVEL if ge(coverage, 0.94, tie=False) else 0.975


# ---------- validity gates and controls ----------
def pc_a_label(k, n, median_ok):
    return PASS if Fraction(k, n) >= Fraction(4, 5) and median_ok else FAIL


def pc_a(cf_o05, median_clause=True):
    cf = np.asarray(cf_o05, float)
    med_ok = ge(np.median(cf), 0.95) if median_clause else True
    return pc_a_label(int(np.sum(success(cf))), len(cf), med_ok)


def pc_b(delta, lo):
    return PASS if ge(delta, 0.10) and gt(lo, 0.0) else FAIL


def pc_c(cf_c0):
    cf = np.asarray(cf_c0, float)
    m = float(cf.mean())
    if le(m, 0.30) and Fraction(int(np.sum(lt(cf, 0.30))), len(cf)) >= Fraction(4, 5):
        return REPRODUCED
    if gt(m, 0.50, tie=True):
        return NOT_REPRODUCED
    return PARTIAL


def pc_k_a(e3_medians):
    """e3_medians: {(family, size): seed-median CF} on the 8 cells shared with E2 (A2)."""
    keys = list(E2_MEANS_A2)
    med = np.array([e3_medians[k] for k in keys], float)
    e2 = np.array([E2_MEANS_A2[k] for k in keys])
    return PASS if ge(np.mean(med - e2), -0.01) and all(ge(med, e2 - 0.10)) else FAIL


def pc_k_b(gru4_a1, lru16_a1):
    return PASS if ge(gru4_a1, PCK_A1["GRU"][1]) and ge(lru16_a1, PCK_A1["LRU"][1]) else FAIL


def pc_k(a_label, b_label):
    return PASS if a_label == PASS and b_label == PASS else FAIL


def pc_o(median_cf):
    return PASS if ge(median_cf, 0.95) else FAIL


def nc(cf_shuffled):
    return PASS if all(le(np.asarray(cf_shuffled, float), 0.0)) else FAIL


def compute_check(cell_median_cpu, hq3m_median_cpu):
    return PASS if all(ge(np.asarray(cell_median_cpu, float), 0.8 * hq3m_median_cpu)) else FAIL


# ---------- family 1 and 2 ----------
def h1(s, pca_pass):
    if not pca_pass:
        return UNINT
    return KILL if s <= 6 else PASS if s >= 9 else PARTIAL


def h2(s):
    return REACHED if s >= 9 else NOT_REACHED if s <= 1 else PARTIAL


def transfer_success(cf_m, cf_o05, cf_c0=None):
    if lt(cf_o05, 0.50, tie=True):
        return False
    ok = ge(cf_m, 0.95 * cf_o05) and ge(cf_m, 0.50)
    return bool(ok and (cf_c0 is None or ge(cf_m, cf_c0 - 0.01)))


def h3(s):
    return KILL if s <= 5 else PASS if s >= 9 else PARTIAL


# ---------- mechanism ----------
def h4(k, n):
    """k of n seeds with CF >= 0.90 at sigma 0.3; exact rational shares, boundaries inclusive as written."""
    f = Fraction(k, n)
    return LOCAL if f <= Fraction(1, 5) else NONLOCAL if f >= Fraction(3, 5) else UNDET


def sigma_star(sigmas, shares, level=0.5):
    """Linear interpolation of the sigma where the success share first drops below `level`."""
    s, f = np.asarray(sigmas, float), np.asarray(shares, float)
    below = np.flatnonzero(f < level)
    if len(below) == 0:
        return None, "above_grid"
    i = int(below[0])
    if i == 0:
        return None, "below_grid"
    return float(s[i - 1] + (f[i - 1] - level) / (f[i - 1] - f[i]) * (s[i] - s[i - 1])), None


def h5_label(mean_c1, lo):
    if ge(mean_c1, 0.80, tie=True):
        return REFUTED
    if ge(lo, 0.40) and le(mean_c1, 0.50):
        return ESTABLISHED
    return UNDET


def h5_combine(label_all, label_no_nan):
    """The with-NaN version is primary; ESTABLISHED also needs the without-NaN version (§6), else UNDETERMINED."""
    if label_all == ESTABLISHED and label_no_nan != ESTABLISHED:
        return UNDET
    return label_all


# ---------- family 3 ----------
def p_from_medians(med, params):
    """med (..., m) seed-median CF per rung with params increasing -> (P_F, code); code 0 crossed, 1 left-, 2 right-censored."""
    med = np.asarray(med, float)
    params = np.asarray(params, float)
    env = np.maximum.accumulate(med, axis=-1)
    hit = np.asarray(ge(env, SUCCESS, tie=True))
    any_hit, first = hit.any(-1), np.argmax(hit, -1)
    i = np.clip(first, 1, len(params) - 1)
    e0 = np.take_along_axis(env, (i - 1)[..., None], -1)[..., 0]
    e1 = np.take_along_axis(env, i[..., None], -1)[..., 0]
    lp = np.log(params)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.clip(np.nan_to_num((SUCCESS - e0) / (e1 - e0), nan=1.0), 0.0, 1.0)
    P = np.exp(lp[i - 1] + frac * (lp[i] - lp[i - 1]))
    P = np.where(first == 0, params[0], P)
    P = np.where(any_hit, P, params[-1])
    code = np.where(any_hit, np.where(first == 0, 1, 0), 2)
    return P, code


def h6_component(R, lb):
    if lt(R, 2.0, tie=True):
        return KILL
    if ge(R, 4.0) and ge(lb, 2.0):
        return COMP_HOLDS
    return NOT_EST_IMP if ge(R, 4.0) else NOT_EST


def h6_ratios(hq_cf, ladders, n_boot=N_BOOT, seed=BOOT_SEED):
    """hq_cf (n,); ladders {family: (params (m,), cf (m, n))}, seeds aligned across HQ and every rung.

    Joint seed bootstrap (HQ and classical resampled together); a replicate whose HQ median is below 0.90 has R = 0.
    """
    hq = np.asarray(hq_cf, float)
    idx = boot_indices(len(hq), n_boot, seed)
    hq_ok = ge(np.median(hq), SUCCESS)
    hq_boot_ok = np.asarray(ge(np.median(hq[idx], -1), SUCCESS))
    out = {}
    for fam, (params, cf) in ladders.items():
        params, cf = np.asarray(params, float), np.asarray(cf, float)
        med = np.median(cf, -1)
        P, code = p_from_medians(med, params)
        R = float(P) / P_HQ if hq_ok else 0.0
        Pb, cb = p_from_medians(np.median(cf[:, idx], -1).T, params)
        Rb = np.where(hq_boot_ok, Pb / P_HQ, 0.0)
        lb = float(np.quantile(Rb, H6_LB_PCT))
        frac_cens = float(np.mean(hq_boot_ok & (cb == 2)))
        out[fam] = {"R": R, "lb": lb, "P_F": float(P), "censored": ["none", "left", "right"][int(code)],
                    "frac_right_censored": frac_cens, "censor_report": bool(frac_cens >= 0.5),
                    "frac_r0": float(np.mean(~hq_boot_ok)), "median_cf": med.tolist(), "params": params.tolist(),
                    "max_rung_median": float(med[-1]), "label": h6_component(R, lb)}
        if fam == "LRU" and int(code) == 2 and abs(med[-1] - LIN_CEIL_A2) <= 0.03:
            out[fam]["note"] = "possible architectural ceiling, not parameter efficiency"
    return out


def h6_rule(component_labels):
    return HOLDS if all(l == COMP_HOLDS for l in component_labels) else NOT_HOLDS


def h6_gate(h1_label, pcb_label, pck_label, compute_label):
    """None when every H6 gate passes, else (label, reason)."""
    if h1_label != PASS or pcb_label != PASS:
        return NOT_JUDGED, f"SC block gated off (H1={h1_label}, PC-B={pcb_label})"
    if pck_label != PASS:
        return UNINT, f"PC-K={pck_label}"
    if compute_label != PASS:
        return NOT_JUDGED_COMPUTE, f"compute check={compute_label}"
    return None


def stop_rule(gate_ok, ratios):
    if not gate_ok:
        return NOT_JUDGED
    return STOP if lt(min(ratios), 2.0, tie=True) else NO_STOP


def h7(delta, lo, hi, oom_opt_failure=False):
    if ge(lo, -0.03) and le(hi, 0.03):
        return EQUIV
    if ge(delta, 0.05) and gt(lo, 0.0):
        return OOM_OPT_FAIL if oom_opt_failure else HQ_BETTER
    if le(delta, -0.05) and lt(hi, 0.0):
        return OOM_BETTER
    return UNCERTAIN


# ---------- run results -> cells (with §8 failure imputation) ----------
def cell_key(r):
    sig = r.get("sigma")
    return (r["process"], r["family"], int(r["size"]), r["recipe"], None if sig is None else round(float(sig), 6))


def build_cells(results):
    """Group records by cell; per seed keep the non-failed run with the lowest val_nll (e.g. C0's two lrs).

    A seed whose runs all failed is imputed against the hypothesis: CF 0 for the tested arm M and the oracle C2,
    otherwise the maximum CF over successful seeds of the same arm and size. >1/10 failed seeds marks the cell.
    A run that hit NaN without a finite checkpoint to score (cf None, nan_flag; only C0's unmodified E2 protocol can
    produce one) is not a job failure (§4): it is imputed the same way but counted as a NaN, not a failure.
    """
    by = {}
    for r in results:
        if r.get("family") == "REF":
            continue
        by.setdefault(cell_key(r), {}).setdefault(int(r["seed"]), []).append(r)
    cells = {}
    for key, seeds in by.items():
        pick = {}
        for s, rs in seeds.items():
            ok = [r for r in rs if not r.get("failed") and r.get("cf") is not None]
            if ok:
                pick[s] = min(ok, key=lambda r: np.inf if r.get("val_nll") is None else r["val_nll"])
        order = sorted(seeds)
        nan_nocf = [s for s in order if s not in pick and any(not r.get("failed") and r.get("nan_flag") for r in seeds[s])]
        failed = [s for s in order if s not in pick and s not in nan_nocf]
        fill = 0.0 if key[3] in ZERO_IMPUTE or not pick else max(float(r["cf"]) for r in pick.values())
        cf = np.array([float(pick[s]["cf"]) if s in pick else fill for s in order])
        nan = np.array([bool(pick[s].get("nan_flag")) if s in pick else s in nan_nocf for s in order])
        cpu = [pick[s].get("cpu_s") for s in pick if pick[s].get("cpu_s") is not None]
        npar = [pick[s].get("n_params") for s in pick if pick[s].get("n_params") is not None]
        cells[key] = {"seeds": order, "cf": cf, "nan": nan, "n_failed": len(failed), "failed_seeds": failed,
                      "nan_imputed_seeds": nan_nocf,
                      "n_nan": int(nan.sum()), "uninterpretable": Fraction(len(failed), len(order)) > Fraction(1, 10),
                      "cpu_med": float(np.median(cpu)) if cpu else None,
                      "n_params": int(np.median(npar)) if npar else None, "records": pick}
    return cells


def lower_bracket(cells, family, cut=()):
    """SENS rung (§4): (size, recipe) of the rung just below the first crossing of 0.90 by the seed-median envelope;
    the largest rung if right-censored, the smallest if the smallest already crosses. None while a cell is missing."""
    rungs = [(s, r) for s, r in LADDER[family] if not (s == 32 and "TOP32" in cut)]
    keys = [("A2", family, s, r, None) for s, r in rungs]
    if any(k not in cells for k in keys):
        return None
    env = np.maximum.accumulate([float(np.median(cells[k]["cf"])) for k in keys])
    hit = np.asarray(ge(env, SUCCESS, tie=True))
    i = len(rungs) - 1 if not hit.any() else max(int(np.argmax(hit)) - 1, 0)
    return rungs[i]


def align(cells, keys):
    """Common seeds across cells -> (seeds, {key: cf array}, {key: nan array}); None if a cell is absent."""
    if any(k not in cells for k in keys):
        return None
    common = sorted(set.intersection(*[set(cells[k]["seeds"]) for k in keys]))
    if not common:
        return None
    pos = {k: [cells[k]["seeds"].index(s) for s in common] for k in keys}
    return (common, {k: cells[k]["cf"][pos[k]] for k in keys}, {k: cells[k]["nan"][pos[k]] for k in keys})


# ---------- summarize ----------
def _ck(process, family, size, recipe, sigma=None):
    return (process, family, size, recipe, None if sigma is None else round(float(sigma), 6))


A2_M, A2_C0, A2_C1, A2_S = (_ck("A2", "HQ", 3, r) for r in ("M", "C0", "C1", "S"))
A1_M, A1_C0, A1_C2 = _ck("A1", "HQ", 2, "M"), _ck("A1", "HQ", 2, "C0"), _ck("A1", "HQ", 2, "C2", 0.05)
A2W_M, A2W_C2 = _ck("A2w", "HQ", 3, "M"), _ck("A2w", "HQ", 3, "C2", 0.05)
A1_HMM4, A1_OOM3 = _ck("A1", "HMM", 4, "HMM-C"), _ck("A1", "OOM", 3, "OOM-C")


def a2_c2(sigma):
    return _ck("A2", "HQ", 3, "C2", sigma)


def _py(x):
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return _py(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, Fraction):
        return float(x)
    return x


def _entry(label, inputs=None, reason=None):
    e = {"label": label}
    if reason:
        e["reason"] = reason
    if inputs is not None:
        e["inputs"] = inputs
    else:
        e["withheld"] = label in (UNINT, NOT_JUDGED, NOT_JUDGED_COMPUTE)
    return e


def _failed_arms(cells, keys):
    bad = [f"{k[0]}/{k[1]}{k[2]}/{k[3]}" + ("" if k[4] is None else f"/s{k[4]}") for k in keys
           if k in cells and cells[k]["uninterpretable"]]
    return f"arm(s) with >1/10 failed seeds: {', '.join(bad)}" if bad else None


def _med(x):
    x = [float(v) for v in x if v is not None]
    return float(np.median(x)) if x else None


def _recs(results, **kw):
    return [r for r in results if not r.get("failed") and all(r.get(k) == v for k, v in kw.items())]


def d7(C, results, cut=()):
    """D7 (registered H6 robustness, report-only): on seeds 20-24 the SENS value replaces the lower bracketing rung
    and R_F is recomputed against CPLUS-M on the same seeds ("H6 not robust": >= 4 -> < 4; "classical curve
    optimisation-limited": rung delta of seed-median CF >= 0.03); plus CF(selected) - CF(replica 0) on the decisive
    cells (single vs R_c replicas). Never changes the H6 label; SENS never enters STOP."""
    out = {"replica_delta_cf": {}}
    for f in LADDER:
        for s, r in LADDER[f][:3]:
            k = _ck("A2", f, s, r)
            if k in C:
                d = [x["cf"] - x["replica_cf"][0] for x in C[k]["records"].values() if x.get("replica_cf")]
                out["replica_delta_cf"][f"{f}{s}"] = _med(d)
    if "SENS" in cut:
        out["sens"] = "cut by G9: only the single- vs multi-replica difference is reported"
        return out
    fams, flags = {}, {"H6_not_robust": False, "classical_optimisation_limited": False}
    for f in LADDER:
        sens = {int(r["seed"]): r for r in _recs(results, recipe="SENS", family=f, process="A2") if r.get("cf") is not None}
        rungs = [(s, r) for s, r in LADDER[f] if not (s == 32 and "TOP32" in cut)]
        keys = [_ck("A2", f, s, r) for s, r in rungs]
        if not sens or any(k not in C for k in keys):
            fams[f] = {"label": NOT_RUN, "reason": "SENS or ladder cells missing"}
            continue
        size = int(next(iter(sens.values()))["size"])
        seeds = sorted(set(sens).intersection(*[set(C[k]["seeds"]) for k in keys]))
        i = [s for s, _ in rungs].index(size)
        base = np.array([[C[k]["cf"][C[k]["seeds"].index(sd)] for sd in seeds] for k in keys])
        sub = base.copy()
        sub[i] = [float(sens[sd]["cf"]) for sd in seeds]
        params = np.array([C[k]["n_params"] for k in keys], float)
        r_base = float(p_from_medians(np.median(base, -1), params)[0]) / P_HQ
        r_sens = float(p_from_medians(np.median(sub, -1), params)[0]) / P_HQ
        dcf = float(np.median(sub[i]) - np.median(base[i]))
        nr, ol = bool(ge(r_base, 4.0) and not ge(r_sens, 4.0)), bool(ge(dcf, 0.03, tie=True))
        exp = lower_bracket(C, f, cut)
        fams[f] = {"rung": size, "rung_matches_lower_bracket": exp is not None and exp[0] == size, "seeds": seeds,
                   "R_cplus_m": r_base, "R_sens": r_sens, "delta_cf": dcf, "not_robust": nr, "optimisation_limited": ol}
        flags["H6_not_robust"] |= nr
        flags["classical_optimisation_limited"] |= ol
    out.update(families=fams, flags=flags)
    return out


def descriptive(C, results, cut=(), g6=True):
    """D1-D8 (§6, report-only, labelled exploratory) and the per-arm CPU report (§10 item 9)."""
    cut, D = set(cut), {}
    D["D1"] = {"label": NOT_RUN, "reason": "the A2 small-budget difference is not defined operationally in revision 2"}
    oom9 = _recs(results, recipe="OOM-C", process="A2", size=9)
    D["D2"] = ({"label": REMOVED_PF, "reason": "G6 failed"} if not g6 else {"label": NOT_RUN, "reason": "cut by G9"}
               if "O" in cut else {"n": len(oom9), **{f"median_{k}": _med([r.get(k) for r in oom9]) for k in
                                                      ("cf", "nr", "cf_raw_unnormalised", "cf_valid_pairs", "cf_spec", "cf_ft")}})
    lru = _recs(results, recipe="CPLUS", process="A2w", family="LRU", size=16)
    refw = [r["ref"] for r in _recs(results, recipe="REF", process="A2w") if r.get("ref")]
    D["D3"] = {"label": NOT_RUN, "reason": "cut by G9"} if "D3" in cut else {
        "n": len(lru), "median_cf_lru16_20k": _med([r.get("cf") for r in lru]),
        "median_linear_hard": _med([x.get("linear_hard") for x in refw]),
        "median_linear_soft": _med([x.get("linear_soft") for x in refw]),
        "note": "A2w HQ (120k sequences) is never tabulated next to this LRU (20k sequences)"}
    d4 = _recs(results, recipe="M", process="A2", family="HQ", size=4)
    D["D4"] = {"label": NOT_RUN, "reason": "cut by G9"} if "D4" in cut else {
        "n": len(d4), "median_cf": _med([r.get("cf") for r in d4])}
    m3 = _recs(results, recipe="M", process="A2", family="HQ", size=3)
    D["D5"] = {"median_mom_loss": _med([r.get("mom_loss") for r in m3]), "median_cf_mom": _med([r.get("cf_mom") for r in m3]),
               "plateau_labels": "not computed: P1 / escaped / P2 are defined in revision 1, which is not available"}
    hq = C.get(A2_M, {}).get("cpu_med")
    cpu = {}
    for k, c in sorted(C.items(), key=lambda kv: str(kv[0])):
        v = [r.get("cpu_s") for r in c["records"].values() if r.get("cpu_s") is not None]
        if v:
            q1, q3 = np.percentile(v, [25, 75])
            cpu["/".join(str(x) for x in k if x is not None)] = {
                "median": float(np.median(v)), "iqr": float(q3 - q1), "total": float(np.sum(v)),
                "ratio_to_hq3m": None if not hq or k[1] not in ("GRU", "LRU") else float(np.median(v)) / hq}
    D["D6"] = cpu
    D["D7"] = d7(C, results, cut)
    a = align(C, [_ck("A3", "HQ", 4, "M"), _ck("A3", "HQ", 4, "C2", 0.0375)])
    if "A3" in cut or a is None:
        D["D8"] = {"label": NOT_RUN, "reason": "cut by G9" if "A3" in cut else "A3 cells missing"}
    else:
        ok = [transfer_success(m, o) for m, o in zip(*a[1].values())]
        D["D8"] = {"share": float(np.mean(ok)), "n": len(ok), "per_seed": ok}
    return D


def brier(G, H, predictions=PREDICTIONS):
    """Brier score of the frozen forecasts against the mechanical labels (rules whose label is outside the forecast
    set, e.g. UNINTERPRETABLE, are not scored)."""
    per = {}
    for name, probs in predictions.items():
        e = H.get(name) or G.get(name)
        if e and e["label"] in probs:
            per[name] = float(sum((p - (lab == e["label"])) ** 2 for lab, p in probs.items()))
    return {"per_rule": per, "mean": float(np.mean(list(per.values()))) if per else None}


def summarize(results, level=CI_LEVEL, n_boot=N_BOOT, median_clause=True, g4=True, pcd=None, trims=(), g6=True,
              h6_registered=True):
    """Mechanical E3 summary; gate statuses come first, contrasts behind failed gates are withheld. JSON-serialisable.

    pcd: the PC-D verdict from FIXTURE.json; anything but True halts every rule (§7: no PC-D pass, no interpretation).
    g4 / g6 False: A2w and H3b / H7 and D2 removed before freeze (G4 / G6). h6_registered False: H6 demoted to
    descriptive before freeze (G8 planted simulation). trims: the G9 cuts; "SC" is the terminal rule (H6 and STOP are
    registered as not run for compute), "C2_0.2" leaves H4 on the sigmas present.
    """
    C = build_cells(results)
    G, H, cut = {}, {}, set(trims)

    G["PC-D"] = _entry({True: PASS, False: FAIL}.get(pcd, "NOT_PROVIDED"), {})
    halt = pcd is not True
    why_halt = "PC-D failed" if pcd is False else "PC-D not provided (no FIXTURE.json verdict)"

    a = align(C, [a2_c2(0.05)])
    if a is None:
        G["PC-A"] = _entry(NOT_RUN, reason="A2 C2 sigma 0.05 missing")
    elif _failed_arms(C, [a2_c2(0.05)]):
        G["PC-A"] = _entry(UNINT, reason=_failed_arms(C, [a2_c2(0.05)]))
    else:
        cf = a[1][a2_c2(0.05)]
        k = int(np.sum(success(cf)))
        G["PC-A"] = _entry(pc_a(cf, median_clause), {"k": k, "n": len(cf), "median": float(np.median(cf)),
                                                     "median_clause": median_clause, "cp_lower": cp_lower(k, len(cf))})

    a = align(C, [A1_M, A1_HMM4])
    if a is None:
        G["PC-B"] = _entry(NOT_RUN, reason="A1 HQ2-M or HMM4-C missing")
    elif _failed_arms(C, [A1_M, A1_HMM4]):
        G["PC-B"] = _entry(UNINT, reason=_failed_arms(C, [A1_M, A1_HMM4]))
    else:
        d = a[1][A1_M] - a[1][A1_HMM4]
        lo, hi = bca(d, level=level, n_boot=n_boot)
        G["PC-B"] = _entry(pc_b(float(d.mean()), lo), {"delta": float(d.mean()), "ci": [lo, hi], "n": len(d)})

    shared = [_ck("A2", f, s, "CPLUS-M" if s <= 8 else "CPLUS") for (f, s) in E2_MEANS_A2]
    b_keys = [_ck("A1", "GRU", 4, "CPLUS"), _ck("A1", "LRU", 16, "CPLUS")]
    if any(k not in C for k in shared + b_keys):
        G["PC-K"] = _entry(NOT_RUN, reason="classical cells missing (SC block not run)")
    elif _failed_arms(C, shared + b_keys):
        G["PC-K"] = _entry(UNINT, reason=_failed_arms(C, shared + b_keys))
    else:
        meds = {(k[1], k[2]): float(np.median(C[k]["cf"])) for k in shared}
        la = pc_k_a(meds)
        g4m, l16m = (float(np.median(C[k]["cf"])) for k in b_keys)
        lb_ = pc_k_b(g4m, l16m)
        G["PC-K"] = _entry(pc_k(la, lb_), {"a": la, "b": lb_, "a_medians": [[f, s, v] for (f, s), v in meds.items()],
                                            "a_mean_diff": float(np.mean([meds[k] - E2_MEANS_A2[k] for k in E2_MEANS_A2])),
                                            "b_gru4": g4m, "b_lru16": l16m})

    if A1_OOM3 not in C:
        G["PC-O"] = _entry(NOT_RUN, reason="A1 OOM3-C missing")
    elif _failed_arms(C, [A1_OOM3]):
        G["PC-O"] = _entry(UNINT, reason=_failed_arms(C, [A1_OOM3]))
    else:
        m = float(np.median(C[A1_OOM3]["cf"]))
        G["PC-O"] = _entry(pc_o(m), {"median": m})

    dec = [_ck("A2", f, s, "CPLUS-M") for f in ("GRU", "LRU") for s in (2, 4, 8)]
    if any(k not in C or C[k]["cpu_med"] is None for k in dec) or A2_M not in C or C[A2_M]["cpu_med"] is None:
        G["COMPUTE"] = _entry(NOT_RUN, reason="decisive-ladder or HQ3-M cpu_s missing")
    else:
        cm = [C[k]["cpu_med"] for k in dec]
        G["COMPUTE"] = _entry(compute_check(cm, C[A2_M]["cpu_med"]),
                              {"cell_median_cpu": cm, "hq3m_median_cpu": C[A2_M]["cpu_med"], "ratio_floor": 0.8})

    shuf = [float(r["cf_shuffled"]) for r in results if r.get("family") == "REF" and r.get("cf_shuffled") is not None]
    G["NC"] = _entry(nc(shuf), {"max_cf_shuffled": max(shuf), "n": len(shuf)}) if shuf else _entry(NOT_RUN, reason="no cf_shuffled")

    if A2_C0 not in C:
        G["PC-C (modifier)"] = _entry(NOT_RUN, reason="A2 C0 missing")
    elif _failed_arms(C, [A2_C0]):
        G["PC-C (modifier)"] = _entry(UNINT, reason=_failed_arms(C, [A2_C0]))
    else:
        cf = C[A2_C0]["cf"]
        G["PC-C (modifier)"] = _entry(pc_c(cf), {"mean": float(cf.mean()), "n_below_030": int(np.sum(lt(cf, 0.30))), "n": len(cf)})

    pca_ok = G["PC-A"]["label"] == PASS

    def gated(keys, fn, need_pca=True):
        if halt:
            return _entry(UNINT, reason=why_halt)
        if need_pca and not pca_ok:
            return _entry(UNINT, reason=f"PC-A={G['PC-A']['label']}")
        a_ = align(C, keys)
        if a_ is None:
            return _entry(NOT_RUN, reason="cells missing: " + ", ".join(str(k) for k in keys if k not in C))
        bad = _failed_arms(C, keys)
        return _entry(UNINT, reason=bad) if bad else fn(*a_)

    def r_h1(seeds, cf, nan):
        x = cf[A2_M]
        s = int(np.sum(success(x)))
        return _entry(h1(s, pca_ok), {"s": s, "n": len(x), "mean": float(x.mean()), "median": float(np.median(x)),
                                      "cp_lower": cp_lower(s, len(x)), "seeds": seeds})
    H["H1"] = gated([A2_M], r_h1)

    def r_h2(seeds, cf, nan):
        x = cf[A2_S]
        s = int(np.sum(success(x)))
        return _entry(h2(s), {"s": s, "n": len(x), "cp_lower": cp_lower(s, len(x)), "cp_upper": cp_upper(s, len(x))})
    H["H2"] = gated([A2_S], r_h2)

    def r_h3a(seeds, cf, nan):
        ok = [transfer_success(m, o, c) for m, o, c in zip(cf[A1_M], cf[A1_C2], cf[A1_C0])]
        s = int(sum(ok))
        return _entry(h3(s), {"s": s, "n": len(ok), "per_seed": ok, "cp_lower": cp_lower(s, len(ok)),
                              "cp_upper": cp_upper(s, len(ok))})
    H["H3a"] = gated([A1_M, A1_C2, A1_C0], r_h3a)

    def r_h3b(seeds, cf, nan):
        ok = [transfer_success(m, o) for m, o in zip(cf[A2W_M], cf[A2W_C2])]
        s = int(sum(ok))
        med = float(np.median(cf[A2W_C2]))
        inp = {"s": s, "n": len(ok), "per_seed": ok, "o05_median": med, "cp_lower": cp_lower(s, len(ok)),
               "cp_upper": cp_upper(s, len(ok))}
        if H["H1"]["label"] == PARTIAL:
            return _entry(DESCRIPTIVE, inp, reason="H1 PARTIAL: described, not judged")
        if H["H1"]["label"] != PASS:
            return _entry(NOT_JUDGED, reason=f"H1={H['H1']['label']}")
        if not ge(med, 0.50):
            return _entry(STAT_LIMITED, inp, reason="O05 median < 0.50 on the confirmatory seeds")
        return _entry(h3(s), inp)
    if halt:
        H["H3b"] = _entry(UNINT, reason=why_halt)
    elif not g4:
        H["H3b"] = _entry(NOT_DECIDABLE_PF, reason="G4 failed: A2w removed before freeze")
    else:
        H["H3b"] = gated([A2W_M, A2W_C2], r_h3b) if H["H1"]["label"] in (PASS, PARTIAL) or not pca_ok \
            else _entry(NOT_JUDGED, reason=f"H1={H['H1']['label']}")

    def r_h4(seeds, cf, nan):
        """The label depends on s(0.3) only; shares and sigma* use the sigmas present (G9 cut 4 removes 0.2)."""
        present = [s for s in SIGMAS if a2_c2(s) in C]
        _, cfs, _ = align(C, [a2_c2(s) for s in present])
        shares = [float(np.mean(success(cfs[a2_c2(s)]))) for s in present]
        k, n = int(np.sum(success(cf[a2_c2(0.3)]))), len(seeds)
        ss, cens = sigma_star(present, shares)
        return _entry(h4(k, n), {"k_at_0.3": k, "n": n, "shares": shares, "sigmas": present, "sigma_star": ss,
                                 "sigmas_missing": [s for s in SIGMAS if s not in present], "sigma_star_censored": cens,
                                 "cp_lower": cp_lower(k, n), "cp_upper": cp_upper(k, n)})
    H["H4"] = gated([a2_c2(0.3)], r_h4)

    def r_h5(seeds, cf, nan):
        X = np.stack([cf[A2_M], cf[A2_C0], cf[A2_C1]], 1)
        keep = ~(nan[A2_M] | nan[A2_C0] | nan[A2_C1])
        vers = {}
        for name, Xv in (("with_nan", X), ("without_nan", X[keep])):
            if len(Xv) < 2:
                vers[name] = {"label": UNDET, "n": int(len(Xv))}
                continue
            lo, hi = bca(Xv, h5_stat, level=level, n_boot=n_boot)
            m = Xv.mean(0)
            vers[name] = {"label": h5_label(m[2], lo), "delta5": float(h5_stat(Xv)), "ci": [lo, hi],
                          "mean_m": float(m[0]), "mean_c0": float(m[1]), "mean_c1": float(m[2]), "n": int(len(Xv))}
        return _entry(h5_combine(vers["with_nan"]["label"], vers["without_nan"]["label"]), vers)
    H["H5"] = gated([A2_M, A2_C0, A2_C1], r_h5)

    gate = h6_gate(H["H1"]["label"], G["PC-B"]["label"], G["PC-K"]["label"], G["COMPUTE"]["label"])
    ladder_keys = {f: [_ck("A2", f, s, r) for s, r in LADDER[f] if _ck("A2", f, s, r) in C] for f in LADDER}
    keys6 = [A2_M] + ladder_keys["GRU"] + ladder_keys["LRU"]
    missing_dec = [k for k in dec if k not in C]
    if halt:
        H["H6"], H["STOP"] = _entry(UNINT, reason=why_halt), _entry(UNINT, reason=why_halt)
    elif "SC" in cut:
        H["H6"] = H["STOP"] = _entry(NOT_RUN_COMPUTE, reason="G9 terminal rule: SC block not run within E3")
    elif gate is not None:
        H["H6"], H["STOP"] = _entry(gate[0], reason=gate[1]), _entry(gate[0], reason=gate[1])
    elif missing_dec or align(C, keys6) is None:
        H["H6"] = H["STOP"] = _entry(NOT_RUN, reason="decisive ladder cells missing")
    elif _failed_arms(C, keys6):
        H["H6"] = H["STOP"] = _entry(UNINT, reason=_failed_arms(C, keys6))
    else:
        seeds, cf, _ = align(C, keys6)
        lad = {f: (np.array([C[k]["n_params"] for k in ks], float), np.stack([cf[k] for k in ks]))
               for f, ks in ladder_keys.items()}
        comp = h6_ratios(cf[A2_M], lad, n_boot=n_boot)
        H["H6"] = _entry(h6_rule([c["label"] for c in comp.values()]), {"components": comp, "P_HQ": P_HQ, "seeds": seeds})
        H["STOP"] = _entry(stop_rule(True, [c["R"] for c in comp.values()]), {"R": {f: c["R"] for f, c in comp.items()}})
    if not h6_registered and not halt and "SC" not in cut:
        H["H6"] = _entry(DESCRIPTIVE, H["H6"].get("inputs"),
                         reason=f"H6 demoted to descriptive before freeze (G8 planted simulation); mechanical={H['H6']['label']}")
        # Author decision 2 (E3-DESIGN, pre-freeze): STOP rests on the same R_F machinery, so it is demoted with H6.
        H["STOP"] = _entry(DESCRIPTIVE, H["STOP"].get("inputs"),
                           reason=f"STOP demoted with H6 before freeze; mechanical={H['STOP']['label']}")

    def r_h7(seeds, cf, nan):
        d = cf[A1_M] - cf[A1_OOM3]
        lo, hi = bca(d, level=level, n_boot=n_boot)
        recs = C[A1_OOM3]["records"].values()
        ft = [r["cf_ft"] for r in recs if r.get("cf_ft") is not None]
        sp = [r["cf_spec"] for r in recs if r.get("cf_spec") is not None]
        flag = bool(ft and sp and lt(np.mean(ft), np.mean(sp)))
        return _entry(h7(float(d.mean()), lo, hi, flag), {"delta7": float(d.mean()), "ci": [lo, hi], "n": len(d),
                                                         "oom_ft_below_spectral": flag})
    if halt:
        H["H7"] = _entry(UNINT, reason=why_halt)
    elif not g6:
        H["H7"] = _entry(REMOVED_PF, reason="G6 failed: H7 and D2 removed before freeze")
    elif G["PC-O"]["label"] != PASS or H["H3a"]["label"] == KILL:
        H["H7"] = _entry(NOT_JUDGED, reason=f"PC-O={G['PC-O']['label']}, H3a={H['H3a']['label']}")
    else:
        H["H7"] = gated([A1_M, A1_OOM3], r_h7, need_pca=False)

    enqueue = {"T": H["H1"]["label"] in (PASS, PARTIAL), "SC": H["H1"]["label"] == PASS and G["PC-B"]["label"] == PASS,
               "O": G["PC-O"]["label"] == PASS and g6}
    arms = {"/".join(str(v) for v in k if v is not None): {"n": len(c["seeds"]), "n_failed": c["n_failed"],
                                                          "n_nan": c["n_nan"], "uninterpretable": c["uninterpretable"]}
            for k, c in sorted(C.items(), key=lambda kv: str(kv[0]))}
    desc = {"withheld": why_halt} if halt else descriptive(C, results, cut, g6)
    return _py({"gates": G, "rules": H, "descriptive": desc, "brier": brier(G, H), "enqueue": enqueue, "arms": arms,
                "ci_level": level, "n_boot": n_boot, "trims": sorted(cut), "g4": g4, "g6": g6,
                "h6_registered": h6_registered})


def report(summary):
    """Text rendering with every gate status printed before any rule."""
    lines = ["GATES"]
    for k, v in summary["gates"].items():
        lines.append(f"  {k}: {v['label']}" + (f"  ({v['reason']})" if v.get("reason") else ""))
    lines.append("RULES")
    for k, v in summary["rules"].items():
        tail = f"  ({v['reason']})" if v.get("reason") else ""
        lines.append(f"  {k}: {v['label']}{tail}")
        if "inputs" in v:
            lines.append("    " + json.dumps(v["inputs"], sort_keys=True)[:600])
    d7_ = summary.get("descriptive", {}).get("D7", {})
    if d7_:
        lines.append("D7 (with H6, label unchanged) " + json.dumps(d7_.get("flags", d7_.get("sens")), sort_keys=True))
    lines.append("DESCRIPTIVE (exploratory) " + json.dumps(summary.get("descriptive", {}), sort_keys=True)[:1200])
    lines.append("ENQUEUE " + json.dumps(summary["enqueue"], sort_keys=True))
    return "\n".join(lines)

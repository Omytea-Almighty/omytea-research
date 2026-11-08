"""Confirmatory analysis per pre-registration §1/§4, with the v1.1 corrections recorded in DEVIATIONS.md D8
(written before any confirmatory run).

  python -m osw.analyze runs/confirm [runs/dev2]  ->  writes runs/confirm/RESULTS.md and RESULTS.json
  (if those already exist — the frozen confirmatory output — it writes RESULTS-RERUN-<UTC>.md/.json instead)

Per seed: window means (extrapolation window k in [51,200], in-range [1,50]); paired differences across
seeds 0-9; 95% CI by bootstrap over seeds (10,000 resamples) and by t-interval; one-sided paired t-tests.
D8 corrections:
  * every test needs all ten confirmatory seeds, otherwise it is INCOMPLETE and cannot pass;
  * PASS needs the effect-size rule AND both 95% CIs excluding 0 AND Holm-adjusted p < 0.05;
  * the Holm family is H1 (2 tasks), H2, H3, H4 (3 tasks); conjunctive hypotheses use the
    intersection-union p-value (the max of their component p-values);
  * H3 is tested on metastable tau_slow = 80 (D6) against the parameter-matched Lindblad model L1 (D9);
    the simple-task tie is an equivalence test (TOST, margin 0.01 nats), not a point estimate;
  * H4's best classical baseline is chosen on the development seed's validation NLL, never on confirmatory data;
    "parity or better" is a non-inferiority test (margin 0.01 nats).
Effect sizes in nats.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import stats

CONFIRM_SEEDS = list(range(10))
ALPHA = 0.05
H1_MIN, H3_MIN, TIE, NONINF = 0.02, 0.01, 0.01, 0.01
H3_META_TAU = 80          # D6
H3_L = "L1"               # D9: Lindblad model with one jump operator, 768 parameters vs C's 769
CLASSICAL = ["HMM", "GRU", "LRU"]


def load(d: Path):
    R = defaultdict(dict)                                    # (task, tau, model) -> seed -> result
    for f in d.glob("*.json"):
        if f.name.startswith("RESULTS"):
            continue
        r = json.loads(f.read_text())
        if r["seed"] in CONFIRM_SEEDS:
            R[(r["task"], float(r["tau"]), r["model"])][r["seed"]] = r
    return R


def series(R, task, tau, model, window="extrap_51_200", metric="kl"):
    runs = R.get((task, float(tau), model), {})
    return {s: runs[s]["windows"][window][metric] for s in runs}


def paired(a: dict, b: dict, rng, margin=0.0):
    """diff = a - b over common seeds. Bootstrap and t 95% intervals; one-sided t-test p-values against `margin`:
    p_greater for H1: mean > margin, p_less for H1: mean < margin."""
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    out = dict(n=len(d), complete=seeds == CONFIRM_SEEDS, mean=float(d.mean()) if len(d) else None)
    if len(d) < 2:
        return out
    boots = rng.choice(d, size=(10_000, len(d)), replace=True).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if d.std() > 0:
        t_lo, t_hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        p_g = stats.ttest_1samp(d, margin, alternative="greater").pvalue
        p_l = stats.ttest_1samp(d, margin, alternative="less").pvalue
    else:
        t_lo = t_hi = d.mean()
        p_g, p_l = (0.0 if d.mean() > margin else 1.0), (0.0 if d.mean() < margin else 1.0)
    out.update(boot_ci=[float(lo), float(hi)], t_ci=[float(t_lo), float(t_hi)], p_greater=float(p_g), p_less=float(p_l))
    return out


def ci_above_zero(h):
    return h.get("boot_ci", [0])[0] > 0 and h.get("t_ci", [0])[0] > 0


def spearman_at_most(x, y, bound: Fraction) -> bool:
    """Exact test of Spearman rho <= bound (bound < 0), using rational arithmetic on average ranks.
    DEVIATIONS.md D13: with four points the float rho -0.7999999999999999 failed a rule that the exact -4/5 meets."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [Fraction(0)] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = Fraction(i + j + 2, 2)              # average of 1-based ranks i+1 .. j+1
            i = j + 1
        return r
    rx, ry = ranks(list(x)), ranks(list(y))
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    return var > 0 and cov < 0 and cov * cov >= bound * bound * var    # rho = cov / sqrt(var) <= bound < 0


def spearman_perm_p(x, y) -> Fraction:
    """Exact one-sided permutation p for a negative rank association: the fraction of the n! orderings of y whose
    Spearman correlation with x is at most the observed one (ties counted). DEVIATIONS.md D14(a): the registered H2
    conjunction (v1.1 §6, D8) needs a p-value for the rank component; four points give p >= 1/24."""
    obs = stats.spearmanr(x, y).correlation
    perms = [stats.spearmanr(x, [y[i] for i in p]).correlation for p in itertools.permutations(range(len(y)))]
    return Fraction(sum(r <= obs + 1e-12 for r in perms), len(perms))


def holm(ps: dict):
    items = sorted((p, k) for k, p in ps.items())
    m, out, running = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def best_classical_on_dev(dev: Path, task, tau):
    """Lowest development-seed validation NLL among the classical baselines (each at its best learning rate)."""
    best = {}
    for f in dev.glob(f"{task}-{tau:g}-*.json"):
        r = json.loads(f.read_text())
        if r["model"] in CLASSICAL and r.get("val_nll") is not None:
            best[r["model"]] = min(best.get(r["model"], np.inf), r["val_nll"])
    if not best:
        raise SystemExit(f"no development results for {task}-{tau:g} in {dev}; H4 selection must not use confirmatory data")
    return min(best, key=best.get), best


def analyze(d: Path, dev: Path):
    rng = np.random.default_rng(20261019)
    R = load(d)
    res, ps = {}, {}
    # H1: U2 - L on the extrapolation-window KL, simple tau=20 and metastable tau=20 (as frozen)
    for task in ["simple", "metastable"]:
        h = paired(series(R, task, 20, "U2"), series(R, task, 20, "L"), rng)
        res[f"H1_{task}"] = h
        ps[f"H1_{task}"] = h.get("p_greater", 1.0) if h["complete"] else 1.0
    # H2: environment-dimension sweep and revival index (simple tau=20)
    dims = {"U1": 1, "U2": 2, "U4": 4, "U8": 8}
    sweep = {m: series(R, "simple", 20, m) for m in dims}
    sweep_complete = all(sorted(v) == CONFIRM_SEEDS for v in sweep.values())
    means = {m: float(np.mean(list(v.values()))) for m, v in sweep.items() if v}
    rho = stats.spearmanr([dims[m] for m in means], list(means.values())).correlation if len(means) >= 3 else None
    rev = lambda m: {s: r["revival"]["revival_index"] for s, r in R.get(("simple", 20.0, m), {}).items() if "revival" in r}
    hrev = paired(rev("U2"), rev("L"), rng)
    res["H2"] = dict(extrap_kl_by_dE={dims[m]: v for m, v in means.items()}, spearman=rho, sweep_complete=sweep_complete,
                     revival_U2_minus_L=hrev)
    # D14(a): registered conjunction — the H2 p-value is the larger of the rank component's exact permutation p and the
    # revival component's p (v1.1 §6; D8 names both components). The frozen code used the revival p alone.
    p_rank = float(spearman_perm_p([dims[m] for m in means], list(means.values()))) if len(means) >= 3 else 1.0
    res["H2"]["rank_perm_p"] = p_rank
    ps["H2"] = max(p_rank, hrev.get("p_greater", 1.0)) if (hrev["complete"] and sweep_complete) else 1.0
    # H3: metastable (tau_slow=80) must favour the matched Lindblad model by >= 0.01; simple (tau=20) must be equivalent
    h3m = paired(series(R, "metastable", H3_META_TAU, "C"), series(R, "metastable", H3_META_TAU, H3_L), rng)
    h3s = paired(series(R, "simple", 20, "C"), series(R, "simple", 20, H3_L), rng)
    p_tost = max(paired(series(R, "simple", 20, "C"), series(R, "simple", 20, H3_L), rng, margin=-TIE).get("p_greater", 1.0),
                 paired(series(R, "simple", 20, "C"), series(R, "simple", 20, H3_L), rng, margin=TIE).get("p_less", 1.0))
    h3s["p_tost"] = float(p_tost)
    res["H3"] = dict(metastable_C_minus_L1=h3m, simple_C_minus_L1=h3s)
    ps["H3"] = max(h3m.get("p_greater", 1.0), p_tost) if (h3m["complete"] and h3s["complete"]) else 1.0
    # secondary (not in the Holm family): the same contrasts against the native-size L (1024 parameters)
    res["H3_native_L"] = dict(metastable_C_minus_L=paired(series(R, "metastable", H3_META_TAU, "C"), series(R, "metastable", H3_META_TAU, "L"), rng),
                              simple_C_minus_L=paired(series(R, "simple", 20, "C"), series(R, "simple", 20, "L"), rng))
    # H4: L vs the classical baseline chosen on the development seed; non-inferiority margin 0.01 nats
    h4 = {}
    for task in ["simple", "metastable", "action"]:
        best, dev_nll = best_classical_on_dev(dev, task, 20)
        diff = paired(series(R, task, 20, "L"), series(R, task, 20, best), rng, margin=NONINF)
        diff.update(best_classical=best, selected_on_dev_val_nll=dev_nll)
        h4[task] = diff
        ps[f"H4_{task}"] = diff.get("p_less", 1.0) if diff["complete"] else 1.0
    res["H4"] = h4
    res["holm"] = adj = holm(ps)
    # verdicts
    for task in ["simple", "metastable"]:
        h = res[f"H1_{task}"]
        h["pass"] = bool(h["complete"] and h["mean"] >= H1_MIN and ci_above_zero(h) and adj[f"H1_{task}"] < ALPHA)
    exact_ok = len(means) >= 3 and spearman_at_most([dims[m] for m in means], list(means.values()), Fraction(-4, 5))
    res["H2"]["spearman_le_-0.8_exact"] = bool(exact_ok)
    res["H2"]["pass_"] = bool(sweep_complete and hrev["complete"] and exact_ok
                              and hrev["mean"] > 0 and ci_above_zero(hrev) and adj["H2"] < ALPHA)
    res["H3"]["pass_"] = bool(h3m["complete"] and h3s["complete"] and h3m["mean"] >= H3_MIN and ci_above_zero(h3m)
                              and adj["H3"] < ALPHA)
    for task, h in h4.items():
        h["parity_or_better"] = bool(h["complete"] and adj[f"H4_{task}"] < ALPHA)
    # descriptive table of every model's window means
    table = defaultdict(dict)
    for (task, tau, model), runs in R.items():
        kl_x = [r["windows"]["extrap_51_200"]["kl"] for r in runs.values()]
        kl_i = [r["windows"]["in_range_1_50"]["kl"] for r in runs.values()]
        table[f"{task}-{tau:g}"][model] = dict(n=len(runs), kl_in=float(np.mean(kl_i)), kl_extrap=float(np.mean(kl_x)),
                                              kl_extrap_sd=float(np.std(kl_x)), params=next(iter(runs.values()))["n_params"])
    res["table"] = table
    res["output_stem"] = stem = output_stem(d)
    (d / f"{stem}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def output_stem(d: Path) -> str:
    """Never overwrite the frozen confirmatory output: if RESULTS.json or RESULTS.md already exists in this folder,
    write RESULTS-RERUN-<UTC time> instead. Decided once per analysis run, from this folder only."""
    if (d / "RESULTS.json").exists() or (d / "RESULTS.md").exists():
        return "RESULTS-RERUN-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return "RESULTS"


def verdict(ok, complete):
    return "PASS" if ok else ("NOT MET" if complete else "INCOMPLETE")


def report(res, d: Path):
    L = ["# Confirmatory results (pre-registration v1 SHA-256 37172954..., with v1.1 corrections D5-D9)", ""]
    for k in ["H1_simple", "H1_metastable"]:
        h = res[k]
        L.append(f"- **{k}** U2 − L extrapolation KL: mean {h.get('mean')} nats, bootstrap CI {h.get('boot_ci')}, "
                 f"t CI {h.get('t_ci')}, Holm p {res['holm'].get(k)} → {verdict(h['pass'], h['complete'])}")
    h2 = res["H2"]
    L.append(f"- **H2** extrapolation KL by d_E {h2['extrap_kl_by_dE']}, Spearman {h2['spearman']}, revival U2−L "
             f"{h2['revival_U2_minus_L'].get('mean')} CI {h2['revival_U2_minus_L'].get('boot_ci')}, Holm p {res['holm'].get('H2')} "
             f"→ {verdict(h2['pass_'], h2['sweep_complete'] and h2['revival_U2_minus_L']['complete'])}")
    h3 = res["H3"]
    m, s = h3["metastable_C_minus_L1"], h3["simple_C_minus_L1"]
    L.append(f"- **H3** metastable(τ_slow={H3_META_TAU}) C−L1 {m.get('mean')} CI {m.get('boot_ci')}; "
             f"simple C−L1 {s.get('mean')} (TOST p {s.get('p_tost')}), Holm p {res['holm'].get('H3')} "
             f"→ {verdict(h3['pass_'], m['complete'] and s['complete'])}")
    n = res["H3_native_L"]
    L.append(f"  - secondary, native-size L: metastable C−L {n['metastable_C_minus_L'].get('mean')} CI {n['metastable_C_minus_L'].get('boot_ci')}; "
             f"simple C−L {n['simple_C_minus_L'].get('mean')}")
    for task, h in res["H4"].items():
        L.append(f"- **H4 {task}** L − {h['best_classical']} (chosen on dev): {h.get('mean')} CI {h.get('boot_ci')}, "
                 f"Holm p {res['holm'].get('H4_' + task)} → "
                 f"{'non-inferior (margin 0.01)' if h['parity_or_better'] else ('not shown non-inferior' if h['complete'] else 'INCOMPLETE')}")
    L += ["", "| task | model | params | KL in [1,50] | KL extrap [51,200] (sd) | n |", "|---|---|---|---|---|---|"]
    for task, ms in sorted(res["table"].items()):
        for mdl, v in sorted(ms.items(), key=lambda kv: kv[1]["kl_extrap"]):
            L.append(f"| {task} | {mdl} | {v['params']} | {v['kl_in']:.4f} | {v['kl_extrap']:.4f} ({v['kl_extrap_sd']:.4f}) | {v['n']} |")
    (d / f"{res['output_stem']}.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/confirm")
    dev = Path(sys.argv[2] if len(sys.argv) > 2 else "runs/dev2")
    report(analyze(d, dev), d)

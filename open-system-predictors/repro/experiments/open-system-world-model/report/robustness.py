"""Post hoc robustness and sensitivity analyses requested by the round-1 internal correctness review.
All are labelled post hoc in the report; none changes a frozen verdict.

  .venv/bin/python report/robustness.py   ->  runs/robustness/ROBUSTNESS.json

1. In-range KL per (configuration, model), and per-seed plateau counts for L and C on simple-20 (review B4).
2. Extrapolation KL of the uninformative (stationary-marginal) predictor on each confirmatory test set (review B3).
3. H4 with the HMM eligible: HMM validation NLL computed on development seed 102 only (review F3/B7).
4. H1 against a converged CPTP model: U2 - K, paired over seeds (review B4).
5. Exact one-sided permutation p for the H2 rank component (review F4/B6).
6. E2 A2 HQ d=3 seed 0 original captured fraction (review B8).
7. Secondary metrics (NLL, Brier, ECE) in the extrapolation window, simple-20 (review B13).
Round-2 internal review additions:
8. Holm family under the registered conjunction rule: H2 p = max(exact Spearman permutation p, revival p).
9. Captured fraction 1 - KL_model / KL_marginal per model and configuration (real signal only at tau = 80).
10. Unitary-family development extrapolation KL at both grid learning rates (simple-20, seed 102) and the chosen rate.
11. E2 positive-control feasibility: HMM curve interpolated at the A1 budgets; HQ captured fraction by size on A1.
12. E1 per-fold K - L differences.
Round-3 internal review additions:
13. Per-seed median in-range KL at simple tau = 80; C's development in-range KL at both learning rates.
14. E1 HMM refitted on the training window only (context + future steps 1-6), against L, per fold (sensitivity).
    Since round 4 the refits are raw files under runs/e1-posthoc/ (osw.run_e1_hmmtw); this script only aggregates them.
Round-4 internal review additions:
    Item 3 also reports the seed-bootstrap interval of L - HMM.
Round-5 internal review additions:
15. H1 contrast with the larger unitary models: U4 - L and U8 - L extrapolation KL, simple tau = 20.
Round-6/7 review additions:
16. Seed-mean per-horizon KL at k = 50, 100, 200 for L, U2, GRU and C on the Figure 1 panels.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from osw import hmm, queue_e2, run, tasks  # noqa: E402

CONF = ROOT / "runs/confirm"
OUT = ROOT / "runs/robustness"
KEY = {"simple": 1, "metastable": 2, "action": 3}


def load(task, tau, model):
    out = {}
    for f in CONF.glob(f"{task}-{tau}-{model}-s*-lr*.json"):
        r = json.loads(f.read_text())
        out[r["seed"]] = r
    return out


def paired(a, b, rng):
    s = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in s])
    boots = rng.choice(d, size=(10_000, len(d))).mean(1)
    lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return dict(n=len(d), mean=float(d.mean()), boot_ci=[float(x) for x in np.percentile(boots, [2.5, 97.5])],
                t_ci=[float(lo), float(hi)], p_greater=float(stats.ttest_1samp(d, 0, alternative="greater").pvalue))


def main():
    rng = np.random.default_rng(20261023)
    res = {}
    configs = [("simple", 20), ("metastable", 20), ("metastable", 80), ("simple", 80), ("action", 20)]
    models = ["U2", "L", "L1", "K", "C", "GRU", "LRU", "HMM", "U1", "U4", "U8"]
    res["in_range_kl"] = {f"{t}-{tau}": {m: float(np.mean([r["windows"]["in_range_1_50"]["kl"] for r in load(t, tau, m).values()]))
                                          for m in models if load(t, tau, m)} for t, tau in configs}
    plate = {}
    for m in ("L", "C", "K", "U2"):
        v = {s: r["windows"]["in_range_1_50"]["kl"] for s, r in load("simple", 20, m).items()}
        plate[m] = dict(per_seed={str(k): round(x, 5) for k, x in sorted(v.items())}, n_above_0_02=int(sum(x > 0.02 for x in v.values())))
    res["simple20_plateau"] = plate
    marg = {}
    for t, tau in configs:
        task = run.make_task(t, tau)
        vals = []
        for seed in range(10):
            _, _, te = run.data(task, seed, KEY[t] * 1000 + tau)
            q = np.clip(tasks.bayes_predictive(task, te[0], te[1], run.K_EVAL), 1e-12, 1)
            vals.append(float((q * np.log(q / 0.25)).sum(-1).mean(0)[50:].mean()))
        marg[f"{t}-{tau}"] = float(np.mean(vals))
    res["marginal_predictor_extrap_kl"] = marg
    h4 = {}
    for t in ("simple", "metastable", "action"):
        task = run.make_task(t, 20)
        tr, va, _ = run.data(task, 102, KEY[t] * 1000 + 20)
        h = hmm.fit_best(task, tr[0][:5000], tr[1][:5000], 102)
        pv = h.predictive(va[0][:1000], va[1][:1000], run.K_TRAIN)
        yv = va[0][:1000, tasks.L_CTX:tasks.L_CTX + run.K_TRAIN]
        dev = {"HMM": float(-np.log(np.clip(np.take_along_axis(pv, yv[..., None], -1), 1e-9, 1)).mean())}
        for f in (ROOT / "runs/dev2").glob(f"{t}-20-*-s102-*.json"):
            r = json.loads(f.read_text())
            if r["model"] in ("GRU", "LRU") and r["val_nll"] is not None:
                dev[r["model"]] = min(dev.get(r["model"], np.inf), r["val_nll"])
        best = min(dev, key=dev.get)
        L = {s: r["windows"]["extrap_51_200"]["kl"] for s, r in load(t, 20, "L").items()}
        B = {s: r["windows"]["extrap_51_200"]["kl"] for s, r in load(t, 20, best).items()}
        d = np.array([L[s] - B[s] for s in sorted(L)])
        lo, hi = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
        # Round-4 review m4: also the seed-bootstrap interval. Own generator, so the shared rng stream is unchanged.
        boots = np.random.default_rng(20261024 + KEY[t]).choice(d, size=(10_000, len(d))).mean(1)
        h4[t] = dict(dev_val_nll=dev, best=best, mean=float(d.mean()), t_ci=[float(lo), float(hi)],
                     boot_ci=[float(x) for x in np.percentile(boots, [2.5, 97.5])],
                     p_noninferior=float(stats.ttest_1samp(d, 0.01, alternative="less").pvalue))
    res["h4_with_hmm_eligible"] = h4
    res["h1_vs_converged_K"] = {t: paired({s: r["windows"]["extrap_51_200"]["kl"] for s, r in load(t, 20, "U2").items()},
                                          {s: r["windows"]["extrap_51_200"]["kl"] for s, r in load(t, 20, "K").items()}, rng)
                                for t in ("simple", "metastable")}
    # Round-5 internal review: H1's contrast with the larger unitary models, simple tau = 20. Own generators, so the
    # shared rng stream (and every other bootstrap here) is unchanged.
    Ls = {s: r["windows"]["extrap_51_200"]["kl"] for s, r in load("simple", 20, "L").items()}
    res["h1_larger_unitary_minus_L"] = {m: paired({s: r["windows"]["extrap_51_200"]["kl"] for s, r in load("simple", 20, m).items()},
                                                  Ls, np.random.default_rng(20261025 + i))
                                        for i, m in enumerate(("U4", "U8"))}
    obs = [float(np.mean([r["windows"]["extrap_51_200"]["kl"] for r in load("simple", 20, m).values()])) for m in ("U1", "U2", "U4", "U8")]
    rho_obs = stats.spearmanr([1, 2, 4, 8], obs).correlation
    perms = [stats.spearmanr([1, 2, 4, 8], [obs[i] for i in p]).correlation for p in itertools.permutations(range(4))]
    res["h2_spearman_permutation_p_one_sided"] = sum(r <= rho_obs + 1e-12 for r in perms) / len(perms)
    r0 = json.loads((ROOT / "runs/e2/e2-A2-HQ3-s0-lr0.003.json").read_text())
    r1 = json.loads((ROOT / "runs/e2/e2-A2-HQ3-s0-lr0.01.json").read_text())
    chosen = min((r0, r1), key=lambda r: r["val_nll"])
    res["e2_A2_HQ3_seed0_original_cf"] = float(queue_e2.metric(chosen))
    sec = {}
    for m in ("U2", "L", "L1", "K", "C", "GRU", "LRU", "HMM"):
        rs = load("simple", 20, m).values()
        sec[m] = {k: float(np.mean([r["windows"]["extrap_51_200"][k] for r in rs])) for k in ("nll", "brier", "ece")}
    res["secondary_simple20_extrap"] = sec
    R = json.loads((CONF / "RESULTS.json").read_text())
    raw = {"H1_simple": R["H1_simple"]["p_greater"], "H1_metastable": R["H1_metastable"]["p_greater"],
           "H2": max(res["h2_spearman_permutation_p_one_sided"], R["H2"]["revival_U2_minus_L"]["p_greater"]),
           "H3": max(R["H3"]["metastable_C_minus_L1"]["p_greater"], R["H3"]["simple_C_minus_L1"]["p_tost"]),
           **{f"H4_{t}": R["H4"][t]["p_less"] for t in ("simple", "metastable", "action")}}
    items = sorted((v, k) for k, v in raw.items())
    adj, run_max = {}, 0.0
    for i, (v, k) in enumerate(items):
        run_max = max(run_max, min(1.0, (len(items) - i) * v)); adj[k] = run_max
    res["holm_registered_conjunction"] = {"raw": raw, "adjusted": adj}
    cf = {}
    for cfg, mk in res["marginal_predictor_extrap_kl"].items():
        t, tau = cfg.split("-")
        cf[cfg] = {m: 1 - float(np.mean([r["windows"]["extrap_51_200"]["kl"] for r in load(t, int(tau), m).values()])) / mk
                   for m in models if load(t, int(tau), m)}
    res["captured_fraction_extrap"] = cf
    ulr = {}
    for m in ("U1", "U2", "U4", "U8"):
        rows = {}
        for f in (ROOT / "runs/dev2").glob(f"simple-20-{m}-s102-lr*.json"):
            r = json.loads(f.read_text())
            rows[str(r["lr"])] = {"extrap_kl": r["windows"]["extrap_51_200"]["kl"], "val_nll": r["val_nll"]}
        chosen = min(rows, key=lambda k: rows[k]["val_nll"])
        ulr[m] = {"by_lr": rows, "chosen_lr": chosen}
    res["unitary_dev_extrap_by_lr"] = ulr
    res["u2_confirm_inrange_min"] = min(plate["U2"]["per_seed"].values())
    e2s = json.loads((ROOT / "runs/e2/SUMMARY.json").read_text())
    def interp(curve, b):
        pts = sorted((np.log(v["params"]), v["mean"]) for v in curve.values())
        return float(np.interp(np.log(b), [x for x, _ in pts], [y for _, y in pts]))
    res["e2_A1_hmm_at_budget"] = {str(b): interp(e2s["A1"]["curves"]["HMM"], b) for b in (64, 256)}
    res["e2_A1_hq_by_size"] = {k: v["mean"] for k, v in e2s["A1"]["curves"]["HQ"].items()}
    res["e2_emp_rpss"] = {a: e2s[a]["curves"]["EMP"]["0"]["mean"] for a in ("B", "B2", "B3")}
    e1 = {}
    for f in (ROOT / "runs/e1").glob("e1-*.json"):
        r = json.loads(f.read_text())
        if r["model"] in ("K", "L"):
            k = (r["held_out"], r["seed"], r["model"])
            if k not in e1 or r["val_nll"] < e1[k]["val_nll"]:
                e1[k] = r
    folds = sorted({k[0] for k in e1})
    res["e1_K_minus_L_by_fold"] = {fo: float(np.mean([e1[(fo, s, "K")]["windows"]["extrap_7_12"]["nll"] - e1[(fo, s, "L")]["windows"]["extrap_7_12"]["nll"]
                                                       for s in range(3)])) for fo in folds}
    res["simple80_inrange_median"] = {m: float(np.median([r["windows"]["in_range_1_50"]["kl"] for r in load("simple", 80, m).values()]))
                                      for m in ("L", "K", "GRU", "C", "U2")}
    cdev = {}
    for tau in (20, 80):
        for f in (ROOT / "runs/dev2").glob(f"simple-{tau}-C-s102-lr*.json"):
            r = json.loads(f.read_text())
            cdev[f"simple-{tau}@{r['lr']:g}"] = r["windows"]["in_range_1_50"]["kl"]
    res["C_dev_inrange_by_lr"] = cdev
    from osw import tasks_ethucy as E
    # The six-step refits are raw run files written by `python -m osw.run_e1_hmmtw` (round-4 review m7), read here.
    trunc = {}
    for f in (ROOT / "runs/e1-posthoc").glob("e1hmmtw-*.json"):
        r = json.loads(f.read_text())
        trunc[(r["held_out"], r["seed"])] = r["windows"]["extrap_7_12"]["nll"]
    assert len(trunc) == 3 * len(E.SCENES), f"expected {3 * len(E.SCENES)} e1hmmtw files, found {len(trunc)}"
    e1L = {}
    for f in (ROOT / "runs/e1").glob("e1-*-L-*.json"):
        r = json.loads(f.read_text())
        k = (r["held_out"], r["seed"])
        if k not in e1L or r["val_nll"] < e1L[k][0]:
            e1L[k] = (r["val_nll"], r["windows"]["extrap_7_12"]["nll"])
    res["e1_hmm_trainwindow"] = {"extrap_nll_mean": float(np.mean(list(trunc.values()))),
                                 "minus_L_by_fold": {fo: float(np.mean([trunc[(fo, s_)] - e1L[(fo, s_)][1] for s_ in range(3)])) for fo in E.SCENES},
                                 "minus_L_mean": float(np.mean([trunc[k] - e1L[k][1] for k in trunc]))}
    # Round-6/7 internal review: seed-mean per-horizon KL at k = 50, 100, 200 for the Figure 1 panels.
    res["fig1_horizon_kl"] = {f"{t}-{tau}": {m: {str(k): float(np.mean([r["per_k"]["kl"][k - 1] for r in load(t, tau, m).values()]))
                                                for k in (50, 100, 200)} for m in ("L", "U2", "GRU", "C")}
                              for t, tau in (("simple", 20), ("metastable", 20), ("metastable", 80), ("simple", 80))}
    # Pre-publication errata (2026-10-29). r9 m5: H1's threshold is defined on U − L, so the development-seed
    # learning-rate check quotes U − L, with L at its registered (validation-NLL) learning rate on development seed 102.
    Ldev = {}
    for f in (ROOT / "runs/dev2").glob("simple-20-L-s102-lr*.json"):
        r = json.loads(f.read_text())
        Ldev[str(r["lr"])] = {"extrap_kl": r["windows"]["extrap_51_200"]["kl"], "val_nll": r["val_nll"]}
    L_lr = min(Ldev, key=lambda k: Ldev[k]["val_nll"])
    res["unitary_dev_minus_L"] = {"L_chosen_lr": L_lr, "L_extrap_kl": Ldev[L_lr]["extrap_kl"],
                                  "by_model": {m: {lr: v["extrap_kl"] - Ldev[L_lr]["extrap_kl"] for lr, v in ulr[m]["by_lr"].items()}
                                               for m in ulr}}
    # r10 m3: movable-matched H1 contrast, U2 (768 movable) - L1 (704 movable), simple tau=20 extrapolation KL; own
    # generator so that every earlier number is unchanged.
    res["h1_movable_matched_U2_minus_L1"] = paired({s: r["windows"]["extrap_51_200"]["kl"] for s, r in load("simple", 20, "U2").items()},
                                                   {s: r["windows"]["extrap_51_200"]["kl"] for s, r in load("simple", 20, "L1").items()},
                                                   np.random.default_rng(20261029))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ROBUSTNESS.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v for k, v in res.items() if k not in ("in_range_kl", "secondary_simple20_extrap")}, indent=1, default=float))


if __name__ == "__main__":
    main()

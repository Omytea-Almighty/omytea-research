"""Post hoc checks from the T1 confirmatory records, added at release after the final correctness review.

None of these is registered. They make three statements in the report checkable:

1. H2 rank component. The registration requires a p-value for the rank component but names no test. The test used is an
   exact permutation of the four seed means, whose p cannot go below 1/24. Here: the best Holm-adjusted p that test could
   reach given the other members of the family, and two tests that use the 10 x 4 seed-level values.
2. Revival index. What the index measures on the records, and its value on synthetic curves through the frozen function.
3. Contraction control. C against L, L1 and K in every configuration, for the comparator named in kill test KT3.

  PYTHONPATH=src .venv/bin/python report/post_review_checks.py      # writes runs/robustness/POST_REVIEW_CHECKS.json
"""
from __future__ import annotations

import glob
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

from osw.run import revival

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/robustness/POST_REVIEW_CHECKS.json"
SEEDS = range(10)
CONFIGS = ("simple-20", "metastable-20", "metastable-80", "simple-80", "action-20")


def rec(cfg, model, seed):
    f = glob.glob(str(ROOT / f"runs/confirm/{cfg}-{model}-s{seed}-*.json"))
    assert len(f) == 1, (cfg, model, seed, f)
    return json.loads(Path(f[0]).read_text())


def extrap(cfg, model):
    return np.array([rec(cfg, model, s)["windows"]["extrap_51_200"]["kl"] for s in SEEDS])


def in_range(cfg, model):
    return np.array([rec(cfg, model, s)["windows"]["in_range_1_50"]["kl"] for s in SEEDS])


def t_ci(d):
    d = np.asarray(d)
    se = d.std(ddof=1) / np.sqrt(len(d))
    t = stats.t.ppf(0.975, len(d) - 1)
    return {"mean": float(d.mean()), "t_ci": [float(d.mean() - t * se), float(d.mean() + t * se)],
            "n_positive": int((d > 0).sum()), "p_two_sided": float(stats.ttest_1samp(d, 0).pvalue)}


def holm(raw):
    items = sorted(raw.items(), key=lambda kv: kv[1])
    out, cur = {}, 0.0
    for i, (k, v) in enumerate(items):
        cur = max(cur, (len(items) - i) * v)
        out[k] = min(1.0, cur)
    return out


def h2():
    ds = [1, 2, 4, 8]
    X = np.array([extrap("simple-20", f"U{d}") for d in ds])                      # (4, 10)
    means = X.mean(1)
    rho = float(stats.spearmanr(ds, means).statistic)
    rvals = [stats.spearmanr(ds, [means[i] for i in p]).statistic for p in itertools.permutations(range(4))]
    p_perm = float(np.mean([r <= rho + 1e-12 for r in rvals]))
    p_floor = float(np.mean([r <= min(rvals) + 1e-12 for r in rvals]))
    rb = json.loads((ROOT / "runs/robustness/ROBUSTNESS.json").read_text())["holm_registered_conjunction"]
    raw = dict(rb["raw"])
    rev_p = float(rb["revival_p"]) if "revival_p" in rb else None
    best = dict(raw)
    best["H2"] = max(p_floor, rev_p if rev_p is not None else 0.0)
    page = stats.page_trend_test(X[::-1].T, method="exact")                       # columns U8, U4, U2, U1; KL rises as d falls
    rs = np.array([stats.spearmanr(ds, X[:, s]).statistic for s in SEEDS])
    t_less = stats.ttest_1samp(rs, 0, alternative="less")
    with_test = lambda p: holm(dict(raw, H2=max(p, rev_p if rev_p is not None else 0.0)))["H2"]
    return {"means_U1_U2_U4_U8": means.tolist(), "rho_of_means": rho, "exact_perm_p": p_perm, "exact_perm_p_floor": p_floor,
            "holm_raw_family": raw, "holm_best_case_with_floor": holm(best)["H2"],
            "page_exact_p_one_sided": float(page.pvalue), "page_holm": with_test(float(page.pvalue)),
            "per_seed_rho": rs.tolist(), "per_seed_rho_mean": float(rs.mean()), "per_seed_rho_t_p_one_sided": float(t_less.pvalue),
            "per_seed_rho_holm": with_test(float(t_less.pvalue)),
            "paired": {"U1-U2": t_ci(X[0] - X[1]), "U1-U4": t_ci(X[0] - X[2]), "U8-U4": t_ci(X[3] - X[2])}}


def revival_checks():
    out = {}
    for model in ("U2", "L", "C"):
        v = [rec("simple-20", model, s)["revival"] for s in SEEDS]
        idx, pk, md = (float(np.mean([x[k] for x in v])) for k in ("revival_index", "mean_peaks", "mean_D_51_200"))
        out[model] = {"index": idx, "mean_peaks": pk, "mean_D_51_200": md}
    K = 600
    k = np.arange(1, K + 1)
    rotation = {}
    for a in (0.08, 0.10):
        p = np.zeros((1, K, 4))
        for o in range(4):
            p[0, :, o] = 0.25 + a * np.cos(2 * np.pi * (k - o) / 4 + 0.3)
        rotation[str(a)] = revival(p)["revival_index"]

    def curve(fn):
        D = np.array([fn(i) for i in k])
        p = np.full((1, K, 4), 0.25)
        p[:, :, 0] += D[None, :]
        p[:, :, 1] -= D[None, :]
        return revival(p)["revival_index"]

    out["synthetic"] = {"constant_amplitude_period4_rotation": rotation,
                        "one_bump_height_0.05": curve(lambda i: 0.05 * np.exp(-((i - 120) / 15) ** 2)),
                        "monotone_decay": curve(lambda i: 0.1 * np.exp(-i / 20))}
    return out


def contraction():
    out = {}
    for cfg in CONFIGS:
        c = extrap(cfg, "C")
        row = {"C-L": t_ci(c - extrap(cfg, "L")), "C-K": t_ci(c - extrap(cfg, "K"))}
        if glob.glob(str(ROOT / f"runs/confirm/{cfg}-L1-s0-*.json")):
            row["C-L1"] = t_ci(c - extrap(cfg, "L1"))
        out[cfg] = row
    out["simple-80_in_range_median"] = {m: float(np.median(in_range("simple-80", m))) for m in ("C", "K", "L")}
    return out


def main() -> None:
    out = {"note": "post hoc; from the T1 confirmatory records; not registered", "h2": h2(), "revival": revival_checks(),
           "contraction": contraction()}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: out["h2"][k] for k in ("rho_of_means", "exact_perm_p", "exact_perm_p_floor", "holm_best_case_with_floor",
                                                "page_exact_p_one_sided", "page_holm", "per_seed_rho_t_p_one_sided", "per_seed_rho_holm",
                                                "paired")}, indent=0))
    print(json.dumps(out["revival"], indent=0))
    print(json.dumps({c: {k: [round(v["mean"], 5), [round(x, 5) for x in v["t_ci"]]] for k, v in out["contraction"][c].items()} for c in CONFIGS}, indent=0))
    print(out["contraction"]["simple-80_in_range_median"])


if __name__ == "__main__":
    main()

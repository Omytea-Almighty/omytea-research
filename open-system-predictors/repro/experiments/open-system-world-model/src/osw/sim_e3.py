"""E3 freeze-gate simulations (E3-DESIGN §9(ii), G8): planted-ratio check of the H6 rule and BCa coverage on paired differences.

Both call the registered machinery in summary_e3 unchanged (seed medians, cumulative-max envelope, log-param
interpolation, joint seed bootstrap with R = 0 replicates, 1st-percentile bound; BCa with the shared resample seed).
"""
from __future__ import annotations

import numpy as np

from osw import summary_e3 as S

LADDERS = {"GRU": np.array([63, 147, 387, 1155, 3843], float), "LRU": np.array([59, 99, 259, 771, 2563], float)}
SIM_SEED = 20261026


def truth_curve(params, p_true, slope=0.25, lo=-0.05, hi=0.99):
    """True seed-median CF per rung: crosses 0.90 exactly at p_true, linear in log params."""
    return np.clip(S.SUCCESS + slope * np.log(np.asarray(params, float) / p_true), lo, hi)


def p_true_for_top(params, top_cf, slope=0.25):
    return float(params[-1] * np.exp(-(top_cf - S.SUCCESS) / slope))


def _draw(rng, mean, sd, n, plateau_rate, plateau_cf):
    mean = np.atleast_1d(np.asarray(mean, float))
    x = mean[:, None] + np.asarray(sd, float).reshape(-1, 1) * rng.standard_normal((len(mean), n))
    return np.where(rng.random(x.shape) < plateau_rate, plateau_cf, x)


def planted_ratio_sim(R_true=None, n_sims=1000, top_cf=None, hq_mean=0.97, hq_sd=0.015, cl_sd=0.08,
                      plateau_rate=0.10, plateau_cf=0.03, slope=0.25, n_seeds=10, n_boot=S.N_BOOT, seed=SIM_SEED):
    """One planted truth (R_true, or the largest rung's true median = top_cf) -> label rates per family and for the rule.

    Per-seed CF ~ N(true median, sd) with a plateau seed (CF = plateau_cf) at rate plateau_rate in every cell, HQ included.
    cl_sd may be a scalar or one value per rung.
    """
    rng = np.random.default_rng(seed)
    p_true = {f: (p_true_for_top(p, top_cf, slope) if top_cf is not None else R_true * S.P_HQ) for f, p in LADDERS.items()}
    truth = {f: truth_curve(p, p_true[f], slope) for f, p in LADDERS.items()}
    hold = {f: 0 for f in LADDERS}
    cens_hold = {f: 0 for f in LADDERS}
    report = {f: 0 for f in LADDERS}
    rule = rule_h1 = n_h1 = 0
    for _ in range(n_sims):
        hq = _draw(rng, [hq_mean], [hq_sd], n_seeds, plateau_rate, plateau_cf)[0]
        lad = {f: (p, _draw(rng, truth[f], np.broadcast_to(cl_sd, p.shape), n_seeds, plateau_rate, plateau_cf))
               for f, p in LADDERS.items()}
        comp = S.h6_ratios(hq, lad, n_boot=n_boot)
        for f, c in comp.items():
            h = c["label"] == S.COMP_HOLDS
            hold[f] += h
            cens_hold[f] += h and c["censored"] == "right"
            report[f] += c["censor_report"]
        held = S.h6_rule([c["label"] for c in comp.values()]) == S.HOLDS
        h1_pass = S.h1(int(np.sum(S.success(hq))), True) == S.PASS
        rule += held
        n_h1 += h1_pass
        rule_h1 += held and h1_pass
    k = float(n_sims)
    return {"R_true": R_true, "top_cf": top_cf, "p_true": p_true, "n_sims": n_sims,
            "component_hold_rate": {f: v / k for f, v in hold.items()},
            "right_censored_hold_rate": {f: v / k for f, v in cens_hold.items()},
            "censor_report_rate": {f: v / k for f, v in report.items()}, "rule_hold_rate": rule / k,
            "h1_pass_rate": n_h1 / k, "rule_hold_rate_given_h1": rule_h1 / n_h1 if n_h1 else None}


def planted_gate(n_sims=1000, n_boot=S.N_BOOT, seed=SIM_SEED, **kw):
    """§9(ii) freeze gate: false-positive rate at R = 1, censored mis-labelling at top-rung median 0.92, detection at R = 8."""
    r1 = planted_ratio_sim(1.0, n_sims, n_boot=n_boot, seed=seed, **kw)
    top = planted_ratio_sim(None, n_sims, top_cf=0.92, n_boot=n_boot, seed=seed + 1, **kw)
    r8 = planted_ratio_sim(8.0, n_sims, n_boot=n_boot, seed=seed + 2, **kw)
    fp = max(r1["component_hold_rate"].values())
    mis = max(top["right_censored_hold_rate"].values())
    det = r8["rule_hold_rate"]
    return {"false_positive_rate_R1": fp, "censored_mislabel_rate": mis,
            "censor_report_rate_top092": max(top["censor_report_rate"].values()), "detection_rate_R8": det,
            "detection_rate_R8_given_h1": r8["rule_hold_rate_given_h1"],
            "fp_ok": bool(S.le(fp, 0.025)), "censored_ok": bool(S.le(mis, 0.025)),
            "h6_stays_registered": bool(S.le(fp, 0.025) and S.le(mis, 0.025)), "underpowered": bool(S.lt(det, 0.80)),
            "detail": {"R1": r1, "top092": top, "R8": r8}}


def coverage_sim(n_sims=2000, n=10, df=5, mu=0.10, scale=0.05, level=S.CI_LEVEL, n_boot=S.N_BOOT, seed=SIM_SEED):
    """Coverage of the registered BCa interval for the mean of n paired differences with t_df tails."""
    rng = np.random.default_rng(seed)
    hit = 0
    for _ in range(n_sims):
        d = mu + scale * rng.standard_t(df, n)
        lo, hi = S.bca(d, level=level, n_boot=n_boot)
        hit += lo <= mu <= hi
    cov = hit / n_sims
    return {"coverage": cov, "n_sims": n_sims, "n": n, "df": df, "level": level, "use_level": S.interval_level(cov)}

"""E3 mechanical decision rules and freeze-gate simulations (src/osw/summary_e3.py, src/osw/sim_e3.py)."""
import itertools
import json
import os

import numpy as np
import pytest

from osw import sim_e3
from osw import summary_e3 as S

SEEDS = list(range(200, 210))
NB = 400


# ---------- counts: every input of the grid gets exactly one, correct label ----------
def test_h1_every_count():
    want = {s: S.KILL if s <= 6 else S.PASS if s >= 9 else S.PARTIAL for s in range(11)}
    for s in range(11):
        assert S.h1(s, True) == want[s]
        assert S.h1(s, False) == S.UNINT
    assert {S.h1(s, True) for s in range(11)} == {S.KILL, S.PARTIAL, S.PASS}


def test_h2_and_h3_every_count():
    for s in range(11):
        assert S.h2(s) == (S.REACHED if s >= 9 else S.NOT_REACHED if s <= 1 else S.PARTIAL)
        assert S.h3(s) == (S.KILL if s <= 5 else S.PASS if s >= 9 else S.PARTIAL)


def test_h4_every_count():
    for k in range(11):
        assert S.h4(k, 10) == (S.LOCAL if k <= 2 else S.NONLOCAL if k >= 6 else S.UNDET)


def test_pc_a_counts_and_median_clause():
    for k in range(11):
        assert S.pc_a_label(k, 10, True) == (S.PASS if k >= 8 else S.FAIL)
        assert S.pc_a_label(k, 10, False) == S.FAIL
    cf = [0.99] * 8 + [0.2, 0.3]
    assert S.pc_a(cf) == S.PASS
    assert S.pc_a([0.95] * 10) == S.FAIL                       # median exactly 0.95: tie goes against
    assert S.pc_a([0.95] * 10, median_clause=False) == S.PASS
    assert S.pc_a([0.99] * 7 + [0.2] * 3) == S.FAIL


def test_operating_characteristics_match_design_table():
    oc = lambda f, p: S.operating_characteristics(f, p)
    rows = {(lambda s: S.h1(s, True), S.PASS): [0.011, 0.046, 0.149, 0.376, 0.736, 0.914],
            (lambda s: S.h1(s, True), S.PARTIAL): [0.161, 0.336, 0.500, 0.503, 0.251, 0.085],
            (lambda s: S.h1(s, True), S.KILL): [0.828, 0.618, 0.350, 0.121, 0.013, 0.001],
            (S.h3, S.PARTIAL): [0.366, 0.587, 0.700, 0.591, 0.262, 0.086],
            (S.h3, S.KILL): [0.623, 0.367, 0.150, 0.033, 0.002, 0.000],
            (lambda s: S.pc_a_label(s, 10, True), S.PASS): [0.055, 0.167, 0.383, 0.678, 0.930, 0.988]}
    for (f, lab), want in rows.items():
        got = [oc(f, p).get(lab, 0.0) for p in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)]
        assert np.allclose(got, want, atol=6e-4), (lab, got)
    assert abs(oc(S.h2, 0.3)[S.NOT_REACHED] - 0.149) < 6e-4 and abs(oc(S.h2, 0.5)[S.NOT_REACHED] - 0.011) < 6e-4
    for p in (0.3, 0.77):
        assert abs(sum(oc(lambda s: S.h1(s, True), p).values()) - 1) < 1e-12


def test_clopper_pearson_bounds():
    assert round(S.cp_lower(9, 10), 2) == 0.61 and round(S.cp_lower(10, 10), 2) == 0.74
    assert round(S.cp_upper(1, 10), 2) == 0.39 and round(S.cp_upper(3, 10), 2) == 0.61
    assert S.cp_lower(0, 10) == 0.0 and S.cp_upper(10, 10) == 1.0
    assert abs(S.cp_lower(10, 10) - 0.05 ** 0.1) < 1e-9


# ---------- boundary ties ----------
def test_tolerance_and_ties_go_against_hypothesis():
    assert not S.success(0.90) and not S.success(0.90 * (1 + 5e-10)) and S.success(0.90 * (1 + 2e-9))
    assert S.success(0.9000001)
    assert S.ge(1.0, 1.0, tie=True) and not S.ge(1.0, 1.0) and S.lt(1.0, 1.0, tie=True) and not S.gt(0.0, 0.0)
    assert S.h5_label(0.80, 0.9) == S.REFUTED
    assert S.h5_label(0.50, 0.9) == S.UNDET and S.h5_label(0.49, 0.40) == S.UNDET and S.h5_label(0.49, 0.41) == S.ESTABLISHED
    assert S.pc_b(0.10, 0.05) == S.FAIL and S.pc_b(0.11, 0.0) == S.FAIL and S.pc_b(0.11, 1e-6) == S.PASS
    assert S.pc_o(0.95) == S.FAIL and S.pc_o(0.9501) == S.PASS
    assert S.nc([0.0]) == S.FAIL and S.nc([-0.01, -0.2]) == S.PASS
    assert S.compute_check([80.0], 100.0) == S.FAIL and S.compute_check([80.01, 120], 100.0) == S.PASS


def test_transfer_success_definition_and_broken_oracle():
    assert S.transfer_success(0.99, 0.99)
    assert not S.transfer_success(0.99, -0.05) and not S.transfer_success(0.99, 0.30)     # broken O05 -> unsuccessful
    assert not S.transfer_success(0.99, 0.50)                                              # O05 tie at 0.50
    assert S.transfer_success(0.99, 0.5001)
    assert not S.transfer_success(0.95 * 0.98, 0.98) and S.transfer_success(0.95 * 0.98 + 1e-6, 0.98)
    assert not S.transfer_success(0.50, 0.51) and S.transfer_success(0.5001, 0.51)
    assert not S.transfer_success(0.90, 0.92, cf_c0=0.91) and S.transfer_success(0.9001, 0.92, cf_c0=0.91)


def test_h5_table_and_nan_versions():
    assert S.h5_combine(S.ESTABLISHED, S.ESTABLISHED) == S.ESTABLISHED
    assert S.h5_combine(S.ESTABLISHED, S.UNDET) == S.UNDET
    assert S.h5_combine(S.ESTABLISHED, S.REFUTED) == S.UNDET          # with-NaN version primary (reviewer fix)
    assert S.h5_combine(S.UNDET, S.REFUTED) == S.UNDET and S.h5_combine(S.REFUTED, S.ESTABLISHED) == S.REFUTED
    assert S.h5_combine(S.UNDET, S.UNDET) == S.UNDET


def test_h7_table():
    assert S.h7(0.0, -0.02, 0.02) == S.EQUIV
    assert S.h7(0.0, -0.03, 0.02) == S.UNCERTAIN                  # bound exactly on the margin
    assert S.h7(0.06, 0.01, 0.11) == S.HQ_BETTER
    assert S.h7(0.06, 0.01, 0.11, oom_opt_failure=True) == S.OOM_OPT_FAIL
    assert S.h7(0.05, 0.0, 0.1) == S.UNCERTAIN
    assert S.h7(-0.06, -0.11, -0.01) == S.OOM_BETTER
    assert S.h7(0.04, 0.01, 0.07) == S.UNCERTAIN
    labs = {S.EQUIV, S.HQ_BETTER, S.OOM_BETTER, S.UNCERTAIN}
    for d, w in itertools.product(np.linspace(-0.2, 0.2, 21), (0.005, 0.02, 0.05, 0.2)):
        assert S.h7(d, d - w, d + w) in labs


def test_pc_c_table():
    assert S.pc_c([0.05] * 8 + [0.5, 0.6]) == S.REPRODUCED
    assert S.pc_c([0.05] * 7 + [0.5, 0.6, 0.31]) == S.PARTIAL
    assert S.pc_c([0.6] * 10) == S.NOT_REPRODUCED
    assert S.pc_c([0.5] * 10) == S.NOT_REPRODUCED                # mean exactly 0.50: tie goes against reproduction
    assert S.pc_c([0.30] * 10) == S.PARTIAL


# ---------- PC-K arithmetic ----------
def test_pc_k_arithmetic():
    e2 = dict(S.E2_MEANS_A2)
    assert S.pc_k_a(e2) == S.PASS
    assert S.pc_k_a({k: v - 0.01 for k, v in e2.items()}) == S.FAIL          # mean diff exactly -0.01
    assert S.pc_k_a({k: v - 0.009 for k, v in e2.items()}) == S.PASS
    one = {**e2, ("GRU", 8): e2[("GRU", 8)] - 0.10}
    assert S.pc_k_a({k: v + (0.02 if k != ("GRU", 8) else 0) for k, v in one.items()}) == S.FAIL   # cell at -0.10
    one = {**e2, ("LRU", 4): e2[("LRU", 4)] - 0.099}
    assert S.pc_k_a({k: v + (0.02 if k != ("LRU", 4) else 0) for k, v in one.items()}) == S.PASS
    one = {**e2, ("LRU", 4): e2[("LRU", 4)] - 0.09}
    assert S.pc_k_a(one) == S.FAIL                                           # mean -0.01125
    assert S.pc_k_b(0.931, 0.859) == S.PASS
    assert S.pc_k_b(0.93, 0.9) == S.FAIL and S.pc_k_b(0.95, 0.858) == S.FAIL
    assert S.pc_k(S.PASS, S.PASS) == S.PASS and S.pc_k(S.PASS, S.FAIL) == S.FAIL


# ---------- H6 machinery ----------
def test_h6_interpolation_and_censoring():
    params = np.array([10.0, 100.0, 1000.0])
    P, code = S.p_from_medians([0.5, 0.8, 0.95], params)
    assert code == 0 and abs(float(P) - 10 ** (2 + 2 / 3)) < 1e-8
    P, code = S.p_from_medians([0.5, 0.92, 0.85], params)         # envelope keeps 0.92
    assert code == 0 and abs(float(P) - 10 ** (1 + 0.4 / 0.42)) < 1e-8
    P, code = S.p_from_medians([0.6, 0.95, 0.80], params)
    P2, _ = S.p_from_medians([0.6, 0.95, 0.99], params)
    assert float(P) == float(P2)                                  # later rungs never move the crossing
    P, code = S.p_from_medians([0.95, 0.97, 0.99], params)
    assert code == 1 and float(P) == 10.0
    P, code = S.p_from_medians([0.5, 0.6, 0.89], params)
    assert code == 2 and float(P) == 1000.0
    P, code = S.p_from_medians([0.5, 0.90, 0.95], params)         # tie at 0.90 counts as crossed (against H6)
    assert code == 0 and abs(float(P) - 100.0) < 1e-6
    Pb, cb = S.p_from_medians(np.array([[0.5, 0.8, 0.95], [0.95, 0.97, 0.99], [0.5, 0.6, 0.7]]), params)
    assert list(cb) == [0, 1, 2] and np.allclose(Pb, [10 ** (2 + 2 / 3), 10, 1000])


def test_h6_component_table():
    assert S.h6_component(1.99, 9) == S.KILL and S.h6_component(2.0, 9) == S.KILL
    assert S.h6_component(3.0, 3) == S.NOT_EST
    assert S.h6_component(4.0, 3) == S.NOT_EST
    assert S.h6_component(4.01, 2.0) == S.NOT_EST_IMP
    assert S.h6_component(4.01, 2.01) == S.COMP_HOLDS
    assert S.h6_rule([S.COMP_HOLDS, S.COMP_HOLDS]) == S.HOLDS and S.h6_rule([S.COMP_HOLDS, S.NOT_EST]) == S.NOT_HOLDS


def test_h6_ratios_r0_replicates_and_point_estimate():
    params = np.array([63.0, 147.0, 387.0, 1155.0, 3843.0])
    cl = np.tile(np.array([0.2, 0.3, 0.7, 0.85, 0.95])[:, None], (1, 10))
    hq_good = np.full(10, 0.98)
    out = S.h6_ratios(hq_good, {"GRU": (params, cl)}, n_boot=2000)["GRU"]
    assert out["frac_r0"] == 0 and abs(out["R"] - np.sqrt(1155 * 3843) / 54) < 1e-6 and out["label"] == S.COMP_HOLDS
    hq_frag = np.array([0.98] * 6 + [0.03] * 4)                    # median >= 0.90 but many replicates fall below
    out = S.h6_ratios(hq_frag, {"GRU": (params, cl)}, n_boot=2000)["GRU"]
    assert out["R"] > 4 and out["frac_r0"] > 0.01 and out["lb"] == 0.0 and out["label"] == S.NOT_EST_IMP
    out = S.h6_ratios(np.array([0.98] * 4 + [0.03] * 6), {"GRU": (params, cl)}, n_boot=500)["GRU"]
    assert out["R"] == 0.0 and out["label"] == S.KILL
    right = np.tile(np.array([0.2, 0.3, 0.5, 0.6, 0.77])[:, None], (1, 10))
    out = S.h6_ratios(hq_good, {"LRU": (params, right)}, n_boot=500)["LRU"]
    assert out["censored"] == "right" and out["censor_report"] and "note" in out and out["R"] == 3843 / 54


def test_h6_gates_and_stop():
    assert S.h6_gate(S.PASS, S.PASS, S.PASS, S.PASS) is None
    assert S.h6_gate(S.PARTIAL, S.PASS, S.PASS, S.PASS)[0] == S.NOT_JUDGED
    assert S.h6_gate(S.PASS, S.FAIL, S.PASS, S.PASS)[0] == S.NOT_JUDGED
    assert S.h6_gate(S.PASS, S.PASS, S.FAIL, S.PASS)[0] == S.UNINT
    assert S.h6_gate(S.PASS, S.PASS, S.PASS, S.FAIL)[0] == S.NOT_JUDGED_COMPUTE
    assert S.stop_rule(True, [1.9, 30]) == S.STOP and S.stop_rule(True, [2.0, 30]) == S.STOP
    assert S.stop_rule(True, [2.01, 30]) == S.NO_STOP and S.stop_rule(False, [1.0]) == S.NOT_JUDGED


def test_sigma_star():
    assert S.sigma_star([0.05, 0.1, 0.2, 0.3], [1.0, 0.9, 0.3, 0.0]) == (pytest.approx(0.1 + 0.4 / 0.6 * 0.1), None)
    assert S.sigma_star([0.05, 0.1], [0.4, 0.0]) == (None, "below_grid")
    assert S.sigma_star([0.05, 0.1], [1.0, 0.8]) == (None, "above_grid")


# ---------- bootstrap ----------
def test_bca_basic_properties():
    lo, hi = S.bca(np.full(10, 0.3), n_boot=200)
    assert lo == hi == pytest.approx(0.3)
    d = np.random.default_rng(201).normal(0.1, 0.05, 10)
    lo, hi = S.bca(d, n_boot=2000)
    lo2, hi2 = S.bca(d, level=0.975, n_boot=2000)
    assert lo < d.mean() < hi and lo2 <= lo and hi2 >= hi
    assert S.bca(d, n_boot=2000) == (lo, hi)
    X = np.stack([d + 0.8, d, d - 0.05], 1)
    lo, hi = S.bca(X, S.h5_stat, n_boot=2000)
    assert lo < float(S.h5_stat(X)) < hi
    assert S.interval_level(0.95) == 0.95 and S.interval_level(0.9399) == 0.975 and S.interval_level(0.94) == 0.975


# ---------- run results -> summary ----------
GRU_P = {2: 63, 4: 147, 8: 387, 16: 1155, 32: 3843}
LRU_P = {2: 59, 4: 99, 8: 259, 16: 771, 32: 2563}


def _rec(process, family, size, recipe, seed, cf, sigma=None, **kw):
    r = dict(block="U", arm=f"{process}-{family}{size}-{recipe}", process=process, family=family, size=size,
             recipe=recipe, seed=seed, sigma=sigma, cf=cf, val_nll=1.0, n_params=kw.pop("n_params", 10),
             cpu_s=kw.pop("cpu_s", 100.0), nan_flag=False, failed=False)
    r.update(kw)
    return r


def make_results(cf_over=None, extra=()):
    cf_over = cf_over or {}
    out = []
    j = lambda s: 0.001 * (s - 205)
    cells = [("A2", "HQ", 3, "M", None, 0.99, 54, 300.0), ("A2", "HQ", 3, "C1", None, 0.05, 54, 300.0),
             ("A2", "HQ", 3, "S", None, 0.10, 54, 300.0), ("A1", "HQ", 2, "M", None, 0.98, 16, 50.0),
             ("A1", "HQ", 2, "C0", None, 0.60, 16, 50.0), ("A1", "HQ", 2, "C2", 0.05, 0.99, 16, 50.0),
             ("A1", "HMM", 4, "HMM-C", None, 0.80, 19, 50.0), ("A1", "OOM", 3, "OOM-C", None, 0.985, 17, 50.0),
             ("A2w", "HQ", 3, "M", None, 0.95, 54, 300.0), ("A2w", "HQ", 3, "C2", 0.05, 0.96, 54, 300.0),
             ("A1", "GRU", 4, "CPLUS", None, 0.95, 130, 50.0), ("A1", "LRU", 16, "CPLUS", None, 0.87, 722, 50.0)]
    cells += [("A2", "HQ", 3, "C2", s, v, 54, 300.0) for s, v in ((0.05, 0.97), (0.1, 0.95), (0.2, 0.5), (0.3, 0.05))]
    for fam, P, cfs in (("GRU", GRU_P, (0.22, 0.30, 0.75, 0.85, 0.95)), ("LRU", LRU_P, (0.51, 0.67, 0.72, 0.78, 0.80))):
        for (h, p), v in zip(P.items(), cfs):
            cells.append(("A2", fam, h, "CPLUS-M" if h <= 8 else "CPLUS", None, v, p, 400.0))
    for (pr, fam, size, rec, sig, v, npar, cpu) in cells:
        v = cf_over.get((pr, fam, size, rec, sig), v)
        for s in SEEDS:
            out.append(_rec(pr, fam, size, rec, s, (v(s) if callable(v) else v) + j(s), sig, n_params=npar, cpu_s=cpu))
    for s in SEEDS:                                                    # C0: two lrs, the lower val_nll is kept
        out.append(_rec("A2", "HQ", 3, "C0", s, 0.10 + j(s), val_nll=1.0, lr=3e-3))
        out.append(_rec("A2", "HQ", 3, "C0", s, 0.25 + j(s), val_nll=1.1, lr=1e-2))
        out.append(_rec("A2", "REF", 0, "REF", s, None, cf_shuffled=-0.1))
    return out + list(extra)


def test_summarize_full_pass_path_and_report_order():
    sm = S.summarize(make_results(), n_boot=NB, pcd=True)
    json.dumps(sm)
    g, h = sm["gates"], sm["rules"]
    assert [g[k]["label"] for k in ("PC-A", "PC-B", "PC-K", "PC-O", "COMPUTE", "NC")] == [S.PASS] * 6
    assert g["PC-C (modifier)"]["label"] == S.REPRODUCED and g["PC-C (modifier)"]["inputs"]["mean"] < 0.11
    want = {"H1": S.PASS, "H2": S.NOT_REACHED, "H3a": S.PASS, "H3b": S.PASS, "H4": S.LOCAL, "H5": S.ESTABLISHED,
            "H6": S.HOLDS, "STOP": S.NO_STOP, "H7": S.EQUIV}
    assert {k: h[k]["label"] for k in want} == want
    gru = h["H6"]["inputs"]["components"]["GRU"]
    assert gru["censored"] == "none" and h["H6"]["inputs"]["components"]["LRU"]["censored"] == "right"
    assert sm["enqueue"] == {"T": True, "SC": True, "O": True}
    txt = S.report(sm)
    assert txt.index("GATES") < txt.index("PC-K") < txt.index("RULES") < txt.index("H1:")


def test_summarize_failed_pc_a_withholds_contrasts():
    sm = S.summarize(make_results({("A2", "HQ", 3, "C2", 0.05): 0.5}), n_boot=NB, pcd=True)
    h = sm["rules"]
    assert sm["gates"]["PC-A"]["label"] == S.FAIL
    for k in ("H1", "H2", "H3a", "H4", "H5"):
        assert h[k]["label"] == S.UNINT and "inputs" not in h[k] and h[k]["withheld"]
    assert h["H6"]["label"] == S.NOT_JUDGED and "inputs" not in h["H6"]
    assert sm["enqueue"]["SC"] is False


def test_failure_imputation_direction():
    res = make_results()
    for r in res:
        if r["process"] == "A2" and r["recipe"] in ("M", "C1") and r["seed"] == 203:
            r.update(failed=True, cf=None)
    C = S.build_cells(res)
    m, c1 = C[S.A2_M], C[S.A2_C1]
    i = m["seeds"].index(203)
    assert m["cf"][i] == 0.0 and m["n_failed"] == 1 and not m["uninterpretable"]
    assert c1["cf"][i] == pytest.approx(max(0.05 + 0.001 * (s - 205) for s in SEEDS if s != 203))
    sm = S.summarize(res, n_boot=NB, pcd=True)
    assert sm["rules"]["H1"]["label"] == S.PASS and sm["rules"]["H1"]["inputs"]["s"] == 9
    for r in res:
        if r["process"] == "A2" and r["recipe"] == "M" and r["seed"] == 204:
            r.update(failed=True, cf=None)
    sm = S.summarize(res, n_boot=NB, pcd=True)
    assert sm["rules"]["H1"]["label"] == S.UNINT and "failed" in sm["rules"]["H1"]["reason"]
    assert sm["arms"]["A2/HQ/3/M"]["n_failed"] == 2


def test_c0_keeps_lower_val_nll_and_survives_one_failed_lr():
    res = make_results()
    for r in res:
        if r["recipe"] == "C0" and r["process"] == "A2" and r["seed"] == 200 and r.get("lr") == 3e-3:
            r["failed"] = True
    C = S.build_cells(res)
    c0 = C[S.A2_C0]
    assert c0["n_failed"] == 0 and c0["cf"][0] == pytest.approx(0.25 - 0.005) and c0["cf"][1] == pytest.approx(0.10 - 0.004)


def test_summarize_compute_mismatch_and_pck_failure():
    res = make_results()
    for r in res:
        if r["recipe"] == "CPLUS-M":
            r["cpu_s"] = 100.0
    assert S.summarize(res, n_boot=NB, pcd=True)["rules"]["H6"]["label"] == S.NOT_JUDGED_COMPUTE
    sm = S.summarize(make_results({("A1", "GRU", 4, "CPLUS", None): 0.90}), n_boot=NB, pcd=True)
    assert sm["gates"]["PC-K"]["label"] == S.FAIL and sm["rules"]["H6"]["label"] == S.UNINT
    assert sm["rules"]["STOP"]["label"] == S.UNINT


def test_summarize_h3b_and_h7_gating():
    sm = S.summarize(make_results({("A2", "HQ", 3, "M", None): lambda s: 0.99 if s < 208 else 0.5}), n_boot=NB, pcd=True)
    assert sm["rules"]["H1"]["label"] == S.PARTIAL and sm["rules"]["H3b"]["label"] == S.DESCRIPTIVE
    assert sm["rules"]["H6"]["label"] == S.NOT_JUDGED and sm["enqueue"] == {"T": True, "SC": False, "O": True}
    sm = S.summarize(make_results(), n_boot=NB, g4=False, pcd=True)
    assert sm["rules"]["H3b"]["label"] == S.NOT_DECIDABLE_PF          # G4 failure: removed before freeze
    sm = S.summarize(make_results({("A1", "OOM", 3, "OOM-C", None): 0.90}), n_boot=NB, pcd=True)
    assert sm["gates"]["PC-O"]["label"] == S.FAIL and sm["rules"]["H7"]["label"] == S.NOT_JUDGED
    res = make_results({("A1", "OOM", 3, "OOM-C", None): 0.90})
    for r in res:
        if r["recipe"] == "OOM-C":
            r.update(cf=0.96 + 0.001 * (r["seed"] - 205) - 0.10, cf_ft=0.80, cf_spec=0.97)
    sm = S.summarize(res, n_boot=NB, pcd=True)
    assert sm["gates"]["PC-O"]["label"] == S.FAIL                     # median 0.86


def test_h7_relabel_in_summary():
    res = make_results({("A1", "OOM", 3, "OOM-C", None): 0.955})
    for r in res:
        if r["recipe"] == "OOM-C":
            r.update(cf_ft=0.90, cf_spec=0.955)
    sm = S.summarize(res, n_boot=NB, pcd=True)
    assert sm["rules"]["H7"]["inputs"]["oom_ft_below_spectral"]
    res2 = make_results({("A1", "OOM", 3, "OOM-C", None): 0.955, ("A1", "HQ", 2, "M", None): 0.99 + 0.03})
    for r in res2:
        if r["recipe"] == "OOM-C":
            r.update(cf_ft=0.90, cf_spec=0.955)
    assert S.summarize(res2, n_boot=NB, pcd=True)["rules"]["H7"]["label"] == S.OOM_OPT_FAIL


def test_h5_nan_version_blocks_established():
    res = make_results({("A2", "HQ", 3, "C1", None): lambda s: 0.05 if s < 205 else 0.6})
    for r in res:
        if r["process"] == "A2" and r["recipe"] == "C1" and r["seed"] < 205:
            r["nan_flag"] = True
    sm = S.summarize(res, n_boot=NB, pcd=True)
    v = sm["rules"]["H5"]["inputs"]
    assert v["without_nan"]["label"] == S.UNDET and v["with_nan"]["label"] == S.ESTABLISHED
    assert sm["rules"]["H5"]["label"] == S.UNDET                       # ESTABLISHED needs both versions
    assert v["with_nan"]["n"] == 10 and v["without_nan"]["n"] == 5


def test_pcd_failure_halts_everything():
    sm = S.summarize(make_results(), n_boot=NB, pcd=False)
    assert all(v["label"] == S.UNINT for v in sm["rules"].values())


# ---------- simulations ----------
def test_planted_sim_fast_and_deterministic():
    a = sim_e3.planted_gate(n_sims=12, n_boot=300)
    b = sim_e3.planted_gate(n_sims=12, n_boot=300)
    assert json.dumps(S._py(a), sort_keys=True) == json.dumps(S._py(b), sort_keys=True)
    assert a["false_positive_rate_R1"] <= 0.025 and a["fp_ok"]
    assert 0 <= a["censored_mislabel_rate"] <= 1 and 0 <= a["detection_rate_R8"] <= 1
    r = sim_e3.planted_ratio_sim(8.0, 6, n_boot=300, plateau_rate=0.0)
    assert r["rule_hold_rate"] == 1.0 and abs(r["p_true"]["GRU"] - 432) < 1e-9
    top = sim_e3.truth_curve(sim_e3.LADDERS["GRU"], sim_e3.p_true_for_top(sim_e3.LADDERS["GRU"], 0.92))
    assert abs(top[-1] - 0.92) < 1e-12


def test_coverage_sim_fast_and_deterministic():
    a = sim_e3.coverage_sim(40, n_boot=500)
    assert a == sim_e3.coverage_sim(40, n_boot=500) and 0.6 <= a["coverage"] <= 1.0
    assert a["use_level"] in (0.95, 0.975)


@pytest.mark.skipif(os.environ.get("E3_SLOW") != "1", reason="slow: set E3_SLOW=1")
def test_planted_gate_full_size():
    g = sim_e3.planted_gate(n_sims=1000)
    assert g["fp_ok"]

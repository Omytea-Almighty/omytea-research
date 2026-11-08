"""Regression tests for the E3 review fixes: the G1-G10 development gate, PC-D enforcement (FIXTURE.json), H4 under the
G9 sigma cut, the H6 terminal rule, the pre-freeze removals (G4 / G6 / H6 demotion / N_test lock), D7, the per-sequence
bootstrap SD of CF, and the minor fixes (C0 NaN, CPLUS NaN probe, lr-grid init, integrity-gated final summary)."""
import copy
import json
from functools import lru_cache

import numpy as np
import pytest

from osw import cplus, queue_e3 as q, run_e3, summary_e3 as S, tasks_e3
from test_e3_summary import NB, make_results

LRS = {"GRU": [3e-3, 1e-2], "LRU": [3e-3, 1e-2]}
SIMS = {"planted": {"h6_stays_registered": True, "underpowered": False, "fp_ok": True, "censored_ok": True},
        "coverage": {"coverage": 0.95}}
REF_FIELDS = dict(cf=None, cf_exact=1.0, cf_marginal=0.0, cf_mix={"0.5": {"cf": 0.5, "cf_direct": 0.5}},
                  marginal_vs_stationary_rel=0.001, n_params_checks={"HQ3": {"got": 54, "want": 54, "ok": True}},
                  kraus_vp={"max_abs_err": 1e-7, "grad_finite": True, "ok": True}, cf_shuffled=-0.1,
                  ref={"markov1": 0.3, "markov2": 0.4, "markov3": 0.45, "linear_hard": 0.6, "linear_soft": 0.75},
                  denominator_k10=0.018, share_k3=0.5, kl_marginal=[0.1] * 20)
TAU_VAL = {0.1: 1.2, 0.3: 1.0, 0.6: 1.0, 0.9: 1.1}
LR_VAL = {"GRU": {1e-3: 1.3, 3e-3: 1.1, 1e-2: 1.0, 3e-2: 1.2}, "LRU": {1e-3: 1.0, 3e-3: 1.05, 1e-2: 1.3, 3e-2: 1.4}}


def _dev_rec(j, **kw):
    r = dict(j, job=dict(j), failed=False, nan_flag=False, cf=0.97, val_nll=1.0, n_params=10, cpu_s=100.0,
             kl=[0.0] * 20, kl_marginal=[0.1] * 20, cf_floor_1e6=0.97, cf_boot_sd=0.005)
    rc, p, f, sz = j["recipe"], j["process"], j["family"], j["size"]
    if rc == "REF":
        r.update(REF_FIELDS)
    elif rc == "M":
        r.update(cpu_s=330.0, mom_cpu_s=100.0)
    elif rc == "C1":
        r.update(cpu_s=100.0 * j["R"], cf=0.1)
    elif rc == "S":
        r.update(cpu_s=120.0 * j["R"], val_nll=TAU_VAL[j["tau0"]], cf=0.1)
    elif rc == "HMM-C":
        r.update(cf=0.80)
    elif rc == "LRGRID":
        r.update(val_nll=LR_VAL[f][j["lr"]])
    elif rc == "CPLUS" and p == "A2":
        r.update(cpu_s={2: 40.0, 32: 100.0}[sz])
    elif rc == "CPLUS" and p == "A1":
        r.update(cf={"GRU": 0.95, "LRU": 0.88}[f])
    r.update(cf_floor_1e6=r["cf"], **kw)
    return r


def dev_records(fn=None):
    js = q.jobs_dev1() + q.jobs_dev2(LRS) + q.jobs_dev3(LRS) + q.jobs_fixture()
    recs = [_dev_rec(j) for j in js]
    if fn:
        for r in recs:
            old = (r["cf"], r.get("cf_floor_1e6"))
            fn(r)
            if r.get("cf_floor_1e6") == old[1] and r["cf"] != old[0]:      # keep the floor rescoring consistent
                r["cf_floor_1e6"] = r["cf"]
    return recs


HANKEL_OK = {"cf": 1.0, "ok": True}


# ---------------------------------------------------------------- MAJOR 1: the development gate exists and is mechanical
def test_dev_gate_derives_every_constant_and_allows_freeze():
    g = q.dev_gate(dev_records(), SIMS, tests_ok=True, hankel=HANKEL_OK)
    C = g["frozen_constants"]
    assert g["freeze_ok"], g["blocking"]
    assert not q.missing_constants(C)
    assert (C["R1"], C["RS"], C["TAU0"]) == (4, 3, 0.3)              # 4 x 100 >= 330; 3 x 120 >= 330; tie -> grid order
    assert C["CPLUS_LRS"] == {"GRU": [3e-3, 1e-2], "LRU": [1e-3, 3e-3]}
    assert C["CPU_FLOOR"] == 330.0 and C["R_C"]["GRU2"] == 9 and C["R_C"]["GRU4"] == 8 and C["R_C"]["LRU32"] == 4
    assert (C["FT_STEPS"], C["MEDIAN_CLAUSE"], C["G4_PASS"], C["G6_PASS"], C["H6_REGISTERED"], C["CI_LEVEL"]) == \
        (3000, True, True, True, True, 0.95)
    assert (C["N_TEST_A2W"], C["N_TEST_A3"], C["TRIMS"]) == (10_000, 4_000, [])
    assert len(q.jobs_U(C)) == 200 and len(q.jobs_SC(C)) == 125
    assert g["gates"]["G1"]["status"] == "PASS" and q.dev_gate(dev_records(), SIMS, hankel=HANKEL_OK)["gates"]["G1"]["status"] == q.PENDING


def test_g2_outcomes_median_drop_redev_and_no_freeze():
    def c2(vals):
        def fn(r):
            if r["recipe"] == "C2" and r["process"] == "A2" and r["block"] == "DEV1" and abs(r["sigma"] - 0.05) < 1e-9:
                r["cf"] = vals[r["seed"] - 200]
        return fn
    assert q.g2(dev_records(c2([0.92] * 5)))["MEDIAN_CLAUSE"] is False
    recs = dev_records(c2([0.95, 0.5, 0.5, 0.95, 0.95]))
    assert q.g2(recs)["status"] == "REDEV_REQUIRED"
    assert not q.dev_gate(recs, SIMS, True, HANKEL_OK)["freeze_ok"]
    redev = [_dev_rec(j, cpu_s=600.0, mom_cpu_s=100.0) for j in q.jobs_redev()]
    g = q.g2(recs + redev)
    assert (g["status"], g["FT_STEPS"], g["MEDIAN_CLAUSE"]) == (q.CHANGED, 6000, True)
    assert q.dev_gate(recs + redev, SIMS, True, HANKEL_OK)["frozen_constants"]["CPU_FLOOR"] == 600.0   # M at 6,000 steps
    bad = [dict(r, cf=0.5) for r in redev]
    assert q.g2(recs + bad)["status"] == q.NO_FREEZE


def test_g4_g5_g6_g8_g10_outcomes():
    def set_(cond, **kw):
        def fn(r):
            if cond(r):
                r.update(kw)
        return fn
    g = q.dev_gate(dev_records(set_(lambda r: r["process"] == "A2w" and r["recipe"] == "C2", cf=0.3)), SIMS, True, HANKEL_OK)
    assert g["frozen_constants"]["G4_PASS"] is False and g["gates"]["G4"]["status"] == q.REMOVES and g["freeze_ok"]
    g = q.dev_gate(dev_records(set_(lambda r: r["family"] == "HMM" and r["size"] == 4, cf=0.90)), SIMS, True, HANKEL_OK)
    assert g["gates"]["G5"]["status"] == q.NO_FREEZE and not g["freeze_ok"]
    g = q.dev_gate(dev_records(set_(lambda r: r["family"] == "OOM" and r["size"] == 3, cf=0.90)), SIMS, True, HANKEL_OK)
    assert g["frozen_constants"]["G6_PASS"] is False and g["freeze_ok"]
    assert q.g6(dev_records(), {"cf": 0.9, "ok": False})["G6_PASS"] is False
    g = q.dev_gate(dev_records(set_(lambda r: r["process"] in ("A2w", "A3") and r["recipe"] == "C2", cf_boot_sd=0.03)),
                   SIMS, True, HANKEL_OK)
    assert (g["frozen_constants"]["N_TEST_A2W"], g["frozen_constants"]["N_TEST_A3"]) == (20_000, 8_000)
    sims = {"planted": dict(SIMS["planted"], h6_stays_registered=False), "coverage": {"coverage": 0.90}}
    C = q.dev_gate(dev_records(), sims, True, HANKEL_OK)["frozen_constants"]
    assert C["H6_REGISTERED"] is False and C["CI_LEVEL"] == 0.975
    assert q.dev_gate(dev_records(), None, True, HANKEL_OK)["gates"]["G8"]["status"] == q.PENDING
    for fn in (set_(lambda r: r["recipe"] == "M", cf_floor_1e6=0.9), set_(lambda r: r["recipe"] == "REF" and r["process"] == "A2",
               kraus_vp={"ok": False, "max_abs_err": 0.1}), set_(lambda r: r["recipe"] == "REF" and r["process"] == "A2",
               ref=dict(REF_FIELDS["ref"], markov3=0.8)), set_(lambda r: r["process"] == "A2" and r["recipe"] == "C2", cf_boot_sd=0.02)):
        assert q.g8(dev_records(fn), SIMS)["status"] == q.NO_FREEZE
    g = q.g10(dev_records(set_(lambda r: r["process"] == "A1" and r["family"] == "LRU" and r["recipe"] == "CPLUS", cf=0.85)), 3000)
    assert g["status"] == q.NO_FREEZE


def test_g9_cuts_in_order_and_terminal_rule():
    base = q.dev_gate(dev_records(), SIMS, True, HANKEL_OK)["frozen_constants"]
    C = dict(base, TRIMS=[])
    p0 = q.projection(dev_records(), C, [])["total_h"]
    assert q.g9(dev_records(), C)["TRIMS"] == [] and p0 <= q.G9_TARGET_H
    heavy = dev_records(lambda r: r.update(cpu_s=r["cpu_s"] * 1.6))
    g = q.g9(heavy, C)
    assert g["TRIMS"] and g["TRIMS"] == list(q.TRIM_ORDER[:len(g["TRIMS"])]) and "SC" not in g["TRIMS"]
    huge = dev_records(lambda r: r.update(cpu_s=r["cpu_s"] * 20))
    g = q.g9(huge, C)
    assert g["TRIMS"] == list(q.TRIM_ORDER) and g["projected_h"] < g["steps"][-2]["total_h"]
    assert q.jobs_SC(dict(C, TRIMS=g["TRIMS"])) == [] and q.jobs_SENS(dict(C, TRIMS=g["TRIMS"]), placeholder=True) == []


def test_exact_hankel_fixture_reaches_cf_one():
    h = q.exact_hankel_fixture()
    assert h["ok"] and abs(h["cf"] - 1) < 1e-4


def test_g1_dev_lists_cover_every_recipe():
    js = q.jobs_dev1() + q.jobs_dev2(LRS) + q.jobs_dev3(LRS)
    kinds = {(j["recipe"], j.get("train_subset") is not None) for j in js if j["seed"] == 200}
    assert {("CPLUS-M", False), ("SENS", False), ("CPLUS", True), ("HMM-C", False)} <= kinds
    assert any(j["family"] == "HMM" and j["size"] == 3 for j in js)
    sens = q.jobs_dev3(LRS)[0]
    assert sens["base"] == run_e3.job_name(next(j for j in js if j["recipe"] == "CPLUS-M"))


# ---------------------------------------------------------------- MAJOR 2: PC-D enforced
def test_pcd_missing_or_failed_halts_all_rules():
    for pcd, why in ((None, "not provided"), (False, "PC-D failed")):
        sm = S.summarize(make_results(), n_boot=NB, pcd=pcd)
        assert all(v["label"] == S.UNINT and why in v["reason"] for v in sm["rules"].values())
    assert S.summarize(make_results(), n_boot=NB, pcd=True)["rules"]["H1"]["label"] == S.PASS


def test_fixture_checks_and_file(tmp_path):
    refs = [_dev_rec(j) for j in q.jobs_fixture()]
    assert q.fixture_checks(refs)["pcd"]
    assert not q.fixture_checks([r for r in refs if r["process"] != "A3"])["pcd"]
    bad = copy.deepcopy(refs)
    bad[0]["cf_exact"] = 1 + 1e-5
    assert not q.fixture_checks(bad)["pcd"]
    bad = copy.deepcopy(refs)
    bad[2]["failed"] = True
    assert not q.fixture_checks(bad)["pcd"]
    assert q.fixture_pcd(tmp_path) is None
    q.write_fixture(tmp_path, dev=q.fixture_checks(refs))
    assert q.fixture_pcd(tmp_path) is True
    q.write_fixture(tmp_path, confirm={"pcd": False})
    fx = q.read_fixture(tmp_path)
    assert fx["dev"]["pcd"] and fx["pcd"] is False and q.fixture_pcd(tmp_path) is False


def test_confirm_refuses_without_manifest_or_dev_pcd(tmp_path):
    with pytest.raises(RuntimeError, match="manifest"):
        q.confirm(q.PLACEHOLDER, tmp_path)
    q.freeze_e2_manifest(tmp_path)
    with pytest.raises(FileExistsError):
        q.freeze_e2_manifest(tmp_path)
    with pytest.raises(RuntimeError, match="PC-D"):
        q.confirm(q.PLACEHOLDER, tmp_path)
    q.write_fixture(tmp_path, dev={"pcd": False})
    with pytest.raises(RuntimeError, match="PC-D"):
        q.confirm(q.PLACEHOLDER, tmp_path)
    assert not list(tmp_path.glob("e3-*.json"))


@lru_cache(maxsize=None)
def _data(process, seed, n_test=None):
    return tasks_e3.data(process, seed, n_test)


def test_ref_job_carries_the_kraus_check_and_feeds_the_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(run_e3, "load_data", _data)
    r = run_e3.run_job(dict(block="FIX", arm="REF", process="A2", family="REF", size=0, recipe="REF", seed=200,
                            sigma=None, ref_steps=10), tmp_path)
    assert not r["failed"] and r["kraus_vp"]["ok"] and r["kraus_vp"]["max_abs_err"] < 2e-3
    row = q.fixture_checks([r])["rows"][0]
    assert row["ok"] and all(row["checks"].values())


# ---------------------------------------------------------------- MAJOR 3: H4 under the sigma = 0.2 cut
def test_h4_judged_on_sigma_03_when_02_cut():
    res = [r for r in make_results() if not (r["recipe"] == "C2" and r["process"] == "A2" and abs((r["sigma"] or 0) - 0.2) < 1e-9)]
    h = S.summarize(res, n_boot=NB, pcd=True, trims=["O", "D4", "SENS", "C2_0.2"])["rules"]["H4"]
    assert h["label"] == S.LOCAL and h["inputs"]["sigmas"] == [0.05, 0.1, 0.3] and h["inputs"]["sigmas_missing"] == [0.2]
    assert h["inputs"]["k_at_0.3"] == 0 and h["inputs"]["cp_upper"] == pytest.approx(S.cp_upper(0, 10))


# ---------------------------------------------------------------- MAJOR 4: H6 / STOP under the G9 terminal rule
def test_h6_and_stop_not_run_compute_when_sc_cut():
    res = [r for r in make_results() if r["family"] not in ("GRU", "LRU")]
    sm = S.summarize(res, n_boot=NB, pcd=True, trims=list(q.TRIM_ORDER))
    assert sm["rules"]["H6"]["label"] == S.NOT_RUN_COMPUTE == sm["rules"]["STOP"]["label"]
    assert S.summarize(res, n_boot=NB, pcd=True)["rules"]["H6"]["label"] == S.UNINT        # without the cut: PC-K missing


# ---------------------------------------------------------------- MAJOR 5: pre-freeze removals and the N_test lock
def test_g4_g6_h6_flags_change_jobs_and_labels():
    C = dict(q.PLACEHOLDER, G4_PASS=False)
    assert not any(j["process"] == "A2w" for j in q.jobs_T(C)) and len(q.jobs_T(C)) == 13
    assert q.jobs_O(dict(q.PLACEHOLDER, G6_PASS=False)) == []
    sm = S.summarize(make_results(), n_boot=NB, pcd=True, g4=False, g6=False)
    assert sm["rules"]["H3b"]["label"] == S.NOT_DECIDABLE_PF and sm["rules"]["H7"]["label"] == S.REMOVED_PF
    assert sm["enqueue"]["O"] is False and sm["descriptive"]["D2"]["label"] == S.REMOVED_PF
    sm = S.summarize(make_results(), n_boot=NB, pcd=True, h6_registered=False)
    assert sm["rules"]["H6"]["label"] == S.DESCRIPTIVE and "HOLDS" in sm["rules"]["H6"]["reason"]
    assert sm["rules"]["H6"]["inputs"]["components"]["GRU"]["label"] == S.COMP_HOLDS
    assert sm["rules"]["STOP"]["label"] == S.DESCRIPTIVE                   # author decision 2: STOP demoted with H6


def test_n_test_lock_reaches_the_data(tmp_path, monkeypatch):
    C = dict(q.PLACEHOLDER, N_TEST_A2W=20_000, N_TEST_A3=8_000)
    js = q.jobs_U(C) + q.jobs_T(C) + q.jobs_SC(C)
    assert {j["n_test"] for j in js if j["process"] == "A2w"} == {20_000}
    assert {j["n_test"] for j in js if j["process"] == "A3"} == {8_000}
    assert all("n_test" not in j for j in js if j["process"] in ("A1", "A2"))
    a, b = tasks_e3.data("A1", 200), tasks_e3.data("A1", 200, n_test=500)
    assert np.array_equal(a[0][0], b[0][0]) and np.array_equal(a[1][0], b[1][0]) and b[2][0].shape[0] == b[3].shape[0] == 500
    seen = []
    monkeypatch.setattr(run_e3, "load_data", lambda p, s, n_test=None: seen.append(n_test) or _data(p, s, n_test))
    run_e3.run_job(dict(block="FIX", arm="REF", process="A1", family="REF", size=0, recipe="REF", seed=200, sigma=None,
                        ref_steps=5, n_test=300), tmp_path)
    assert seen == [300]


# ---------------------------------------------------------------- MAJOR 6: D7 and per-replica CF
def _ladder_recs(meds, sens=None, replica=True):
    out = []
    P = {"GRU": (63, 147, 387, 1155, 3843), "LRU": (59, 99, 259, 771, 2563)}
    for f, ms in meds.items():
        for (sz, rc), npar, m in zip(S.LADDER[f], P[f], ms):
            for s in range(20, 30):
                r = dict(process="A2", family=f, size=sz, recipe=rc, seed=s, sigma=None, cf=m, val_nll=1.0,
                         n_params=npar, failed=False)
                if rc == "CPLUS-M" and replica:
                    r["replica_cf"] = [m - 0.02, m]
                out.append(r)
    for f, (sz, v) in (sens or {}).items():
        out += [dict(process="A2", family=f, size=sz, recipe="SENS", seed=s, sigma=None, cf=v, val_nll=0.9, failed=False)
                for s in range(20, 25)]
    return out


def test_d7_flags_and_replica_delta():
    meds = {"GRU": (0.22, 0.30, 0.75, 0.85, 0.95), "LRU": (0.5, 0.6, 0.7, 0.95, 0.96)}
    recs = _ladder_recs(meds, {"GRU": (16, 0.92), "LRU": (8, 0.99)})
    d = S.d7(S.build_cells(recs), recs)
    g, lr = d["families"]["GRU"], d["families"]["LRU"]
    assert g["rung_matches_lower_bracket"] and lr["rung_matches_lower_bracket"] and g["seeds"] == list(range(20, 25))
    assert g["optimisation_limited"] and not g["not_robust"] and g["delta_cf"] == pytest.approx(0.07)
    assert lr["R_cplus_m"] >= 4 > lr["R_sens"] and lr["not_robust"]
    assert d["flags"] == {"H6_not_robust": True, "classical_optimisation_limited": True}
    assert d["replica_delta_cf"]["GRU2"] == pytest.approx(0.02)
    cut = S.d7(S.build_cells(recs), recs, cut=["SENS"])
    assert "families" not in cut and "cut by G9" in cut["sens"] and cut["replica_delta_cf"]["LRU8"] == pytest.approx(0.02)


def test_cplus_m_job_records_every_replica_cf(tmp_path, monkeypatch):
    monkeypatch.setattr(run_e3, "load_data", _data)
    r = run_e3.run_job(dict(block="DEV2", arm="it", process="A1", family="LRU", size=2, recipe="CPLUS-M", seed=200,
                            sigma=None, R=2, lrs=[3e-3, 1e-2], steps_probe=5, steps_total=10), tmp_path)
    assert not r["failed"], r.get("error")
    assert len(r["replica_cf"]) == 2 and r["cf"] == pytest.approx(r["replica_cf"][r["selected_replica"]])
    assert "replica_preds" not in r


# ---------------------------------------------------------------- MAJOR 7: per-sequence bootstrap SD of CF
def test_cf_boot_sd_in_score():
    tr, va, te, exact = _data("A1", 200)
    marg = exact[:, -1].mean(0)
    pred = 0.5 * exact + 0.5 * marg
    m = run_e3.score(pred, exact, te)
    assert run_e3.score(exact, exact, te)["cf_boot_sd"] == 0.0
    assert 0 < m["cf_boot_sd"] < 0.05 and m["cf_boot_sd"] == run_e3.score(pred, exact, te)["cf_boot_sd"]
    q_ = np.clip(exact, 1e-12, 1)
    a = (q_ * (np.log(q_) - np.log(np.clip(pred, 1e-9, 1)))).sum(-1)[:, :10].mean(1)
    b = (q_ * np.log(q_ / marg)).sum(-1)[:, :10].mean(1)
    assert 1 - a.mean() / b.mean() == pytest.approx(m["cf"], abs=1e-12)
    rng = np.random.default_rng(S.BOOT_SEED)
    cf = np.concatenate([1 - a[i].mean(1) / b[i].mean(1) for i in (rng.integers(0, len(a), (100, len(a))) for _ in range(10))])
    assert m["cf_boot_sd"] == pytest.approx(cf.std(ddof=1), rel=1e-12)


# ---------------------------------------------------------------- minor fixes
def test_c0_nan_without_cf_is_not_a_job_failure():
    res = make_results()
    for r in res:
        if r["recipe"] == "C0" and r["process"] == "A2" and r["seed"] in (200, 201):
            r.update(cf=None, val_nll=None, nan_flag=True)
    c = S.build_cells(res)[S.A2_C0]
    assert c["n_failed"] == 0 and c["n_nan"] == 2 and not c["uninterpretable"] and c["nan_imputed_seeds"] == [200, 201]
    assert c["cf"][0] == pytest.approx(max(c["cf"][2:]))
    sm = S.summarize(res, n_boot=NB, pcd=True)
    assert sm["gates"]["PC-C (modifier)"]["label"] != S.UNINT and sm["rules"]["H5"]["label"] != S.UNINT


def test_cplus_nan_probe_is_not_continued(monkeypatch):
    tr, va, te, _ = _data("A1", 200)
    orig, calls = cplus.run_phase, []

    def nan_probe(model, opt, gen, data, lr0, steps, decay, st, on_ckpt=None):
        calls.append(decay)
        ok = orig(model, opt, gen, data, lr0, steps, decay, st, on_ckpt)
        if not decay:
            st.update(nan_flag=True, nan_step=st["step"])
            return False
        return ok
    monkeypatch.setattr(cplus, "run_phase", nan_probe)
    r = cplus.cplus_single("GRU", 2, tr, va, te, 2, 200, 0, (3e-3, 1e-2), steps_probe=5, steps_total=10)
    assert calls == [False] * 4 and r["continued"] is False and r["nan_flag"] and r["adam_step"] == 5


def test_lrgrid_shares_one_init_across_lrs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_e3, "load_data", _data)
    rs = [run_e3.run_job(dict(block="DEV1", arm="G10-lr", process="A1", family="GRU", size=4, recipe="LRGRID", seed=200,
                              sigma=None, lr=lr, lr_index=i, steps=5), tmp_path) for i, lr in enumerate((1e-3, 3e-2))]
    assert rs[0]["init_seed"] == rs[1]["init_seed"] == cplus.init_seed(200, 4, run_e3.LRGRID_R0)
    assert rs[0]["cf"] != rs[1]["cf"]


def test_lru_ceiling_note_uses_point_estimate():
    hq = np.full(10, 0.97)
    lad = {"LRU": (np.array([59, 99, 259, 771, 2563.0]), np.array([[0.5] * 10, [0.6] * 10, [0.7] * 10, [0.75] * 10,
                                                                   [0.6] * 5 + [0.95] * 5]))}
    out = S.h6_ratios(hq, lad, n_boot=300)["LRU"]
    assert out["censored"] == "right" and "note" in out                   # envelope top 0.775 within 0.768 +- 0.03


def test_integrity_failure_writes_no_final_summary(tmp_path, monkeypatch):
    q.freeze_e2_manifest(tmp_path)
    q.write_fixture(tmp_path, dev=q.fixture_checks([_dev_rec(j) for j in q.jobs_fixture()]))
    fake = [dict(r, block="U", design_sha256="d", code_manifest_sha256="c1" if i % 2 else "c2")
            for i, r in enumerate(make_results()) if r["seed"] >= 205]
    for r in fake:
        r["seed"] = r["seed"] - 185                                         # 20-24
        if r["recipe"] == "REF":
            r.update({k: v for k, v in REF_FIELDS.items() if k != "cf"})
    fake += [dict(fake[-1], process=p) for p in ("A1", "A2w", "A3")]
    monkeypatch.setattr(q, "run_jobs", lambda *a, **k: {})
    monkeypatch.setattr(q, "all_done", lambda *a, **k: True)
    monkeypatch.setattr(q, "records", lambda out=None, blocks=None: fake)
    res = q.confirm(q.PLACEHOLDER, tmp_path)
    assert not res["integrity"]["ok"] and (tmp_path / "INTEGRITY-FAIL.json").exists()
    assert not (tmp_path / "SUMMARY.json").exists() and (tmp_path / "GATES-U.json").exists()
    assert json.loads((tmp_path / "FIXTURE.json").read_text())["confirm"]["pcd"] is True


def test_brier_scores_registered_labels():
    sm = S.summarize(make_results(), n_boot=NB, pcd=True)
    b = sm["brier"]["per_rule"]
    lab = sm["rules"]["H1"]["label"]
    want = sum((p - (k == lab)) ** 2 for k, p in S.PREDICTIONS["H1"].items())
    assert b["H1"] == pytest.approx(want) and "H6" in b and 0 <= sm["brier"]["mean"] <= 2

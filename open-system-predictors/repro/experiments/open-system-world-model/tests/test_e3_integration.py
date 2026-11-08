"""E3 integration: run_e3 jobs end to end (dev seed 200, minimal steps), C0 isolation from runs/e2, the poisoning test,
bitwise reruns, the dry-run job table and the seed namespaces (E3-DESIGN §4, §5, §9 iii, §16)."""
import hashlib
import json
from functools import lru_cache

import numpy as np
import pytest
import torch

from osw import cplus, hmm_e3, hq_init, oom_e3, queue_e3, run_e3, stiefel, tasks_e3

LRS = [3e-3, 1e-2]
BASE = dict(block="DEV1", arm="it", process="A1", seed=200, sigma=None)
FAST = {
    "M": dict(family="HQ", size=2, recipe="M", restarts=2, mom_iters=5, steps=20),
    "C1": dict(family="HQ", size=2, recipe="C1", R=1, steps=20),
    "S": dict(family="HQ", size=2, recipe="S", R=1, tau0=0.3, steps=20),
    "C2": dict(family="HQ", size=2, recipe="C2", sigma=0.05, steps=20),
    "CPLUS": dict(family="GRU", size=2, recipe="CPLUS", lrs=LRS, steps_probe=10, steps_total=20),
    "CPLUS-M": dict(family="LRU", size=2, recipe="CPLUS-M", R=2, lrs=LRS, steps_probe=10, steps_total=20),
    "SENS": dict(family="LRU", size=2, recipe="SENS", R=1, lrs=LRS, steps_probe=10, steps_total=20,
                 base="e3-DEV1-A1-LRU2-CPLUS-M-s200.json"),
    "LRGRID": dict(family="GRU", size=4, recipe="LRGRID", lr=3e-3, lr_index=1, steps=20),
    "HMM-C": dict(family="HMM", size=3, recipe="HMM-C", restarts=2, mom_iters=5, em_iters=2, train_subset=2000),
    "OOM-C": dict(family="OOM", size=3, recipe="OOM-C", restarts=2, mom_iters=5, steps=20),
    "REF": dict(family="REF", size=0, recipe="REF", ref_steps=20),
}
COMMON = ("block", "arm", "process", "family", "size", "recipe", "seed", "sigma", "n_params", "val_nll", "cf", "kl",
          "kl_marginal", "cpu_s", "job_cpu_s", "wall_s", "nan_flag", "failed", "attempt", "design_sha256",
          "code_manifest_sha256", "result_sha256")
VOLATILE = ("cpu_s", "job_cpu_s", "wall_s", "mom_cpu_s", "started_utc", "result_sha256")


@lru_cache(maxsize=None)
def _data(process, seed):
    return tasks_e3.data(process, seed)


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    out = tmp_path_factory.mktemp("e3")
    mp = pytest.MonkeyPatch()
    mp.setattr(run_e3, "load_data", _data)
    recs = {k: run_e3.run_job(dict(BASE, **v), out) for k, v in FAST.items()}      # SENS after its CPLUS-M base
    mp.undo()
    return out, recs


def _strip(r):
    return {k: v for k, v in r.items() if k not in VOLATILE}


# (1) every recipe end to end
@pytest.mark.parametrize("recipe", list(FAST))
def test_recipe_end_to_end(e2e, recipe):
    out, recs = e2e
    r = recs[recipe]
    assert not r["failed"], r.get("error")
    for k in COMMON:
        assert k in r or (recipe == "REF" and k == "kl"), k            # REF has several predictors, no single kl
    for k, v in FAST[recipe].items():
        assert r[k] == v and r["job"][k] == v                 # job fields survive the fit diagnostics
    on_disk = json.loads((out / run_e3.job_name(r)).read_text())
    assert on_disk == r
    body = {k: v for k, v in r.items() if k != "result_sha256"}
    assert r["result_sha256"] == hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    assert r["design_sha256"] == run_e3.design_sha256() and r["code_manifest_sha256"] == run_e3.code_manifest_sha256()
    assert len(r["kl_marginal"]) == 20 and r["attempt"] == 1 and r["threads"] == 1
    if recipe == "REF":
        assert abs(r["cf_exact"] - 1) < 1e-6 and abs(r["cf_marginal"]) < 1e-6 and r["cf_shuffled"] <= 0
        assert all(abs(v["cf"] - v["cf_direct"]) < 1e-9 for v in r["cf_mix"].values())
        assert r["marginal_vs_stationary_rel"] <= 0.02 and all(v["ok"] for v in r["n_params_checks"].values())
        assert r["kraus_vp"]["ok"] and queue_e3.fixture_checks([r])["rows"][0]["ok"]
    else:
        assert np.isfinite(r["cf"]) and np.isfinite(r["val_nll"]) and len(r["kl"]) == 20 and r["cpu_s"] > 0
        assert 0 <= r["cf_boot_sd"] < 1                                 # §9 (i) test-sequence bootstrap SD
    if recipe in ("M", "C1", "S", "C2", "HMM-C", "OOM-C"):
        assert r["n_params_closed_form_ok"] is True


def test_sens_merges_with_base(e2e):
    _, recs = e2e
    s, b = recs["SENS"], recs["CPLUS-M"]
    assert s["val_nll_base"] == b["val_nll"] and s["cf_base"] == b["cf"]
    assert s["sens_source"] == ("base" if b["val_nll"] <= s["val_nll_extra"] else "extra")
    assert s["cf"] == (b["cf"] if s["sens_source"] == "base" else s["cf_extra"])
    assert s["cpu_s_with_base"] == pytest.approx(s["cpu_s"] + b["cpu_s"])


def test_oom_and_hmm_report_variants(e2e):
    _, recs = e2e
    o = recs["OOM-C"]
    assert set(o["cand_cf"]) >= {"spectral", "spectral_ft"} and o["cf_spec"] == o["cand_cf"]["spectral"]
    assert o["cf"] == pytest.approx(o["cand_cf"][o["winner"]]) and 0 <= o["nr"] <= 1
    assert np.isfinite(recs["HMM-C"]["cf_mom_only"]) and np.isfinite(recs["M"]["cf_mom"])


def test_unnormalised_prediction_fails_the_job(tmp_path, monkeypatch):
    monkeypatch.setattr(run_e3, "load_data", _data)

    def bad(j, tr, va, te, exact):
        return {"pred": 1.1 * exact, "val_nll": 0.0, "n_params": 1, "cpu_s": 0.0}
    monkeypatch.setitem(run_e3.RECIPES, "M", bad)
    r = run_e3.run_job(dict(BASE, **FAST["M"]), tmp_path)
    assert r["failed"] and "not normalised" in r["error"] and r["cf"] is None


# (2) C0 runs the unmodified run_e2.main and leaves runs/e2 untouched
def test_c0_isolated_from_runs_e2(tmp_path):
    before = queue_e3.e2_manifest()
    r = run_e3.run_job(dict(BASE, family="HQ", size=2, recipe="C0", lr=3e-3, steps=20), tmp_path)
    assert queue_e3.e2_manifest() == before and len(before) > 0
    assert not r["failed"], r.get("error")
    raw = tmp_path / "c0" / "e2-A1-HQ2-s200-lr0.003.json"
    assert raw.exists() and json.loads(raw.read_text())["result_sha256"] == r["e2_result_sha256"]
    assert np.isfinite(r["cf"]) and r["cpu_s"] > 0 and r["n_params"] == 16
    for bad in (run_e3.E2_RUNS, run_e3.E2_RUNS / "sub"):
        with pytest.raises(ValueError):
            run_e3.run_job(dict(BASE, family="HQ", size=2, recipe="C0", lr=3e-3, steps=20), bad)
    assert queue_e3.e2_manifest() == before


# (3) poisoning: val / test / exact never reach the MoM fits, the spectral init or r_hat
def _spy(monkeypatch, mod, name, sink):
    fn = getattr(mod, name)

    def wrapped(*a, **k):
        res = fn(*a, **k)
        sink.setdefault(name, []).append(res)
        return res
    monkeypatch.setattr(mod, name, wrapped)


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a if k != "cpu_s")
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def test_poisoning_leaves_selection_free_fits_bit_identical(tmp_path, monkeypatch):
    jobs = [dict(BASE, family="HQ", size=2, recipe="M", restarts=2, mom_iters=5, steps=5),
            dict(BASE, family="HMM", size=3, recipe="HMM-C", restarts=2, mom_iters=5, em_iters=1, train_subset=1000),
            dict(BASE, family="OOM", size=3, recipe="OOM-C", restarts=2, mom_iters=5, steps=5),
            dict(BASE, family="LRU", size=2, recipe="CPLUS", lrs=LRS, steps_probe=2, steps_total=4)]

    def poisoned(process, seed):
        tr, va, te, exact = _data(process, seed)
        O = tasks_e3.process(process).O
        return tr, ((va[0] + 1) % O, va[1]), ((te[0] + 1) % O, te[1]), np.full_like(exact, np.nan)
    sinks = []
    for loader in (_data, poisoned):
        sink = {}
        with monkeypatch.context() as m:
            for mod, name in ((hq_init, "fit_mom"), (hmm_e3, "fit_hmm_mom"), (oom_e3, "fit_oom_mom"),
                              (oom_e3, "spectral_init"), (cplus, "spectral_lambdas")):
                _spy(m, mod, name, sink)
            m.setattr(run_e3, "load_data", loader)
            for j in jobs:
                run_e3.run_job(j, tmp_path / loader.__name__.strip("_"))
        sinks.append(sink)
    clean, dirty = sinks
    assert set(clean) == {"fit_mom", "fit_hmm_mom", "fit_oom_mom", "spectral_init", "spectral_lambdas"}
    for name in clean:
        assert _same(clean[name], dirty[name]), name
    assert clean["spectral_lambdas"][0]["r_hat"] == dirty["spectral_lambdas"][0]["r_hat"]


# (4) bitwise rerun with one thread
def test_m_job_bitwise_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(run_e3, "load_data", _data)
    j = dict(BASE, family="HQ", size=2, recipe="M", restarts=2, mom_iters=8, steps=40)
    a = run_e3.run_job(j, tmp_path / "a")
    b = run_e3.run_job(j, tmp_path / "b")
    assert torch.get_num_threads() == 1 and not a["failed"]
    assert _strip(a) == _strip(b)


# (5) dry-run job table (design §5 / §13)
def test_dry_run_counts(capsys):
    c = queue_e3.counts()
    assert {b: c[b]["jobs"] for b in ("U", "T", "SC", "O", "DEV")} == {"U": 200, "T": 33, "SC": 135, "O": 10, "DEV": 88}
    assert c["confirm_total"] == 378 and c["FIX"]["jobs"] == 12
    assert c["U"]["cpu_h"] == pytest.approx(6.77, abs=0.01) and c["SC"]["cpu_h"] == pytest.approx(7.77, abs=0.01)
    assert queue_e3.main(["--dry-run"]) == 0 and "U    jobs= 200" in capsys.readouterr().out


def test_confirmatory_build_refused_while_constants_unset():
    assert not queue_e3.missing_constants()                # the dev gate has filled them (frozen 2026-10-24)
    unset = dict(queue_e3.FROZEN_CONSTANTS, R1=None, R_C=dict(queue_e3.FROZEN_CONSTANTS["R_C"], GRU2=None))
    assert queue_e3.missing_constants(unset) == ["R1", "R_C.GRU2"]
    for fn in (queue_e3.jobs_U, queue_e3.jobs_T, queue_e3.jobs_SC, queue_e3.jobs_O):
        with pytest.raises(ValueError, match="frozen constants not set"):
            fn(unset)
    trimmed = dict(queue_e3.PLACEHOLDER, TRIMS=["O", "D4", "SENS", "C2_0.2", "A3", "D3", "TOP32"])
    assert len(queue_e3.jobs_U(trimmed)) == 190 and len(queue_e3.jobs_T(trimmed)) == 20
    assert len(queue_e3.jobs_SC(trimmed)) == 100


# (6) seed namespaces
def test_seed_namespaces_disjoint():
    C = queue_e3.PLACEHOLDER
    conf = queue_e3.jobs_U(C) + queue_e3.jobs_T(C) + queue_e3.jobs_SC(C) + queue_e3.jobs_SENS(C, placeholder=True) \
        + queue_e3.jobs_O(C)
    dev = queue_e3.jobs_dev1() + queue_e3.jobs_dev2(C["CPLUS_LRS"]) + queue_e3.jobs_dev3(C["CPLUS_LRS"]) \
        + queue_e3.jobs_fixture() + queue_e3.jobs_redev()
    assert {j["seed"] for j in conf} == set(range(20, 30)) and {j["seed"] for j in dev} == set(range(200, 210))
    f = cplus.init_seed
    for s in range(20, 30):
        assert stiefel.init_seed(s, 7) == f(s, 2, 7) == oom_e3.init_seed(s, 2, 7) == hmm_e3.init_seed(s, 2, 7)
    init_conf = {f(j["seed"], c, r) for j in conf for c in range(10) for r in (0, 999)}
    init_dev = {f(j["seed"], c, r) for j in dev for c in range(10) for r in (0, 999)}
    assert min(init_conf) >= 10_200_000 and max(init_conf) <= 10_299_999
    assert min(init_dev) >= 12_000_000 and max(init_dev) <= 12_099_999
    e2_run, v11_run = set(range(5)), set(range(10)) | set(range(100, 110))   # run (= data) seeds; exploratory >= 300
    e2_hmm = {s * 101 + r for s in range(5) for r in range(3)}               # run_e2's numpy HMM init seeds
    explore_ns_min = f(300, 0, 0)                                            # namespaced exploratory seeds
    for x in init_conf | init_dev:
        assert x not in e2_run | v11_run | e2_hmm and 10 ** 7 <= x < explore_ns_min   # raw exploratory seeds < 10**7
    run_seeds = {j["seed"] for j in conf + dev}                              # data seeds; C0 / M batch generators
    assert not run_seeds & (e2_run | v11_run) and max(run_seeds) < 300


# queue bookkeeping
def test_status_resume_and_gating(tmp_path):
    j = dict(BASE, **FAST["M"])
    sh = ("d" * 64, "c" * 64)
    assert queue_e3.status(j, tmp_path, sh) == "todo"
    run_e3.write_record(dict(j, job=dict(j), attempt=1, failed=True, design_sha256=sh[0], code_manifest_sha256=sh[1]), tmp_path)
    assert queue_e3.status(j, tmp_path, sh) == "retry"
    run_e3.write_record(dict(j, job=dict(j), attempt=2, failed=True, design_sha256=sh[0], code_manifest_sha256=sh[1]), tmp_path)
    assert queue_e3.status(j, tmp_path, sh) == "done"
    assert queue_e3.status(j, tmp_path, ("e" * 64, sh[1])) == "stale"
    s = {"enqueue": {"T": True, "SC": True, "O": False}}
    assert queue_e3.gated_blocks(s) == ["T", "SC"] and queue_e3.gated_blocks(s, ["SC"]) == ["T"]


def test_lower_bracket_rung():
    def recs(f, meds):
        return [dict(process="A2", family=f, size=s, recipe=r, seed=sd, sigma=None, cf=m, val_nll=1.0, failed=False)
                for (s, r), m in zip(queue_e3.LADDER[f], meds) for sd in range(20, 30)]
    assert queue_e3.lower_bracket(recs("GRU", [0.2, 0.5, 0.95, 0.9, 0.97]), "GRU") == (4, "CPLUS-M")
    assert queue_e3.lower_bracket(recs("LRU", [0.2, 0.5, 0.7, 0.8, 0.85]), "LRU") == (32, "CPLUS")
    assert queue_e3.lower_bracket(recs("LRU", [0.95] * 5), "LRU") == (2, "CPLUS-M")
    assert queue_e3.lower_bracket(recs("GRU", [0.2] * 4)[:-10], "GRU") is None

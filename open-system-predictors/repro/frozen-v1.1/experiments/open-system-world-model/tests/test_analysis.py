"""The confirmatory analysis must find a planted effect, must NOT find one when none is planted, must refuse to
pass on incomplete seed sets, and must choose H4's classical baseline on development data only (DEVIATIONS.md D8)."""
import json

import numpy as np
import pytest

from osw import analyze

CONFIGS = [("simple", 20), ("metastable", 20), ("metastable", 80), ("action", 20)]


def _write(d, base, override=None, seed=0, drop=()):
    rng = np.random.default_rng(seed)
    for task, tau in CONFIGS:
        for m, v in base.items():
            v = (override or {}).get((task, tau, m), v)
            for s in range(10):
                if (task, tau, m, s) in drop:
                    continue
                r = {"task": task, "tau": float(tau), "model": m, "seed": s, "n_params": 1000,
                     "windows": {"extrap_51_200": {"kl": v + rng.normal(0, 0.005)}, "in_range_1_50": {"kl": 0.01}},
                     "revival": {"revival_index": (0.5 if m.startswith("U") and v > 0.05 else 0.01) + rng.normal(0, 0.01)}}
                (d / f"{task}-{tau}-{m}-s{s}.json").write_text(json.dumps(r))


def _dev(d, nll=None):
    d.mkdir(exist_ok=True)
    nll = nll or {"HMM": 1.33, "GRU": 1.32, "LRU": 1.325}
    for task, tau in CONFIGS:
        for m, v in nll.items():
            (d / f"{task}-{tau}-{m}-s102-lr0.003.json").write_text(json.dumps(
                {"task": task, "tau": float(tau), "model": m, "seed": 102, "val_nll": v}))


BASE = {"U1": 0.05, "U2": 0.05, "U4": 0.05, "U8": 0.05, "L": 0.05, "L1": 0.05, "K": 0.05, "C": 0.05,
        "GRU": 0.05, "LRU": 0.05, "HMM": 0.05, "BAYES": 0.0}
PLANTED = dict(BASE, U1=0.20, U2=0.10, U4=0.07, U8=0.06)


def test_planted_effect_is_found(tmp_path):
    conf, dev = tmp_path / "c", tmp_path / "d"
    conf.mkdir(); _dev(dev)
    _write(conf, PLANTED, {("metastable", 80, "C"): 0.08})
    res = analyze.analyze(conf, dev)
    assert res["H1_simple"]["pass"] and res["H1_metastable"]["pass"]
    assert res["H2"]["pass_"] and res["H3"]["pass_"]
    assert all(h["parity_or_better"] for h in res["H4"].values())


def test_planted_null_is_not_found(tmp_path):
    conf, dev = tmp_path / "c", tmp_path / "d"
    conf.mkdir(); _dev(dev)
    _write(conf, BASE, seed=1)
    res = analyze.analyze(conf, dev)
    assert not res["H1_simple"]["pass"] and not res["H1_metastable"]["pass"]
    assert not res["H2"]["pass_"]
    assert not res["H3"]["pass_"]


def test_incomplete_seed_set_cannot_pass(tmp_path):
    conf, dev = tmp_path / "c", tmp_path / "d"
    conf.mkdir(); _dev(dev)
    _write(conf, PLANTED, {("metastable", 80, "C"): 0.08}, drop={("simple", 20, "U2", 9)})
    res = analyze.analyze(conf, dev)
    assert not res["H1_simple"]["complete"] and not res["H1_simple"]["pass"]
    assert res["H1_metastable"]["pass"]


def test_h3_ignores_the_frozen_tau20_metastable_contrast(tmp_path):
    """D6/D9: H3 is decided on metastable tau_slow=80 against L1; an effect planted only at tau=20 must not count."""
    conf, dev = tmp_path / "c", tmp_path / "d"
    conf.mkdir(); _dev(dev)
    _write(conf, PLANTED, {("metastable", 20, "C"): 0.08})
    assert not analyze.analyze(conf, dev)["H3"]["pass_"]


def test_h4_baseline_is_chosen_on_development_data(tmp_path):
    conf, dev = tmp_path / "c", tmp_path / "d"
    conf.mkdir(); _dev(dev, {"HMM": 1.40, "GRU": 1.30, "LRU": 1.35})
    _write(conf, PLANTED, {(t, tau, "HMM"): 0.01 for t, tau in CONFIGS})   # HMM best on confirm data, worst on dev
    res = analyze.analyze(conf, dev)
    assert {h["best_classical"] for h in res["H4"].values()} == {"GRU"}


def test_h4_refuses_without_development_data(tmp_path):
    conf = tmp_path / "c"
    conf.mkdir()
    _write(conf, PLANTED)
    with pytest.raises(SystemExit):
        analyze.analyze(conf, tmp_path / "missing")

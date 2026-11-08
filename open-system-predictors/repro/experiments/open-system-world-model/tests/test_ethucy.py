"""E1 data pipeline (E1-DESIGN.zh.md): symbol rules, rotation invariance, and no pedestrian leakage."""
import numpy as np
import pytest

from osw import tasks_ethucy as E

pytestmark = pytest.mark.skipif(not (E.RAW / "eth_obsmat.txt").exists(), reason="ETH/UCY raw data not present")


def _walk(headings_deg, speed=0.5):
    """Positions for a sequence of step headings (degrees), one step per heading."""
    steps = [speed * np.array([np.cos(np.radians(h)), np.sin(np.radians(h))]) for h in headings_deg]
    return np.vstack([np.zeros(2), np.cumsum(steps, axis=0)])


def test_sector_rules():
    s = E.symbolize(_walk([0] * 8 + [0, 90, -90, 180, 44, 46, 0, 0, 0, 0, 0, 0]))
    assert list(s[:8]) == [0] * 8
    assert list(s[8:14]) == [0, 2, 6, 4, 1, 1]          # straight, left, right, back, +44 -> sector 1, +46 -> sector 1


def test_stationary_and_reference_heading():
    pos = _walk([30] * 7 + [0] + [30] * 12)
    pos[-3:] = pos[-4]                                    # last three displacements stationary
    s = E.symbolize(pos)
    assert s[7] == 0 and s[8] == 1                        # reference = last moving context step (heading 0)
    assert list(s[-3:]) == [E.STATIONARY] * 3
    assert E.symbolize(np.zeros((E.WIN, 2))) is None      # never moves in the context -> dropped


def test_rotation_invariance():
    rng = np.random.default_rng(0)
    pos = np.cumsum(rng.normal(0, 0.3, size=(E.WIN, 2)), axis=0)
    th = 1.234
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert np.array_equal(E.symbolize(pos), E.symbolize(pos @ R.T + np.array([5.0, -3.0])))


def test_window_counts_and_alphabet():
    o, p, dropped = E.windows("eth")
    assert o.shape[1] == E.CTX + E.HORIZON and len(o) == len(p)
    assert len(o) + dropped == 2343                       # feasibility count in E1-DESIGN.zh.md §1
    assert o.min() >= 0 and o.max() < E.N_SYM


def test_loso_has_no_pedestrian_leakage():
    (tr, _), (va, _), (te, _), keys = E.loso("zara01", seed=0)
    assert len(tr) == len(keys["train"]) > 0 and len(va) == len(keys["val"]) > 0 and len(te) == len(keys["test"]) > 0
    train, val, test = set(keys["train"]), set(keys["val"]), set(keys["test"])
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert {sc for sc, _ in test} == {"zara01"} and "zara01" not in {sc for sc, _ in train | val}


def test_importing_the_e1_runner_leaves_t1_context_length_alone():
    from osw import hmm, models, run_ethucy  # noqa: F401
    from osw.tasks import L_CTX
    assert models.L_CTX == hmm.L_CTX == L_CTX == 32


def test_e1_runner_end_to_end(tmp_path):
    import json
    import os
    import subprocess
    import sys
    env = dict(os.environ, PYTHONPATH=str(E.RAW.parents[2] / "experiments/open-system-world-model/src"))
    r = subprocess.run([sys.executable, "-m", "osw.run_ethucy", "--held-out", "hotel", "--model", "MARKOV1", "--seed", "0",
                        "--out", str(tmp_path)], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-800:]
    res = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert res["e1_design_sha256"].startswith("f06879ca") and res["n_test"] > 0
    assert set(res["windows"]) == {"in_range_1_6", "extrap_7_12"}
    assert len(res["per_k"]["nll"]) == 12 and len(res["D_curve_7_40"]) == 34

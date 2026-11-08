"""E2 (E2-DESIGN.zh.md): generators, exact predictive, the HQMM model, the weather split, scores and the decision rule."""
import numpy as np
import pytest
import torch

from osw import models_e2, queue_e2, tasks_aiwq, tasks_e2, tasks_series


@pytest.mark.parametrize("proc", [tasks_e2.probability_clock(), tasks_e2.qutrit_clock()])
def test_generator_is_valid_and_exact_predictive_is_calibrated(proc):
    assert np.allclose(sum(k.conj().T @ k for k in proc.K), np.eye(proc.d))
    s = proc.stationary()
    assert np.allclose(proc.channel(s), s) and np.isclose(np.trace(s).real, 1)
    obs, _ = tasks_e2.sample(proc, 20_000, 2, np.random.default_rng(3))
    q = tasks_e2.exact_predictive(proc, obs, 2)
    assert np.allclose(q.sum(-1), 1)
    p0, y0 = q[:, 0, 0], obs[:, tasks_e2.L_CTX] == 0
    edges = np.quantile(p0, [0.25, 0.5, 0.75])
    idx = np.digitize(p0, edges)
    for b in np.unique(idx):
        m, e, n = p0[idx == b].mean(), y0[idx == b].mean(), (idx == b).sum()
        assert abs(e - m) / np.sqrt(m * (1 - m) / n) < 4


def test_relaxation_times_match_the_design():
    assert abs(tasks_e2.probability_clock().relaxation_time() - 2.71) < 0.01
    assert abs(tasks_e2.qutrit_clock().relaxation_time() - 20.0) < 0.05


@pytest.mark.parametrize("proc", [tasks_e2.probability_clock(), tasks_e2.qutrit_clock()])
def test_hq_model_with_the_true_kraus_operators_reproduces_the_exact_predictive(proc):
    m = models_e2.build("HQ", proc.d, proc.O, tasks_e2.L_CTX)
    G = np.concatenate(proc.K, 0)                                      # already an isometry, so V = G
    with torch.no_grad():
        m.g_re.copy_(torch.tensor(G.real)); m.g_im.copy_(torch.tensor(G.imag))
    obs, act = tasks_e2.sample(proc, 200, 12, np.random.default_rng(5))
    pred = m(torch.as_tensor(obs), torch.as_tensor(act), 12).detach().numpy()
    exact = tasks_e2.exact_predictive(proc, obs, 12)
    # the model starts from the maximally mixed state, the exact filter from the stationary state; 32 context
    # steps make the difference negligible for these mixing processes
    assert np.abs(pred - exact).max() < 2e-3


aiwq = pytest.mark.skipif(not tasks_aiwq.DATA.exists(), reason="weather-quest categories not prepared")


@aiwq
def test_weather_split_is_strictly_separated():
    tr, va, te = tasks_aiwq.split(0, n_train=3000, n_val=500, n_test=500)
    assert tr[2]["start"].max() + tasks_aiwq.WIN - 1 <= tasks_aiwq.TRAIN_LAST
    assert va[2]["start"].max() + tasks_aiwq.WIN - 1 <= tasks_aiwq.TRAIN_LAST
    assert te[2]["start"].min() >= tasks_aiwq.TEST_FIRST
    assert set(tr[2]["point"]).isdisjoint(set(va[2]["point"]))
    assert tr[0].min() >= 0 and tr[0].max() < tasks_aiwq.N_SYM and tr[0].shape[1] == tasks_aiwq.WIN


def test_rps_known_values():
    y = np.zeros((1, 1), dtype=int)
    assert np.isclose(tasks_aiwq.rps(np.eye(5)[None, None, 0], y)[0, 0], 0)
    assert np.isclose(tasks_aiwq.rps(np.full((1, 1, 5), 0.2), y)[0, 0], 0.64 + 0.36 + 0.16 + 0.04)


def _fake(arm, fam, size, seed, params, m):
    kl_marg = [0.1] * 20
    return {"arm": arm, "family": fam, "size": size, "seed": seed, "n_params": params, "val_nll": 1.0,
            "per_k": {"kl": [0.1 * (1 - m)] * 20, "kl_marginal": kl_marg}}


def _ladder(cf):
    rng = np.random.default_rng(0)
    R = []
    for fam, sizes in {"HQ": [(2, 16), (4, 64), (8, 256)], "GRU": [(2, 54), (8, 354)],
                       "LRU": [(2, 50), (8, 234), (16, 722)], "HMM": [(2, 5), (16, 271), (32, 1055)]}.items():
        for size, params in sizes:
            for s in range(5):
                R.append(_fake("A1", fam, size, s, params, cf[fam] + rng.normal(0, 0.01)))
    return R


def test_decision_rule_finds_a_planted_advantage_and_not_a_planted_null():
    cell = queue_e2.collect(_ladder({"HQ": 0.9, "GRU": 0.6, "LRU": 0.6, "HMM": 0.5}))
    r = queue_e2.compare(cell["A1"], 64, "HQ", ["GRU", "LRU"], np.random.default_rng(1), reps=500)
    assert r["advantage"] and r["delta"] > 0.25
    cell = queue_e2.collect(_ladder({"HQ": 0.7, "GRU": 0.7, "LRU": 0.68, "HMM": 0.5}))
    r = queue_e2.compare(cell["A1"], 64, "HQ", ["GRU", "LRU"], np.random.default_rng(1), reps=500)
    assert not r["advantage"]


def test_importing_the_e2_runner_leaves_t1_context_length_alone():
    from osw import hmm, models, run_e2  # noqa: F401
    assert models.L_CTX == hmm.L_CTX == 32


series = pytest.mark.skipif(not (tasks_series.ROOT / "exp_e3/nyc_taxi.csv").exists(), reason="series data not present")


@series
@pytest.mark.parametrize("arm", ["B2", "B3"])
def test_series_split_is_time_ordered_and_thresholds_use_training_only(arm):
    v = tasks_series.load(arm)
    n = len(v); a, b = int(0.6 * n), int(0.7 * n)
    tr, va, te, meta = tasks_series.split(arm)
    assert len(tr[0]) == a - tasks_series.WIN + 1 and len(va[0]) == b - a - tasks_series.WIN + 1
    assert len(te[0]) == n - b - tasks_series.WIN + 1
    c, q = tasks_series.symbolise(v, a)
    v2 = v.copy(); v2[a:] = v2[a:] * 10 + 1e6                   # change the future: thresholds must not move
    assert np.allclose(tasks_series.symbolise(v2, a)[1], q)
    assert np.array_equal(te[0][0], c[b:b + tasks_series.WIN]) and np.array_equal(tr[0][-1], c[a - tasks_series.WIN:a])

"""E3 processes, data, metric and REF predictors (src/osw/tasks_e3.py). Dev-style seeds only (200-209, 290)."""
import numpy as np
import pytest

from osw import run_e2, tasks_e2, tasks_e3 as t

L, K = tasks_e2.L_CTX, 20


@pytest.fixture(scope="module")
def a2():
    return t.data("A2", 201)


def _direct_cf(pred, exact):
    p, q = np.clip(pred, 1e-9, 1), np.clip(exact, 1e-12, 1)
    kl = (q * (np.log(q) - np.log(p))).sum(-1).mean(0)
    marg = exact[:, -1].mean(0)
    klm = (q * np.log(q / marg)).sum(-1).mean(0)
    return 1 - kl[:10].mean() / klm[:10].mean()


def test_qudit_clock_d3_is_qutrit_clock_bit_for_bit():
    a, b = t.qudit_clock(3, tasks_e2.QUTRIT_EPS), tasks_e2.qutrit_clock()
    assert len(a.K) == len(b.K) == 3
    assert all(np.array_equal(x, y) for x, y in zip(a.K, b.K))


def test_a3_eps_bisection_and_relaxation():
    assert abs(t.eps_for_relaxation(4) - t.A3_EPS) < 1e-9
    p = t.process("A3")
    assert abs(p.relaxation_time() - 20.0) < 1e-6
    assert np.allclose(sum(k.conj().T @ k for k in p.K), np.eye(4), atol=1e-12)


def test_process_table():
    assert sorted(c["code"] for c in t.PROC.values()) == [1, 2, 3, 4]
    dims = {n: (t.process(n).d, t.process(n).O) for n in t.PROC}
    assert dims == {"A1": (2, 2), "A2": (3, 3), "A2w": (3, 3), "A3": (4, 4)}
    assert abs(t.process("A2").relaxation_time() - 20.0) < 0.05
    assert abs(t.process("A2w").relaxation_time() - 93.8) < 0.05
    assert (t.PROC["A2w"]["n_train"], t.PROC["A2w"]["n_test"], t.PROC["A3"]["n_train"]) == (120_000, 10_000, 40_000)


@pytest.mark.parametrize("arm", ["A1", "A2"])
def test_data_matches_e2_runner(arm):
    tr, va, te, ex = t.data(arm, 200)
    tr2, va2, te2, ex2, _ = run_e2.data(arm, 200)
    for x, y in zip((tr, va, te), (tr2, va2, te2)):
        assert np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
    assert np.array_equal(ex, ex2)
    assert tr[0].shape == (20_000, L + K) and ex.shape == (2_000, K, t.process(arm).O)


def test_cf_fixture_identities(a2):
    _, _, te, ex = a2
    y = te[0][:, L:L + K]
    assert abs(t.cf_and_kl(ex, ex, y)["cf"] - 1) < 1e-6
    marg = np.broadcast_to(ex[:, -1].mean(0), ex.shape).copy()
    r = t.cf_and_kl(marg, ex, y)
    assert abs(r["cf"]) < 1e-6 and len(r["kl"]) == len(r["kl_marginal"]) == K
    big = 1.1 * ex
    assert t.cf_and_kl(big, ex, y, check=False)["cf"] > 1
    with pytest.raises(ValueError):
        t.assert_normalised(big)
    with pytest.raises(ValueError):
        t.cf_and_kl(big, ex, y)
    nan = ex.copy(); nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        t.assert_normalised(nan)
    t.assert_normalised(ex)
    for lam in (0.5, 0.9):
        mix = lam * ex + (1 - lam) * marg
        cf = t.cf_and_kl(mix, ex, y)["cf"]
        assert abs(cf - _direct_cf(mix, ex)) < 1e-10
        assert 0 < cf < 1


def test_denominators_and_stationary_marginal():
    d2 = t.denominators("A2")
    assert abs(d2 - 0.0180) < 0.0008
    assert abs(t.denominators("A1") - 0.019) < 0.0015
    for name in ("A1", "A2"):
        proc = t.process(name)
        obs, _ = tasks_e2.sample(proc, 4000, K, np.random.default_rng([290, 7]))
        e2_marg = tasks_e2.exact_predictive(proc, obs, K)[:, -1].mean(0)
        assert np.abs(e2_marg / t.stationary_marginal(proc) - 1).max() < 0.02


def _cyclic(n, pattern, rng):
    pattern = np.asarray(pattern)
    ph = rng.integers(0, len(pattern), n)
    return pattern[(ph[:, None] + np.arange(L + K)) % len(pattern)]


def test_markov_k_counts_and_rollout():
    rng = np.random.default_rng(205)
    tr, te = rng.integers(0, 3, (300, L + K)), rng.integers(0, 3, (50, L + K))
    C = np.full((3, 3), 0.5)
    for row in tr:
        for a, b in zip(row[:-1], row[1:]):
            C[a, b] += 1
    P = C / C.sum(1, keepdims=True)
    pred = t.markov_k(tr, te, 1, K, 3)
    for k in (0, 1, 6):
        assert np.allclose(pred[:, k], np.eye(3)[te[:, L - 1]] @ np.linalg.matrix_power(P, k + 1), atol=1e-12)
    t.assert_normalised(pred)
    tr4, te4 = _cyclic(2000, [0, 0, 1, 1], rng), _cyclic(100, [0, 0, 1, 1], rng)
    truth = np.eye(2)[te4[:, L:L + K]]
    p2, p1 = t.markov_k(tr4, te4, 2, K, 2), t.markov_k(tr4, te4, 1, K, 2)
    assert (p2 * truth).sum(-1).min() > 0.99
    assert (p1[:, 0] * truth[:, 0]).sum(-1).max() < 0.6
    assert t.markov_k(tr4, te4, 3, K, 2).shape == (100, K, 2)


def test_linear_context_on_cyclic_and_a2(a2):
    rng = np.random.default_rng(206)
    tr4, te4 = _cyclic(500, [0, 0, 1, 1], rng), _cyclic(100, [0, 0, 1, 1], rng)
    p = t.linear_context(tr4, te4, K, 2, steps=100)
    t.assert_normalised(p)
    assert (p * np.eye(2)[te4[:, L:L + K]]).sum(-1).min() > 0.99
    tr, _, te, ex = a2
    sub = slice(0, 5000)
    p = t.linear_context(tr[0][sub], te[0], K, 3)
    cf_hard = t.cf_and_kl(p, ex, te[0][:, L:L + K])["cf"]
    assert np.array_equal(p, t.linear_context(tr[0][sub], te[0], K, 3))
    proc = t.process("A2")
    soft = tasks_e2.exact_predictive(proc, tr[0][sub], K)
    cf_soft = t.cf_and_kl(t.linear_context(tr[0][sub], te[0], K, 3, soft=soft), ex, te[0][:, L:L + K])["cf"]
    assert cf_hard < cf_soft < 0.85 and cf_soft > 0.6


def test_markov_cf_small_on_a2(a2):
    tr, _, te, ex = a2
    cfs = [t.cf_and_kl(t.markov_k(tr[0], te[0], m, K, 3), ex, te[0][:, L:L + K])["cf"] for m in (1, 2, 3)]
    assert max(cfs) < 0.75 and min(cfs) > -0.1


def test_shuffled_exact_is_negative_control(a2):
    _, _, te, ex = a2
    proc = t.process("A2")
    p = t.shuffled_exact(proc, te[0], K, np.random.default_rng(207))
    perm = np.random.default_rng(207).permutation(len(te[0]))
    assert np.allclose(p, tasks_e2.exact_predictive(proc, te[0][perm], K), atol=1e-12)
    t.assert_normalised(p)
    assert t.cf_and_kl(p, ex, te[0][:, L:L + K])["cf"] <= 0

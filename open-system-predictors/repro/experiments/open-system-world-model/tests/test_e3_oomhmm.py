"""E3 baselines: OOM-C (src/osw/oom_e3.py) and HMM-C (src/osw/hmm_e3.py)."""
import numpy as np
import torch

from osw import hmm, hmm_e3, hq_init, models_e2, oom_e3, tasks_e2
from osw.run import metrics

L = tasks_e2.L_CTX
T2 = np.array([[0.9, 0.1], [0.2, 0.8]])
B2 = np.array([[0.85, 0.15], [0.25, 0.75]])


def _exact_word_prob(proc):
    Ks, rho = np.stack(proc.K), proc.stationary()

    def prob(word):
        r = rho
        for x in word:
            r = Ks[x] @ r @ Ks[x].conj().T
        return float(np.trace(r).real)
    return prob


def _cf(pred, exact, obs):
    q = np.clip(exact, 1e-12, 1)
    km = (q * np.log(q / exact[:, -1].mean(0))).sum(-1).mean(0)
    return 1 - metrics(pred, exact, obs[:, L:L + 20])["kl"][:10].mean() / km[:10].mean()


def _sample_hmm(T, B, n, length, rng):
    pi = np.linalg.matrix_power(T, 500)[0]
    s = (rng.random(n)[:, None] > np.cumsum(pi)).sum(1)
    obs = np.zeros((n, length), dtype=np.int64)
    for t in range(length):
        obs[:, t] = (rng.random(n)[:, None] > np.cumsum(B[s], 1)).sum(1).clip(0, B.shape[1] - 1)
        s = (rng.random(n)[:, None] > np.cumsum(T[s], 1)).sum(1).clip(0, T.shape[0] - 1)
    return obs


# ---------------------------------------------------------------- OOM
def test_oom_parameter_count_and_exact_normalisation():
    for O, r, n in ((2, 3, 17), (3, 9, 242)):
        m = oom_e3.OOMWM(O, r, L)
        assert sum(p.numel() for p in m.parameters()) == n == oom_e3.n_params(O, r)
        torch.manual_seed(200)
        with torch.no_grad():
            for p in m.parameters():
                p.normal_(0, 0.3)
        A = m.ops()
        assert torch.allclose(A.sum(0)[0], torch.eye(r, dtype=torch.float64)[0], atol=1e-14)
        assert float(m.w0().detach()[0]) == 1.0
        obs = torch.randint(0, O, (16, L + 20), generator=torch.Generator().manual_seed(201))
        raw = m(obs, None, 20, raw=True)
        assert torch.allclose(raw.sum(-1), torch.ones(16, 20, dtype=torch.float64), atol=1e-8)
        p = m(obs, None, 20)
        assert (p > 0).all() and torch.allclose(p.sum(-1), torch.ones(16, 20, dtype=torch.float64))


def test_spectral_on_exact_hankel_reproduces_A1_predictive():
    proc = tasks_e2.probability_clock()
    w0, A, sv = oom_e3.spectral(_exact_word_prob(proc), proc.O, 3)
    assert sv[2] > 1e-3 and sv[3] < 1e-10                              # Hankel rank 3
    m = oom_e3.OOMWM(proc.O, 3, L)
    m.load(w0, A)
    obs, _ = tasks_e2.sample(proc, 200, 20, np.random.default_rng(202))
    pred, raw = oom_e3.predict(m, obs, 20)
    exact = tasks_e2.exact_predictive(proc, obs, 20)
    assert np.abs(raw - exact).max() < 1e-6
    assert np.abs(pred - exact).max() < 1e-5


def test_oom_tables_equal_hq_tables_at_exact_spectral_model():
    proc = tasks_e2.probability_clock()
    _, A, _ = oom_e3.spectral(_exact_word_prob(proc), proc.O, 3)
    P, T3 = oom_e3.oom_tables(torch.tensor(A))
    P2, T32 = hq_init.model_tables(torch.tensor(np.stack(proc.K), dtype=torch.complex128))
    assert (P - P2).abs().max() < 1e-10 and (T3 - T32).abs().max() < 1e-10


def test_spectral_init_from_training_samples_and_mom_fit():
    proc = tasks_e2.probability_clock()
    rng = np.random.default_rng(10_000 * 203 + 1)
    tr, te = tasks_e2.sample(proc, 20_000, 20, rng)[0], tasks_e2.sample(proc, 1_000, 20, rng)[0]
    exact = tasks_e2.exact_predictive(proc, te, 20)
    m = oom_e3.OOMWM(proc.O, 3, L)
    m.load(*oom_e3.spectral_init(tr, 3, proc.O)[:2])
    assert _cf(oom_e3.predict(m, te)[0], exact, te) > 0.95
    tabs = hq_init.empirical_tables(tr, proc.O)
    fit = oom_e3.fit_oom_mom(tabs, proc.O, 3, restarts=3, iters=100, init_seed0=oom_e3.init_seed(203, 6, 0))
    floor = float(hq_init.mom_loss(torch.tensor(np.stack(proc.K), dtype=torch.complex128), tabs))
    assert fit["mom_loss"] < floor + 1e-3 and fit["n_failed"] == 0
    assert fit["losses_sorted"] == sorted(fit["losses_sorted"])
    m.load(fit["w0"], fit["A"])
    assert _cf(oom_e3.predict(m, te)[0], exact, te) > 0.9


def test_fit_OOM_end_to_end_small():
    proc = tasks_e2.probability_clock()
    rng = np.random.default_rng(10_000 * 204 + 1)
    tr, va, te = (tasks_e2.sample(proc, n, 20, rng) for n in (3_000, 400, 300))
    res = oom_e3.fit_OOM(tr, va, te, proc.O, 3, seed=204, restarts=2, steps=60, mom_iters=30)
    assert set(res["cand_val_nll"]) == {"spectral", "spectral_ft", "mom_ft"}
    assert res["val_nll"] == min(res["cand_val_nll"].values()) and res["winner"] in res["cand_val_nll"]
    assert res["pred"].shape == (300, 20, 2) and np.allclose(res["pred"].sum(-1), 1, atol=1e-5)
    assert res["pred_raw"].shape == (300, 20, 2) and res["nr"] == float((res["pred_raw"] < 0).mean())
    assert res["n_params"] == 17 and res["cpu_s"] > 0 and len(res["mom_losses_sorted"]) == 2
    res2 = oom_e3.fit_OOM(tr, va, te, proc.O, 3, seed=204, restarts=2, steps=60, mom_iters=30)
    assert np.array_equal(res["pred"], res2["pred"]) and res["cand_val_nll"] == res2["cand_val_nll"]


def test_finetune_restores_last_finite_checkpoint_on_nan():
    proc = tasks_e2.probability_clock()
    obs, _ = tasks_e2.sample(proc, 500, 20, np.random.default_rng(205))
    m = oom_e3.OOMWM(proc.O, 3, L)
    m.load(*oom_e3.spectral(_exact_word_prob(proc), proc.O, 3)[:2])
    calls = {"n": 0}
    orig = m.forward

    def poisoned(*a, **k):
        calls["n"] += 1
        p = orig(*a, **k)
        return p * float("nan") if calls["n"] > 150 else p
    m.forward = poisoned
    out = oom_e3.finetune(m, obs, steps=300, seed=205)
    assert out["nan_flag"] and all(torch.isfinite(p).all() for p in m.parameters())
    assert out["train_curve"][-1][0] == 100


def test_seed_namespace():
    assert oom_e3.init_seed(20, 6, 0) == 10_206_000 and hmm_e3.init_seed(29, 7, 47) == 10_297_047
    assert hmm_e3.init_seed(204, 5, 2) == 12_045_002
    conf = {hmm_e3.init_seed(s, c, r) for s in range(20, 30) for c in (5, 6, 7) for r in range(48)}
    assert min(conf) >= 10_200_000 and max(conf) <= 10_299_999


# ---------------------------------------------------------------- HMM
def test_hmm_exact_tables_match_empirical_tables():
    obs = _sample_hmm(T2, B2, 20_000, 52, np.random.default_rng(206))
    tabs = hq_init.empirical_tables(obs, 2)
    P, T3 = hmm_e3.hmm_tables(torch.tensor(T2), torch.tensor(B2))
    assert torch.allclose(P.sum((1, 2)), torch.ones(10, dtype=torch.float64), atol=1e-12)
    assert torch.allclose(T3.sum((1, 2, 3)), torch.ones(9, dtype=torch.float64), atol=1e-12)
    assert (P - tabs["pairs"]).abs().max() < 0.01 and (T3 - tabs["triples"]).abs().max() < 0.01
    assert float(hmm_e3.table_kl(P, T3, tabs)) < 2e-3


def test_hmm_mom_recovers_two_state_tables():
    obs = _sample_hmm(T2, B2, 20_000, 52, np.random.default_rng(207))
    tabs = hq_init.empirical_tables(obs, 2)
    fit = hmm_e3.fit_hmm_mom(tabs, 2, 2, restarts=4, iters=100, init_seed0=hmm_e3.init_seed(207, 7, 0))
    true = dict(zip(("pairs", "triples"), hmm_e3.hmm_tables(torch.tensor(T2), torch.tensor(B2))))
    got = hmm_e3.hmm_tables(torch.tensor(fit["T"]), torch.tensor(fit["B"]))
    assert float(hmm_e3.table_kl(*got, true)) < 1e-3
    assert np.allclose(fit["pi"] @ fit["T"], fit["pi"], atol=1e-10)


def test_fit_HMM_uses_all_training_rows_and_mom_start(monkeypatch):
    seen = []
    orig = hmm.HMM.fit

    def spy(self, obs, act, iters=60, tol=1e-5):
        seen.append((obs.shape[0], self.P[0].copy(), iters, tol))
        return orig(self, obs, act, iters, tol)
    monkeypatch.setattr(hmm.HMM, "fit", spy)
    proc = tasks_e2.probability_clock()
    rng = np.random.default_rng(10_000 * 208 + 1)
    tr, va, te = (tasks_e2.sample(proc, n, 20, rng) for n in (1_500, 300, 200))
    res = hmm_e3.fit_HMM(tr, va, te, 2, 3, seed=208, restarts=2, em_iters=5, mom_iters=30)
    assert [s[0] for s in seen] == [1_500] * 4 and res["n_fit"] == [1_500] * 4
    assert all(s[2] == 5 and s[3] == 1e-6 for s in seen)
    mom = hmm_e3.fit_hmm_mom(hq_init.empirical_tables(tr[0], 2), 3, 2, 2, 30, hmm_e3.init_seed(208, 7, 0))
    assert np.allclose(seen[0][1], mom["T"])
    assert res["fit_names"] == ["mom", "em0", "em1", "em2"]
    assert res["winner"] == res["fit_names"][int(np.argmax(res["train_ll"]))]
    assert res["pred"].shape == (200, 20, 2) and np.allclose(res["pred"].sum(-1), 1, atol=1e-5)
    assert res["n_params"] == models_e2.hmm_params(3, 2) == 11 and len(res["fit_val_nll"]) == 4

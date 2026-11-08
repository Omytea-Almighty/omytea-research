"""E3 building blocks: method-of-moments warm start for HQ (src/osw/hq_init.py)."""
import numpy as np
import torch

from osw import hq_init, models, models_e2, tasks_e2
from osw.run import predict_torch

L = tasks_e2.L_CTX


def _true_K(proc):
    return torch.tensor(np.stack(proc.K), dtype=torch.complex128)


def test_model_tables_match_empirical_tables_at_truth():
    proc = tasks_e2.qutrit_clock()
    obs, _ = tasks_e2.sample(proc, 20_000, 20, np.random.default_rng(12345))
    tabs = hq_init.empirical_tables(obs, proc.O)
    P, T3 = hq_init.model_tables(_true_K(proc))
    assert torch.allclose(P.sum((1, 2)), torch.ones(len(tabs["pair_lags"]), dtype=torch.float64), atol=1e-10)
    assert torch.allclose(T3.sum((1, 2, 3)), torch.ones(len(tabs["triple_lags"]), dtype=torch.float64), atol=1e-10)
    loss_truth = float(hq_init.mom_loss(_true_K(proc), tabs))
    assert loss_truth < 2e-3                                    # sampling floor only (about 1e-4 to 4e-4 measured)
    G = torch.complex(torch.randn(9, 3, dtype=torch.float64), torch.randn(9, 3, dtype=torch.float64))
    assert float(hq_init.mom_loss(hq_init.polar(G).reshape(3, 3, 3), tabs)) > 10 * loss_truth


def test_stationary_state_of_probability_clock():
    proc = tasks_e2.probability_clock()
    s = hq_init.stationary(_true_K(proc)).numpy()
    assert np.allclose(s, proc.stationary(), atol=1e-8)


def test_warm_start_reproduces_truth_with_finite_gradients():
    proc = tasks_e2.qutrit_clock()
    models.L_CTX = L
    m = models_e2.HQMMWM(proc.O, proc.d, L)
    V = _true_K(proc).reshape(proc.O * proc.d, proc.d)
    hq_init.load_warm_start(m, V)
    assert torch.allclose(m.kraus().to(torch.complex128), _true_K(proc), atol=1e-5)
    obs, act = tasks_e2.sample(proc, 400, 20, np.random.default_rng(7))
    pred = predict_torch(m, obs, act, 20)
    exact = tasks_e2.exact_predictive(proc, obs, 20)
    assert np.abs(pred - exact).max() < 2e-3
    p = m(torch.as_tensor(obs[:64]), torch.as_tensor(act[:64]), 20)
    loss = -torch.log(torch.gather(p, 2, torch.as_tensor(obs[:64, L:L + 20]).unsqueeze(-1))).mean()
    loss.backward()
    assert all(torch.isfinite(t.grad).all() for t in (m.g_re, m.g_im))


def test_fit_mom_uses_tables_only_and_finds_probability_clock():
    proc = tasks_e2.probability_clock()
    obs, _ = tasks_e2.sample(proc, 20_000, 20, np.random.default_rng(99))
    tabs = hq_init.empirical_tables(obs, proc.O)
    fit = hq_init.fit_mom(tabs, d=2, n_obs=2, restarts=6, iters=100, init_seed0=1000)
    floor = float(hq_init.mom_loss(_true_K(proc), tabs))
    assert fit["mom_loss"] < floor + 1e-3
    true_slow = hq_init.channel_spectrum(proc.K)[1][0]
    fit_slow = hq_init.channel_spectrum(fit["V"].reshape(2, 2, 2))[1][0]
    assert abs(true_slow - fit_slow) < 0.05

"""Unit tests that must pass before any confirmatory run (pre-registration §4b, PROGRESS.md VERIFIER)."""
import numpy as np
import pytest
import torch

from osw import models, tasks
from osw.models import CDT

torch.manual_seed(0)


def rand_density(B, D):
    A = torch.randn(B, D, D, dtype=CDT)
    rho = A @ A.conj().transpose(-1, -2)
    return rho / torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).real[:, None, None].to(CDT)


# --- tasks / exact Bayes reference ---------------------------------------------------------------

@pytest.mark.parametrize("tau", [5, 20, 80])
def test_closed_form_power(tau):
    t = tasks.t1_simple(tau)
    for k in [1, 2, 7, 50, 200]:
        assert np.allclose(np.linalg.matrix_power(t.P[0], k), tasks.closed_form_power(t, k))


@pytest.mark.parametrize("mk", [lambda: tasks.t1_simple(20), lambda: tasks.t1_metastable(20), lambda: tasks.t1_action(20)])
def test_transition_matrices_are_stochastic(mk):
    t = mk()
    for P in t.P:
        assert np.allclose(P.sum(1), 1) and (P >= 0).all()
        assert np.allclose(np.ones(8) / 8 @ P, np.ones(8) / 8)        # uniform is stationary


def test_bayes_predictive_matches_closed_form_and_beats_marginal():
    t = tasks.t1_simple(20)
    rng = np.random.default_rng(1)
    obs, act = tasks.sample(t, 4000, 60, rng)
    pred = tasks.bayes_predictive(t, obs, act, 60)
    assert np.allclose(pred.sum(-1), 1)
    b = tasks.bayes_filter(t, obs, act)
    for k in [1, 10, 60]:
        assert np.allclose(pred[:, k - 1], b @ tasks.closed_form_power(t, k) @ t.B)
    # the exact predictive must score better than the stationary marginal on held-out samples
    y = obs[:, tasks.L_CTX:tasks.L_CTX + 60]
    nll_bayes = -np.log(np.take_along_axis(pred, y[..., None], -1)).mean()
    marg = np.ones(8) / 8 @ t.B
    nll_marg = -np.log(marg[y]).mean()
    assert nll_bayes < nll_marg


def test_long_horizon_predictive_relaxes_to_stationary():
    t = tasks.t1_simple(5)
    obs, act = tasks.sample(t, 100, 200, np.random.default_rng(2))
    pred = tasks.bayes_predictive(t, obs, act, 200)
    assert np.abs(pred[:, -1] - np.ones(8) / 8 @ t.B).max() < 1e-6


# --- quantum-structured models -----------------------------------------------------------------

def test_instrument_completeness():
    inst = models.Instrument(8, 4)
    M = inst.ops()
    S = (M.conj().transpose(-1, -2) @ M).sum(0)
    assert torch.allclose(S, torch.eye(8, dtype=CDT), atol=1e-4)


@pytest.mark.parametrize("d_e", [1, 2, 4])
def test_unitary_preserves_joint_spectrum(d_e):
    m = models.UnitaryWM(4, 2, d_e=d_e)
    Us = m.prepare()
    rho = rand_density(5, 8 * d_e)
    out = m.evolve(rho, torch.tensor([0, 1, 0, 1, 1]), Us)
    ev0 = torch.linalg.eigvalsh(rho)
    ev1 = torch.linalg.eigvalsh(out)
    assert torch.allclose(ev0, ev1, atol=1e-4)


def _check_channel(m, D, n_act=2):
    cache = m.prepare()
    rho = rand_density(6, D)
    a = torch.tensor([0, 1, 0, 1, 0, 1]) % n_act
    out = m.evolve(rho, a, cache)
    tr = torch.diagonal(out, dim1=-2, dim2=-1).sum(-1).real
    assert torch.allclose(tr, torch.ones(6), atol=1e-4)
    assert torch.linalg.eigvalsh(out).min() > -1e-4
    assert torch.allclose(out, out.conj().transpose(-1, -2), atol=1e-5)


def test_lindblad_is_cptp_on_states():
    _check_channel(models.LindbladWM(4, 2), 8)


def test_kraus_is_cptp():
    m = models.KrausWM(4, 2)
    A = m.prepare()
    S = torch.einsum("arij,arik->ajk", A.conj(), A)
    assert torch.allclose(S, torch.eye(8, dtype=CDT).expand(2, 8, 8), atol=1e-4)
    _check_channel(m, 8)


def test_contraction_control_is_cptp():
    _check_channel(models.ContractionWM(4, 2), 8)


@pytest.mark.parametrize("name", ["U1", "U2", "U4", "L", "L1", "K", "C", "GRU", "LRU"])
def test_models_output_valid_distributions(name):
    t = tasks.t1_action(20)
    obs, act = tasks.sample(t, 7, 30, np.random.default_rng(3))
    m = models.build(name, 4, 2)
    p = m(torch.as_tensor(obs), torch.as_tensor(act), 30)
    assert p.shape == (7, 30, 4)
    assert torch.allclose(p.sum(-1), torch.ones(7, 30), atol=1e-4) and (p >= 0).all()
    loss = -torch.log(p).mean()
    loss.backward()                                                  # gradients flow
    assert any(q.grad is not None and torch.isfinite(q.grad).all() for q in m.parameters())


def test_recurrence_vs_relaxation_signature():
    """Sanity check of the mechanism the experiment tests: an untrained unitary latent keeps oscillating at
    long horizons, while a dissipative (Lindblad) latent settles. This is a property of the dynamics classes,
    not a result."""
    torch.manual_seed(4)
    t = tasks.t1_simple(20)
    obs, act = tasks.sample(t, 16, 400, np.random.default_rng(5))
    o, a = torch.as_tensor(obs), torch.as_tensor(act)
    with torch.no_grad():
        u = models.UnitaryWM(4, 1, d_e=2)
        for p_ in (u.h0_0, u.h0_1):
            p_.mul_(20)                                              # a non-trivial rotation
        lb = models.LindbladWM(4, 1)
        lb.j_re.mul_(10); lb.j_im.mul_(10)
        pu = u(o, a, 400)
        pl = lb(o, a, 400)
    late = slice(300, 400)
    assert pu[:, late].std(1).mean() > 10 * pl[:, late].std(1).mean()


def test_liouvillian_matches_gksl_rhs():
    """The row-major superoperator must reproduce -i[H,rho] + sum_j (L rho L^† - 1/2{L^†L, rho}) exactly."""
    torch.manual_seed(7)
    m = models.LindbladWM(4, 1)
    H = models.hermitian(m.h0_re, m.h0_im).to(CDT).detach()
    Lsup = m.liouvillian(H).detach()
    rho = rand_density(3, 8)
    Lj = torch.complex(m.j_re, m.j_im).detach()
    rhs = -1j * (H @ rho - rho @ H)
    for k in range(Lj.shape[0]):
        Lk = Lj[k]
        LdL = Lk.conj().T @ Lk
        rhs = rhs + Lk @ rho @ Lk.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
    got = (Lsup @ rho.reshape(3, 64, 1)).reshape(3, 8, 8)
    assert torch.allclose(got, rhs, atol=1e-5)


def test_amplitude_damping_closed_form():
    """Qubit amplitude damping L = sqrt(g) sigma_-: rho_ee(t) = rho_ee(0) e^{-g t}, rho_eg(t) = rho_eg(0) e^{-g t/2}."""
    m = models.LindbladWM(2, 1, d_s=2, n_jumps=1)
    g = 0.7
    with torch.no_grad():
        m.h0_re.zero_(); m.h0_im.zero_(); m.j_re.zero_(); m.j_im.zero_()
        m.j_re[0, 1, 0] = g ** 0.5          # sigma_- |e><g| convention: basis (g=0, e=1) -> L = |0><1|
        m.j_re[0] = m.j_re[0].T.clone()     # L[0,1] = sqrt(g): maps |1> (excited) to |0> (ground)
    E = m.prepare()[0]
    rho0 = torch.tensor([[0.3, 0.2 + 0.1j], [0.2 - 0.1j, 0.7]], dtype=CDT)
    rho1 = (E @ rho0.reshape(4, 1)).reshape(2, 2)
    assert abs(rho1[1, 1].real - 0.7 * np.exp(-g)) < 1e-5
    assert abs(rho1[0, 1] - (0.2 + 0.1j) * np.exp(-g / 2)) < 1e-5


def test_parameter_budgets_match_within_ten_percent():
    """Pre-registration §3 / DEVIATIONS.md D9: each pre-registered comparison within +-10% trainable parameters."""
    for na in (1, 2):
        n = {m: models.n_params(models.build(m, 4, na)) for m in ["U2", "L", "L1", "C", "GRU", "LRU"]}
        close = lambda a, b: abs(n[a] - n[b]) <= 0.10 * n[b]
        assert close("GRU", "L") and close("LRU", "L") and close("C", "L1")
        if na == 1:
            assert close("U2", "L")                   # H1 tasks have one action



def test_queue_names_match_run_names(tmp_path, monkeypatch):
    """End to end: the filename run.py actually writes must be the one the queue/gate looks for (BAYES needs no
    training, so this is cheap). Regression for the round-2 gate's false 'missing' HMM/BAYES files."""
    from osw import queue, run
    cfg = {"steps": 6000, "init": "identity", "lrs": [3e-3, 1e-2]}
    monkeypatch.setattr(queue, "train_config", lambda: cfg)
    j = dict(task="simple", tau=20, model="BAYES", seed=102, lr=3e-3)
    run.main(["--task", "simple", "--tau", "20", "--model", "BAYES", "--seed", "102", "--lr", "0.003",
              "--steps", str(cfg["steps"]), "--init", cfg["init"], "--out", str(tmp_path)])
    assert [f.name for f in tmp_path.glob("*.json")] == [queue.out_name(j)]


@pytest.mark.parametrize("name", ["L", "U2", "GRU"])
def test_v11_diagnostics_do_not_change_training(name):
    """v1.1 adds an initial validation NLL and gradient-norm logging; the trained model must be bit-identical."""
    from osw import run
    task = tasks.t1_simple(20)
    tr, va, _ = run.data(task, 7, 1020)
    tr, va = (tr[0][:500], tr[1][:500]), (va[0][:200], va[1][:200])
    outs = []
    for diag in (None, {}):
        torch.manual_seed(7)
        m = models.build(name, tasks.O, 1)
        hist, v = run.train_torch(m, tr, va, 1e-2, 30, 16, 7, log_every=10, diag=diag)
        outs.append((hist, v, [p.detach().clone() for p in m.parameters()], diag))
    (h0, v0, p0, _), (h1, v1, p1, d1) = outs
    assert h0 == h1 and v0 == v1 and all(torch.equal(a, b) for a, b in zip(p0, p1))
    assert np.isfinite(d1["init_val_nll"]) and len(d1["grad_norm_curve"]) == 3

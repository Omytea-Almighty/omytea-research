"""E3 recipe S: Stiefel/Cayley optimiser for HQ (src/osw/stiefel.py)."""
import json

import numpy as np
import torch

from osw import hq_init, models, models_e2, run_e2, stiefel, tasks_e2
from osw.run import predict_torch

L = tasks_e2.L_CTX


def _small(proc, n, rng_seed):
    return tasks_e2.sample(proc, n, 20, np.random.default_rng(rng_seed))


class _NoExact:
    def __getattr__(self, name):
        raise AssertionError("fit_S read the exact predictive")

    def __getitem__(self, k):
        raise AssertionError("fit_S read the exact predictive")


def test_init_seed_namespace():
    assert stiefel.init_seed(20, 0) == 10_202_000
    conf = [stiefel.init_seed(s, r) for s in range(20, 30) for r in (0, 999)]
    assert all(10_200_000 <= x <= 10_299_999 for x in conf)
    dev = [stiefel.init_seed(s, r) for s in range(200, 210) for r in (0, 999)]
    assert all(12_000_000 <= x <= 12_099_999 for x in dev)


def test_forward_matches_hqmmwm_for_same_isometry():
    proc = tasks_e2.qutrit_clock()
    models.L_CTX = L
    V = stiefel.random_isometry(proc.O, proc.d, stiefel.init_seed(201, 0))
    assert stiefel.isometry_error(V) < 1e-12
    s = stiefel.HQStiefel(proc.O, proc.d, L, V)
    h = models_e2.HQMMWM(proc.O, proc.d, L)
    hq_init.load_warm_start(h, V)                                  # G = V diag(linspace(1, 1.6, d))
    obs, act = _small(proc, 300, 201)
    ps, ph = predict_torch(s, obs, act, 20), predict_torch(h, obs, act, 20)
    assert ps.shape == (300, 20, proc.O)
    assert np.abs(ps.sum(-1) - 1).max() < 1e-12
    assert np.abs(ps - ph).max() < 1e-5


def test_isometry_kept_after_50_steps():
    proc = tasks_e2.qutrit_clock()
    tr, va = _small(proc, 2000, 202), _small(proc, 1000, 203)
    m = stiefel.HQStiefel(proc.O, proc.d, L, stiefel.random_isometry(proc.O, proc.d, stiefel.init_seed(202, 0)))
    V0 = m.V.clone()
    out = stiefel.train_stiefel(m, tr, va, steps=50, tau0=0.9, seed=stiefel.init_seed(202, 0), log_every=10)
    assert not out["nan_flag"] and out["ckpt_step"] == 50
    assert stiefel.isometry_error(out["V"]) < 1e-10
    assert torch.equal(out["V"], m.V) and not torch.allclose(out["V"], V0)


def test_deterministic_rerun():
    proc = tasks_e2.qutrit_clock()
    tr, va = _small(proc, 2000, 204), _small(proc, 1000, 205)
    V0 = stiefel.random_isometry(proc.O, proc.d, stiefel.init_seed(204, 1))
    outs = []
    for seed in (stiefel.init_seed(204, 1), stiefel.init_seed(204, 1), stiefel.init_seed(204, 2)):
        m = stiefel.HQStiefel(proc.O, proc.d, L, V0)
        outs.append(stiefel.train_stiefel(m, tr, va, steps=50, tau0=0.3, seed=seed))
    assert torch.equal(outs[0]["V"], outs[1]["V"]) and outs[0]["history"] == outs[1]["history"]
    assert not torch.equal(outs[0]["V"], outs[2]["V"])                # the batch seed matters


def test_loss_decreases_on_a1():
    tr, va, _, _, _ = run_e2.data("A1", 200)
    s = stiefel.init_seed(200, 0)
    m = stiefel.HQStiefel(2, 2, L, stiefel.random_isometry(2, 2, s))
    out = stiefel.train_stiefel(m, tr, va, steps=300, tau0=0.3, seed=s)
    assert not out["nan_flag"]
    assert out["val_nll"] < out["init_val_nll"] - 0.01


class _NaNAfter(stiefel.HQStiefel):
    def __init__(self, *a, after):
        super().__init__(*a)
        self.calls, self.after = 0, after

    def probs(self, obs, K, V=None, raw=False):
        p = super().probs(obs, K, V, raw)
        if V is not None and V.requires_grad:
            self.calls += 1
            if self.calls > self.after:
                return p * float("nan")
        return p


def test_nan_stops_and_keeps_last_checkpoint():
    proc = tasks_e2.probability_clock()
    tr, va = _small(proc, 2000, 206), _small(proc, 1000, 207)
    s = stiefel.init_seed(206, 0)
    V0 = stiefel.random_isometry(2, 2, s)
    bad = _NaNAfter(2, 2, L, V0, after=150)
    out = stiefel.train_stiefel(bad, tr, va, steps=300, tau0=0.3, seed=s)
    assert out["nan_flag"] and out["nan_step"] == 151 and out["ckpt_step"] == 100
    ref = stiefel.train_stiefel(stiefel.HQStiefel(2, 2, L, V0), tr, va, steps=100, tau0=0.3, seed=s)
    assert torch.equal(out["V"], ref["V"]) and np.isfinite(out["val_nll"])


def test_fit_S_contract_and_selection():
    proc = tasks_e2.probability_clock()
    tr, va, te = _small(proc, 1000, 208), _small(proc, 1000, 209), _small(proc, 200, 208 * 7)
    res = stiefel.fit_S(tr, va, te, _NoExact(), d=2, O=2, seed=208, RS=2, tau0=0.6, steps=30)
    assert res["pred"].shape == (200, 20, 2)
    assert np.abs(res["pred"].sum(-1) - 1).max() <= 1e-5
    assert res["n_params"] == 16 and stiefel.n_params(3, 3) == 54 and stiefel.n_params(3, 4) == 96
    assert res["init_seeds"] == [stiefel.init_seed(208, 0), stiefel.init_seed(208, 1)]
    assert res["selected_restart"] == int(np.argmin(res["restart_val_nll"]))
    assert res["val_nll"] == min(res["restart_val_nll"]) and res["cpu_s"] > 0
    assert res["nan_count"] == 0 and res["isometry_err"] < 1e-10 and 0 <= res["clamp_frac"] <= 1
    json.dumps({k: v for k, v in res.items() if k != "pred"})
    again = stiefel.fit_S(tr, va, te, None, d=2, O=2, seed=208, RS=2, tau0=0.6, steps=30)
    assert np.array_equal(again["pred"], res["pred"])                 # exact argument has no effect

"""E3 recipes in src/osw/cplus.py: CPLUS / CPLUS-M / SENS (GRU, LRU) and HQ M / C1 / C2. Dev seed 200 only."""
import copy
from functools import lru_cache

import numpy as np
import pytest
import torch
import torch.nn as nn

from osw import cplus, hq_init, models, models_e2, run_e2, tasks_e2

L = cplus.L


@lru_cache(maxsize=None)
def _data(arm):
    tr, va, te, _, _ = run_e2.data(arm, 200)                  # the exact predictive is discarded: never an input
    return tr, va, te


def _normalised(pred, n, O):
    assert pred.shape == (n, cplus.K, O) and pred.dtype == np.float64
    assert np.abs(pred.sum(-1) - 1).max() <= 1e-5


def test_seed_namespace():
    assert cplus.init_seed(20, cplus.RECIPE["CPLUS"], 10 * 1 + 3) == 10_000_000 + 200_000 + 4_000 + 13
    conf = {cplus.init_seed(s, c, r) for s in range(20, 30) for c in range(9) for r in (0, 999)}
    dev = {cplus.init_seed(s, c, r) for s in range(200, 210) for c in range(9) for r in (0, 999)}
    assert min(conf) >= 10_200_000 and max(conf) <= 10_299_999
    assert min(dev) >= 12_000_000 and max(dev) <= 12_099_999
    with pytest.raises(AssertionError):
        cplus.init_seed(20, 4, 1000)


class _Toy(nn.Module):
    """Constant predictive; returns NaN on the `bad`-th forward call (to exercise the checkpoint rule)."""

    def __init__(self, O=2, bad=None):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(cplus.K, O))
        self.calls, self.bad = 0, bad

    def forward(self, obs, act, K):
        self.calls += 1
        p = torch.softmax(self.w, -1).expand(obs.shape[0], K, -1)
        return p * float("nan") if self.calls == self.bad else p


def _toy_data(n=256, O=2):
    rng = np.random.default_rng(201)
    obs = rng.integers(0, O, size=(n, L + cplus.K))
    return obs, np.zeros_like(obs)


def test_linear_schedule_reaches_zero():
    assert cplus.linear_lr(0, 50, 1e-2) == 1e-2 and cplus.linear_lr(50, 50, 1e-2) == 0.0
    m = _Toy()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    st = cplus.new_state(m, opt)
    ok = cplus.run_phase(m, opt, torch.Generator().manual_seed(202), cplus._tensors(_toy_data()), 1e-2, 250, True, st)
    assert ok and st["step"] == 250 and opt.param_groups[0]["lr"] == 0.0
    lrs = [c[2] for c in st["curve"]]
    assert [c[0] for c in st["curve"]] == [100, 200, 250]
    assert np.allclose(lrs, [1e-2 * (1 - 99 / 250), 1e-2 * (1 - 199 / 250), 1e-2 * (1 - 249 / 250)])


def test_nan_restores_last_finite_checkpoint():
    m = _Toy(bad=150)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    st = cplus.new_state(m, opt)
    ref = _Toy()
    ref_opt = torch.optim.Adam(ref.parameters(), lr=1e-2)
    data, ref_st = cplus._tensors(_toy_data()), cplus.new_state(ref, ref_opt)
    cplus.run_phase(ref, ref_opt, torch.Generator().manual_seed(203), data, 1e-2, 100, False, ref_st)
    ok = cplus.run_phase(m, opt, torch.Generator().manual_seed(203), data, 1e-2, 300, False, st)
    assert not ok and st["nan_flag"] and st["nan_step"] == 150 and st["step"] == 100
    assert torch.equal(m.w, ref.w) and cplus.adam_step(opt) == 100


def test_cplus_single_gru_a1_runs_and_carries_adam_state():
    tr, va, te = _data("A1")
    r = cplus.cplus_single("GRU", 4, tr, va, te, 2, seed=200, replica=1, lrs=(3e-3, 1e-2),
                           steps_probe=20, steps_total=60)
    _normalised(r["pred"], len(te[0]), 2)
    assert r["n_params"] == 130 and np.isfinite(r["val_nll"]) and r["cpu_s"] > 0
    assert len(r["probe_val_nlls"]) == 4 and r["selected_probe"] == int(np.argmin(r["probe_val_nlls"]))
    assert r["probe_seeds"] == [cplus.init_seed(200, 4, 10 + p) for p in range(4)]
    assert r["probe_lrs"] == [3e-3, 1e-2, 3e-3, 1e-2]
    assert r["adam_step"] == 60                                # 20 probe steps + 40 continued: same optimizer
    assert r["lr_end"] == 0.0 and not r["nan_flag"]
    assert "r_hat" not in r


def test_cplus_continuation_equals_uninterrupted_run():
    """Continuation keeps model, Adam state and batch generator: probe + decayed phase == one manual run."""
    tr, va, te = _data("A1")
    r = cplus.cplus_single("GRU", 2, tr, va, te, 2, seed=200, replica=0, lrs=(1e-2, 1e-2),
                           steps_probe=10, steps_total=30)
    p = r["selected_probe"]
    torch.manual_seed(cplus.init_seed(200, 4, 2 * (p // 2)))
    m = models_e2.build("GRU", 2, 2, L)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    gen = torch.Generator().manual_seed(cplus.init_seed(200, 4, p))
    st, data = cplus.new_state(m, opt), cplus._tensors(tr)
    cplus.run_phase(m, opt, gen, data, 1e-2, 10, False, st)
    cplus.run_phase(m, opt, gen, data, 1e-2, 20, True, st)
    assert np.array_equal(cplus.checked_pred(m, te), r["pred"])


def test_lru_spectral_init_on_a2():
    tr, va, te = _data("A2")
    models.L_CTX = L
    torch.manual_seed(cplus.init_seed(200, 4, 0))
    m = models_e2.build("LRU", 4, 3, L)
    before = copy.deepcopy(m.state_dict())
    info = {}
    r_hat = cplus.lru_spectral_lambda_init(m, tr[0], 3, info=info)
    assert isinstance(r_hat, int) and 1 <= r_hat <= 9
    k = len(info["written"])
    assert k == min(4, len(info["lambdas"]))
    assert torch.isfinite(m.nu).all() and torch.isfinite(m.theta).all()
    lam = torch.exp(torch.complex(-torch.exp(m.nu), m.theta)).detach()
    for i, (mod, ang) in enumerate(info["written"]):
        assert mod <= 0.995 and abs(abs(lam[i]) - mod) < 1e-5
    assert torch.equal(m.nu[k:], before["nu"][k:]) and torch.equal(m.b_re, before["b_re"])
    assert all(x[1] >= 0 for x in info["lambdas"])            # one member per conjugate pair


def test_spectral_rank_uses_training_data_only_and_finds_a1_mode():
    tr, va, te = _data("A1")
    sp = cplus.spectral_lambdas(tr[0], 2)
    sp2 = cplus.spectral_lambdas(tr[0].copy(), 2)
    assert sp == sp2
    true = hq_init.channel_spectrum(tasks_e2.probability_clock().K)
    assert sp["r_hat"] == 3                                   # A1 Hankel rank (design §2)
    assert abs(sp["lambdas"][0][0] - true[1][0]) < 0.05 and abs(sp["lambdas"][0][1] - abs(true[1][1])) < 0.05


def test_cplus_lru_records_rank_and_cplus_m_selects():
    tr, va, te = _data("A1")
    r = cplus.cplus_m("LRU", 2, tr, va, te, 2, seed=200, lrs=(3e-3, 1e-2), R_c=2, steps_probe=5, steps_total=10)
    _normalised(r["pred"], len(te[0]), 2)
    assert len(r["replica_val_nlls"]) == 2 and r["val_nll"] == min(r["replica_val_nlls"])
    assert r["selected_replica"] == int(np.argmin(r["replica_val_nlls"])) and r["r_hat"] == 3
    assert r["cpu_s"] >= r["replica_cpu_s"] > 0
    s = cplus.sens("LRU", 2, tr, va, te, 2, seed=200, lrs=(3e-3, 1e-2), R_c=1, base=r, steps_probe=5, steps_total=10)
    assert s["val_nll"] == min(r["val_nll"], s["replica_val_nlls_extra"][0])
    assert s["recipe"] == (cplus.RECIPE["CPLUS"] if s["sens_source"] == "base" else cplus.RECIPE["SENS"])


def test_fit_M_a1_end_to_end():
    tr, va, te = _data("A1")
    r = cplus.fit_M(tr, va, te, d=2, O=2, seed=200, restarts=4, steps=300)
    _normalised(r["pred"], len(te[0]), 2)
    _normalised(r["pred_mom"], len(te[0]), 2)
    assert r["n_params"] == 16 and len(r["mom_losses_sorted"]) == 4 and r["mom_loss"] == r["mom_losses_sorted"][0]
    assert r["init_seed0"] == cplus.init_seed(200, 0, 0) and 0 <= r["selected_restart"] < 4
    Kr = np.array(r["kraus"])
    Kc = Kr[..., 0] + 1j * Kr[..., 1]
    assert Kc.shape == (2, 2, 2)
    assert np.allclose(sum(k.conj().T @ k for k in Kc), np.eye(2), atol=1e-4)
    assert len(r["channel_spectrum"]) == 4 and abs(r["channel_spectrum"][0][0] - 1) < 1e-4
    assert r["adam_step"] == 300 and r["lr_end"] == 0.0 and not r["nan_flag"]
    assert r["val_nll"] <= r["init_val_nll"] + 0.01 and r["clamp_frac_test"] < 1e-4


def test_fit_M_deterministic():
    tr, va, te = _data("A1")
    a = cplus.fit_M(tr, va, te, d=2, O=2, seed=200, restarts=2, iters=30, steps=50)
    b = cplus.fit_M(tr, va, te, d=2, O=2, seed=200, restarts=2, iters=30, steps=50)
    assert np.array_equal(a["pred"], b["pred"]) and a["kraus"] == b["kraus"] and a["mom_loss"] == b["mom_loss"]


def test_fit_C2_writes_perturbed_G_exactly():
    proc = tasks_e2.qutrit_clock()
    models.L_CTX = L
    m = models_e2.HQMMWM(3, 3, L)
    ns = cplus.init_seed(200, cplus.RECIPE["C2"], 0)
    G0 = cplus.c2_start(m, np.stack(proc.K), 0.05, ns)
    g = torch.Generator().manual_seed(ns)
    V = torch.tensor(np.stack(proc.K).reshape(9, 3), dtype=torch.complex128)
    zr = torch.randn(9, 3, generator=g, dtype=torch.float64)
    zi = torch.randn(9, 3, generator=g, dtype=torch.float64)
    expect = V + 0.05 * torch.complex(zr, zi)
    assert torch.equal(G0, expect)
    assert torch.equal(m.g_re, expect.real.float()) and torch.equal(m.g_im, expect.imag.float())


def test_fit_C2_and_C1_run():
    tr, va, te = _data("A1")
    proc = tasks_e2.probability_clock()
    c2 = cplus.fit_C2(tr, va, te, d=2, O=2, seed=200, sigma=0.05, V_true=np.stack(proc.K), steps=40)
    _normalised(c2["pred"], len(te[0]), 2)
    assert c2["uses_oracle"] is True and c2["noise_seed"] == cplus.init_seed(200, 3, 0) and c2["adam_step"] == 40
    c1 = cplus.fit_C1(tr, va, te, d=2, O=2, seed=200, R1=2, steps=30)
    _normalised(c1["pred"], len(te[0]), 2)
    assert c1["init_seeds"] == [cplus.init_seed(200, 1, r) for r in range(2)]
    assert c1["val_nll"] == min(c1["restart_val_nlls"]) and c1["lr_end"] == 0.0

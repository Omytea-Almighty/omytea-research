"""E3 recipe S: COSM-style Stiefel-manifold optimiser for the HQ density-matrix sequence model (design §1, §4 S).

The stacked isometry V (O*d, d), V^dag V = I, is the state itself (no polar map, unlike models_e2.HQMMWM). One step
(Wen & Yin 2013 Cayley retraction with the normalised-momentum scheme of Adhikary et al., AISTATS 2020):

  G = dL/dRe V + i dL/dIm V              Euclidean (Wirtinger) gradient of the E2 multi-step hard-target NLL
  M = 0.9 M + G/|G|_F,  M = M/|M|_F      gradient normalised before and momentum normalised after accumulation
  A = M V^dag - V M^dag                  skew-Hermitian, so the Cayley factor below is unitary
  V <- (I + tau/2 A)^{-1} (I - tau/2 A) V,   tau_t = tau0 * 0.9**(t/313)

Filtering and prediction are identical to HQMMWM.forward (rho0 = I/d, clamp_min 1e-8 then renormalise), in complex128.
NaN rule (design §4): checkpoint every 100 steps; a non-finite loss, gradient or update stops training and the last
checkpoint is kept with nan_flag set. fit_S never touches the exact predictive (argument `exact_unused`).
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from . import hq_init
from .run import predict_torch

C128 = torch.complex128
F64 = torch.float64
EPS = 1e-8
RECIPE_S = 2


def init_seed(seed: int, r: int, code: int = RECIPE_S) -> int:
    """Design §4 namespace: 10**7 + 10**4*seed + 10**3*recipe_code + r."""
    return 10 ** 7 + 10 ** 4 * seed + 10 ** 3 * code + r


def random_isometry(O: int, d: int, seed: int, scale: float = 0.5) -> torch.Tensor:
    """Polar of a complex Gaussian G (O*d, d) with std `scale` per real component."""
    g = torch.Generator().manual_seed(seed)
    re = torch.randn(O * d, d, generator=g, dtype=F64) * scale
    im = torch.randn(O * d, d, generator=g, dtype=F64) * scale
    return hq_init.polar(torch.complex(re, im))


class HQStiefel(nn.Module):
    kind = "density"

    def __init__(self, n_obs, d, ctx, V=None):
        super().__init__()
        self.n_obs, self.d, self.ctx = n_obs, d, ctx
        if V is None:
            V = random_isometry(n_obs, d, 0)
        self.register_buffer("V", V.detach().to(C128).clone())

    def kraus(self, V=None):
        return (self.V if V is None else V).reshape(self.n_obs, self.d, self.d)

    def probs(self, obs, K, V=None, raw=False):
        B = obs.shape[0]
        Ks = self.kraus(V)
        rho = (torch.eye(self.d, dtype=C128) / self.d).expand(B, self.d, self.d).clone()
        for t in range(self.ctx):
            Ky = Ks[obs[:, t]]
            r = Ky @ rho @ Ky.conj().transpose(-1, -2)
            tr = torch.diagonal(r, dim1=-2, dim2=-1).sum(-1).real.clamp_min(EPS)
            rho = r / tr.to(C128)[:, None, None]
        out = []
        for _ in range(K):
            p = torch.einsum("yij,bjk,yik->by", Ks, rho, Ks.conj()).real
            if not raw:
                p = p.clamp_min(EPS)
                p = p / p.sum(1, keepdim=True)
            out.append(p)
            rho = torch.einsum("yij,bjk,ylk->bil", Ks, rho, Ks.conj())
        return torch.stack(out, 1)

    def forward(self, obs, act, K):
        return self.probs(obs, K)


def n_params(O: int, d: int) -> int:
    return 2 * O * d * d                                       # real components of G, as HQMMWM (design §9: A2 HQ3 54)


def isometry_error(V: torch.Tensor) -> float:
    return float((V.conj().T @ V - torch.eye(V.shape[1], dtype=V.dtype)).abs().max())


def _val_nll(model, va, ctx, K):
    pv = predict_torch(model, va[0][:1000], va[1][:1000], K)
    yv = va[0][:1000, ctx:ctx + K]
    return float(-np.log(np.clip(np.take_along_axis(pv, yv[..., None], -1), 1e-9, 1)).mean())


def train_stiefel(model, tr, va, ctx=32, K=20, steps=6000, tau0=0.3, batch=64, seed=0, log_every=200):
    """Cayley-retraction training of model.V in place; batches from torch.Generator().manual_seed(seed)."""
    torch.set_num_threads(1)
    g = torch.Generator().manual_seed(seed)
    obs, act = torch.as_tensor(tr[0]), torch.as_tensor(tr[1])
    y = obs[:, ctx:ctx + K]
    V = model.V.clone()
    I = torch.eye(V.shape[0], dtype=C128)
    M = torch.zeros_like(V)
    ckpt, ckpt_step, nan_flag, nan_step = V.clone(), 0, False, None
    hist, gn_curve = [], []
    init = _val_nll(model, va, ctx, K)
    for step in range(1, steps + 1):
        idx = torch.randint(0, obs.shape[0], (batch,), generator=g)
        re, im = V.real.clone().requires_grad_(), V.imag.clone().requires_grad_()
        p = model.probs(obs[idx], K, V=torch.complex(re, im))
        loss = -torch.log(torch.gather(p, 2, y[idx].unsqueeze(-1)).clamp_min(1e-9)).mean()
        ok = bool(torch.isfinite(loss))
        if ok:
            loss.backward()
            G = torch.complex(re.grad, im.grad)
            gn = float(torch.linalg.norm(G))
            ok = np.isfinite(gn)
        if ok:
            M = 0.9 * M + G / max(gn, 1e-300)
            M = M / torch.linalg.norm(M).clamp_min(1e-300)
            A = M @ V.conj().T - V @ M.conj().T
            tau = tau0 * 0.9 ** ((step - 1) / 313)
            Vn = torch.linalg.solve(I + 0.5 * tau * A, (I - 0.5 * tau * A) @ V)
            ok = bool(torch.isfinite(torch.view_as_real(Vn)).all())
        if not ok:
            nan_flag, nan_step = True, step
            break
        V = Vn
        if step % log_every == 0 or step == steps:
            hist.append((step, float(loss.detach()))); gn_curve.append((step, gn))
        if step % 100 == 0 or step == steps:
            ckpt, ckpt_step = V.clone(), step
    model.V.copy_(ckpt)
    return {"history": hist, "val_nll": _val_nll(model, va, ctx, K), "init_val_nll": init, "V": ckpt,
            "nan_flag": nan_flag, "nan_step": nan_step, "ckpt_step": ckpt_step, "grad_norm_curve": gn_curve,
            "isometry_err": isometry_error(ckpt)}


def fit_S(tr, va, te, exact_unused, d, O, seed, RS, tau0, steps=6000, ctx=32, K=20, batch=64):
    """RS random-isometry restarts (namespace recipe 2), each trained by train_stiefel; keep the lowest val NLL.
    The exact predictive is never read."""
    torch.set_num_threads(1)
    t0 = time.process_time()
    runs = []
    for r in range(RS):
        s = init_seed(seed, r)
        m = HQStiefel(O, d, ctx, random_isometry(O, d, s))
        out = train_stiefel(m, tr, va, ctx=ctx, K=K, steps=steps, tau0=tau0, batch=batch, seed=s)
        runs.append((m, out, s))
    vals = [o["val_nll"] if np.isfinite(o["val_nll"]) else float("inf") for _, o, _ in runs]
    sel = int(np.argmin(vals))
    m, out, _ = runs[sel]
    pred = predict_torch(m, te[0], te[1], K)
    assert np.abs(pred.sum(-1) - 1).max() <= 1e-5
    with torch.no_grad():
        raw = m.probs(torch.as_tensor(te[0]), K, raw=True).numpy()
    return {"pred": pred, "val_nll": out["val_nll"], "n_params": n_params(O, d), "cpu_s": time.process_time() - t0,
            "tau0": tau0, "RS": RS, "steps": steps, "selected_restart": sel, "init_seeds": [s for _, _, s in runs],
            "restart_val_nll": vals, "restart_init_val_nll": [o["init_val_nll"] for _, o, _ in runs],
            "nan_flags": [bool(o["nan_flag"]) for _, o, _ in runs], "nan_count": sum(o["nan_flag"] for _, o, _ in runs),
            "train_curve": out["history"], "grad_norm_curve": out["grad_norm_curve"],
            "isometry_err": out["isometry_err"], "clamp_frac": float((raw < EPS).mean()),
            "channel_spectrum": [list(x) for x in hq_init.channel_spectrum(m.kraus())]}

"""Method-of-moments (MoM) warm start for the HQ density-matrix sequence model (models_e2.HQMMWM).

Why: on E2 arm A2 gradient training of HQ(3) from a random isometry stalls near ln 3 although the class contains the
true model and 20,000 training sequences suffice (exploratory diagnosis 2026-10-24, seeds >= 300, independently
reproduced). Fitting the Kraus operators first to low-order statistics of the TRAINING sequences only - pair tables
P(y_t, y_{t+k}) and triple tables P(y_t, y_{t+k1}, y_{t+k1+k2}) - lands in the right basin; NLL fine-tuning then
recovers the exact predictive.

  tabs = empirical_tables(train_obs, n_obs)                 # training sequences only; no oracle object
  fit = fit_mom(tabs, d, n_obs, restarts=48)               # L-BFGS restarts, selected by the moment loss alone
  load_warm_start(model, fit["V"])                         # G = V P with non-degenerate P (see below)

Model tables use the model's own stationary state (power iteration of the channel), because the sequences are
sampled from the stationary state. The isometry map is the same polar map as HQMMWM, V = G (G^dag G)^{-1/2}.
Pitfall handled here: the eigh-based polar backward is NaN at an exactly isometric G (repeated eigenvalue 1), so the
warm start loads G = V P with P = diag(linspace(1, 1.6, d)). The Kraus operators are unchanged: for an isometry V and
positive-definite P, (V P)((V P)^dag (V P))^{-1/2} = V P (P^2)^{-1/2} = V.
"""
from __future__ import annotations

import time

import numpy as np
import torch

C128 = torch.complex128
F64 = torch.float64
PAIR_LAGS = tuple(range(1, 11))
TRIPLE_LAGS = tuple((k1, k2) for k1 in range(1, 4) for k2 in range(1, 4))


def empirical_tables(obs: np.ndarray, n_obs: int, pair_lags=PAIR_LAGS, triple_lags=TRIPLE_LAGS):
    """Pair and triple frequency tables pooled over all positions of the given sequences (n, T)."""
    obs = np.asarray(obs)
    T = obs.shape[1]
    pairs = []
    for k in pair_lags:
        J = np.zeros((n_obs, n_obs))
        np.add.at(J, (obs[:, :T - k].ravel(), obs[:, k:].ravel()), 1)
        pairs.append(J / J.sum())
    trips = []
    for k1, k2 in triple_lags:
        J = np.zeros((n_obs, n_obs, n_obs))
        np.add.at(J, (obs[:, :T - k1 - k2].ravel(), obs[:, k1:T - k2].ravel(), obs[:, k1 + k2:].ravel()), 1)
        trips.append(J / J.sum())
    return {"pairs": torch.tensor(np.stack(pairs), dtype=F64), "triples": torch.tensor(np.stack(trips), dtype=F64),
            "pair_lags": tuple(pair_lags), "triple_lags": tuple(triple_lags), "n_obs": n_obs}


def polar(G: torch.Tensor) -> torch.Tensor:
    """V = G (G^dag G)^{-1/2}, as models_e2.HQMMWM.kraus (eigh-based)."""
    w, U = torch.linalg.eigh(G.conj().transpose(-1, -2) @ G)
    w = w.clamp_min(1e-6)
    return G @ ((U * w.rsqrt().to(G.dtype).unsqueeze(-2)) @ U.conj().transpose(-1, -2))


def _chan(K, r):
    return torch.einsum("yij,...jk,ylk->...il", K, r, K.conj())


def stationary(K: torch.Tensor, steps: int = 400) -> torch.Tensor:
    d = K.shape[-1]
    r = torch.eye(d, dtype=K.dtype) / d
    for _ in range(steps):
        r = _chan(K, r)
    return r / torch.diagonal(r).sum().real


def model_tables(K: torch.Tensor, pair_lags=PAIR_LAGS, triple_lags=TRIPLE_LAGS, steps: int = 400):
    """Exact pair and triple tables of the HQMM with Kraus operators K (O, d, d) started in its stationary state."""
    E = torch.einsum("yji,yjk->yik", K.conj(), K)                        # E_y = K_y^dag K_y
    prob = lambda X: torch.einsum("cji,...ij->...c", E, X).real          # Tr(E_c X) over the trailing (d, d)
    s = stationary(K, steps)
    first = torch.einsum("aij,jk,alk->ail", K, s, K.conj())              # unnormalised post-measurement states
    pairs = []
    X = first
    for k in range(1, max(pair_lags) + 1):
        if k > 1:
            X = _chan(K, X)
        if k in pair_lags:
            pairs.append(prob(X))
    trips = []
    for k1, k2 in triple_lags:
        X = first
        for _ in range(k1 - 1):
            X = _chan(K, X)
        Y = torch.einsum("bij,ajk,blk->abil", K, X, K.conj())
        for _ in range(k2 - 1):
            Y = _chan(K, Y)
        trips.append(prob(Y))
    return torch.stack(pairs), torch.stack(trips)


def mom_loss(K: torch.Tensor, tabs: dict) -> torch.Tensor:
    P, T3 = model_tables(K, tabs["pair_lags"], tabs["triple_lags"])
    kl = lambda e, m: (e * (torch.log(e.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum()
    return kl(tabs["pairs"], P) + kl(tabs["triples"], T3)


def fit_mom(tabs: dict, d: int, n_obs: int, restarts: int = 48, iters: int = 150, init_seed0: int = 0,
            scale: float = 0.5) -> dict:
    """L-BFGS on the moment loss from `restarts` random G (seeds init_seed0 + r); keep the lowest moment loss.
    Selection uses the moment loss on the given (training) tables only."""
    t0 = time.process_time()
    fits = []
    for r in range(restarts):
        g = torch.Generator().manual_seed(init_seed0 + r)
        re = (torch.randn(n_obs * d, d, generator=g, dtype=F64) * scale).requires_grad_()
        im = (torch.randn(n_obs * d, d, generator=g, dtype=F64) * scale).requires_grad_()
        opt = torch.optim.LBFGS([re, im], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            loss = mom_loss(polar(torch.complex(re, im)).reshape(n_obs, d, d), tabs)
            loss.backward()
            return loss
        try:
            opt.step(closure)
        except Exception:                                                  # a failed restart is recorded, not fatal
            fits.append((float("inf"), r, None))
            continue
        with torch.no_grad():
            V = polar(torch.complex(re, im))
            loss = float(mom_loss(V.reshape(n_obs, d, d), tabs))
        fits.append((loss if np.isfinite(loss) else float("inf"), r, V.detach()))
    ok = sorted((f for f in fits if f[2] is not None and np.isfinite(f[0])), key=lambda f: f[0])
    if not ok:
        raise RuntimeError("every MoM restart failed")
    best = ok[0]
    return {"V": best[2], "mom_loss": best[0], "selected_restart": best[1], "losses_sorted": [f[0] for f in ok],
            "n_failed": sum(1 for f in fits if f[2] is None or not np.isfinite(f[0])), "cpu_s": time.process_time() - t0}


def nondegenerate_p(d: int) -> torch.Tensor:
    return torch.diag(torch.linspace(1.0, 1.6, d, dtype=F64)).to(C128)


def load_warm_start(model, V: torch.Tensor) -> None:
    """Write G = V P into an HQMMWM so that model.kraus() == V (up to float32) and the polar backward is finite."""
    G = V.to(C128) @ nondegenerate_p(model.d)
    with torch.no_grad():
        model.g_re.copy_(G.real.to(model.g_re.dtype))
        model.g_im.copy_(G.imag.to(model.g_im.dtype))


def channel_spectrum(K: np.ndarray | torch.Tensor) -> list:
    """Eigenvalues of the superoperator sum_y K_y (x) conj(K_y), sorted by modulus: [(modulus, angle), ...]."""
    K = K.detach().cpu().numpy() if isinstance(K, torch.Tensor) else np.asarray(K)
    S = sum(np.kron(k, k.conj()) for k in K)
    w = np.linalg.eigvals(S)
    w = w[np.argsort(-np.abs(w))]
    return [(float(abs(x)), float(np.angle(x))) for x in w]

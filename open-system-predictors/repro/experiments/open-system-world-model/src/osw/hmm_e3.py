"""E3 HMM-C baseline (E3-DESIGN §4 HMM-C, §7 PC-B, §10 item 4): HMM with a symmetric method-of-moments start.

  T = softmax(transition logits) (n, n), B = softmax(emission logits) (n, O), pi = stationary distribution of T.
  1. Fit the hq_init pair/triple tables of the TRAINING sequences (pair lags 1..10, triple lags {1,2,3}^2, KL sum):
     48 restarts (init seeds recipe 7), L-BFGS 150 iterations, selection by moment loss alone.
  2. Load (T, B, pi) into hmm.HMM and run Baum-Welch on ALL training sequences (<= 100 iterations, tol 1e-6).
  3. Three random-initialisation EM fits (init seeds recipe 5) on all training sequences.
  Keep the fit with the highest training log-likelihood, recomputed at its final parameters; report all fits.
"""
from __future__ import annotations

import time

import numpy as np
import torch

from . import hmm, hq_init, models_e2

F64 = torch.float64
RECIPE_HMM_EM, RECIPE_HMM_MOM = 5, 7


def init_seed(seed: int, recipe: int, r: int) -> int:
    return 10 ** 7 + 10 ** 4 * seed + 10 ** 3 * recipe + r


def stationary(T: torch.Tensor, squarings: int = 9) -> torch.Tensor:
    """Power iteration of T from the uniform distribution (T^(2^squarings) by repeated squaring)."""
    M = T
    for _ in range(squarings):
        M = M @ M
    pi = torch.full((T.shape[0],), 1.0 / T.shape[0], dtype=T.dtype) @ M
    return pi / pi.sum()


def hmm_tables(T: torch.Tensor, B: torch.Tensor, pi: torch.Tensor | None = None, pair_lags=hq_init.PAIR_LAGS,
               triple_lags=hq_init.TRIPLE_LAGS):
    """Exact pair and triple tables of the HMM started in pi (default: stationary), same lags as hq_init."""
    pi = stationary(T) if pi is None else pi
    first = B.T * pi[None]                                               # (a, i) = P(y_t = a, s_t = i)
    pairs, X = [], first
    for k in range(1, max(pair_lags) + 1):
        X = X @ T
        if k in pair_lags:
            pairs.append(X @ B)
    trips = []
    for k1, k2 in triple_lags:
        X = first @ torch.linalg.matrix_power(T, k1)
        Y = (X[:, None, :] * B.T[None]) @ torch.linalg.matrix_power(T, k2)
        trips.append(Y @ B)
    return torch.stack(pairs), torch.stack(trips)


def table_kl(P, T3, tabs) -> torch.Tensor:
    kl = lambda e, m: (e * (torch.log(e.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum()
    return kl(tabs["pairs"], P) + kl(tabs["triples"], T3)


def fit_hmm_mom(tabs: dict, n: int, n_obs: int, restarts: int = 48, iters: int = 150, init_seed0: int = 0,
                scale: float = 1.0) -> dict:
    """L-BFGS on the moment loss from `restarts` random logits (N(0, scale^2), seeds init_seed0 + i); keep the lowest
    moment loss. Uses the given (training) tables only."""
    t0 = time.process_time()
    fits = []
    for i in range(restarts):
        g = torch.Generator().manual_seed(init_seed0 + i)
        tl = (torch.randn(n, n, generator=g, dtype=F64) * scale).requires_grad_()
        bl = (torch.randn(n, n_obs, generator=g, dtype=F64) * scale).requires_grad_()
        opt = torch.optim.LBFGS([tl, bl], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            loss = table_kl(*hmm_tables(tl.softmax(1), bl.softmax(1)), tabs)
            loss.backward()
            return loss
        try:
            opt.step(closure)
            with torch.no_grad():
                T, B = tl.softmax(1), bl.softmax(1)
                loss, pi = float(table_kl(*hmm_tables(T, B), tabs)), stationary(T)
            ok = np.isfinite(loss) and bool(torch.isfinite(T).all() and torch.isfinite(B).all())
        except Exception:                                                  # a failed restart is recorded, not fatal
            ok = False
        fits.append((loss, i, (T.numpy(), B.numpy(), pi.numpy())) if ok else (float("inf"), i, None))
    good = sorted((f for f in fits if f[2] is not None), key=lambda f: f[0])
    if not good:
        raise RuntimeError("every HMM MoM restart failed")
    best = good[0]
    T, B, pi = best[2]
    return {"T": T, "B": B, "pi": pi, "mom_loss": best[0], "selected_restart": best[1],
            "init_seed": init_seed0 + best[1], "losses_sorted": [f[0] for f in good],
            "n_failed": restarts - len(good), "cpu_s": time.process_time() - t0}


def train_ll(h: hmm.HMM, obs, act) -> float:
    """Exact training log-likelihood at the current parameters (hmm.HMM.fit stores the one before its last M-step)."""
    _, _, c = h._forward_backward(obs, act)
    return float(np.log(c).sum())


def val_nll(h: hmm.HMM, va_obs, K: int = 20, ctx: int = 32) -> float:
    o = va_obs[:1000]
    pv = h.predictive(o, np.zeros_like(o), K)
    y = o[:, ctx:ctx + K]
    return float(-np.log(np.clip(np.take_along_axis(pv, y[..., None], -1), 1e-9, 1)).mean())


def _obs(x):
    return np.asarray(x[0] if isinstance(x, (tuple, list)) else x)


def fit_HMM(tr, va, te, O: int, n: int, seed: int, restarts: int = 48, em_iters: int = 100, tol: float = 1e-6,
            mom_iters: int = 150, ctx: int = 32, K: int = 20) -> dict:
    """HMM-C: MoM-initialised EM plus 3 random-init EM fits, all on every training sequence; best training LL wins.
    tr/va/te: (obs, act) or obs (actions are all zero)."""
    torch.set_num_threads(1)
    assert hmm.L_CTX == ctx, "hmm.L_CTX must equal the E3 context length"
    t0 = time.process_time()
    tro, vao, teo = _obs(tr), _obs(va), _obs(te)
    tra = np.zeros_like(tro)
    mom = fit_hmm_mom(hq_init.empirical_tables(tro, O), n, O, restarts, mom_iters, init_seed(seed, RECIPE_HMM_MOM, 0))
    h = hmm.HMM(n_states=n, n_obs=O, n_actions=1, seed=mom["init_seed"])
    h.P[0], h.B, h.pi = mom["T"].copy(), mom["B"].copy(), mom["pi"].copy()
    mom_val = val_nll(h, vao, K, ctx)
    pred_mom = h.predictive(teo, np.zeros_like(teo), K)
    fits = [("mom", h.fit(tro, tra, iters=em_iters, tol=tol))]
    for r in range(3):
        h = hmm.HMM(n_states=n, n_obs=O, n_actions=1, seed=init_seed(seed, RECIPE_HMM_EM, r))
        fits.append((f"em{r}", h.fit(tro, tra, iters=em_iters, tol=tol)))
    lls = [train_ll(h, tro, tra) for _, h in fits]
    vals = [val_nll(h, vao, K, ctx) for _, h in fits]
    i = int(np.argmax(lls))
    cpu = time.process_time() - t0
    best = fits[i][1]
    pred = best.predictive(teo, np.zeros_like(teo), K)
    assert np.allclose(pred.sum(-1), 1, atol=1e-5), "HMM predictive does not sum to 1"
    return {"pred": pred, "val_nll": vals[i], "n_params": models_e2.hmm_params(n, O), "cpu_s": cpu,
            "winner": fits[i][0], "fit_names": [k for k, _ in fits], "train_ll": lls,
            "train_ll_em": [float(h.train_ll) for _, h in fits], "fit_val_nll": vals,
            "n_fit": [int(tro.shape[0])] * len(fits), "mom_loss": mom["mom_loss"],
            "mom_losses_sorted": mom["losses_sorted"], "mom_n_failed": mom["n_failed"],
            "mom_selected_restart": mom["selected_restart"], "mom_cpu_s": mom["cpu_s"], "mom_val_nll": mom_val,
            "pred_mom_only": pred_mom}

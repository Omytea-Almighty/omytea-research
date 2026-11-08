"""E3 OOM-C baseline (E3-DESIGN §3 OOM evaluation variant, §4 OOM-C, §7 PC-O, §6 H7, D2): observable operator model.

  state w (r,), normaliser sigma = e1, learned initial state w0 with w0[0] = 1 (r - 1 free);
  operators A_y (r, r); the first row of A_{O-1} is set so that sigma^T sum_y A_y = sigma^T exactly.
  Free parameters O r^2 - 1 (A1 r = 3: 17; A2 r = 9: 242).
  filtering  w <- A_y w / (sigma^T A_y w);  prediction  p_k(y) = sigma^T A_y S^{k-1} w,  S = sum_y A_y.

Raw outputs can be negative. Registered variant: p = clamp_min(p_raw, 1e-6), renormalised per (seq, k); it is used for
the training loss, the validation selection and the reported predictive. NR = fraction of raw entries < 0.
Initialisations, both from training sequences only: Hankel spectral learning (prefix/suffix words of length <= 2, SVD
truncated to r, then a basis change that makes sigma = e1) and the method of moments on the hq_init pair/triple
tables (48 random restarts from random r-state HMMs written as OOMs, L-BFGS, selection by moment loss alone).
"""
from __future__ import annotations

import copy
import itertools
import time

import numpy as np
import torch
import torch.nn as nn

from . import hq_init

F64 = torch.float64
P_MIN = 1e-6                      # registered clamp of the evaluation/training variant
DEN_EPS = 1e-8                    # sign-preserving guard on the filtering normaliser sigma^T A_y w
RECIPE_OOM_MOM = 6


def init_seed(seed: int, recipe: int, r: int) -> int:
    return 10 ** 7 + 10 ** 4 * seed + 10 ** 3 * recipe + r


class OOMWM(nn.Module):
    kind = "oom"

    def __init__(self, n_obs: int, r: int, ctx: int = 32):
        super().__init__()
        self.n_obs, self.r, self.ctx = n_obs, r, ctx
        self.w_free = nn.Parameter(torch.zeros(r - 1, dtype=F64))
        self.a_head = nn.Parameter(torch.zeros(n_obs - 1, r, r, dtype=F64))
        self.a_last = nn.Parameter(torch.zeros(r - 1, r, dtype=F64))     # rows 1..r-1 of A_{O-1}

    def w0(self):
        return torch.cat([torch.ones(1, dtype=F64), self.w_free])

    def ops(self):
        return assemble(self.a_head, self.a_last)

    def load(self, w0, A) -> None:
        """Write (w0, A) given in the sigma = e1 basis; w0 is rescaled to w0[0] = 1 and A_{O-1}[0] is re-derived."""
        w0, A = torch.as_tensor(np.asarray(w0), dtype=F64), torch.as_tensor(np.asarray(A), dtype=F64)
        with torch.no_grad():
            self.w_free.copy_(w0[1:] / w0[0])
            self.a_head.copy_(A[:-1])
            self.a_last.copy_(A[-1, 1:])

    def forward(self, obs, act=None, K: int = 20, raw: bool = False):
        A = self.ops()
        B = obs.shape[0]
        w = self.w0().expand(B, self.r)
        for t in range(self.ctx):
            v = torch.einsum("bij,bj->bi", A[obs[:, t]], w)
            den = v[:, 0]
            den = torch.where(den >= 0, den.clamp_min(DEN_EPS), den.clamp_max(-DEN_EPS))
            w = v / den[:, None]
        S, rows = A.sum(0), A[:, 0, :]                                   # rows[y] = sigma^T A_y
        out = []
        for _ in range(K):
            out.append(w @ rows.T)
            w = w @ S.T
        p = torch.stack(out, 1)
        if raw:
            return p
        p = p.clamp_min(P_MIN)
        return p / p.sum(-1, keepdim=True)


def assemble(a_head, a_last):
    e1 = torch.zeros(a_head.shape[-1], dtype=a_head.dtype)
    e1[0] = 1
    first = (e1 - a_head[:, 0].sum(0)).unsqueeze(0)
    return torch.cat([a_head, torch.cat([first, a_last], 0).unsqueeze(0)], 0)


def n_params(n_obs: int, r: int) -> int:
    return n_obs * r * r - 1


def to_e1(sigma, w0, A):
    """Basis change M with first row sigma^T (rest: orthonormal complement), so that sigma^T M^{-1} = e1^T."""
    sigma, w0, A = (np.asarray(x, dtype=float) for x in (sigma, w0, A))
    Q = np.linalg.svd(sigma[None])[2][1:]
    M = np.vstack([sigma, Q])
    Mi = np.linalg.inv(M)
    w = M @ w0
    return w / w[0], np.einsum("ij,yjk,kl->yil", M, A, Mi)


# ---------------------------------------------------------------- spectral (Hankel) initialisation
def words(n_obs: int, maxlen: int):
    return [w for L in range(maxlen + 1) for w in itertools.product(range(n_obs), repeat=L)]


def empirical_word_probs(obs: np.ndarray, n_obs: int, maxlen: int):
    """Word frequencies pooled over all positions (the sequences are stationary); returns word-tuple -> probability."""
    obs = np.asarray(obs)
    T = obs.shape[1]
    tabs = {0: np.ones(1)}
    for L in range(1, maxlen + 1):
        code = np.zeros((obs.shape[0], T - L + 1), dtype=np.int64)
        for i in range(L):
            code = code * n_obs + obs[:, i:T - L + 1 + i]
        f = np.bincount(code.ravel(), minlength=n_obs ** L).astype(float)
        tabs[L] = f / f.sum()

    def prob(word):
        c = 0
        for x in word:
            c = c * n_obs + int(x)
        return float(tabs[len(word)][c])
    return prob


def spectral(prob, n_obs: int, r: int, maxlen: int = 2):
    """Hankel spectral learning (Hsu-Kakade-Zhang form) from a word-probability function. Returns (w0, A, sv) in the
    sigma = e1 basis; P(x_1..x_t) = sigma^T A_{x_t} ... A_{x_1} w0."""
    W = words(n_obs, maxlen)
    H = np.array([[prob(p + s) for p in W] for s in W])                  # rows suffixes, columns prefixes
    Hy = np.array([[[prob(p + (y,) + s) for p in W] for s in W] for y in range(n_obs)])
    U, sv, _ = np.linalg.svd(H)
    U = U[:, :r]
    F = np.linalg.pinv(U.T @ H)                                          # (prefixes, r)
    b1, binf = U.T @ H[:, 0], F.T @ H[0, :]
    A = np.einsum("si,ysp,pj->yij", U, Hy, F)
    w0, A = to_e1(binf, b1, A)
    return w0, A, sv


def spectral_init(train_obs: np.ndarray, r: int, n_obs: int, maxlen: int = 2):
    return spectral(empirical_word_probs(train_obs, n_obs, 2 * maxlen + 1), n_obs, r, maxlen)


# ---------------------------------------------------------------- method of moments
def stationary(A: torch.Tensor, squarings: int = 9) -> torch.Tensor:
    """Power iteration of S = sum_y A_y from e1 (S^(2^squarings) e1 by repeated squaring), normalised by sigma = e1."""
    M = A.sum(0)
    for _ in range(squarings):
        M = M @ M
    w = M[:, 0]
    return w / w[0]


def oom_tables(A: torch.Tensor, pair_lags=hq_init.PAIR_LAGS, triple_lags=hq_init.TRIPLE_LAGS):
    """Pair and triple tables of the OOM started in its stationary state (same lags as hq_init.model_tables)."""
    S, rows = A.sum(0), A[:, 0, :]
    first = A @ stationary(A)                                            # (a, r): A_a w_s
    pairs, X = [], first
    for k in range(1, max(pair_lags) + 1):
        if k > 1:
            X = X @ S.T
        if k in pair_lags:
            pairs.append(X @ rows.T)
    trips = []
    for k1, k2 in triple_lags:
        X = first @ torch.linalg.matrix_power(S, k1 - 1).T
        Y = torch.einsum("bij,aj->abi", A, X) @ torch.linalg.matrix_power(S, k2 - 1).T
        trips.append(Y @ rows.T)
    return torch.stack(pairs), torch.stack(trips)


def table_kl(P, T3, tabs) -> torch.Tensor:
    kl = lambda e, m: (e * (torch.log(e.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum()
    return kl(tabs["pairs"], P) + kl(tabs["triples"], T3)


def random_oom(n_obs: int, r: int, seed: int):
    """A random r-state HMM written as an OOM (A_y = T^T diag(B[:, y]), sigma = 1), moved to the sigma = e1 basis."""
    rng = np.random.default_rng(seed)
    T, B = rng.dirichlet(np.ones(r), size=r), rng.dirichlet(np.ones(n_obs), size=r)
    A = np.stack([T.T * B[:, y][None] for y in range(n_obs)])
    return to_e1(np.ones(r), np.ones(r) / r, A)


def fit_oom_mom(tabs: dict, n_obs: int, r: int, restarts: int = 48, iters: int = 150, init_seed0: int = 0) -> dict:
    """L-BFGS on the moment loss from `restarts` random OOMs (seeds init_seed0 + i); keep the lowest moment loss.
    Uses the given (training) tables only. w0 of the result is the model's stationary state."""
    t0 = time.process_time()
    fits = []
    for i in range(restarts):
        _, A0 = random_oom(n_obs, r, init_seed0 + i)
        head = torch.tensor(A0[:-1], dtype=F64).requires_grad_()
        last = torch.tensor(A0[-1, 1:], dtype=F64).requires_grad_()
        opt = torch.optim.LBFGS([head, last], lr=1, max_iter=iters, history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            loss = table_kl(*oom_tables(assemble(head, last)), tabs)
            loss.backward()
            return loss
        try:
            opt.step(closure)
            with torch.no_grad():
                A = assemble(head, last)
                loss, w = float(table_kl(*oom_tables(A), tabs)), stationary(A)
            ok = np.isfinite(loss) and bool(torch.isfinite(A).all() and torch.isfinite(w).all())
        except Exception:                                                  # a failed restart is recorded, not fatal
            ok = False
        fits.append((loss, i, (w.numpy(), A.numpy())) if ok else (float("inf"), i, None))
    good = sorted((f for f in fits if f[2] is not None), key=lambda f: f[0])
    if not good:
        raise RuntimeError("every OOM MoM restart failed")
    best = good[0]
    return {"w0": best[2][0], "A": best[2][1], "mom_loss": best[0], "selected_restart": best[1],
            "losses_sorted": [f[0] for f in good], "n_failed": restarts - len(good), "cpu_s": time.process_time() - t0}


# ---------------------------------------------------------------- training, prediction, selection
def predict(model: OOMWM, obs: np.ndarray, K: int = 20, batch: int = 500):
    """Returns (clamped-renormalised, raw) predictives, both (n, K, O) numpy."""
    cl, rw = [], []
    with torch.no_grad():
        for i in range(0, obs.shape[0], batch):
            p = model(torch.as_tensor(obs[i:i + batch]), None, K, raw=True)
            q = p.clamp_min(P_MIN)
            cl.append((q / q.sum(-1, keepdim=True)).numpy())
            rw.append(p.numpy())
    return np.concatenate(cl), np.concatenate(rw)


def val_nll(model: OOMWM, va_obs: np.ndarray, K: int = 20) -> float:
    pv, _ = predict(model, va_obs[:1000], K)
    yv = va_obs[:1000, model.ctx:model.ctx + K]
    v = float(-np.log(np.clip(np.take_along_axis(pv, yv[..., None], -1), 1e-9, 1)).mean())
    return v if np.isfinite(v) else float("inf")


def finetune(model: OOMWM, tr_obs: np.ndarray, K: int = 20, lr: float = 3e-3, steps: int = 3000, batch: int = 64,
             seed: int = 0, ckpt_every: int = 100) -> dict:
    """E2 multi-step NLL on the clamped-renormalised outputs; Adam, linear decay to 0, clip 5, last checkpoint.
    A non-finite loss or gradient stops training and restores the last finite checkpoint (nan_flag)."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1 - s / steps)
    obs = torch.as_tensor(tr_obs)
    y = obs[:, model.ctx:model.ctx + K]
    ckpt, nan_flag, curve = copy.deepcopy(model.state_dict()), False, []
    for step in range(1, steps + 1):
        idx = torch.randint(0, obs.shape[0], (batch,), generator=g)
        p = model(obs[idx], None, K)
        loss = -torch.log(torch.gather(p, 2, y[idx].unsqueeze(-1)).clamp_min(1e-9)).mean()
        opt.zero_grad()
        if not torch.isfinite(loss):
            nan_flag = True
            break
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(gn):
            nan_flag = True
            break
        opt.step()
        sched.step()
        if step % ckpt_every == 0 or step == steps:
            if all(torch.isfinite(t).all() for t in model.parameters()):
                ckpt = copy.deepcopy(model.state_dict())
            else:
                nan_flag = True
                break
            curve.append((step, float(loss.detach())))
    if nan_flag:
        model.load_state_dict(ckpt)
    return {"nan_flag": nan_flag, "train_curve": curve}


def _obs(x):
    return np.asarray(x[0] if isinstance(x, (tuple, list)) else x)


def fit_OOM(tr, va, te, O: int, r: int, seed: int, restarts: int = 48, steps: int = 3000, mom_iters: int = 150,
            ctx: int = 32, K: int = 20, lr: float = 3e-3) -> dict:
    """OOM-C: candidates {spectral, spectral + fine-tune, MoM + fine-tune}, selected by validation NLL (first 1000
    validation sequences, horizons 1..K). Fine-tune batches use the run seed (as recipe M). tr/va/te: (obs, act) or obs."""
    torch.set_num_threads(1)
    t0 = time.process_time()
    tro, vao, teo = _obs(tr), _obs(va), _obs(te)
    cands, diag = {}, {}
    w0, A, sv = spectral_init(tro, r, O)
    m = OOMWM(O, r, ctx)
    m.load(w0, A)
    cands["spectral"] = (m, {"nan_flag": False, "train_curve": []})
    m2 = copy.deepcopy(m)
    cands["spectral_ft"] = (m2, finetune(m2, tro, K, lr, steps, 64, seed))
    try:
        mom = fit_oom_mom(hq_init.empirical_tables(tro, O), O, r, restarts, mom_iters, init_seed(seed, RECIPE_OOM_MOM, 0))
        m3 = OOMWM(O, r, ctx)
        m3.load(mom["w0"], mom["A"])
        diag.update(mom_loss=mom["mom_loss"], mom_losses_sorted=mom["losses_sorted"], mom_n_failed=mom["n_failed"],
                    mom_selected_restart=mom["selected_restart"], mom_cpu_s=mom["cpu_s"], mom_init_val_nll=val_nll(m3, vao, K))
        cands["mom_ft"] = (m3, finetune(m3, tro, K, lr, steps, 64, seed))
    except RuntimeError:
        diag.update(mom_loss=None, mom_losses_sorted=[], mom_n_failed=restarts)
    vals = {k: val_nll(mm, vao, K) for k, (mm, _) in cands.items()}
    winner = min(vals, key=lambda k: (vals[k], list(vals).index(k)))
    cpu = time.process_time() - t0
    cand_pred, cand_nr = {}, {}
    for k, (mm, _) in cands.items():
        pc, pr = predict(mm, teo, K)
        cand_pred[k], cand_nr[k] = pc, float((pr < 0).mean())
    pred, pred_raw = predict(cands[winner][0], teo, K)
    assert np.allclose(pred.sum(-1), 1, atol=1e-5), "OOM predictive does not sum to 1"
    return {"pred": pred, "pred_raw": pred_raw, "nr": float((pred_raw < 0).mean()), "val_nll": vals[winner],
            "n_params": n_params(O, r), "cpu_s": cpu, "winner": winner, "cand_val_nll": vals,
            "cand_nan_flag": {k: v[1]["nan_flag"] for k, v in cands.items()}, "cand_nr": cand_nr,
            "cand_pred": cand_pred, "nan_flag": cands[winner][1]["nan_flag"],
            "spectral_singular_values": [float(x) for x in sv], **diag}

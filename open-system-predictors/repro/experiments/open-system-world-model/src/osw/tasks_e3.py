"""E3 processes, data, metric and reference predictors (design: E3-DESIGN.zh.md §2, §3, §4 REF, §9).

  * A1, A2 reuse the E2 processes and data codes, so data(A1|A2, seed) equals run_e2.data bit for bit.
  * A2w is the weaker qutrit clock (eps = 0.15); A3 the d = 4 qudit clock with eps bisected to relaxation time 20.
  * CF = 1 - mean_{k<=10} KL(exact||model) / mean_{k<=10} KL(exact||marginal), computed by queue_e2.metric on
    run.metrics KLs and the E2 kl_marginal; every prediction must sum to 1 +- 1e-5 before the metric is computed.
  * REF: training-count Markov-1/2/3 (add-1/2), the linear-context (softmax) predictor, and the
    context-shuffled exact predictor (negative control).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sl
import torch

from . import queue_e2, tasks_e2
from .run import metrics

L_CTX, K_EVAL = tasks_e2.L_CTX, 20
A3_EPS = 0.2812632554285184
PROC = {"A1": dict(code=1, n_train=20_000, n_val=2_000, n_test=2_000),
        "A2": dict(code=2, n_train=20_000, n_val=2_000, n_test=2_000),
        "A2w": dict(code=3, n_train=120_000, n_val=2_000, n_test=10_000),
        "A3": dict(code=4, n_train=40_000, n_val=2_000, n_test=4_000)}


def qudit_clock(d: int, eps: float, theta: float = 0.6) -> tasks_e2.HQMMProcess:
    F = np.exp(2j * np.pi * np.outer(np.arange(d), np.arange(d)) / d) / np.sqrt(d)
    U = sl.expm(-1j * theta * (F @ np.diag(np.sqrt(np.arange(d))) @ F.conj().T))
    Ks = []
    for y in range(d):
        E = (1 - eps) * np.eye(d) / d
        E[y, y] += eps
        Ks.append(sl.sqrtm(E) @ U)
    return tasks_e2.HQMMProcess(f"qudit{d}-clock(eps={eps:g})", tuple(Ks))


def eps_for_relaxation(d: int, target: float = 20.0, theta: float = 0.6, lo=0.05, hi=0.95, iters=60) -> float:
    """Relaxation time decreases with eps; bisection on [lo, hi]."""
    for _ in range(iters):
        m = (lo + hi) / 2
        if qudit_clock(d, m, theta).relaxation_time() > target:
            lo = m
        else:
            hi = m
    return (lo + hi) / 2


def process(name: str) -> tasks_e2.HQMMProcess:
    return {"A1": tasks_e2.probability_clock, "A2": tasks_e2.qutrit_clock,
            "A2w": lambda: tasks_e2.qutrit_clock(eps=0.15), "A3": lambda: qudit_clock(4, A3_EPS)}[name]()


def data(name: str, seed: int, n_test: int | None = None):
    """(tr, va, te, exact); tr/va/te are (obs, act) pairs of length L_CTX + 20; exact is (n_test, 20, O).
    n_test overrides PROC's test size (the G8 lock: A2w 20k, A3 8k); train and validation draws are unchanged."""
    cfg, proc = dict(PROC[name]), process(name)
    if n_test is not None:
        cfg["n_test"] = int(n_test)
    rng = np.random.default_rng(10_000 * seed + cfg["code"])
    tr, va, te = (tasks_e2.sample(proc, cfg[k], K_EVAL, rng) for k in ("n_train", "n_val", "n_test"))
    return tr, va, te, tasks_e2.exact_predictive(proc, te[0], K_EVAL)


def assert_normalised(pred: np.ndarray, tol: float = 1e-5):
    s = np.asarray(pred).sum(-1)
    bad = ~(np.abs(s - 1) <= tol)                                  # NaN counts as bad
    if bad.any():
        raise ValueError(f"prediction not normalised: {int(bad.sum())} (seq, k) cells, max |sum-1| = {np.nanmax(np.abs(s - 1))}")


def kl_marginal(exact: np.ndarray) -> np.ndarray:
    marg = exact[:, -1].mean(0)                                    # verbatim run_e2.main lines 126-128
    q = np.clip(exact, 1e-12, 1)
    return (q * np.log(q / marg)).sum(-1).mean(0)


def cf_and_kl(pred, exact, y, check: bool = True) -> dict:
    if check:
        assert_normalised(pred)
    kl, klm = metrics(pred, exact, y)["kl"], kl_marginal(exact)
    cf = queue_e2.metric({"arm": "A", "per_k": {"kl": kl, "kl_marginal": klm}})
    return {"cf": float(cf), "kl": [float(x) for x in kl], "kl_marginal": [float(x) for x in klm]}


def stationary_marginal(proc: tasks_e2.HQMMProcess) -> np.ndarray:
    r = proc.stationary()
    return np.array([np.trace(k @ r @ k.conj().T).real for k in proc.K])


def denominators(name: str, n: int = 20_000, seed: int = 290) -> float:
    """Mean kl_marginal over k <= 10 on a fresh sample (stream [seed, code], disjoint from data())."""
    proc = process(name)
    obs, _ = tasks_e2.sample(proc, n, K_EVAL, np.random.default_rng([seed, PROC[name]["code"]]))
    return float(kl_marginal(tasks_e2.exact_predictive(proc, obs, K_EVAL))[:10].mean())


def _normalise(p):
    p = np.clip(p, 0, None)
    return p / p.sum(-1, keepdims=True)


def markov_k(tr_obs, te_obs, order: int, K: int, n_obs: int, ctx: int = L_CTX) -> np.ndarray:
    """Order-m Markov chain from all transitions in the training sequences (add-1/2), rolled out K steps
    by exact marginalisation over the m-gram state, starting from the last m context symbols."""
    m, S = order, n_obs ** order
    tr = np.asarray(tr_obs)
    codes = sum(tr[:, i:tr.shape[1] - m + 1 + i] * n_obs ** (m - 1 - i) for i in range(m))
    C = np.full((S, n_obs), 0.5)
    np.add.at(C, (codes[:, :-1].ravel(), tr[:, m:].ravel()), 1)
    P = C / C.sum(1, keepdims=True)
    T = np.zeros((S, S))
    for s in range(S):
        T[s, (s * n_obs + np.arange(n_obs)) % S] += P[s]
    te = np.asarray(te_obs)
    b = np.eye(S)[sum(te[:, ctx - m + i] * n_obs ** (m - 1 - i) for i in range(m))]
    out = np.zeros((len(te), K, n_obs))
    for k in range(K):
        out[:, k] = b @ P
        b = b @ T
    return _normalise(out)


def linear_context(tr_obs, te_obs, K: int, n_obs: int, ctx: int = L_CTX, steps: int = 200, soft=None) -> np.ndarray:
    """Per-horizon multinomial logistic regression on the one-hot of all ctx context symbols, fitted by L-BFGS
    (strong Wolfe, float64, no penalty) on the training hard targets y_{ctx+k-1}, k = 1..K. soft: optional
    (n_train, K, n_obs) target distributions (e.g. the exact predictive) for the population ceiling instead."""
    torch.set_num_threads(1)
    tr, te = np.asarray(tr_obs), np.asarray(te_obs)

    def feats(o):
        return torch.as_tensor(np.eye(n_obs)[o[:, :ctx]].reshape(len(o), ctx * n_obs))
    X, Y = feats(tr), torch.as_tensor(tr[:, ctx:ctx + K])
    Q = None if soft is None else torch.as_tensor(np.asarray(soft, dtype=np.float64))
    W = torch.zeros(ctx * n_obs, K * n_obs, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(K * n_obs, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], lr=1, max_iter=steps, history_size=20, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        logp = torch.log_softmax((X @ W + b).view(-1, K, n_obs), -1)
        loss = -torch.gather(logp, 2, Y.unsqueeze(-1)).mean() if soft is None else -(Q * logp).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        p = torch.softmax((feats(te) @ W + b).view(-1, K, n_obs), -1).numpy()
    return _normalise(p)


def shuffled_exact(proc: tasks_e2.HQMMProcess, te_obs, K: int, rng: np.random.Generator) -> np.ndarray:
    """Exact predictive of a context taken from a randomly permuted test sequence (negative control)."""
    te = np.asarray(te_obs)
    return _normalise(tasks_e2.exact_predictive(proc, te[rng.permutation(len(te))], K))

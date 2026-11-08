"""T1 synthetic partially observed Markov processes with an exact Bayes reference.

Pre-registration §2 (frozen 2026-10-19, SHA-256 37172954...):
  * S = 8 hidden states, O = 4 observation symbols, emission B (8x4) of moderate information.
  * T1-simple:      P = lam*C + (1-lam)*Pi, C the 8-cycle, Pi = 1 pi^T with pi uniform, lam = exp(-1/tau).
                    Because C Pi = Pi C = Pi, P^k = lam^k C^k + (1-lam^k) Pi (closed form).
  * T1-metastable:  two blocks of 4 states; fast rotation+relaxation inside a block (tau_fast = 3),
                    slow switching to the global uniform (tau_slow); P^k by matrix power.
  * T1-action:      action a in {0, 1} picks the rotation direction (C or C^T) in T1-simple.
  * Protocol: start from the stationary distribution, context L = 32 observations, then blind
    rollout: predict the observation distribution k = 1..K steps ahead with no new observations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

S, O, L_CTX = 8, 4, 32


def emission_matrix() -> np.ndarray:
    """State s emits symbol s mod 4 with probability 0.7, each other symbol 0.1.
    States s and s+4 are therefore indistinguishable from one observation (partial observability);
    the order of symbols along the cycle carries the information."""
    B = np.full((S, O), 0.1)
    for s in range(S):
        B[s, s % O] = 0.7
    return B


def emission_metastable() -> np.ndarray:
    """DEVIATIONS.md D6: block identity must be observable, otherwise the slow inter-block dynamics are invisible.
    Block A (states 0-3) emits symbol (s mod 2) w.p. 0.7, block B (states 4-7) emits 2 + (s mod 2) w.p. 0.7,
    each other symbol 0.1. Within-block position is partly visible through the parity of the symbol."""
    B = np.full((S, O), 0.1)
    for s in range(S):
        B[s, (s % 2) + (0 if s < 4 else 2)] = 0.7
    return B


def cycle(n: int, direction: int = 1) -> np.ndarray:
    return np.roll(np.eye(n), direction, axis=1)


@dataclass(frozen=True)
class Task:
    name: str
    P: tuple            # transition matrices, one per action (tuple of np.ndarray)
    B: np.ndarray       # emission matrix
    n_actions: int

    def step_matrix(self, a: int) -> np.ndarray:
        return self.P[a]


def t1_simple(tau: float) -> Task:
    lam = float(np.exp(-1.0 / tau))
    Pi = np.ones((S, S)) / S
    P = lam * cycle(S) + (1 - lam) * Pi
    return Task(f"T1-simple(tau={tau:g})", (P,), emission_matrix(), 1)


def t1_metastable(tau_slow: float, tau_fast: float = 3.0) -> Task:
    lam_f = float(np.exp(-1.0 / tau_fast))
    eps = 1.0 - float(np.exp(-1.0 / tau_slow))
    W = lam_f * cycle(4) + (1 - lam_f) * np.ones((4, 4)) / 4
    blk = np.zeros((S, S))
    blk[:4, :4] = W
    blk[4:, 4:] = W
    P = (1 - eps) * blk + eps * np.ones((S, S)) / S
    return Task(f"T1-metastable(tau_slow={tau_slow:g})", (P,), emission_metastable(), 1)


def t1_action(tau: float) -> Task:
    lam = float(np.exp(-1.0 / tau))
    Pi = np.ones((S, S)) / S
    P0 = lam * cycle(S, +1) + (1 - lam) * Pi
    P1 = lam * cycle(S, -1) + (1 - lam) * Pi
    return Task(f"T1-action(tau={tau:g})", (P0, P1), emission_matrix(), 2)


def stationary(task: Task) -> np.ndarray:
    return np.ones(S) / S   # all T1 chains are doubly stochastic -> uniform stationary law


def sample(task: Task, n: int, horizon: int, rng: np.random.Generator):
    """Sample n sequences: context of L_CTX observations then `horizon` future observations.
    Returns obs (n, L_CTX+horizon) int, actions (n, L_CTX+horizon) int (action taken before each step)."""
    T = L_CTX + horizon
    obs = np.zeros((n, T), dtype=np.int64)
    act = rng.integers(0, task.n_actions, size=(n, T)) if task.n_actions > 1 else np.zeros((n, T), dtype=np.int64)
    s = rng.choice(S, size=n, p=stationary(task))
    cumB = np.cumsum(task.B, axis=1)
    for t in range(T):
        if t > 0:
            P = np.stack(task.P)[act[:, t]]                       # (n, S, S)
            cum = np.cumsum(P[np.arange(n), s], axis=1)
            s = (rng.random(n)[:, None] > cum).sum(1)
        obs[:, t] = (rng.random(n)[:, None] > cumB[s]).sum(1)
    return obs, act


def bayes_filter(task: Task, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
    """Exact posterior over the hidden state after the L_CTX context observations. Returns (n, S)."""
    n = obs.shape[0]
    b = np.tile(stationary(task), (n, 1))
    for t in range(L_CTX):
        if t > 0:
            P = np.stack(task.P)[act[:, t]]
            b = np.einsum("ns,nst->nt", b, P)
        b = b * task.B[:, obs[:, t]].T
        b /= b.sum(1, keepdims=True)
    return b


def bayes_predictive(task: Task, obs: np.ndarray, act: np.ndarray, K: int) -> np.ndarray:
    """Exact k-step predictive observation distributions p(o_{L+k-1} | context, actions), k = 1..K.
    Returns (n, K, O). Step k uses the action recorded at time index L_CTX + k - 1."""
    b = bayes_filter(task, obs, act)
    out = np.zeros((obs.shape[0], K, O))
    for k in range(1, K + 1):
        P = np.stack(task.P)[act[:, L_CTX + k - 1]]
        b = np.einsum("ns,nst->nt", b, P)
        out[:, k - 1] = b @ task.B
    return out


def closed_form_power(task: Task, k: int) -> np.ndarray:
    """Closed form P^k for T1-simple (used by the unit tests)."""
    lam = (task.P[0][0, 1] - 1.0 / S) / (1 - 1.0 / S)
    Pi = np.ones((S, S)) / S
    return lam**k * np.linalg.matrix_power(cycle(S), k) + (1 - lam**k) * Pi

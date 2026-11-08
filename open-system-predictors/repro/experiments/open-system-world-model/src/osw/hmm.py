"""HMM baseline fitted by Baum-Welch (EM) — the true model class on T1, used as the 'how good is good' yardstick
(pre-registration §3). Action-conditioned transitions when the task has actions."""
from __future__ import annotations

import numpy as np

from .tasks import L_CTX, Task


class HMM:
    kind = "hmm"

    def __init__(self, n_states=8, n_obs=4, n_actions=1, seed=0):
        rng = np.random.default_rng(seed)
        self.S, self.O, self.A = n_states, n_obs, n_actions
        self.P = rng.dirichlet(np.ones(n_states), size=(n_actions, n_states))
        self.B = rng.dirichlet(np.ones(n_obs), size=n_states)
        self.pi = np.ones(n_states) / n_states

    def n_params(self):
        return self.A * self.S * (self.S - 1) + self.S * (self.O - 1) + (self.S - 1)

    def _forward_backward(self, obs, act):
        n, T = obs.shape
        alpha = np.zeros((n, T, self.S)); c = np.zeros((n, T))
        a = self.pi[None] * self.B[:, obs[:, 0]].T
        c[:, 0] = a.sum(1); alpha[:, 0] = a / c[:, 0, None]
        for t in range(1, T):
            a = np.einsum("ns,nst->nt", alpha[:, t - 1], self.P[act[:, t]]) * self.B[:, obs[:, t]].T
            c[:, t] = a.sum(1); alpha[:, t] = a / c[:, t, None]
        beta = np.ones((n, T, self.S))
        for t in range(T - 2, -1, -1):
            beta[:, t] = np.einsum("nst,nt->ns", self.P[act[:, t + 1]], self.B[:, obs[:, t + 1]].T * beta[:, t + 1]) / c[:, t + 1, None]
        return alpha, beta, c

    def fit(self, obs, act, iters=60, tol=1e-5):
        prev = -np.inf
        for _ in range(iters):
            alpha, beta, c = self._forward_backward(obs, act)
            ll = np.log(c).sum()
            gamma = alpha * beta
            gamma /= gamma.sum(2, keepdims=True)
            # transitions: xi_t(s, s') proportional to alpha_{t-1}(s) P(s,s') B(s', o_t) beta_t(s')
            num = np.zeros((self.A, self.S, self.S))
            for t in range(1, obs.shape[1]):
                w = self.B[:, obs[:, t]].T * beta[:, t] / c[:, t, None]                 # (n, S')
                xi = alpha[:, t - 1, :, None] * self.P[act[:, t]] * w[:, None, :]      # (n, S, S')
                for a_ in range(self.A):
                    m = act[:, t] == a_
                    if m.any():
                        num[a_] += xi[m].sum(0)
            self.P = (num + 1e-6) / (num + 1e-6).sum(2, keepdims=True)
            Bn = np.zeros((self.S, self.O))
            for o in range(self.O):
                Bn[:, o] = (gamma * (obs == o)[..., None]).sum((0, 1))
            self.B = (Bn + 1e-6) / (Bn + 1e-6).sum(1, keepdims=True)
            self.pi = gamma[:, 0].mean(0)
            if ll - prev < tol * abs(ll):
                break
            prev = ll
        self.train_ll = ll
        return self

    def predictive(self, obs, act, K):
        n = obs.shape[0]
        b = np.tile(self.pi, (n, 1))
        for t in range(L_CTX):
            if t > 0:
                b = np.einsum("ns,nst->nt", b, self.P[act[:, t]])
            b = b * self.B[:, obs[:, t]].T
            b /= b.sum(1, keepdims=True)
        out = np.zeros((n, K, self.O))
        for k in range(1, K + 1):
            b = np.einsum("ns,nst->nt", b, self.P[act[:, L_CTX + k - 1]])
            out[:, k - 1] = b @ self.B
        return out


def fit_best(task: Task, obs, act, seed, restarts=3):
    best = None
    for r in range(restarts):
        h = HMM(n_actions=task.n_actions, seed=seed * 101 + r).fit(obs, act)
        if best is None or h.train_ll > best.train_ll:
            best = h
    return best

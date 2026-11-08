"""E2 model families at several sizes, so prediction quality can be compared at matched parameter counts.

  HQ(d)    hidden quantum Markov model: one Kraus operator per observation (w = 1), all taken from one isometry
           V = G (G^dag G)^{-1/2} so that sum_y K_y^dag K_y = I (Stiefel parameterisation, as in Adhikary et al. 2020).
           Filtering rho <- K_y rho K_y^dag / p(y); prediction p(y_{t+k}) = Tr(K_y Phi^{k-1}(rho) K_y^dag).
           This is the canonical density-matrix sequence model. The T1 models (shared instrument + observation-
           independent dynamics) cannot represent observation-dependent Kraus operators, so they are not used here.
  GRU(h), LRU(n)  classical recurrent models (models.GRUWM / models.LRUWM with other sizes).
  HMM(n)   classical hidden Markov model fitted by EM (hmm.HMM), for the known HQMM-versus-HMM contrast.

The context length is passed explicitly (`ctx`), so nothing here depends on the T1 module constant.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from . import models
from .models import CDT, _c, _cparam, inv_sqrt_psd

EPS = 1e-8


class HQMMWM(nn.Module):
    kind = "density"

    def __init__(self, n_obs, d, ctx):
        super().__init__()
        self.n_obs, self.d, self.ctx = n_obs, d, ctx
        self.g_re, self.g_im = _cparam(n_obs * d, d, scale=0.5)

    def kraus(self):
        G = _c(self.g_re, self.g_im)
        V = G @ inv_sqrt_psd(G.conj().transpose(-1, -2) @ G)               # (O*d, d), V^dag V = I
        return V.reshape(self.n_obs, self.d, self.d)

    def forward(self, obs, act, K):
        B = obs.shape[0]
        Ks = self.kraus()
        rho = (torch.eye(self.d, dtype=CDT) / self.d).expand(B, self.d, self.d).clone()
        for t in range(self.ctx):
            Ky = Ks[obs[:, t]]                                              # (B, d, d)
            r = Ky @ rho @ Ky.conj().transpose(-1, -2)
            tr = torch.diagonal(r, dim1=-2, dim2=-1).sum(-1).real.clamp_min(EPS)
            rho = r / tr.to(CDT)[:, None, None]
        out = []
        for _ in range(K):
            p = torch.einsum("yij,bjk,yik->by", Ks, rho, Ks.conj()).real.clamp_min(EPS)
            out.append(p / p.sum(1, keepdim=True))
            rho = torch.einsum("yij,bjk,ylk->bil", Ks, rho, Ks.conj())
        return torch.stack(out, 1)


SIZES = {                     # the parameter ladder (sizes per family); counts are reported per run
    "HQ": [2, 3, 4, 6, 8],
    "GRU": [2, 4, 8, 16],
    "LRU": [2, 4, 8, 16],
    "HMM": [2, 4, 8, 16, 32],
}


def build(family: str, size: int, n_obs: int, ctx: int):
    if family == "HQ":
        return HQMMWM(n_obs, size, ctx)
    if family == "GRU":
        return models.GRUWM(n_obs, 1, hidden=size)
    if family == "LRU":
        return models.LRUWM(n_obs, 1, n_state=size, readout=max(4, size))
    raise ValueError(family)


def hmm_params(n, n_obs):
    return (n - 1) + n * (n - 1) + n * (n_obs - 1)                       # initial + transition + emission (free)

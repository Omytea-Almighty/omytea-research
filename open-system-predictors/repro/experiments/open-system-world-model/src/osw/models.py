"""World models compared in the pre-registration (§3). All share one interface:

    forward(obs (B,T) long, act (B,T) long, K) -> predictive (B, K, O) probabilities

The context is obs[:, :L_CTX]; step k of the blind rollout uses the action act[:, L_CTX + k - 1]
(the same convention as tasks.bayes_predictive).

Density-matrix models (U, L, K, C-ctrl) share one quantum instrument {M_o} with sum_o M_o^† M_o = I
(DEVIATIONS.md D3): readout p(o) = Tr(M_o rho M_o^†), update rho <- M_o rho M_o^† / p(o).
They differ only in the dynamics between observations:
  U(d_E)  unitary conjugation on system (x) environment, H(a) = H0 + a*H1  (UWM-JEPA predictor, arXiv 2605.25313 §3)
  L       GKSL / Lindblad channel exp(Liouvillian), H(a) = H0 + a*H1 plus J jump operators
  K       discrete Kraus channel (HQMM-style), one CPTP map per action
  C-ctrl  Phi(rho) = e^{-g} U rho U^† + (1 - e^{-g}) sigma   ("ordinary contraction" control)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tasks import L_CTX

CDT = torch.complex64
EPS = 1e-8


def _cparam(*shape, scale=0.3):
    return nn.Parameter(torch.randn(*shape) * scale), nn.Parameter(torch.randn(*shape) * scale)


def _c(re, im):
    return torch.complex(re, im)


def hermitian(re, im):
    A = _c(re, im)
    return 0.5 * (A + A.conj().transpose(-1, -2))


def inv_sqrt_psd(S):
    w, V = torch.linalg.eigh(S)
    w = w.clamp_min(1e-6)
    return (V * w.rsqrt().to(V.dtype).unsqueeze(-2)) @ V.conj().transpose(-1, -2)


def action_values(act, n_actions):
    """Map discrete actions to the scalar a in H(a) = H0 + a*H1: no-action tasks -> 0, two actions -> -1/+1."""
    if n_actions == 1:
        return torch.zeros_like(act, dtype=torch.float32)
    return act.float() * 2.0 - 1.0


class Instrument(nn.Module):
    """O measurement operators on a d-dimensional system with sum_o M_o^† M_o = I."""

    INIT = "identity"                       # "identity" (start uninformative) or "random"; set by build(init=...)

    def __init__(self, d, n_obs):
        super().__init__()
        self.re, self.im = _cparam(n_obs, d, d)
        if Instrument.INIT == "identity":
            with torch.no_grad():
                self.re.add_(torch.eye(d))     # start near the identity: an uninformative but valid instrument

    def ops(self):
        A = _c(self.re, self.im)
        S = (A.conj().transpose(-1, -2) @ A).sum(0)
        return A @ inv_sqrt_psd(S)          # (O, d, d)


class DensityWM(nn.Module):
    """Shared filtering / readout for density-matrix world models on a d_S (x) d_E space."""

    kind = "density"

    def __init__(self, n_obs, n_actions, d_s=8, d_e=1):
        super().__init__()
        self.n_obs, self.n_actions, self.d_s, self.d_e = n_obs, n_actions, d_s, d_e
        self.D = d_s * d_e
        self.instrument = Instrument(d_s, n_obs)

    # --- dynamics hook -------------------------------------------------------------------------
    def prepare(self):
        """Build per-call operators (e.g. matrix exponentials) once per forward pass."""
        return None

    def evolve(self, rho, a_idx, cache):
        raise NotImplementedError

    @staticmethod
    def per_action(fn, rho, a_idx, n_actions):
        """Apply fn(rho, a) for each action value and select per sample (no per-sample operator copies)."""
        if n_actions == 1:
            return fn(rho, 0)
        out = fn(rho, 0)
        for a in range(1, n_actions):
            m = (a_idx == a).view(-1, *([1] * (rho.dim() - 1)))
            out = torch.where(m, fn(rho, a), out)
        return out

    # --- shared machinery ----------------------------------------------------------------------
    def reduced(self, rho):
        B = rho.shape[0]
        r = rho.reshape(B, self.d_s, self.d_e, self.d_s, self.d_e)
        return torch.einsum("bieje->bij", r)

    def probs(self, rho, M):
        rs = self.reduced(rho)
        E = M.conj().transpose(-1, -2) @ M                     # (O, d_s, d_s) POVM elements
        p = torch.einsum("oij,bji->bo", E, rs).real
        p = p.clamp_min(EPS)
        return p / p.sum(1, keepdim=True)

    def update(self, rho, o, M):
        B = rho.shape[0]
        Mo = M[o]                                              # (B, d_s, d_s)
        r = rho.reshape(B, self.d_s, self.d_e, self.d_s, self.d_e)
        r = torch.einsum("bij,bjekf,blk->bielf", Mo, r, Mo.conj())
        rho = r.reshape(B, self.D, self.D)
        tr = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1).real.clamp_min(EPS)
        rho = rho / tr.to(rho.dtype)[:, None, None]
        return 0.5 * (rho + rho.conj().transpose(-1, -2))

    def forward(self, obs, act, K):
        B = obs.shape[0]
        M = self.instrument.ops()
        cache = self.prepare()
        rho = (torch.eye(self.D, dtype=CDT) / self.D).expand(B, self.D, self.D).clone()
        for t in range(L_CTX):
            if t > 0:
                rho = self.evolve(rho, act[:, t], cache)
            rho = self.update(rho, obs[:, t], M)
        out = []
        for k in range(1, K + 1):
            rho = self.evolve(rho, act[:, L_CTX + k - 1], cache)
            out.append(self.probs(rho, M))
        return torch.stack(out, 1)


class UnitaryWM(DensityWM):
    """UWM-JEPA predictor: rho <- U(a) rho U(a)^†, U(a) = exp(-i (H0 + a H1)) on system (x) environment."""

    def __init__(self, n_obs, n_actions, d_s=8, d_e=2):
        super().__init__(n_obs, n_actions, d_s, d_e)
        self.h0 = _cparam(self.D, self.D, scale=0.3 / math.sqrt(self.D))
        self.h1 = _cparam(self.D, self.D, scale=0.3 / math.sqrt(self.D)) if n_actions > 1 else None
        for i, p in enumerate(self.h0):
            self.register_parameter(f"h0_{i}", p)
        if self.h1:
            for i, p in enumerate(self.h1):
                self.register_parameter(f"h1_{i}", p)

    def prepare(self):
        H0 = hermitian(*self.h0)
        vals = [0.0] if self.n_actions == 1 else [-1.0, 1.0]
        Us = []
        for a in vals:
            H = H0 if self.h1 is None else H0 + a * hermitian(*self.h1)
            Us.append(torch.linalg.matrix_exp(-1j * H.to(CDT)))
        return torch.stack(Us)                                  # (n_actions, D, D)

    def evolve(self, rho, a_idx, Us):
        return self.per_action(lambda r, a: Us[a] @ r @ Us[a].conj().T, rho, a_idx, self.n_actions)


class LindbladWM(DensityWM):
    """GKSL channel exp(L) with L(rho) = -i[H(a), rho] + sum_j (L_j rho L_j^† - 1/2 {L_j^† L_j, rho})."""

    def __init__(self, n_obs, n_actions, d_s=8, n_jumps=3):
        super().__init__(n_obs, n_actions, d_s, 1)
        d = d_s
        self.h0_re, self.h0_im = _cparam(d, d, scale=0.3 / math.sqrt(d))
        if n_actions > 1:
            self.h1_re, self.h1_im = _cparam(d, d, scale=0.3 / math.sqrt(d))
        self.j_re, self.j_im = _cparam(n_jumps, d, d, scale=0.15 / math.sqrt(d))

    def liouvillian(self, H):
        """Row-major vectorisation: vec(A X B) = (A kron B^T) vec(X)."""
        d = self.d_s
        I = torch.eye(d, dtype=CDT)
        Lj = _c(self.j_re, self.j_im)
        Lsup = -1j * (torch.kron(H, I) - torch.kron(I, H.transpose(-1, -2).contiguous()))
        for j in range(Lj.shape[0]):
            Lk = Lj[j]
            LdL = Lk.conj().transpose(-1, -2) @ Lk
            Lsup = Lsup + torch.kron(Lk, Lk.conj().resolve_conj()) - 0.5 * torch.kron(LdL, I) - 0.5 * torch.kron(I, LdL.transpose(-1, -2).contiguous())
        return Lsup

    def prepare(self):
        H0 = hermitian(self.h0_re, self.h0_im).to(CDT)
        vals = [0.0] if self.n_actions == 1 else [-1.0, 1.0]
        chans = []
        for a in vals:
            H = H0 if self.n_actions == 1 else H0 + a * hermitian(self.h1_re, self.h1_im).to(CDT)
            chans.append(torch.linalg.matrix_exp(self.liouvillian(H)))
        return torch.stack(chans)                               # (n_actions, d^2, d^2)

    def evolve(self, rho, a_idx, chans):
        B, d = rho.shape[0], self.d_s
        rho = self.per_action(lambda r, a: (r.reshape(B, d * d) @ chans[a].T).reshape(B, d, d), rho, a_idx, self.n_actions)
        return 0.5 * (rho + rho.conj().transpose(-1, -2))


class KrausWM(DensityWM):
    """Discrete CPTP map per action: Kraus operators from an isometry V = G (G^† G)^{-1/2}."""

    def __init__(self, n_obs, n_actions, d_s=8, rank=4):
        super().__init__(n_obs, n_actions, d_s, 1)
        self.rank = rank
        self.g_re, self.g_im = _cparam(n_actions, rank * d_s, d_s, scale=0.3)
        with torch.no_grad():
            self.g_re[:, :d_s, :].add_(torch.eye(d_s) * 2.0)

    def prepare(self):
        G = _c(self.g_re, self.g_im)
        V = G @ inv_sqrt_psd(G.conj().transpose(-1, -2) @ G)        # (n_actions, r*d, d), V^†V = I
        return V.reshape(self.n_actions, self.rank, self.d_s, self.d_s)

    def evolve(self, rho, a_idx, kraus):
        rho = self.per_action(lambda r, a: torch.einsum("rij,bjk,rlk->bil", kraus[a], r, kraus[a].conj()),
                              rho, a_idx, self.n_actions)
        return 0.5 * (rho + rho.conj().transpose(-1, -2))


class ContractionWM(DensityWM):
    """Control: unitary rotation then contraction toward a learned fixed state sigma at rate g."""

    def __init__(self, n_obs, n_actions, d_s=8):
        super().__init__(n_obs, n_actions, d_s, 1)
        d = d_s
        self.h0_re, self.h0_im = _cparam(d, d, scale=0.3 / math.sqrt(d))
        if n_actions > 1:
            self.h1_re, self.h1_im = _cparam(d, d, scale=0.3 / math.sqrt(d))
        self.s_re, self.s_im = _cparam(d, d, scale=0.3)
        self.g = nn.Parameter(torch.tensor(-2.0))

    def prepare(self):
        H0 = hermitian(self.h0_re, self.h0_im).to(CDT)
        vals = [0.0] if self.n_actions == 1 else [-1.0, 1.0]
        Us = []
        for a in vals:
            H = H0 if self.n_actions == 1 else H0 + a * hermitian(self.h1_re, self.h1_im).to(CDT)
            Us.append(torch.linalg.matrix_exp(-1j * H))
        Bm = _c(self.s_re, self.s_im)
        sigma = Bm @ Bm.conj().transpose(-1, -2)
        sigma = sigma / torch.diagonal(sigma).sum().real
        keep = torch.exp(-F.softplus(self.g))
        return torch.stack(Us), sigma, keep

    def evolve(self, rho, a_idx, cache):
        Us, sigma, keep = cache
        rho = self.per_action(lambda r, a: Us[a] @ r @ Us[a].conj().T, rho, a_idx, self.n_actions)
        return keep.to(CDT) * rho + (1 - keep).to(CDT) * sigma


class GRUWM(nn.Module):
    """Classical recurrent baseline. Blind rollout feeds a 'no observation' token plus the action."""

    kind = "classical"

    def __init__(self, n_obs, n_actions, hidden=15):          # D9: 15 keeps GRU within +-10% of L's parameters
        super().__init__()
        self.n_obs, self.n_actions = n_obs, n_actions
        self.cell = nn.GRUCell(n_obs + 1 + n_actions, hidden)
        self.head = nn.Linear(hidden, n_obs)

    def _inp(self, o, a, blind):
        oh = F.one_hot(o, self.n_obs + 1).float() if not blind else F.one_hot(torch.full_like(o, self.n_obs), self.n_obs + 1).float()
        return torch.cat([oh, F.one_hot(a, self.n_actions).float()], 1)

    def forward(self, obs, act, K):
        B = obs.shape[0]
        h = torch.zeros(B, self.cell.hidden_size)
        for t in range(L_CTX):
            h = self.cell(self._inp(obs[:, t], act[:, t], False), h)
        out = []
        for k in range(1, K + 1):
            h = self.cell(self._inp(obs[:, 0], act[:, L_CTX + k - 1], True), h)
            out.append(F.softmax(self.head(h), -1))
        return torch.stack(out, 1)


class LRUWM(nn.Module):
    """Linear recurrent unit (diagonal complex, |lambda| < 1) with an MLP readout; blind rollout as in GRUWM."""

    kind = "classical"

    def __init__(self, n_obs, n_actions, n_state=20, readout=16):   # D9: 20 keeps LRU within +-10% of L
        super().__init__()
        self.n_obs, self.n_actions = n_obs, n_actions
        din = n_obs + 1 + n_actions
        self.nu = nn.Parameter(torch.log(-torch.log(torch.rand(n_state) * 0.5 + 0.5)))   # |lambda| in (0.5, 1)
        self.theta = nn.Parameter(torch.rand(n_state) * 2 * math.pi)
        self.b_re, self.b_im = _cparam(n_state, din, scale=1 / math.sqrt(din))
        self.head = nn.Sequential(nn.Linear(2 * n_state, readout), nn.Tanh(), nn.Linear(readout, n_obs))

    def _inp(self, o, a, blind):
        oh = F.one_hot(torch.full_like(o, self.n_obs) if blind else o, self.n_obs + 1).float()
        return torch.cat([oh, F.one_hot(a, self.n_actions).float()], 1)

    def forward(self, obs, act, K):
        B = obs.shape[0]
        lam = torch.exp(torch.complex(-torch.exp(self.nu), self.theta))
        Bm = _c(self.b_re, self.b_im)
        h = torch.zeros(B, lam.shape[0], dtype=CDT)
        for t in range(L_CTX):
            h = lam * h + self._inp(obs[:, t], act[:, t], False).to(CDT) @ Bm.T
        out = []
        for k in range(1, K + 1):
            h = lam * h + self._inp(obs[:, 0], act[:, L_CTX + k - 1], True).to(CDT) @ Bm.T
            out.append(F.softmax(self.head(torch.cat([h.real, h.imag], 1)), -1))
        return torch.stack(out, 1)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build(name: str, n_obs: int, n_actions: int, init: str = "identity") -> nn.Module:
    """Model registry. Sizes chosen so the density models sit near ~1k real parameters (pre-registration §3).
    init: instrument initialisation for density models ("identity" or "random"; DEVIATIONS.md D5)."""
    Instrument.INIT = init
    if name.startswith("U"):                     # e.g. "U2" -> d_E = 2
        return UnitaryWM(n_obs, n_actions, d_e=int(name[1:]))
    if name == "L":
        return LindbladWM(n_obs, n_actions)
    if name == "L1":                             # D9: one jump operator, parameter-matched to C for H3
        return LindbladWM(n_obs, n_actions, n_jumps=1)
    if name == "K":
        return KrausWM(n_obs, n_actions)
    if name == "C":
        return ContractionWM(n_obs, n_actions)
    if name == "GRU":
        return GRUWM(n_obs, n_actions)
    if name == "LRU":
        return LRUWM(n_obs, n_actions)
    raise ValueError(name)

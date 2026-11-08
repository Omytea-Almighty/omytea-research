"""Post hoc diagnostics of the trained U2 predictors (not registered, no retraining), added at release.

Uses the shipped confirmatory checkpoints runs/confirm/ckpt/{simple,metastable}-20-U2-s*.pt and answers two questions the
report would otherwise leave to argument:

1. Marginalised rollout. H1 concerns a blind rollout that applies the unitary between steps and never the non-selective
   channel of the observation instrument. Here the same trained weights are also rolled out with that channel applied
   after every blind step: rho -> sum_o M_o U rho U^dag M_o^dag. The windows and the Bayes reference are those of H1.
   A numpy version of the plain rollout is compared with the model's own forward pass, so the marginalised rollout runs
   on a checked implementation.
2. Returns. For the plain rollout of each trained predictor: dominant period of the predicted probabilities, the first
   return to within total-variation tolerance 0.1 or 0.05 of p_5 and of p_51 after first leaving it, the share of
   sequences that return by k = 200, and the size of the oscillation early and late.

  PYTHONPATH=src .venv/bin/python report/rollout_diagnostics.py      # writes runs/robustness/ROLLOUT_DIAGNOSTICS.json
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import torch

from osw import models, tasks

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/robustness/ROLLOUT_DIAGNOSTICS.json"
N_KL, H_KL, SEED_KL = 300, 200, 5
N_RET, H_RET, SEED_RET = 200, 1200, 3
TASKS = {"simple-20": lambda: tasks.t1_simple(20), "metastable-20": lambda: tasks.t1_metastable(20)}


def kl(p, q):
    p = np.clip(p, 1e-9, 1)
    q = np.clip(q, 1e-12, 1)
    return (q * (np.log(q) - np.log(p))).sum(-1).mean(0)


def rollouts(U, Ms, obs, horizon):
    """Plain and marginalised blind rollouts in numpy, after the model's conditioning on the context."""
    D = U.shape[0]
    plain, marg = np.zeros((obs.shape[0], horizon, 4)), np.zeros((obs.shape[0], horizon, 4))
    for i in range(obs.shape[0]):
        rho = np.eye(D, dtype=complex) / D
        for t in range(tasks.L_CTX):
            if t > 0:
                rho = U @ rho @ U.conj().T
            Mo = Ms[obs[i, t]]
            rho = Mo @ rho @ Mo.conj().T
            rho /= np.trace(rho).real
        r_p, r_m = rho.copy(), rho.copy()
        for k in range(horizon):
            r_p = U @ r_p @ U.conj().T
            pr = np.array([np.trace(M @ r_p @ M.conj().T).real for M in Ms])
            plain[i, k] = pr / pr.sum()
            r_m = U @ r_m @ U.conj().T
            pr = np.array([np.trace(M @ r_m @ M.conj().T).real for M in Ms])
            marg[i, k] = pr / pr.sum()
            r_m = sum(M @ r_m @ M.conj().T for M in Ms)
    return plain, marg


def first_returns(p, ref_k, tol):
    """First k > ref_k (1-indexed) with TV(p_k, p_ref) < tol after the prediction has left that ball; a large number if never."""
    out = []
    for i in range(p.shape[0]):
        d = 0.5 * np.abs(p[i] - p[i, ref_k - 1]).sum(-1)
        left = np.nonzero(d[ref_k:] >= tol)[0]
        if not len(left):
            out.append(10 ** 9)
            continue
        l0 = ref_k + left[0]
        back = np.nonzero(d[l0:] < tol)[0]
        out.append(int(l0 + back[0] + 1) if len(back) else 10 ** 9)
    return np.array(out)


def load(pattern, name):
    f = glob.glob(str(ROOT / f"runs/confirm/ckpt/{pattern}"))
    assert len(f) == 1, (pattern, f)
    m = models.build(name, 4, 1)
    m.load_state_dict(torch.load(f[0], weights_only=True))
    return m


def main() -> None:
    torch.set_num_threads(1)
    out = {"note": "post hoc diagnostic on the shipped checkpoints; no retraining; fresh test sequences, so levels differ "
                   "slightly from the run records", "seed_kl": SEED_KL, "n_kl": N_KL, "seed_returns": SEED_RET,
           "n_returns_per_seed": N_RET, "tasks": {}}
    for name, make in TASKS.items():
        task = make()
        obs, act = tasks.sample(task, N_KL, H_KL, np.random.default_rng(SEED_KL))
        q = tasks.bayes_predictive(task, obs, act, H_KL)
        rows = []
        for s in range(10):
            m = load(f"{name}-U2-s{s}-lr*.pt", "U2")
            with torch.no_grad():
                M = m.instrument.ops().numpy().astype(np.complex128)
                U = m.prepare()[0].numpy().astype(np.complex128)
                forward = m(torch.as_tensor(obs), torch.as_tensor(act), H_KL).numpy()
            d_e = U.shape[0] // M.shape[1]
            Ms = [np.kron(Mo, np.eye(d_e)) for Mo in M]
            plain, marg = rollouts(U, Ms, obs, H_KL)
            k_plain, k_marg = kl(plain, q), kl(marg, q)
            phi = sum(np.kron(Mk.conj(), Mk) for Mk in Ms) @ np.kron(U.conj(), U)
            ev = np.sort(np.abs(np.linalg.eigvals(phi)))[::-1]
            lam2 = float(ev[1])
            L = load(f"{name}-L-s{s}-lr*.pt", "L")
            with torch.no_grad():
                k_L = kl(L(torch.as_tensor(obs), torch.as_tensor(act), H_KL).numpy(), q)
            row = {"seed": s, "numpy_vs_forward_max_abs": float(np.abs(plain - forward).max()),
                   "in_range_plain": float(k_plain[:50].mean()), "in_range_marg": float(k_marg[:50].mean()),
                   "extrap_plain": float(k_plain[50:].mean()), "extrap_marg": float(k_marg[50:].mean()),
                   "extrap_L": float(k_L[50:].mean()), "abs_lambda2_marg_channel": lam2}
            ob2, ac2 = tasks.sample(task, N_RET, H_RET, np.random.default_rng(SEED_RET))
            with torch.no_grad():
                p = m(torch.as_tensor(ob2), torch.as_tensor(ac2), H_RET).numpy()
            x = p[:, 50:H_RET] - p[:, 50:H_RET].mean(1, keepdims=True)
            spec = (np.abs(np.fft.rfft(x, axis=1)) ** 2).sum(-1).mean(0)
            freqs = np.fft.rfftfreq(x.shape[1])
            row["dominant_period"] = float(1 / freqs[1 + np.argmax(spec[1:])])
            for label, ref_k, tol in (("p5_tv0.1", 5, 0.1), ("p5_tv0.05", 5, 0.05), ("p51_tv0.1", 51, 0.1), ("p51_tv0.05", 51, 0.05)):
                rets = first_returns(p, ref_k, tol)
                row[f"median_return_{label}"] = float(np.median(rets))
                row[f"share_returned_by_200_{label}"] = float((rets <= 200).mean())
            mean_late = p[:, 600:1200].mean(1, keepdims=True)
            row.update(tv_to_own_mean_k51_200=float(0.5 * np.abs(p[:, 50:200] - mean_late).sum(-1).mean()),
                       tv_to_own_mean_k1001_1200=float(0.5 * np.abs(p[:, 1000:1200] - mean_late).sum(-1).mean()))
            rows.append(row)
        keys = [k for k in rows[0] if k != "seed"]
        out["tasks"][name] = {"per_seed": rows, "range": {k: [min(r[k] for r in rows), max(r[k] for r in rows)] for k in keys},
                              "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys}}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    for name, d in out["tasks"].items():
        print(name, json.dumps({k: [round(d["mean"][k], 5), [round(x, 5) for x in d["range"][k]]] for k in d["mean"]}, indent=0))


if __name__ == "__main__":
    main()

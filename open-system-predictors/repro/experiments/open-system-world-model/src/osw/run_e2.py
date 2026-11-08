"""E2 runner (design: E2-DESIGN.zh.md). One run = (arm, family, size, seed, lr).

  python -m osw.run_e2 --arm A1 --family HQ --size 2 --seed 0 --lr 1e-2 --out runs/e2
  arms: A1 probability clock, A2 weakly measured qutrit clock (exact predictive known), B weather quintiles
  families: HQ GRU LRU (trained, Adam), HMM (EM), and for arm B the references CLIM EMP MARKOV1

The T1 recurrent models and the HMM read the context length from a module constant; this runner sets it for its own
process inside main() only (never at import), so T1 code and results are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import hmm, models, models_e2, tasks_aiwq, tasks_e2, tasks_series
from .run import git_commit, metrics, predict_torch

ARMS = {"A1": dict(ctx=32, k_train=20, k_eval=20, n_obs=2), "A2": dict(ctx=32, k_train=20, k_eval=20, n_obs=3),
        "B": dict(ctx=tasks_aiwq.CTX, k_train=tasks_aiwq.H, k_eval=tasks_aiwq.H, n_obs=tasks_aiwq.N_SYM),
        # E2.1 addendum: two more real streams, same window shape and alphabet as B
        "B2": dict(ctx=tasks_series.CTX, k_train=tasks_series.H, k_eval=tasks_series.H, n_obs=tasks_series.N_SYM),
        "B3": dict(ctx=tasks_series.CTX, k_train=tasks_series.H, k_eval=tasks_series.H, n_obs=tasks_series.N_SYM)}
N_TRAIN, N_VAL, N_TEST = 20_000, 2_000, 2_000


def data(arm, seed):
    if arm == "B":
        (tro, tra, _), (vao, vaa, _), (teo, tea, meta) = tasks_aiwq.split(seed)
        return (tro, tra), (vao, vaa), (teo, tea), None, meta
    if arm in tasks_series.SERIES:                     # time split is fixed; the seed varies initialisation and batches
        tr, va, te, _ = tasks_series.split(arm)
        return tr, va, te, None, None
    proc = tasks_e2.probability_clock() if arm == "A1" else tasks_e2.qutrit_clock()
    rng = np.random.default_rng(10_000 * seed + (1 if arm == "A1" else 2))
    k = ARMS[arm]["k_eval"]
    tr, va, te = (tasks_e2.sample(proc, n, k, rng) for n in (N_TRAIN, N_VAL, N_TEST))
    return tr, va, te, tasks_e2.exact_predictive(proc, te[0], k), None


def train(model, tr, va, ctx, K, lr, steps, batch, seed, log_every=200):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    obs, act = torch.as_tensor(tr[0]), torch.as_tensor(tr[1])
    y = obs[:, ctx:ctx + K]
    hist, gn_curve = [], []

    def val_nll():
        pv = predict_torch(model, va[0][:1000], va[1][:1000], K)
        yv = va[0][:1000, ctx:ctx + K]
        return float(-np.log(np.clip(np.take_along_axis(pv, yv[..., None], -1), 1e-9, 1)).mean())
    init = val_nll()
    for step in range(1, steps + 1):
        idx = torch.randint(0, obs.shape[0], (batch,), generator=g)
        p = model(obs[idx], act[idx], K)
        loss = -torch.log(torch.gather(p, 2, y[idx].unsqueeze(-1)).clamp_min(1e-9)).mean()
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % log_every == 0 or step == steps:
            hist.append((step, float(loss.detach()))); gn_curve.append((step, float(gn)))
    return hist, val_nll(), init, gn_curve


def references(family, tr, te, ctx, K):
    n_obs = tasks_aiwq.N_SYM
    if family == "CLIM":
        return np.full((len(te[0]), K, n_obs), 1 / n_obs), 0
    if family == "EMP":
        f = np.bincount(tr[0][:, ctx:].ravel(), minlength=n_obs) + 1.0
        return np.broadcast_to(f / f.sum(), (len(te[0]), K, n_obs)).copy(), n_obs - 1
    T = np.ones((n_obs, n_obs))
    for row in tr[0]:
        np.add.at(T, (row[:-1], row[1:]), 1)
    T /= T.sum(1, keepdims=True)
    b = np.eye(n_obs)[te[0][:, ctx - 1]]
    out = np.zeros((len(te[0]), K, n_obs))
    for k in range(K):
        b = b @ T; out[:, k] = b
    return out, n_obs * (n_obs - 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS), required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--size", type=int, default=0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="runs/e2")
    a = ap.parse_args(argv)
    cfg = ARMS[a.arm]
    ctx, K = cfg["ctx"], cfg["k_eval"]
    models.L_CTX = hmm.L_CTX = ctx                      # this process only; see module docstring
    torch.manual_seed(a.seed)
    tr, va, te, exact, meta = data(a.arm, a.seed)
    y = te[0][:, ctx:ctx + K]
    t0, hist, val, init, gn = time.time(), [], None, None, []
    if a.family in ("CLIM", "EMP", "MARKOV1"):
        pred, n_par = references(a.family, tr, te, ctx, K)
    elif a.family == "HMM":
        best = None
        for r in range(3):
            h = hmm.HMM(n_states=a.size, n_obs=cfg["n_obs"], n_actions=1, seed=a.seed * 101 + r).fit(tr[0][:5000], tr[1][:5000])
            best = h if best is None or h.train_ll > best.train_ll else best
        pred, n_par = best.predictive(te[0], te[1], K), models_e2.hmm_params(a.size, cfg["n_obs"])
    else:
        m = models_e2.build(a.family, a.size, cfg["n_obs"], ctx)
        n_par = models.n_params(m)
        hist, val, init, gn = train(m, tr, va, ctx, cfg["k_train"], a.lr, a.steps, a.batch, a.seed)
        pred = predict_torch(m, te[0], te[1], K)
    ref = exact if exact is not None else pred
    met = metrics(pred, ref, y)
    if exact is None:
        met.pop("kl")
        met["rps"] = tasks_aiwq.rps(pred, y).mean(0)
        met["rps_clim"] = tasks_aiwq.rps(np.full_like(pred, 1 / tasks_aiwq.N_SYM), y).mean(0)
    else:
        marg = exact[:, -1].mean(0)
        q = np.clip(exact, 1e-12, 1)
        met["kl_marginal"] = (q * np.log(q / marg)).sum(-1).mean(0)       # the predictable information per horizon
    res = {"arm": a.arm, "family": a.family, "size": a.size, "seed": a.seed, "lr": a.lr, "steps": a.steps,
           "n_params": int(n_par), "val_nll": val, "init_val_nll": init, "train_curve": hist, "grad_norm_curve": gn,
           "wall_s": round(time.time() - t0, 1), "per_k": {k: [round(float(x), 7) for x in v] for k, v in met.items()},
           "land_frac_test": float(meta["land"].mean()) if meta else None, "code_commit": git_commit()}
    res["result_sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = out / f"e2-{a.arm}-{a.family}{a.size or ''}-s{a.seed}-lr{a.lr:g}.json"
    f.write_text(json.dumps(res, indent=1))
    print(f"{f.name}: params={n_par} val={val} " + " ".join(f"{k}[1]={v[0]:.4f}" for k, v in met.items() if k in ("kl", "nll", "rps")))


if __name__ == "__main__":
    main()

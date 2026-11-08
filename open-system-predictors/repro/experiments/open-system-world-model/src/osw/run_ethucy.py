"""E1 (exploratory) runner: ETH/UCY relative headings, per E1-DESIGN.zh.md (frozen 2026-10-21, SHA-256 f06879ca...).

  python -m osw.run_ethucy --held-out zara01 --model L --seed 0 --lr 1e-2 --steps 6000 --out runs/e1

Context 8 symbols, train on k in [1,6], evaluate k in [1,12] (extrapolation window [7,12]); a 60-step rollout on the
test contexts feeds the descriptive revival diagnostic. MARG (per-k training marginal) and MARKOV1 (empirical
first-order transition from the last context symbol, propagated k steps) are the two reference predictors.

Implementation note: the T1 models read the context length from a module global (`L_CTX = 32`). This runner sets it
to 8 inside main() only (never at import), so the T1 code and every T1 result stay byte-for-byte untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import hmm, models
from . import tasks_ethucy as E
from .run import git_commit, metrics, predict_torch

K_TRAIN, K_EVAL, K_LONG = 6, 12, 60
E1_SHA256 = "f06879caf9a0559799c660de13b76f4195fb7e0b37a9c1b6e826408c3c64d0b0"


def extend(o, k_total):
    """Pad the symbol arrays so a rollout of k_total steps has action indices (all zero) to read."""
    pad = max(0, E.CTX + k_total - o.shape[1])
    return np.pad(o, ((0, 0), (0, pad)))


def train(model, tr, va, lr, steps, batch, seed, log_every=200):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    obs, act = torch.as_tensor(tr[0]), torch.as_tensor(tr[1])
    y = obs[:, E.CTX:E.CTX + K_TRAIN]
    hist = []
    for step in range(1, steps + 1):
        idx = torch.randint(0, obs.shape[0], (batch,), generator=g)
        p = model(obs[idx], act[idx], K_TRAIN)
        loss = -torch.log(torch.gather(p, 2, y[idx].unsqueeze(-1)).clamp_min(1e-9)).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % log_every == 0 or step == steps:
            hist.append((step, float(loss.detach())))
    pv = predict_torch(model, va[0], va[1], K_TRAIN)
    yv = va[0][:, E.CTX:E.CTX + K_TRAIN]
    return hist, float(-np.log(np.clip(np.take_along_axis(pv, yv[..., None], -1), 1e-9, 1)).mean())


def marginal(tr, n):
    m = np.stack([np.bincount(tr[0][:, E.CTX + k], minlength=E.N_SYM) for k in range(K_EVAL)]).astype(float) + 1
    m /= m.sum(1, keepdims=True)
    m = np.vstack([m, np.repeat(m[-1:], K_LONG - K_EVAL, 0)])                 # beyond k=12: hold the k=12 marginal
    return lambda obs, K: np.broadcast_to(m[:K], (obs.shape[0], K, E.N_SYM)).copy()


def markov1(tr):
    T = np.ones((E.N_SYM, E.N_SYM))
    for row in tr[0]:
        np.add.at(T, (row[:-1], row[1:]), 1)
    T /= T.sum(1, keepdims=True)

    def pred(obs, K):
        b = np.eye(E.N_SYM)[obs[:, E.CTX - 1]]
        out = np.zeros((obs.shape[0], K, E.N_SYM))
        for k in range(K):
            b = b @ T
            out[:, k] = b
        return out
    return pred


def rps_moving(pred, y):
    """Ranked probability score on moving targets: sectors ordered -135..180 degrees, renormalised over the 8 headings."""
    order = [5, 6, 7, 0, 1, 2, 3, 4]
    mv = y != E.STATIONARY
    p = pred[..., order]
    p = p / np.clip(p.sum(-1, keepdims=True), 1e-12, None)
    rank = np.argsort(order)[np.where(mv, y, 0)]
    cdf_p = np.cumsum(p, -1)
    cdf_y = (np.arange(8)[None, None, :] >= rank[..., None]).astype(float)
    r = ((cdf_p - cdf_y) ** 2).sum(-1) / 7
    return np.array([r[mv[:, k], k].mean() if mv[:, k].any() else np.nan for k in range(y.shape[1])])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--held-out", choices=E.SCENES, required=True)
    ap.add_argument("--model", required=True, help="U2 L K C GRU LRU HMM MARG MARKOV1")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="runs/e1")
    a = ap.parse_args(argv)
    models.L_CTX = hmm.L_CTX = E.CTX              # this process only; see module docstring
    torch.manual_seed(a.seed)
    tr, va, te, keys = E.loso(a.held_out, a.seed)
    te_long = (extend(te[0], K_LONG), extend(te[1], K_LONG))
    y = te[0][:, E.CTX:E.CTX + K_EVAL]
    t0, hist, val_nll = time.time(), [], None
    if a.model == "MARG":
        f, n_par = marginal(tr, len(te[0])), 0
        pred, pred_long = f(te[0], K_EVAL), f(te[0], K_LONG)
    elif a.model == "MARKOV1":
        f, n_par = markov1(tr), E.N_SYM * (E.N_SYM - 1)
        pred, pred_long = f(te[0], K_EVAL), f(te[0], K_LONG)
    elif a.model == "HMM":
        best = None
        for r in range(3):
            h = hmm.HMM(n_states=8, n_obs=E.N_SYM, n_actions=1, seed=a.seed * 101 + r).fit(tr[0][:5000], tr[1][:5000])
            best = h if best is None or h.train_ll > best.train_ll else best
        pred, pred_long, n_par = best.predictive(te[0], te[1], K_EVAL), best.predictive(*te_long, K_LONG), best.n_params()
    else:
        m = models.build(a.model, E.N_SYM, 1)
        n_par = models.n_params(m)
        hist, val_nll = train(m, tr, va, a.lr, a.steps, a.batch, a.seed)
        pred, pred_long = predict_torch(m, te[0], te[1], K_EVAL), predict_torch(m, *te_long, K_LONG)
    met = {k: v for k, v in metrics(pred, pred, y).items() if k != "kl"}
    met["rps_moving"] = rps_moving(pred, y)
    win = {name: {k: float(np.nanmean(v[sl])) for k, v in met.items()}
           for name, sl in {"in_range_1_6": slice(0, 6), "extrap_7_12": slice(6, 12)}.items()}
    p_inf = pred_long[:, 40:60].mean(1, keepdims=True)
    D = 0.5 * np.abs(pred_long[:, 6:40] - p_inf).sum(-1)
    res = {"held_out": a.held_out, "model": a.model, "seed": a.seed, "lr": a.lr, "steps": a.steps, "n_params": n_par,
           "n_train": len(tr[0]), "n_val": len(va[0]), "n_test": len(te[0]), "val_nll": val_nll, "train_curve": hist,
           "wall_s": round(time.time() - t0, 1), "windows": win,
           "per_k": {k: [round(float(x), 6) for x in v] for k, v in met.items()},
           "D_curve_7_40": [round(float(x), 6) for x in D.mean(0)],
           "code_commit": git_commit(), "e1_design_sha256": E1_SHA256}
    res["result_sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = out / f"e1-{a.held_out}-{a.model}-s{a.seed}-lr{a.lr:g}-st{a.steps}.json"
    f.write_text(json.dumps(res, indent=1))
    w = win["extrap_7_12"]
    print(f"{f.name}: params={n_par} val_nll={val_nll} extrap NLL={w['nll']:.4f} RPS={w['rps_moving']:.4f}")


if __name__ == "__main__":
    main()

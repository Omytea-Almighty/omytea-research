"""Train one model on one task with one seed and evaluate it against the exact Bayes predictive.

Pre-registration §2–§4 (frozen). Usage:
  python -m osw.run --task simple --tau 20 --model L --seed 100 --lr 3e-3 --out runs/dev
Confirmatory seeds are 0-9; development seeds 100-109 (never enter conclusions).
Each run writes <out>/<task>-<tau>-<model>-s<seed>-lr<lr>.json with per-horizon metrics, window means,
parameter count, wall time, the code commit and the SHA-256 of the result payload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from . import hmm, models, tasks

K_TRAIN, K_EVAL = 50, 200
N_TRAIN, N_VAL, N_TEST = 20_000, 2_000, 2_000


def make_task(name, tau):
    return {"simple": tasks.t1_simple, "metastable": tasks.t1_metastable, "action": tasks.t1_action}[name](tau)


def data(task, seed, task_key):
    rng = np.random.default_rng(10_000 * seed + task_key)
    tr = tasks.sample(task, N_TRAIN, K_TRAIN, rng)
    va = tasks.sample(task, N_VAL, K_TRAIN, rng)
    te = tasks.sample(task, N_TEST, K_EVAL, rng)
    return tr, va, te


def metrics(pred, bayes, y):
    """pred, bayes: (n, K, O) probabilities; y: (n, K) observed future symbols."""
    p = np.clip(pred, 1e-9, 1)
    q = np.clip(bayes, 1e-12, 1)
    kl = (q * (np.log(q) - np.log(p))).sum(-1).mean(0)                         # (K,)
    nll = -np.log(np.take_along_axis(p, y[..., None], -1)[..., 0]).mean(0)
    onehot = np.eye(p.shape[-1])[y]
    brier = ((p - onehot) ** 2).sum(-1).mean(0)
    conf, hit = p.max(-1), (p.argmax(-1) == y)
    ece = np.zeros(p.shape[1])
    for k in range(p.shape[1]):
        bins = np.minimum((conf[:, k] * 10).astype(int), 9)
        e = 0.0
        for b_ in range(10):
            m = bins == b_
            if m.any():
                e += m.mean() * abs(hit[m, k].mean() - conf[m, k].mean())
        ece[k] = e
    return {"kl": kl, "nll": nll, "brier": brier, "ece": ece}


def windows(m):
    out = {}
    for name, sl in {"in_range_1_50": slice(0, 50), "extrap_51_200": slice(50, 200)}.items():
        out[name] = {k: float(v[sl].mean()) for k, v in m.items()}
    return out


def revival(pred_long):
    """Pre-registration §4 revival index (operationalised in DEVIATIONS.md D4).
    pred_long: (n, 600, O). p_inf = mean over k in [400, 600]; D(k) = TV(p_k, p_inf) for k in [51, 200];
    per sequence: (#strict local maxima of D on [51,200]) x (mean height of those maxima); averaged over sequences."""
    p_inf = pred_long[:, 399:600].mean(1, keepdims=True)
    D = 0.5 * np.abs(pred_long[:, 50:200] - p_inf).sum(-1)                     # (n, 150)
    mid = D[:, 1:-1]
    peaks = (mid > D[:, :-2]) & (mid > D[:, 2:]) & (mid > 1e-6)
    count = peaks.sum(1)
    height = np.where(count > 0, (mid * peaks).sum(1) / np.maximum(count, 1), 0.0)
    return {"revival_index": float((count * height).mean()), "mean_peaks": float(count.mean()),
            "mean_D_51_200": float(D.mean()), "D_curve": [round(float(x), 6) for x in D.mean(0)]}


PREREG_V11 = Path(__file__).resolve().parents[4] / "docs/research/2026-10-19-open-system-world-model/EXPERIMENT-PREREG-v1.1.zh.md"


def prereg_v11_sha256():
    """SHA-256 of the v1.1 addendum as it is on disk when the run starts (compare with docs/timestamps/)."""
    return hashlib.sha256(PREREG_V11.read_bytes()).hexdigest() if PREREG_V11.exists() else None


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent).stdout.strip()
    except Exception:
        return "unknown"


def predict_torch(model, obs, act, K, batch=500):
    outs = []
    with torch.no_grad():
        for i in range(0, obs.shape[0], batch):
            outs.append(model(torch.as_tensor(obs[i:i + batch]), torch.as_tensor(act[i:i + batch]), K).numpy())
    return np.concatenate(outs)


def val_nll_of(model, va):
    pv = predict_torch(model, va[0][:1000], va[1][:1000], K_TRAIN)
    yv = va[0][:1000, tasks.L_CTX:tasks.L_CTX + K_TRAIN]
    return float(-np.log(np.clip(np.take_along_axis(pv, yv[..., None], -1), 1e-9, 1)).mean())


def train_torch(model, tr, va, lr, steps, batch, seed, log_every=200, diag=None):
    """diag (optional dict) receives v1.1 diagnostics: initial validation NLL and the pre-clipping gradient norm at
    each logged step. Recording them uses no random numbers, so the trained model is identical with or without."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    obs, act = torch.as_tensor(tr[0]), torch.as_tensor(tr[1])
    y = obs[:, tasks.L_CTX:tasks.L_CTX + K_TRAIN]
    hist = []
    if diag is not None:
        diag["init_val_nll"], diag["grad_norm_curve"] = val_nll_of(model, va), []
    for step in range(1, steps + 1):
        idx = torch.randint(0, obs.shape[0], (batch,), generator=g)
        p = model(obs[idx], act[idx], K_TRAIN)
        loss = -torch.log(torch.gather(p, 2, y[idx].unsqueeze(-1)).clamp_min(1e-9)).mean()
        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % log_every == 0 or step == steps:
            hist.append((step, float(loss.detach())))
            if diag is not None:
                diag["grad_norm_curve"].append((step, float(gn)))
    return hist, val_nll_of(model, va)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["simple", "metastable", "action"], required=True)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--model", required=True, help="U1 U2 U4 U8 L K C GRU LRU HMM BAYES")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", default="runs/dev")
    ap.add_argument("--init", choices=["identity", "random"], default="identity", help="instrument init (D5)")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed); torch.set_num_threads(max(1, torch.get_num_threads()))
    task = make_task(a.task, a.tau)
    task_key = {"simple": 1, "metastable": 2, "action": 3}[a.task] * 1000 + int(a.tau)
    tr, va, te = data(task, a.seed, task_key)
    bayes = tasks.bayes_predictive(task, te[0], te[1], K_EVAL)
    y = te[0][:, tasks.L_CTX:tasks.L_CTX + K_EVAL]
    t0 = time.time()
    hist, val_nll, n_par, diag = [], None, None, {}
    if a.model == "BAYES":
        pred, n_par = bayes, 0
    elif a.model == "HMM":
        h = hmm.fit_best(task, tr[0][:5000], tr[1][:5000], a.seed)
        pred, n_par = h.predictive(te[0], te[1], K_EVAL), h.n_params()
    else:
        m = models.build(a.model, tasks.O, task.n_actions, init=a.init)
        n_par = models.n_params(m)
        diag = {}
        hist, val_nll = train_torch(m, tr, va, a.lr, a.steps, a.batch, a.seed, diag=diag)
        pred = predict_torch(m, te[0], te[1], K_EVAL)
    met = metrics(pred, bayes, y)
    # long rollout (k up to 600) on 500 extra test contexts for the revival index (pre-registration §4)
    rng_long = np.random.default_rng(10_000 * a.seed + task_key + 777)
    lo, la = tasks.sample(task, 500, 600, rng_long)
    if a.model == "BAYES":
        pred_long = tasks.bayes_predictive(task, lo, la, 600)
    elif a.model == "HMM":
        pred_long = h.predictive(lo, la, 600)
    else:
        pred_long = predict_torch(m, lo, la, 600)
    rev = revival(pred_long)
    res = {
        "task": a.task, "tau": a.tau, "model": a.model, "seed": a.seed, "lr": a.lr, "steps": a.steps, "init": a.init,
        "n_params": n_par, "val_nll": val_nll, "train_curve": hist, "wall_s": round(time.time() - t0, 1),
        "windows": windows(met), "revival": rev, "per_k": {k: [round(float(x), 6) for x in v] for k, v in met.items()},
        "code_commit": git_commit(), "prereg_sha256": "37172954eacf4eee0b4e4cf62d031372a7d306f3e6502705f74a5e12c593350c",
        "prereg_v11_sha256": prereg_v11_sha256(), **diag,
    }
    payload = json.dumps(res, sort_keys=True).encode()
    res["result_sha256"] = hashlib.sha256(payload).hexdigest()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = "" if (a.steps == 2000 and a.init == "identity") else f"-st{a.steps}-{a.init}"
    f = out / f"{a.task}-{a.tau:g}-{a.model}-s{a.seed}-lr{a.lr:g}{tag}.json"
    f.write_text(json.dumps(res, indent=1))
    if a.model not in ("BAYES", "HMM"):                       # v1.1: keep the trained parameters for later audit
        (out / "ckpt").mkdir(exist_ok=True)
        torch.save(m.state_dict(), out / "ckpt" / f"{f.stem}.pt")
    w = res["windows"]
    print(f"{f.name}: params={n_par} wall={res['wall_s']}s val_nll={val_nll} "
          f"KL in={w['in_range_1_50']['kl']:.4f} extrap={w['extrap_51_200']['kl']:.4f}")


if __name__ == "__main__":
    main()

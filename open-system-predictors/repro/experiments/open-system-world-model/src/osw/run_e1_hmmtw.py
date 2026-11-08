"""E1 post hoc sensitivity (round-4 review, Claude leg m7): the HMM refitted on the six-step training window only.

  python -m osw.run_e1_hmmtw --out runs/e1-posthoc

The registered E1 HMM was fitted by EM on whole training sequences (context + future steps 1-12), so it also saw steps
7-12 of the training sequences (D14(d)). This runner refits it on the context plus future steps 1-6, exactly as
report/robustness.py item 14 did in-process, and writes one raw file per (held-out scene, seed) so the sensitivity
value can be recomputed raw-first. Same fit as before: 8 states, best of 3 EM restarts (seed * 101 + r) by training
log-likelihood, first 5,000 training windows. Labelled post hoc; it changes no frozen verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from . import hmm
from . import tasks_ethucy as E
from .models_e2 import hmm_params
from .run import git_commit
from .run_ethucy import E1_SHA256, K_EVAL, K_TRAIN

N_STATES, N_FIT, RESTARTS = 8, 5000, 3


def fit_and_score(fold: str, seed: int):
    tr, _, te, _ = E.loso(fold, seed)
    best = None
    for r in range(RESTARTS):
        h = hmm.HMM(n_states=N_STATES, n_obs=E.N_SYM, n_actions=1, seed=seed * 101 + r).fit(
            tr[0][:N_FIT, :E.CTX + K_TRAIN], tr[1][:N_FIT, :E.CTX + K_TRAIN])
        best = h if best is None or h.train_ll > best.train_ll else best
    pr = best.predictive(te[0], te[1], K_EVAL)
    y = te[0][:, E.CTX:E.CTX + K_EVAL]
    nll_k = -np.log(np.clip(np.take_along_axis(pr, y[..., None], -1)[..., 0], 1e-9, 1)).mean(0)
    return nll_k, len(te[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/e1-posthoc")
    a = ap.parse_args(argv)
    hmm.L_CTX = E.CTX                                   # this process only, as in run_ethucy.main
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for fold in E.SCENES:
        for seed in range(3):
            nll_k, n_test = fit_and_score(fold, seed)
            res = {"held_out": fold, "model": "HMM-tw6", "seed": seed, "n_states": N_STATES,
                   "n_params": int(hmm_params(N_STATES, E.N_SYM)), "n_fit": N_FIT, "fit_window": f"context + steps 1-{K_TRAIN}",
                   "n_test": n_test, "per_k": {"nll": [round(float(x), 7) for x in nll_k]},
                   "windows": {"in_range_1_6": {"nll": float(nll_k[:K_TRAIN].mean())},
                               "extrap_7_12": {"nll": float(nll_k[K_TRAIN:K_EVAL].mean())}},
                   "posthoc": "round-4 review m7: HMM refitted on the six-step training window (sensitivity)",
                   "code_commit": git_commit(), "e1_design_sha256": E1_SHA256}
            res["result_sha256"] = hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()
            f = out / f"e1hmmtw-{fold}-HMM{N_STATES}-s{seed}.json"
            f.write_text(json.dumps(res, indent=1))
            print(f"{f.name}: extrap_7_12 nll={res['windows']['extrap_7_12']['nll']:.4f}")


if __name__ == "__main__":
    main()

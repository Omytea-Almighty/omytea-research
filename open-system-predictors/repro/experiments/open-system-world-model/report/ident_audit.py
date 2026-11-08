"""Local identifiable dimension per T1 model family — pre-publication erratum r9 m1.

  .venv/bin/python report/ident_audit.py --task simple --names U1,U2   ->  runs/robustness/ident/<name>_<n_actions>.json
  .venv/bin/python report/ident_audit.py --merge                          ->  runs/robustness/IDENT_AUDIT.json

PARAM_AUDIT counts the directions Adam can move. A movable direction can still leave every prediction unchanged
(gauge freedoms: the joint system-basis conjugation, the instrument normalisation, U's environment basis, ...), so the
movable count overstates the dimension of the function class. Here the local identifiable dimension is the numerical
rank of d log p / d theta at a random parameter point (the same initialisation as PARAM_AUDIT, torch seed 7): the model
is cast to float64, log p is taken over the blind rollout of the first N_CTX development-seed-102 sequences for K
horizons and all outcomes, and every parameter coordinate is perturbed by central differences (step H). The rank is the
number of singular values above RANK_TOL times the largest; the ratio of the singular values on either side of that
rank (the spectral gap) is reported, and a rank without a clear gap (ratio < GAP_MIN) is flagged instead of trusted.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from osw import models as M, run, tasks  # noqa: E402

OUT = ROOT / "runs/robustness/IDENT_AUDIT.json"
NAMES = ("U1", "U2", "L", "L1", "K", "C", "GRU", "LRU")
N_CTX, K, H = 48, 16, 1e-6
RANK_TOL, GAP_MIN = 1e-7, 1e3


def flat_params(m):
    return [p for p in m.parameters() if p.requires_grad]


def logp(m, obs, act):
    with torch.no_grad():
        p = m(obs, act, K)
    return torch.log(p.clamp_min(1e-300)).reshape(-1).numpy()


def jacobian(m, obs, act):
    ps = flat_params(m)
    sizes = [p.numel() for p in ps]
    cols = []
    for p in ps:
        flat = p.data.view(-1)
        for i in range(flat.numel()):
            old = flat[i].item()
            flat[i] = old + H
            up = logp(m, obs, act)
            flat[i] = old - H
            dn = logp(m, obs, act)
            flat[i] = old
            cols.append((up - dn) / (2 * H))
    return np.stack(cols, 1), int(sum(sizes))


def audit(name, task, tr):
    torch.manual_seed(7)
    m = M.build(name, tasks.O, task.n_actions).double()
    if hasattr(m, "_inp"):                        # GRU / LRU build their one-hot inputs in float32: cast in this process
        orig = m._inp
        m._inp = lambda o, a, blind: orig(o, a, blind).double()
    obs = torch.as_tensor(tr[0][:N_CTX])
    act = torch.as_tensor(tr[1][:N_CTX])
    t0 = time.process_time()
    J, nominal = jacobian(m, obs, act)
    s = np.linalg.svd(J, compute_uv=False)
    r = int((s > RANK_TOL * s[0]).sum())
    gap = float(s[r - 1] / s[r]) if 0 < r < len(s) and s[r] > 0 else None
    return {"nominal": nominal, "rank": r, "gap_ratio_at_rank": gap, "clear_gap": bool(gap is not None and gap >= GAP_MIN),
            "s_at_rank": [float(s[r - 1]), float(s[r]) if r < len(s) else None], "s_max": float(s[0]),
            "n_outputs": int(J.shape[0]), "cpu_s": time.process_time() - t0}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("simple", "action"))
    ap.add_argument("--names", default=",".join(NAMES))
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args(argv)
    part = OUT.parent / "ident"
    if a.merge:
        res = {"method": " ".join(__doc__.split("\n\n")[1].split()), "n_ctx": N_CTX, "K": K, "step": H,
               "rank_tol": RANK_TOL, "gap_min": GAP_MIN,
               "counts": {f.stem: json.loads(f.read_text()) for f in sorted(part.glob("*.json"))}}
        OUT.write_text(json.dumps(res, indent=1))
        print(json.dumps({k: {x: v[x] for x in ("nominal", "movable", "rank", "clear_gap")} for k, v in res["counts"].items()}, indent=1))
        return
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)        # this process only: hidden states created with the default dtype
    M.L_CTX = tasks.L_CTX
    M.CDT = torch.complex128                      # this process only: the model code's complex dtype, to match .double()
    mov = json.loads((ROOT / "runs/robustness/PARAM_AUDIT.json").read_text())["counts"]
    task = run.make_task(a.task, 20)
    tr, _, _ = run.data(task, 102, {"simple": 1, "action": 3}[a.task])
    part.mkdir(parents=True, exist_ok=True)
    for name in a.names.split(","):
        r = audit(name, task, tr)
        r["movable"] = mov[f"{name}_{task.n_actions}"]["movable"]
        (part / f"{name}_{task.n_actions}.json").write_text(json.dumps(r, indent=1))
        print(name, task.n_actions, {k: r[k] for k in ("nominal", "movable", "rank", "gap_ratio_at_rank", "clear_gap")}, flush=True)


if __name__ == "__main__":
    main()

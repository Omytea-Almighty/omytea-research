"""Parameters that Adam training can change, per T1 model family (post hoc; requested by internal review rounds 5 and 7).

  .venv/bin/python report/param_audit.py   ->  runs/robustness/PARAM_AUDIT.json

Adam updates each coordinate from its own gradient history (zero-initialised moments, shared hyperparameters; global
gradient clipping scales every coordinate alike). A linear direction therefore never moves if (a) a coordinate's
gradient is always zero, (b) a group of coordinates has identical gradients on every batch (a group of size m fixes
m - 1 differences), or (c) two coordinates have exactly opposite gradients on every batch (their sum is fixed). Other
linear relations are broken by Adam's element-wise normalisation. We detect (a)-(c) from the gradients of 24 training
batches (development seed 102 data, the registered loss) and report nominal, fixed and movable counts. Examples found:
the anti-Hermitian half of each Hermitian-projected H (n^2 reals for n x n), and in the GRU the duplicated
input/hidden biases of the reset and update gates plus, with one action, the constant action input that duplicates
the input biases. Gauge directions that Adam does move (normalisations, polar maps, trace of H, sigma's unitary
freedom) are not subtracted: they are movable but not identifiable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from osw import models as M, run, tasks  # noqa: E402

OUT = ROOT / "runs/robustness/PARAM_AUDIT.json"
NAMES = ("U1", "U2", "U4", "U8", "L", "L1", "K", "C", "GRU", "LRU")
N_BATCH, RTOL = 24, 1e-4


def gradients(model, tr, n_batch, seed):
    g = torch.Generator().manual_seed(seed)
    obs, act = torch.as_tensor(tr[0]), torch.as_tensor(tr[1])
    y = obs[:, tasks.L_CTX:tasks.L_CTX + run.K_TRAIN]
    params = [p for p in model.parameters() if p.requires_grad]
    rows = []
    for _ in range(n_batch):
        idx = torch.randint(0, obs.shape[0], (64,), generator=g)
        p = model(obs[idx], act[idx], run.K_TRAIN)
        loss = -torch.log(torch.gather(p, 2, y[idx].unsqueeze(-1)).clamp_min(1e-9)).mean()
        rows.append(torch.cat([x.reshape(-1) for x in torch.autograd.grad(loss, params)]).double().numpy())
    return np.stack(rows)


def fixed_directions(G, rtol=RTOL):
    """(zero-gradient coordinates, tied directions) by union-find over identical-or-opposite gradient columns."""
    scale = np.abs(G).max(0)
    zero = scale <= 1e-12
    idx = np.where(~zero)[0]
    U = G[:, idx]
    first = np.argmax(np.abs(U) > 1e-12 * scale[idx], axis=0)
    V = U * np.sign(U[first, np.arange(U.shape[1])])                  # sign-canonical: opposite columns coincide
    parent = list(range(V.shape[1]))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    ref = np.abs(V).max(0)
    for off in (0.0, 0.5):                                            # two offset grids catch bin-edge pairs
        keys = np.floor(V[:3] / (1e-3 * ref.max()) + off).astype(np.int64)
        buckets = {}
        for j in range(V.shape[1]):
            buckets.setdefault(keys[:, j].tobytes(), []).append(j)
        for b in buckets.values():
            if len(b) < 2:
                continue
            B = V[:, b]
            for i in range(len(b)):
                close = np.abs(B - B[:, [i]]).max(0) <= rtol * np.maximum(ref[b], ref[b[i]])
                for c in np.where(close)[0]:
                    ra, rc = find(b[i]), find(b[c])
                    if ra != rc:
                        parent[ra] = rc
    sizes = {}
    for j in range(V.shape[1]):
        r = find(j)
        sizes[r] = sizes.get(r, 0) + 1
    return int(zero.sum()), sum(s - 1 for s in sizes.values())


def main():
    torch.set_num_threads(2)
    M.L_CTX = tasks.L_CTX
    res = {"method": __doc__.split("\n\n")[1].replace("\n", " "), "n_batch": N_BATCH, "rtol": RTOL, "counts": {}}
    for task_name, key in (("simple", 1), ("action", 3)):
        task = run.make_task(task_name, 20)
        tr, _, _ = run.data(task, 102, key)
        for name in NAMES:
            torch.manual_seed(7)
            m = M.build(name, tasks.O, task.n_actions)
            nz, tied = fixed_directions(gradients(m, tr, N_BATCH, 11))
            nominal = M.n_params(m)
            res["counts"][f"{name}_{task.n_actions}"] = {"nominal": nominal, "zero_grad": nz, "tied": tied,
                                                          "movable": nominal - nz - tied}
            print(name, task.n_actions, res["counts"][f"{name}_{task.n_actions}"], flush=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

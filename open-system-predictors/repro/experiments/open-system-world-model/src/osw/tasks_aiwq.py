"""E2 arm B: weekly 2 m temperature quintile categories (ERA5T observations against the 20-year weekly climatology),
the target variable of the ECMWF AI Weather Quest. Built by scripts/prep_aiwq.py into data/aiwq_tas_quintiles.npz.

Windows: CTX = 26 observed weeks, then forecast k = 1..H = 8 weeks ahead (the competition scores weeks 3-6).
  * training period: windows entirely inside weeks 0..80 (2024-01-15 .. 2025-07-28);
  * test period: windows entirely inside weeks 81..133 (2025-08-04 .. 2026-08-03), i.e. strictly later in time;
  * validation: training-period windows on a held-out 10 % of grid points (never used for training).
Sampling is area-weighted (probability proportional to cos latitude), so plain means estimate area-weighted means.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[2] / "data/aiwq_tas_quintiles.npz"
CTX, H, N_SYM = 26, 8, 5
WIN = CTX + H
TRAIN_LAST, TEST_FIRST = 80, 81


def load():
    z = np.load(DATA)
    return z["cat"], z["lat"], z["land"]


def split(seed: int, n_train=20_000, n_val=2_000, n_test=5_000):
    cat, lat, land = load()
    W, nlat, nlon = cat.shape
    rng = np.random.default_rng(seed)
    pts = np.arange(nlat * nlon)
    w = np.repeat(np.cos(np.radians(lat)), nlon).clip(1e-6)
    val_pts = rng.choice(pts, size=len(pts) // 10, replace=False, p=w / w.sum())
    is_val = np.zeros(len(pts), bool); is_val[val_pts] = True
    flat = cat.reshape(W, -1)

    def draw(n, pool, first, last):
        p = w[pool] / w[pool].sum()
        pt = rng.choice(pool, size=n, p=p)
        st = rng.integers(first, last - WIN + 2, size=n)
        obs = np.stack([flat[s:s + WIN, q] for s, q in zip(st, pt)]).astype(np.int64)
        return obs, np.zeros_like(obs), {"point": pt, "start": st, "land": land.reshape(-1)[pt] > 0.5}

    train = draw(n_train, pts[~is_val], 0, TRAIN_LAST)
    val = draw(n_val, pts[is_val], 0, TRAIN_LAST)
    test = draw(n_test, pts, TEST_FIRST, W - 1)
    return train, val, test


def rps(pred, y):
    """Ranked probability score over the ordered quintiles: sum_j (F_pred,j - F_obs,j)^2 for j = 1..4."""
    F = np.cumsum(pred, -1)[..., :-1]
    O = (np.arange(N_SYM - 1)[None, None, :] >= y[..., None]).astype(float)
    return ((F - O) ** 2).sum(-1)

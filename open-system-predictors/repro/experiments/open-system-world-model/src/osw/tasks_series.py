"""E2 addendum (E2.1-ADDENDUM.zh.md): two more real streams for arm B, symbolised into quintiles.

  B2  household appliance energy use (UCI "Appliances energy prediction"), 10-minute steps, 19,735 points
  B3  New York City taxi demand, 30-minute steps, 10,320 points

Each series is split by time: training = first 60 %, validation = next 10 %, test = last 30 %. Quintile thresholds come
from the training part only; category = number of thresholds strictly below the value (ties make the categories
unequal, which is reported, not corrected). Windows (CTX + H consecutive steps) never cross a split boundary;
every window of each part is used (stride 1).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4] / "data"
SERIES = {"B2": ("exp_e9/energydata_complete.csv", "date", "Appliances"),
          "B3": ("exp_e3/nyc_taxi.csv", "timestamp", "value")}
CTX, H, N_SYM = 26, 8, 5
WIN = CTX + H


def load(arm):
    path, _, col = SERIES[arm]
    with open(ROOT / path) as f:
        return np.array([float(r[col]) for r in csv.DictReader(f)])


def symbolise(v, n_train):
    q = np.quantile(v[:n_train], [0.2, 0.4, 0.6, 0.8])
    return (v[:, None] > q[None]).sum(1).astype(np.int64), q


def windows(c, lo, hi):
    """All windows lying entirely in [lo, hi)."""
    starts = np.arange(lo, hi - WIN + 1)
    return np.stack([c[s:s + WIN] for s in starts])


def split(arm):
    v = load(arm)
    n = len(v)
    a, b = int(0.6 * n), int(0.7 * n)
    c, q = symbolise(v, a)
    pack = lambda o: (o, np.zeros_like(o))
    return pack(windows(c, 0, a)), pack(windows(c, a, b)), pack(windows(c, b, n)), {"thresholds": q.tolist(), "n": n}

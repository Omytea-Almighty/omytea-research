"""E1 (exploratory): ETH/UCY pedestrian headings as a symbol stream, per E1-DESIGN.zh.md (frozen 2026-10-21,
SHA-256 f06879ca...).

  * raw obsmat, 0.4 s per step; a window is 21 positions = 20 displacements = 8 context + 12 future symbols;
  * symbol 8 = stationary (|displacement| < 0.05 m); otherwise the turn relative to the reference heading,
    in 8 sectors of 45 degrees, sector 0 centred on straight ahead;
  * reference heading = the last moving displacement inside the context; the whole window is rotated by it,
    so the scene's absolute orientation never reaches the model; windows with an all-stationary context are dropped;
  * LOSO: one held-out scene is the test set (all windows, stride 1); the other four are split by pedestrian 90/10.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

RAW = Path(__file__).resolve().parents[4] / "data/ethucy/raw"
SCENES = ["eth", "hotel", "univ", "zara01", "zara02"]
CTX, HORIZON, STILL = 8, 12, 0.05
N_SYM, STATIONARY = 9, 8
WIN = CTX + HORIZON + 1                      # positions per window


def load_tracks(scene: str) -> dict:
    """pedestrian id -> list of (T, 2) position arrays; a track is split wherever its frame spacing breaks."""
    a = np.loadtxt(RAW / f"{scene}_obsmat.txt")
    out = {}
    for pid in np.unique(a[:, 1]).astype(int):
        r = a[a[:, 1] == pid]
        r = r[np.argsort(r[:, 0])]
        step = np.median(np.diff(r[:, 0])) if len(r) > 1 else 0
        cut = np.where(np.diff(r[:, 0]) != step)[0] + 1
        out[pid] = [seg[:, [2, 4]] for seg in np.split(r, cut)]
    return out


def symbolize(pos: np.ndarray):
    """(21, 2) positions -> (20,) symbols, or None when the context never moves."""
    d = np.diff(pos, axis=0)
    moving = np.linalg.norm(d, axis=1) >= STILL
    ctx_moving = np.where(moving[:CTX])[0]
    if len(ctx_moving) == 0:
        return None
    ref = d[ctx_moving[-1]]
    ang = np.arctan2(d[:, 1], d[:, 0]) - np.arctan2(ref[1], ref[0])
    sector = np.floor(np.mod(ang + np.pi / 8, 2 * np.pi) / (np.pi / 4)).astype(int) % 8
    return np.where(moving, sector, STATIONARY)


def windows(scene: str, stride: int = 1):
    """All windows of a scene -> (symbols (N, 20) int64, pedestrian ids (N,), number dropped)."""
    obs, ped, dropped = [], [], 0
    for pid, segs in load_tracks(scene).items():
        for seg in segs:
            for t in range(0, len(seg) - WIN + 1, stride):
                s = symbolize(seg[t:t + WIN])
                if s is None:
                    dropped += 1
                    continue
                obs.append(s); ped.append(pid)
    return np.array(obs, dtype=np.int64).reshape(-1, CTX + HORIZON), np.array(ped), dropped


def loso(held_out: str, seed: int, val_frac: float = 0.1):
    """Train / validation from the other four scenes, split by (scene, pedestrian); test = the held-out scene.
    Each split is (obs (N, 20), act (N, 20) zeros): one action, so the action-conditioned models stay unchanged.
    Also returns the (scene, pedestrian) keys of every window per split, so leakage can be checked directly."""
    rng = np.random.default_rng(seed)
    tr, va, keys = [], [], {"train": [], "val": [], "test": []}
    for sc in SCENES:
        if sc == held_out:
            continue
        o, p, _ = windows(sc)
        ids = np.unique(p)
        val_ids = set(rng.choice(ids, size=max(1, int(round(val_frac * len(ids)))), replace=False).tolist())
        mask = np.array([q in val_ids for q in p])
        tr.append(o[~mask]); va.append(o[mask])
        keys["train"] += [(sc, int(q)) for q in p[~mask]]
        keys["val"] += [(sc, int(q)) for q in p[mask]]
    te, pt, _ = windows(held_out)
    keys["test"] = [(held_out, int(q)) for q in pt]
    pack = lambda o: (o, np.zeros_like(o))
    return pack(np.concatenate(tr)), pack(np.concatenate(va)), pack(te), keys

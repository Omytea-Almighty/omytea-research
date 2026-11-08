"""E2 arm B preprocessing (run with the weather-quest venv, which has xarray):

  ../../scripts/ai_weather_quest/.venv/bin/python scripts/prep_aiwq.py

Observed ERA5T weekly-mean 2 m temperature (1.5 deg grid, 134 consecutive weeks from 2024-01-15) is mapped to its
week's 20-year climatological quintile category 0..4 (number of quintile thresholds below the observation).
Output: data/aiwq_tas_quintiles.npz with cat (weeks, lat, lon) int8, lat, lon, week dates, and the land-sea mask.
Read-only on the weather-quest data; writes only inside this experiment directory.
"""
import glob
import hashlib
from pathlib import Path

import numpy as np
import xarray as xr

SRC = Path(__file__).resolve().parents[3] / "scripts/ai_weather_quest/data"
OUT = Path(__file__).resolve().parents[1] / "data/aiwq_tas_quintiles.npz"

obs = sorted(glob.glob(str(SRC / "tas_obs_WEEKLYMEAN_*.nc")))
cats, dates = [], []
for f in obs:
    day = f[-11:-3]
    q = xr.open_dataset(SRC / f"tas_20yrCLIM_WEEKLYMEAN_quintiles_{day}.nc")["tas"].isel(time=0).values   # (4, lat, lon)
    o = xr.open_dataset(f)["t2m"].isel(time=0).values                                                      # (lat, lon)
    c = (o[None] > q).sum(0).astype(np.int8)
    c[~np.isfinite(o)] = -1
    cats.append(c); dates.append(day)
ref = xr.open_dataset(obs[0])
lsm = xr.open_dataset(SRC / "land_sea_mask_1pt5DEG.nc")
lsm_v = lsm[list(lsm.data_vars)[0]].squeeze().values
OUT.parent.mkdir(exist_ok=True)
np.savez_compressed(OUT, cat=np.stack(cats), lat=ref.latitude.values, lon=ref.longitude.values,
                    dates=np.array(dates), land=lsm_v.astype(np.float32))
print(OUT, "sha256", hashlib.sha256(OUT.read_bytes()).hexdigest(), "shape", np.stack(cats).shape)

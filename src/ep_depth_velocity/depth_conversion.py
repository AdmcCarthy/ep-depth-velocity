"""Depth conversion: the linear velocity function V0-k, average and interval velocities, and the interval-velocity check of well tops.  Pure functions.

Definitions (depth z is measured DOWN from the datum, one-way time t from the same datum; two-way time is twice t):
* instantaneous velocity  v(z) = dz/dt = V0 + k z   (V0 at the datum, k the gradient in 1/s)
* time-depth relation     z(t) = (V0 / k) (exp(k t) - 1)  and  t(z) = ln(1 + k z / V0) / k   (k -> 0: z = V0 t)
* average velocity        v_av = z / t
* interval velocity       v_int = (z2 - z1) / (t2 - t1); for the linear function  v_int = k (z2 - z1) / ln((V0 + k z2) / (V0 + k z1)), the logarithmic mean of the velocity at the two depths:
  it is NOT the instantaneous velocity at the mid depth (the two differ by a few per cent over a thick zone).
The datum of V0 matters: V0 is the velocity at the datum depth z = 0, so a water layer is handled separately (z and t below the seabed) or V0 is a fitted parameter of a function from sea level that has no physical meaning at the surface.
A calibration of V0 (and k) to well tops leaves residuals; a residual is a finding about the velocity model (or the tops), never a reason to move a pick.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

SCHEMA_VERSIONS = {"depth_conversion_fit": "1.0", "depth_conversion_interval": "1.0"}


def v0k_depth(t_one_way_s: np.ndarray, v0: float, k: float) -> np.ndarray:
    """Depth below the datum [m] from one-way time [s]: z = (V0 / k) (exp(k t) - 1); V0 t when k is (nearly) zero."""
    t = np.asarray(t_one_way_s, float)
    return v0 * t if abs(k) < 1e-9 else v0 / k * np.expm1(k * t)


def v0k_time(z_m: np.ndarray, v0: float, k: float) -> np.ndarray:
    """One-way time [s] from depth below the datum: t = ln(1 + k z / V0) / k; z / V0 when k is (nearly) zero."""
    z = np.asarray(z_m, float)
    return z / v0 if abs(k) < 1e-9 else np.log1p(k * z / v0) / k


def v0k_instantaneous(z_m: np.ndarray, v0: float, k: float) -> np.ndarray:
    """The instantaneous velocity V0 + k z [m/s]."""
    return v0 + k * np.asarray(z_m, float)


def v0k_average(z_m: np.ndarray, v0: float, k: float) -> np.ndarray:
    """The average velocity z / t(z) [m/s] from the datum to depth z (V0 at z = 0)."""
    z = np.asarray(z_m, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(z > 0, z / v0k_time(np.where(z > 0, z, 1.0), v0, k), v0)


def v0k_interval(z1_m: np.ndarray, z2_m: np.ndarray, v0: float, k: float) -> np.ndarray:
    """The interval velocity between two depths of a V0-k medium: the logarithmic mean of the instantaneous velocities (not the velocity at the mid depth)."""
    z1, z2 = np.asarray(z1_m, float), np.asarray(z2_m, float)
    if abs(k) < 1e-9:
        return np.full(np.broadcast(z1, z2).shape, float(v0))
    return k * (z2 - z1) / np.log((v0 + k * z2) / (v0 + k * z1))


def fit_v0k(z_m: np.ndarray, t_one_way_s: np.ndarray, x0: tuple[float, float] = (1700.0, 0.5)) -> dict[str, Any]:
    """Least-squares V0 and k of z = (V0 / k)(exp(k t) - 1) to depth-time pairs (depths below the datum, one-way times).  The residual is the model depth minus the observed depth [m] (negative: the model is shallower than the observation).
    Returns the parameters, the residuals, their mean and standard deviation and the number of pairs.  Two pairs or fewer give no fit."""
    z, t = np.asarray(z_m, float), np.asarray(t_one_way_s, float)
    ok = np.isfinite(z) & np.isfinite(t) & (t > 0)
    if ok.sum() < 3:
        return {"v0": np.nan, "k": np.nan, "n": int(ok.sum()), "residual": np.full(len(z), np.nan), "residual_mean_m": np.nan, "residual_sd_m": np.nan}
    r = least_squares(lambda p: v0k_depth(t[ok], p[0], p[1]) - z[ok], x0=list(x0))
    res = np.full(len(z), np.nan)
    res[ok] = r.fun
    return {"v0": float(r.x[0]), "k": float(r.x[1]), "n": int(ok.sum()), "residual": res, "residual_mean_m": float(r.fun.mean()), "residual_sd_m": float(r.fun.std())}


def zone_interval_velocities(picks: pd.DataFrame, min_thickness_m: float = 20.0, min_twt_ms: float = 8.0) -> pd.DataFrame:
    """Interval velocities of the zones between consecutive picks of each well.  `picks` has `well`, `surface_name`, `z_m` (depth below the datum, positive down) and `twt_ms`.  Picks are ordered by depth within a well; a zone thinner than `min_thickness_m`
    or with less than `min_twt_ms` of two-way time is skipped (the velocity is then dominated by the pick time resolution).  The zone is named `top -> base`; the interval velocity is dz / (dTWT / 2) [m/s], vertical depth over one-way time."""
    rows = []
    for w, g in picks.dropna(subset=["z_m", "twt_ms"]).sort_values(["well", "z_m"]).drop_duplicates(["well", "surface_name"]).groupby("well"):
        g = g.reset_index(drop=True)
        for a, b in zip(g.itertuples(), g.iloc[1:].itertuples()):
            dz, dt = b.z_m - a.z_m, (b.twt_ms - a.twt_ms) / 2000.0
            if dz >= min_thickness_m and dt * 2000.0 >= min_twt_ms:
                rows.append({"well": w, "zone": f"{a.surface_name} -> {b.surface_name}", "top_z_m": a.z_m, "base_z_m": b.z_m, "thickness_m": dz, "twt_ms": dt * 2000.0, "interval_velocity_m_s": dz / dt, "mid_depth_m": 0.5 * (a.z_m + b.z_m)})
    return pd.DataFrame(rows, columns=["well", "zone", "top_z_m", "base_z_m", "thickness_m", "twt_ms", "interval_velocity_m_s", "mid_depth_m"])


def flag_interval_outliers(iv: pd.DataFrame, min_wells: int = 5, z_limit: float = 3.0, mad_floor_m_s: float = 30.0) -> pd.DataFrame:
    """Each zone velocity against the other wells' velocity of the same zone: the median and the median absolute deviation (MAD, floored at `mad_floor_m_s` so that a tight zone does not flag small differences) give a robust z-score `(v - median) / (1.4826 MAD)`;
    a zone seen in fewer than `min_wells` wells is not judged.  `flag` is `low` or `high` beyond `z_limit`.  Also `inversion`: the zone is slower than the zone above it in the same well by more than the MAD floor (velocity inversions can be real (a shale below a sand) and are a prompt to look, not an error).
    A flag says that a top, a time or the zone itself is unusual; it does not say which, and nothing is corrected."""
    out = iv.copy()
    g = out.groupby("zone")["interval_velocity_m_s"]
    out["wells_in_zone"] = g.transform("size")
    out["zone_median_m_s"] = g.transform("median")
    out["zone_mad_m_s"] = out.groupby("zone")["interval_velocity_m_s"].transform(lambda s: float((s - s.median()).abs().median()))
    out["robust_z"] = (out["interval_velocity_m_s"] - out["zone_median_m_s"]) / (1.4826 * out["zone_mad_m_s"].clip(lower=mad_floor_m_s))
    judged = out["wells_in_zone"] >= min_wells
    out["flag"] = np.where(judged & (out["robust_z"] > z_limit), "high", np.where(judged & (out["robust_z"] < -z_limit), "low", ""))
    out = out.sort_values(["well", "top_z_m"])
    prev = out.groupby("well")["interval_velocity_m_s"].shift(1)
    out["inversion"] = (prev - out["interval_velocity_m_s"]) > mad_floor_m_s
    return out.sort_index()

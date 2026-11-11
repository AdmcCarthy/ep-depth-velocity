"""Sonic log against check shots: the integrated sonic two-way time, the drift (check-shot time minus integrated sonic time) at every check-shot level, and what the drift curve looks like.
Pure functions; only the pipeline and its stage modules (`src/pipeline.py`, `src/stages/`) reads and writes.

A sonic log gives a detailed time-depth relationship by integrating the slowness; a check shot gives the seismic-frequency time at a depth.  Their difference, the drift, is the quantity that is removed when a sonic log is calibrated to check shots.  Here it is a FINDING: neither record is corrected and no
calibrated time-depth table is made.  Rules: the slowness is integrated against the vertical depth of the selected deviation survey (an isotropic medium is assumed, so the slowness along the borehole equals the vertical slowness: anisotropy is a limit); the integration starts at the top of the sonic
interval (the drift is zero there by construction, as in the usual workflow) and only inside the interval where the curve exists; a gap in the curve up to `max_gap_m` is bridged by linear interpolation of the slowness and counted, a longer gap ends the segment; a slowness outside the plausible range is
counted and kept; the check-shot time is read at the same measured depth (an offset of the table's depth reference, tested elsewhere, is applied as a given).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSIONS = {"well_sonic_drift": "1.0", "well_sonic_drift_summary": "1.0", "well_checkshot_table_compare": "1.0", "well_timedepth_calibrated": "1.0", "well_timedepth_calibrated_summary": "1.0"}
US_PER_FT_TO_US_PER_M = 1.0 / 0.3048


def sonic_twt(curve: pd.DataFrame, survey: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The integrated two-way time of the sonic log.  `curve` has `md_m` and `value` (slowness [us/ft]); `survey` has `md_m` and `tvdss_m`.  Returns a frame `md_m`, `tvdss_m`, `sonic_twt_ms` (relative to the top of the sonic interval) and the statistics: interval, samples, bridged gap length,
    samples outside the plausible slowness range and the longest gap.  The slowness is linearly interpolated over a gap of at most `max_gap_m`; the segment that holds most of the samples is kept when a longer gap occurs."""
    c = curve.dropna(subset=["value"]).sort_values("md_m")
    md = c["md_m"].to_numpy(float)
    sl = c["value"].to_numpy(float)
    lo, hi = cfg["plausible_us_ft"]
    stats: dict[str, Any] = {"samples": int(len(md)), "outside_plausible": int(((sl < lo) | (sl > hi)).sum()), "bridged_gap_m": 0.0, "longest_gap_m": 0.0, "segments": 1}
    if len(md) < 3:
        return pd.DataFrame(columns=["md_m", "tvdss_m", "sonic_twt_ms"]), {**stats, "interval_md_m": (np.nan, np.nan)}
    gaps = np.diff(md)
    step = float(np.median(gaps))
    big = np.where(gaps > cfg["max_gap_m"] + step)[0]
    stats["longest_gap_m"] = float(gaps.max() - step) if gaps.max() > 1.5 * step else 0.0
    if len(big):                                                                      # a gap longer than the limit ends the segment: keep the longest
        edges = np.r_[0, big + 1, len(md)]
        k = int(np.argmax(np.diff(edges)))
        md, sl = md[edges[k]:edges[k + 1]], sl[edges[k]:edges[k + 1]]
        stats["segments"] = int(len(edges) - 1)
    grid = np.arange(md[0], md[-1] + 1e-9, step)
    s_grid = np.interp(grid, md, sl)
    stats["bridged_gap_m"] = float(max(0.0, (grid[-1] - grid[0]) - (len(md) - 1) * step))
    z = np.interp(grid, survey["md_m"].to_numpy(float), survey["tvdss_m"].to_numpy(float), left=np.nan, right=np.nan)
    ok = np.isfinite(z)
    grid, s_grid, z = grid[ok], s_grid[ok], z[ok]
    if len(grid) < 3:
        return pd.DataFrame(columns=["md_m", "tvdss_m", "sonic_twt_ms"]), {**stats, "interval_md_m": (np.nan, np.nan)}
    dz = np.diff(z)
    one_way_ms = np.r_[0.0, np.cumsum(0.5 * (s_grid[1:] + s_grid[:-1]) * US_PER_FT_TO_US_PER_M * dz / 1000.0)]
    return pd.DataFrame({"md_m": grid, "tvdss_m": z, "sonic_twt_ms": 2.0 * one_way_ms}), {**stats, "interval_md_m": (float(grid[0]), float(grid[-1]))}


def drift_levels(table: pd.DataFrame, son: pd.DataFrame, md_offset_m: float = 0.0) -> pd.DataFrame:
    """The drift at each check-shot level inside the sonic interval: `drift_ms` = (check-shot time - its time at the top of the sonic interval) - integrated sonic time.  A negative drift means the integrated sonic is slower (longer) than the check shot.
    Also the interval velocities of both between successive levels (from the vertical depth of the survey) and the drift per level.  `table` has `md_m`, `twt_ms`; `md_m` + `md_offset_m` is the measured depth from the rotary table."""
    t = table.assign(md_rt=table["md_m"].to_numpy(float) + md_offset_m).sort_values("md_rt")
    lo, hi = float(son["md_m"].min()), float(son["md_m"].max())
    t = t[(t["md_rt"] >= lo) & (t["md_rt"] <= hi)].reset_index(drop=True)
    if len(t) < 3:
        return pd.DataFrame()
    cs0 = float(t["twt_ms"].iloc[0])
    sn = np.interp(t["md_rt"], son["md_m"], son["sonic_twt_ms"])
    z = np.interp(t["md_rt"], son["md_m"], son["tvdss_m"])
    sn = sn - sn[0]
    out = pd.DataFrame({"md_m": t["md_rt"], "tvdss_m": z, "checkshot_twt_ms": t["twt_ms"].to_numpy(float), "checkshot_rel_ms": t["twt_ms"].to_numpy(float) - cs0, "sonic_rel_ms": sn})
    out["drift_ms"] = out["checkshot_rel_ms"] - out["sonic_rel_ms"]
    dz = np.diff(z)
    dt_cs, dt_sn = np.diff(out["checkshot_rel_ms"].to_numpy()), np.diff(sn)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["v_int_checkshot_m_s"] = np.r_[np.nan, 2000.0 * dz / dt_cs]
        out["v_int_sonic_m_s"] = np.r_[np.nan, 2000.0 * dz / dt_sn]
    out["drift_step_ms"] = out["drift_ms"].diff()
    out["reference_md_m"] = float(t["md_rt"].iloc[0])
    return out


def drift_summary(levels: pd.DataFrame, stats: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """One row per table: the interval, the number of levels, the drift at the end, the mean drift rate (ms per km of vertical depth) from a straight line, its smoothness (the standard deviation of the drift step between successive levels and the share of steps larger than `step_ms`),
    whether the drift keeps one sign, the ratio of the mean interval velocities (check shot over sonic) and the quality of the sonic (samples outside the plausible slowness range, bridged and longest gap).  Findings only."""
    if levels.empty:
        return {"levels": 0}
    d = levels["drift_ms"].to_numpy(float)
    z = levels["tvdss_m"].to_numpy(float)
    slope = float(np.polyfit(z, d, 1)[0] * 1000.0) if len(d) > 2 and np.ptp(z) > 0 else np.nan
    step = levels["drift_step_ms"].dropna().to_numpy(float)
    vc = (levels["tvdss_m"].iloc[-1] - levels["tvdss_m"].iloc[0]) / ((levels["checkshot_rel_ms"].iloc[-1] - levels["checkshot_rel_ms"].iloc[0]) / 2000.0)
    vs = (levels["tvdss_m"].iloc[-1] - levels["tvdss_m"].iloc[0]) / ((levels["sonic_rel_ms"].iloc[-1] - levels["sonic_rel_ms"].iloc[0]) / 2000.0)
    return {"levels": int(len(levels)), "md_from_m": float(levels["md_m"].iloc[0]), "md_to_m": float(levels["md_m"].iloc[-1]), "tvdss_from_m": float(z[0]), "tvdss_to_m": float(z[-1]), "drift_end_ms": float(d[-1]), "drift_min_ms": float(d.min()), "drift_max_ms": float(d.max()),
            "drift_rate_ms_per_km": slope, "step_std_ms": float(step.std()) if len(step) else np.nan, "share_steps_over_limit": float((np.abs(step) > cfg["step_ms"]).mean()) if len(step) else np.nan,
            "one_sign": bool((d[1:] <= 0).all() or (d[1:] >= 0).all()), "v_checkshot_m_s": float(vc), "v_sonic_m_s": float(vs), "v_ratio_checkshot_over_sonic": float(vc / vs), "sonic_samples": stats["samples"], "sonic_outside_plausible": stats["outside_plausible"],
            "sonic_bridged_gap_m": stats["bridged_gap_m"], "sonic_longest_gap_m": stats["longest_gap_m"]}


def table_compare(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Check-shot tables of one wellbore against each other at the measured depths they share (within 0.5 m): the median and largest absolute difference of the two-way time and the number of shared levels.  A difference between two processings is a finding about processing, not an error of either."""
    rows = []
    names = sorted(tables)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ta, tb = tables[a].sort_values("md_m"), tables[b].sort_values("md_m")
            j = np.searchsorted(tb["md_m"].to_numpy(float), ta["md_m"].to_numpy(float))
            j = np.clip(j, 1, len(tb) - 1)
            near = np.where(np.abs(tb["md_m"].to_numpy(float)[j - 1] - ta["md_m"].to_numpy(float)) < np.abs(tb["md_m"].to_numpy(float)[j] - ta["md_m"].to_numpy(float)), j - 1, j)
            dm = np.abs(tb["md_m"].to_numpy(float)[near] - ta["md_m"].to_numpy(float))
            ok = dm <= 0.5
            if ok.sum():
                diff = tb["twt_ms"].to_numpy(float)[near][ok] - ta["twt_ms"].to_numpy(float)[ok]
                rows.append({"table_a": a, "table_b": b, "shared_levels": int(ok.sum()), "median_diff_ms": float(np.median(diff)), "max_abs_diff_ms": float(np.abs(diff).max()), "md_from_m": float(ta["md_m"].to_numpy(float)[ok].min()), "md_to_m": float(ta["md_m"].to_numpy(float)[ok].max())})
    return pd.DataFrame(rows, columns=["table_a", "table_b", "shared_levels", "median_diff_ms", "max_abs_diff_ms", "md_from_m", "md_to_m"])


def calibrated_table(son: pd.DataFrame, lv: pd.DataFrame, cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    """A sonic-calibrated time-depth table: the integrated sonic time plus the drift curve, from the first check-shot level to the end of the sonic.  The drift between check-shot levels is interpolated linearly (the usual first choice; the knees at the levels are not smoothed); below the last level the last
    drift is held constant (flagged `extrapolated_below_last_level`, no trend is extended); above the first level there is no table.  The time at the first level is the check-shot time there, so the calibrated time equals the check shot at every level by construction (`max_level_residual_ms` says so).
    This is a MODELLED relationship (a sonic with an interpolated correction), not a measurement: it carries the sonic's depth detail and the check shot's seismic-frequency times, and nothing here says which of the two is right where they differ.
    A near-horizontal stretch (the vertical depth changes by less than `near_horizontal_dtvd_dmd` per metre of measured depth over 10 m, an assumption) makes the relationship between time and vertical depth ill-conditioned: a correction that is small in time is large against the tiny depth step, so the time can run backwards with depth; such samples are flagged (`near_horizontal`, `time_reversal`) and carry no interval velocity.
    Returns `md_m`, `tvdss_m`, `twt_ms` (calibrated), `sonic_uncorrected_twt_ms` (the same start, no drift), `drift_applied_ms`, `extrapolated_below_last_level`, `near_horizontal`, `time_reversal`, `interval_velocity_m_s` (over `velocity_window_m` metres of measured depth, default 50, centred) and the statistics."""
    limit = float((cfg or {}).get("near_horizontal_dtvd_dmd", 0.2))
    first, last = float(lv["md_m"].iloc[0]), float(lv["md_m"].iloc[-1])
    s = son[son["md_m"] >= first].reset_index(drop=True)
    if len(s) < 11:
        return pd.DataFrame(), {}
    t0 = float(lv["checkshot_twt_ms"].iloc[0])
    rel = s["sonic_twt_ms"].to_numpy(float) - float(np.interp(first, son["md_m"], son["sonic_twt_ms"]))
    drift = np.interp(s["md_m"].to_numpy(float), lv["md_m"].to_numpy(float), lv["drift_ms"].to_numpy(float), left=float(lv["drift_ms"].iloc[0]), right=float(lv["drift_ms"].iloc[-1]))
    out = pd.DataFrame({"md_m": s["md_m"], "tvdss_m": s["tvdss_m"], "twt_ms": t0 + rel + drift, "sonic_uncorrected_twt_ms": t0 + rel, "drift_applied_ms": drift, "extrapolated_below_last_level": s["md_m"].to_numpy(float) > last})
    z, t, m = out["tvdss_m"].to_numpy(float), out["twt_ms"].to_numpy(float), out["md_m"].to_numpy(float)
    w = int(round(float((cfg or {}).get("velocity_window_m", 50.0))))                                  # samples of the 1 m grid; the velocity and the flat test use the same window
    grad = np.r_[np.full(w // 2, np.nan), (z[w:] - z[:-w]) / (m[w:] - m[:-w]), np.full(w - w // 2, np.nan)]
    flat = np.where(np.isfinite(grad), grad < limit, False)
    flat = np.convolve(flat.astype(int), np.ones(w + 1, dtype=int), mode="same") > 0                  # a sample near a flat stretch is flagged too
    with np.errstate(divide="ignore", invalid="ignore"):
        v = 2000.0 * (z[w:] - z[:-w]) / (t[w:] - t[:-w])
    vel = np.r_[np.full(w // 2, np.nan), v, np.full(w - w // 2, np.nan)]
    vel[flat] = np.nan
    out["near_horizontal"] = flat
    out["time_reversal"] = np.r_[False, np.diff(t) <= 0]
    out["interval_velocity_m_s"] = vel
    at_lv = np.interp(lv["md_m"].to_numpy(float), out["md_m"].to_numpy(float), out["twt_ms"].to_numpy(float))
    vv = out["interval_velocity_m_s"].dropna().to_numpy(float)
    stats = {"samples": int(len(out)), "md_from_m": float(out["md_m"].iloc[0]), "md_to_m": float(out["md_m"].iloc[-1]), "tvdss_from_m": float(z[0]), "tvdss_to_m": float(z[-1]), "twt_from_ms": float(t[0]), "twt_to_ms": float(t[-1]),
             "drift_applied_min_ms": float(drift.min()), "drift_applied_max_ms": float(drift.max()), "share_extrapolated": float(out["extrapolated_below_last_level"].mean()),
             "max_level_residual_ms": float(np.abs(at_lv - lv["checkshot_twt_ms"].to_numpy(float)).max()), "share_near_horizontal": float(flat.mean()), "time_reversal_samples": int(out["time_reversal"].sum()), "v_int_min_m_s": float(vv.min()) if len(vv) else np.nan, "v_int_max_m_s": float(vv.max()) if len(vv) else np.nan,
             "v_int_max_step_m_s": float(np.abs(np.diff(vv)).max()) if len(vv) > 1 else np.nan}
    return out, stats

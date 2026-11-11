"""Time-depth relations from VSP / check-shot tables and their checks against the deviation survey and the interpretation team's well picks.  Pure functions; only the pipeline and its stage modules (`src/pipeline.py`, `src/stages/`) reads and writes.

A check-shot table ties depth in the well to two-way time on the seismic.  It is only as good as (1) its depth reference (MD from the rotary table or from sea level: tested against the survey, never assumed),
(2) its time reference (MSL, as the files state) and (3) the first-break picking.  A difference to the picks is a finding, never corrected: the picks come from another interpretation and may rest on a different velocity model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSIONS = {"well_timedepth": "1.0", "well_timedepth_summary": "1.0", "well_timedepth_pick_check": "1.0"}


def timedepth_table(parts: list[tuple[str, str, dict[str, Any], pd.DataFrame]], folder_wellbores: dict[str, str]) -> pd.DataFrame:
    """The curves of all members stacked: `member`, `well_folder`, `wellbore_name` (adapter mapping, blank when the folder holds several wellbores), the layout facts, depth, TVD below MSL, two-way time, the average velocity
    `2 x TVDSS / TWT` and the interval velocity between neighbouring rows."""
    out = []
    for member, folder, meta, d in parts:
        if d.empty:
            continue
        x = d.copy()
        x["member"], x["well_folder"] = member, folder
        x["wellbore_name"] = folder_wellbores.get(folder, "")
        x["layout"], x["md_basis"], x["curve"] = meta["layout"], meta["md_basis"], meta.get("curve", "")
        x["v_avg_m_s"] = 2000.0 * x["tvdss_m"] / x["twt_ms"].replace(0, np.nan)
        dz, dt = x["tvdss_m"].diff(), x["twt_ms"].diff()
        x["v_int_m_s"] = 2000.0 * dz / dt.replace(0, np.nan)
        out.append(x)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def timedepth_summary(td: pd.DataFrame, v_range: tuple[float, float]) -> pd.DataFrame:
    """One row per member: samples, depth and time range, whether time increases with depth, the range of the average velocity and the number of interval velocities outside `v_range` (m/s; an assumed plausibility band)."""
    rows = []
    lo, hi = v_range
    for member, g in td.groupby("member"):
        rows.append({"member": member, "well_folder": g["well_folder"].iloc[0], "wellbore_name": g["wellbore_name"].iloc[0], "layout": g["layout"].iloc[0], "curve": g["curve"].iloc[0], "md_basis": g["md_basis"].iloc[0], "samples": len(g),
                     "md_min_m": g["md_m"].min(), "md_max_m": g["md_m"].max(), "tvdss_max_m": g["tvdss_m"].max(), "twt_min_ms": g["twt_ms"].min(), "twt_max_ms": g["twt_ms"].max(),
                     "time_monotonic": bool(g["twt_ms"].is_monotonic_increasing), "v_avg_min_m_s": g["v_avg_m_s"].min(), "v_avg_max_m_s": g["v_avg_m_s"].max(),
                     "v_int_outside_band": int(((g["v_int_m_s"] < lo) | (g["v_int_m_s"] > hi)).sum())})
    return pd.DataFrame(rows)


def survey_check(td: pd.DataFrame, survey_tvdss, rt_m: dict[str, float]) -> pd.DataFrame:
    """The check-shot depth against the selected deviation survey at the same MD.  A table whose depth is `MSL` or not stated is tested with both offsets (0 and the rotary-table elevation) and the better one is reported with
    BOTH residuals: the choice is evidence, not an assumption.  `survey_tvdss(wellbore_name, md_array)` gives the survey TVD below MSL (NaN outside the surveyed interval).  Residual = check shot minus survey [m]."""
    rows = []
    for member, g in td.groupby("member"):
        name = g["wellbore_name"].iloc[0]
        if not name:
            continue
        rt = float(rt_m.get(name, np.nan))
        offsets = {"RKB": [0.0]}.get(g["md_basis"].iloc[0], [0.0, rt])
        res = {}
        for off in offsets:
            if np.isnan(off):
                continue
            s = np.asarray(survey_tvdss(name, g["md_m"].to_numpy(float) + off), dtype=float)
            r = g["tvdss_m"].to_numpy(float) - s
            res[off] = r[~np.isnan(r)]
        if not res or all(len(v) == 0 for v in res.values()):
            rows.append({"member": member, "wellbore_name": name, "status": "no survey at these depths"})
            continue
        best = min(res, key=lambda k: np.median(np.abs(res[k])) if len(res[k]) else np.inf)
        r = res[best]
        rows.append({"member": member, "wellbore_name": name, "md_offset_used_m": best, "md_offset_note": "none (MD from the rotary table)" if g["md_basis"].iloc[0] == "RKB" else ("0 = depth from sea level" if best == 0.0 else "rotary-table elevation added"),
                     "n": len(r), "residual_median_m": float(np.median(r)), "residual_abs_max_m": float(np.max(np.abs(r))), "residual_alt_median_m": [float(np.median(v)) for k, v in res.items() if k != best and len(v)][0] if len(res) > 1 and any(k != best and len(v) for k, v in res.items()) else np.nan,
                     "status": "agrees within 5 m" if np.median(np.abs(r)) <= 5 else "differs"})
    return pd.DataFrame(rows)


def pick_check(td: pd.DataFrame, picks: pd.DataFrame, candidates: dict[str, list[str]]) -> pd.DataFrame:
    """Each pick of the wellbore (or of every candidate wellbore of a folder that holds several) against the check shot: the pick's depth below MSL (`-tvdss` in the picks file, negative downwards) is looked up in the table
    inside its depth range, and `residual_ms` = pick TWT minus check-shot TWT (positive: the pick is later).  Picks without TWT or depth, and picks outside the table, are listed with the reason."""
    rows = []
    for member, g in td.groupby("member"):
        names = [g["wellbore_name"].iloc[0]] if g["wellbore_name"].iloc[0] else candidates.get(g["well_folder"].iloc[0], [])
        z, t = g["tvdss_m"].to_numpy(float), g["twt_ms"].to_numpy(float)
        order = np.argsort(z)
        z, t = z[order], t[order]
        for nm in names:
            p = picks[picks["well"].eq(nm)]
            for r in p.itertuples():
                depth = -r.tvdss if pd.notna(r.tvdss) else np.nan
                if pd.isna(depth) or pd.isna(r.twt):
                    why = "pick has no depth or TWT"
                elif depth < z.min() or depth > z.max():
                    why = "outside the check-shot depth range"
                else:
                    why = ""
                cs = float(np.interp(depth, z, t)) if why == "" else np.nan
                rows.append({"member": member, "wellbore_candidate": nm, "surface_name": r.surface_name, "qualifier": r.qualifier, "pick_depth_below_msl_m": depth, "pick_twt_ms": r.twt, "checkshot_twt_ms": cs,
                             "residual_ms": r.twt - cs if why == "" else np.nan, "not_checked_because": why})
    return pd.DataFrame(rows)


def pick_check_summary(chk: pd.DataFrame, tolerance_ms: float) -> pd.DataFrame:
    """Per member and candidate wellbore: picks checked, median and largest residual, and the share within `tolerance_ms` (an assumed tolerance).  For a folder with several candidate wellbores the smaller median says which one the table belongs to."""
    c = chk[chk["not_checked_because"].eq("")]
    g = c.groupby(["member", "wellbore_candidate"]).agg(picks_checked=("residual_ms", "size"), residual_median_ms=("residual_ms", "median"), residual_abs_max_ms=("residual_ms", lambda s: float(s.abs().max())),
                                                       share_within_tolerance=("residual_ms", lambda s: float((s.abs() <= tolerance_ms).mean()))).reset_index()
    skipped = chk.groupby(["member", "wellbore_candidate"]).apply(lambda x: int(x["not_checked_because"].ne("").sum()), include_groups=False).rename("picks_not_checked").reset_index()
    return g.merge(skipped, on=["member", "wellbore_candidate"], how="outer")

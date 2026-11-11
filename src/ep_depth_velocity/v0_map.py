"""A map of V0 from the wells: the velocity at the top of a layer (V0 of a V0-k function, or the constant interval velocity) interpolated between the wells, and judged by prediction of left-out wells.
Pure functions; only the pipeline and its stage modules (`src/pipeline.py`, `src/stages/`) reads and writes.

Each well gives one V0 per layer, solved exactly from its own thickness and one-way time for a given k (`layer_cake`).  A map is a function of position fitted to those values.  Methods (all fitted on the training wells only): `area` (the median: no map, the baseline of notebook 46),
`plane` (a plane in easting and northing, ordinary least squares), `idw` (inverse distance weighting, power 2), `krige` (ordinary kriging with an exponential covariance whose range is chosen by prediction of the training wells themselves) and `plane_krige` (the plane plus kriging of its residuals).
Wells of one SLOT lie within a few hundred metres of each other and share their velocity almost exactly, so leaving out one wellbore would leave its neighbours in the training set and flatter every spatial method: the evaluation LEAVES OUT A WHOLE GROUP of wells within `group_distance_m` of each other (single linkage).  The error is the predicted minus the observed thickness of the layer with the
well's own two-way time; a map is judged against `area` by a paired bootstrap over groups (an interval that contains zero means the map cannot be told from one area value).  A prediction is limited to the range of the training values widened by `clip_fraction` of their spread (a plane extrapolates quickly).
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd

from ep_depth_velocity import depth_conversion as dc
from ep_depth_velocity import layer_cake as lc

SCHEMA_VERSIONS = {"v0_map_eval": "1.0", "v0_map_well": "1.0", "v0_map_surface": "1.0", "v0_map_team_check": "1.0"}
METHODS = ["area", "plane", "idw", "krige", "plane_krige"]


def layer_observations_xy(bt: pd.DataFrame, top: str, base: str, min_thickness_m: float) -> pd.DataFrame:
    """`layer_cake.layer_observations` with the position of the well for the layer: the mean of the picks of the top and the base (easting `x`, northing `y`); wells without a position are dropped."""
    o = lc.layer_observations(bt, top, base, min_thickness_m)
    a, b = bt[bt["boundary"] == top].set_index("well"), bt[bt["boundary"] == base].set_index("well")
    xs = pd.concat([a["x_m"].reindex(o["well"]).reset_index(drop=True), b["x_m"].reindex(o["well"]).reset_index(drop=True)], axis=1).mean(axis=1)
    ys = pd.concat([a["y_m"].reindex(o["well"]).reset_index(drop=True), b["y_m"].reindex(o["well"]).reset_index(drop=True)], axis=1).mean(axis=1)
    return o.assign(x=xs.to_numpy(), y=ys.to_numpy()).dropna(subset=["x", "y"]).reset_index(drop=True)


def groups_by_distance(xy: np.ndarray, max_distance_m: float) -> np.ndarray:
    """Group labels: wells that are linked by a chain of neighbours closer than `max_distance_m` belong to one group (single linkage)."""
    n = len(xy)
    lab = np.arange(n)
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] <= max_distance_m and lab[i] != lab[j]:
                lab[lab == lab[j]] = lab[i]
    _, inv = np.unique(lab, return_inverse=True)
    return inv


def _cov(h: np.ndarray, sill: float, range_m: float) -> np.ndarray:
    return sill * np.exp(-3.0 * h / range_m)


def _krige_fit(xy: np.ndarray, v: np.ndarray, range_m: float, nugget_fraction: float) -> Callable[[np.ndarray], np.ndarray]:
    sill = float(max(v.var(), 1e-9))
    n = len(v)
    h = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    k = np.zeros((n + 1, n + 1))
    k[:n, :n] = _cov(h, sill * (1 - nugget_fraction), range_m) + np.eye(n) * sill * nugget_fraction
    k[:n, n] = k[n, :n] = 1.0
    kinv = np.linalg.pinv(k)

    def pred(pts: np.ndarray) -> np.ndarray:
        out = np.empty(len(pts))
        for s in range(0, len(pts), 5000):
            p = pts[s:s + 5000]
            hp = np.hypot(p[:, None, 0] - xy[None, :, 0], p[:, None, 1] - xy[None, :, 1])
            rhs = np.hstack([_cov(hp, sill * (1 - nugget_fraction), range_m), np.ones((len(p), 1))])
            w = rhs @ kinv
            out[s:s + 5000] = w[:, :n] @ v
        return out
    return pred


def fit_surface(method: str, xy: np.ndarray, v: np.ndarray, cfg: dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    """A function of position (an array of easting, northing) fitted to the V0 values of the training wells.  With fewer than `min_wells_map` training wells every method falls back to the median."""
    med = float(np.median(v))
    lo, hi = float(v.min()), float(v.max())
    margin = float(cfg["clip_fraction"]) * (hi - lo)

    def clipped(f: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
        return lambda p: np.clip(f(p), lo - margin, hi + margin)

    if method == "area" or len(v) < int(cfg["min_wells_map"]):
        return lambda p: np.full(len(p), med)
    cx, cy = float(xy[:, 0].mean()), float(xy[:, 1].mean())
    a = np.column_stack([np.ones(len(xy)), (xy[:, 0] - cx) / 1000.0, (xy[:, 1] - cy) / 1000.0])
    plane = lambda p: np.column_stack([np.ones(len(p)), (p[:, 0] - cx) / 1000.0, (p[:, 1] - cy) / 1000.0]) @ np.linalg.lstsq(a, v, rcond=None)[0]
    if method == "plane":
        return clipped(plane)
    if method == "idw":
        def idw(p: np.ndarray) -> np.ndarray:
            d = np.maximum(np.hypot(p[:, None, 0] - xy[None, :, 0], p[:, None, 1] - xy[None, :, 1]), 50.0)
            w = 1.0 / d ** 2
            return (w * v[None, :]).sum(1) / w.sum(1)
        return clipped(idw)
    ranges = [float(r) for r in cfg["krige_ranges_m"]]
    nug = float(cfg["nugget_fraction"])

    def best_range(xy_: np.ndarray, v_: np.ndarray, target: np.ndarray | None = None) -> float:           # the range with the smallest error in predicting each training well from the others
        errs = []
        for r in ranges:
            e = []
            for i in range(len(v_)):
                m = np.ones(len(v_), bool)
                m[i] = False
                e.append(float(_krige_fit(xy_[m], v_[m], r, nug)(xy_[i:i + 1])[0]) - v_[i])
            errs.append(float(np.sqrt(np.mean(np.square(e)))))
        return ranges[int(np.argmin(errs))]

    if method == "krige":
        return clipped(_krige_fit(xy, v, best_range(xy, v), nug))
    if method == "plane_krige":
        res = v - np.asarray(plane(xy))
        kr = _krige_fit(xy, res, best_range(xy, res), nug)
        return clipped(lambda p: plane(p) + kr(p))
    raise ValueError(f"unknown method {method}")


def evaluate_layer(obs: pd.DataFrame, cfg: dict[str, Any], methods: list[str], k_mode: str, lcfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-group-out prediction of the thickness of one layer for every method.  Returns `well` (observed V0, the prediction and the thickness error per well and method) and `eval` (per method: RMS, bias, 90 % interval of the RMS, the paired difference against `area`, and a verdict).
    k is chosen inside every training set as in `layer_cake` (`k_mode`: `const` = 0, `v0k` = the best k) and V0 of the training wells is solved at that k."""
    xy = obs[["x", "y"]].to_numpy(float)
    grp = groups_by_distance(xy, float(cfg["group_distance_m"]))
    rows = []
    for g in np.unique(grp):
        te = np.where(grp == g)[0]
        tr = np.where(grp != g)[0]
        if len(tr) < int(cfg["min_wells_map"]):
            continue
        k, _ = lc.choose_k(obs.iloc[tr].reset_index(drop=True), lcfg, k_mode)
        v_tr = lc._v0(obs["dz"].to_numpy(float)[tr], obs["dt"].to_numpy(float)[tr], k)
        for m in methods:
            f = fit_surface(m, xy[tr], v_tr, cfg)
            v_te = f(xy[te])
            for i, vt in zip(te, v_te):
                dz_hat = float(dc.v0k_depth(np.array([obs["dt"].iloc[i]]), float(vt), k)[0])
                rows.append({"well": obs["well"].iloc[i], "group": int(g), "method": m, "k": float(k), "v0_observed": float(lc._v0(np.array([obs["dz"].iloc[i]]), np.array([obs["dt"].iloc[i]]), k)[0]), "v0_predicted": float(vt),
                             "dz_observed_m": float(obs["dz"].iloc[i]), "dz_predicted_m": dz_hat, "error_m": dz_hat - float(obs["dz"].iloc[i]), "x": float(xy[i, 0]), "y": float(xy[i, 1])})
    well = pd.DataFrame(rows)
    if well.empty:
        return well, pd.DataFrame()
    base = well[well["method"] == "area"].set_index("well")["error_m"]
    out = []
    ngroups = int(well["group"].nunique())
    for m in methods:
        e = well[well["method"] == m].set_index("well")["error_m"]
        com = e.index.intersection(base.index)
        lo, hi = lc.bootstrap_rmse(e.loc[com].to_numpy())
        d = lc.paired_rmse_difference(e.loc[com].to_numpy(), base.loc[com].to_numpy()) if m != "area" else {"diff_m": 0.0, "p05_m": 0.0, "p95_m": 0.0}
        verdict = "baseline" if m == "area" else ("not distinguishable" if d["p05_m"] <= 0 <= d["p95_m"] else ("closer than area" if d["p95_m"] < 0 else "worse than area"))
        out.append({"method": m, "wells": int(len(com)), "groups": ngroups, "rmse_m": float(np.sqrt(np.mean(e.loc[com] ** 2))), "rmse_p05_m": lo, "rmse_p95_m": hi, "bias_m": float(e.loc[com].mean()), "diff_vs_area_m": d["diff_m"], "diff_p05_m": d["p05_m"], "diff_p95_m": d["p95_m"], "verdict": verdict})
    return well, pd.DataFrame(out)


def recommended_method(ev: pd.DataFrame) -> str:
    """The method to use for a layer: the one with the smallest RMS among those whose paired difference against `area` excludes zero on the side of improvement; otherwise `area` (a map that cannot be told from one value is not used)."""
    better = ev[ev["verdict"] == "closer than area"]
    return "area" if better.empty else str(better.sort_values("rmse_m").iloc[0]["method"])


def surface_grid(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, y: np.ndarray, cell_m: float) -> pd.DataFrame:
    """The map on square cells of `cell_m` that contain at least one of the points (`x`, `y`: for example the nodes of a horizon): `cell_x`, `cell_y`, `v0`."""
    cx, cy = (x // cell_m) * cell_m + cell_m / 2, (y // cell_m) * cell_m + cell_m / 2
    cells = np.unique(np.column_stack([cx, cy]), axis=0)
    return pd.DataFrame({"cell_x": cells[:, 0], "cell_y": cells[:, 1], "v0": fn(cells)})

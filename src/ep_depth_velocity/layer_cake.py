"""Layer-cake depth models: a velocity function per layer (a constant interval velocity or V0-k with V0 at the layer top), calibrated to well tops and judged by leave-one-well-out prediction of the depth at the base of a layer or a chain of layers.
Pure functions; only the pipeline and its stage modules (`src/pipeline.py`, `src/stages/`) reads and writes.

Each layer is described by the wells that have its top and its base: depth below the layer top `dz` and one-way time below it `dt` (the picks give depth; the time is the picks' two-way time, the seismic horizon's, or the seabed's from the water velocity).  For a trial k the velocity at the top
of the layer in each well is solved exactly from its own (dz, dt): V0 = k dz / (exp(k dt) - 1) (V0 = dz / dt for k = 0).  A held-out well is predicted with the MEDIAN V0 of the other wells (V0 as one value for the area: no kriging, so any lateral velocity trend is left in the error) and its own time
`dt`, and the error is predicted minus observed `dz`.  k is chosen by the smallest leave-one-well-out RMS; in the chain evaluation k is chosen again inside every training set (nested), so that a held-out well never takes part in its own calibration.  The mode `const` is k = 0, `v0k` is the best k, `auto` is V0-k only
when it lowers the RMS by at least `min_improvement` against the constant velocity (parsimony).  A model is a chain of layers and its error is the error of the depth at the base of the last layer when the time is known at every boundary of the chain.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ep_depth_velocity import depth_conversion as dc

SCHEMA_VERSIONS = {"depth_model_boundary": "1.0", "depth_model_layer": "1.0", "depth_model_layer_profile": "1.0", "depth_model_chain": "1.0", "depth_model_chain_well": "1.0", "depth_model_compare": "1.0"}


def k_values(cfg: dict[str, Any]) -> np.ndarray:
    lo, hi, n = cfg["k_grid"]
    k = np.linspace(float(lo), float(hi), int(n))
    return np.unique(np.r_[k, 0.0])                                                         # 0 (constant velocity) is always on the grid


def boundary_table(picks: pd.DataFrame, scheme: dict[str, Any], water_velocity: float, horizon_twt: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """The boundaries of a scheme at each well: the first available surface of the boundary's list with a depth (and, for `time: pick`, a pick time; for `time: horizon`, a horizon time at the well), its depth `z_m` below sea level, its two-way time `t_ms` and the position of the pick (`x_m`, `y_m`: easting and northing of the penetration, when the picks carry them).
    `picks` has `well`, `surface_name`, `z_m` (depth below sea level, positive down) and `pick_twt_ms`; `horizon_twt[name]` has `well` and `twt_ms`.  `time: water` gives 2000 z / v of the seawater (assumed velocity).  `time_basis` says which."""
    rows = []
    horizon_twt = horizon_twt or {}
    for w, g in picks.groupby("well"):
        for b in scheme["boundaries"]:
            for s in b["surfaces"]:
                h = g[g["surface_name"] == s].dropna(subset=["z_m"])
                if h.empty:
                    continue
                z = float(h["z_m"].iloc[0])
                if b["time"] == "water":
                    t, basis = 2000.0 * z / water_velocity, "water velocity"
                elif b["time"] == "pick":
                    tt = h["pick_twt_ms"].dropna()
                    if tt.empty:
                        continue
                    t, basis = float(tt.iloc[0]), "pick"
                else:
                    ht = horizon_twt.get(b["name"])
                    hv = ht[ht["well"] == w]["twt_ms"].dropna() if ht is not None else pd.Series(dtype=float)
                    if hv.empty:
                        continue
                    t, basis = float(hv.iloc[0]), "horizon"
                rows.append({"well": w, "boundary": b["name"], "group": b["group"], "surface_used": s, "z_m": z, "t_ms": t, "time_basis": basis,
                             "x_m": float(h["easting"].iloc[0]) if "easting" in h and pd.notna(h["easting"].iloc[0]) else np.nan, "y_m": float(h["northing"].iloc[0]) if "northing" in h and pd.notna(h["northing"].iloc[0]) else np.nan})
                break
    return pd.DataFrame(rows, columns=["well", "boundary", "group", "surface_used", "z_m", "t_ms", "time_basis", "x_m", "y_m"])


def layer_observations(bt: pd.DataFrame, top: str, base: str, min_thickness_m: float) -> pd.DataFrame:
    """Per well with both boundaries: depth and one-way time below the top of the layer (`dz` [m], `dt` [s]) and the interval velocity.  A thinner layer than `min_thickness_m` or a non-positive time is dropped."""
    a, b = bt[bt["boundary"] == top].set_index("well"), bt[bt["boundary"] == base].set_index("well")
    j = a[["z_m", "t_ms"]].join(b[["z_m", "t_ms"]], lsuffix="_top", rsuffix="_base", how="inner")
    out = pd.DataFrame({"well": j.index, "z_top_m": j["z_m_top"].to_numpy(), "z_base_m": j["z_m_base"].to_numpy(), "dz": (j["z_m_base"] - j["z_m_top"]).to_numpy(), "dt": ((j["t_ms_base"] - j["t_ms_top"]) / 2000.0).to_numpy()})
    out = out[(out["dz"] >= min_thickness_m) & (out["dt"] > 0)].reset_index(drop=True)
    out["v_int_m_s"] = out["dz"] / out["dt"]
    return out


def _v0(dz: np.ndarray, dt: np.ndarray, k: float) -> np.ndarray:
    return dz / dt if abs(k) < 1e-9 else k * dz / np.expm1(k * dt)


def lowo_profile(obs: pd.DataFrame, ks: np.ndarray) -> pd.DataFrame:
    """For each trial k the leave-one-well-out RMS and mean error of the layer thickness [m]: V0 of every well from its own thickness and time, the held-out well predicted with the median V0 of the others."""
    dz, dt = obs["dz"].to_numpy(float), obs["dt"].to_numpy(float)
    rows = []
    n = len(dz)
    if n < 3:                                                                               # with fewer than three wells a held-out prediction rests on one well: no profile
        return pd.DataFrame({"k": ks, "rmse_m": np.nan, "mean_error_m": np.nan, "max_abs_error_m": np.nan})
    for k in ks:
        v0 = _v0(dz, dt, float(k))
        err = np.empty(n)
        for i in range(n):
            med = float(np.median(np.delete(v0, i)))
            err[i] = float(dc.v0k_depth(np.array([dt[i]]), med, float(k))[0]) - dz[i]
        rows.append({"k": float(k), "rmse_m": float(np.sqrt(np.mean(err ** 2))), "mean_error_m": float(err.mean()), "max_abs_error_m": float(np.abs(err).max())})
    return pd.DataFrame(rows)


def choose_k(obs: pd.DataFrame, cfg: dict[str, Any], mode: str) -> tuple[float, dict[str, float]]:
    """The k of a mode: 0 for `const`; the smallest leave-one-well-out RMS for `v0k`; for `auto` that k only when it improves on k = 0 by `min_improvement`, else 0.  Also the RMS at k = 0, at the chosen k and the relative improvement."""
    prof = lowo_profile(obs, k_values(cfg))
    if prof["rmse_m"].isna().all():
        return 0.0, {"rmse_const_m": np.nan, "rmse_best_m": np.nan, "k_best": 0.0, "improvement": 0.0}
    r0 = float(prof.loc[prof["k"].abs() < 1e-9, "rmse_m"].iloc[0])
    best = prof.loc[prof["rmse_m"].idxmin()]
    imp = (r0 - float(best["rmse_m"])) / r0 if r0 > 0 else 0.0
    if mode == "const":
        k = 0.0
    elif mode == "v0k":
        k = float(best["k"])
    else:
        k = float(best["k"]) if imp >= float(cfg["min_improvement"]) else 0.0
    return k, {"rmse_const_m": r0, "rmse_best_m": float(best["rmse_m"]), "k_best": float(best["k"]), "improvement": imp}


def fit_layer(obs: pd.DataFrame, cfg: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    """One layer: the leave-one-well-out profile over k, the best k, the V0 (median and spread over wells) at k = 0 and at the best k, the improvement of V0-k over the constant velocity, the relative spread of the thickness (k is only identifiable when the thickness varies between wells) and the number of wells.  A best k on the edge of the searched grid is no optimum (the profile is monotonic over the grid): `k_at_grid_edge`, and then k is not called identifiable."""
    prof = lowo_profile(obs, k_values(cfg))
    r0 = float(prof.loc[prof["k"].abs() < 1e-9, "rmse_m"].iloc[0])
    best = prof.loc[prof["rmse_m"].idxmin()]
    kb = float(best["k"])
    v0c, v0k = _v0(obs["dz"].to_numpy(float), obs["dt"].to_numpy(float), 0.0), _v0(obs["dz"].to_numpy(float), obs["dt"].to_numpy(float), kb)
    imp = (r0 - float(best["rmse_m"])) / r0 if r0 > 0 else 0.0
    stats = {"wells": int(len(obs)), "thickness_median_m": float(obs["dz"].median()), "top_depth_min_m": float(obs["z_top_m"].min()), "top_depth_max_m": float(obs["z_top_m"].max()), "top_depth_sd_m": float(obs["z_top_m"].std()),
             "v_int_median_m_s": float(obs["v_int_m_s"].median()), "v_int_mad_m_s": float((obs["v_int_m_s"] - obs["v_int_m_s"].median()).abs().median()), "rmse_const_m": r0, "k_best": kb, "rmse_best_m": float(best["rmse_m"]),
             "improvement": imp, "v0_const_median_m_s": float(np.median(v0c)), "v0_best_median_m_s": float(np.median(v0k)), "v0_best_sd_m_s": float(np.std(v0k)),
             "bias_const_m": float(prof.loc[prof["k"].abs() < 1e-9, "mean_error_m"].iloc[0]), "bias_best_m": float(best["mean_error_m"]),
             "thickness_sd_rel": float(obs["dz"].std() / obs["dz"].median()), "k_at_grid_edge": bool(kb <= float(cfg["k_grid"][0]) + 1e-9 or kb >= float(cfg["k_grid"][1]) - 1e-9),
             "k_identifiable": bool(imp >= float(cfg["min_improvement"]) and float(obs["dz"].std() / obs["dz"].median()) > 0.05 and not (kb <= float(cfg["k_grid"][0]) + 1e-9 or kb >= float(cfg["k_grid"][1]) - 1e-9))}
    return stats, prof


def predict_layer(train: pd.DataFrame, dt_s: float, cfg: dict[str, Any], mode: str) -> float:
    """The thickness predicted for a layer with one-way time `dt_s` from the training wells' observations: k by the mode (chosen inside the training set), V0 the median of the training wells' V0 at that k."""
    k, _ = choose_k(train, cfg, mode)
    v0 = float(np.median(_v0(train["dz"].to_numpy(float), train["dt"].to_numpy(float), k)))
    return float(dc.v0k_depth(np.array([dt_s]), v0, k)[0])


def chain_evaluation(bt: pd.DataFrame, layers: list[list[str]], cfg: dict[str, Any], modes: list[str], eval_wells: set[str] | None = None) -> pd.DataFrame:
    """The leave-one-well-out error of the depth at the base of a chain of layers.  `layers` are (top, base) pairs that follow each other; `modes[i]` is the mode of layer i.  Only wells that have every boundary of the chain (and are in `eval_wells` when given) are judged; every other well that has a layer still trains it.
    For each such well: the observed depth at the end of the chain, the depth predicted by chaining the layers (each from the other wells' observations, nested k choice), and the error (predicted minus observed); `errors_by_layer` are the signed thickness errors."""
    obs = [layer_observations(bt, t, b, float(cfg["min_thickness_m"])) for t, b in layers]
    wells = set(obs[0]["well"])
    for o in obs[1:]:
        wells &= set(o["well"])
    if eval_wells is not None:                                                              # the wells that are judged (a common set for variants); every other well with the layer still trains
        wells &= set(eval_wells)
    rows = []
    for w in sorted(wells):
        pred_total, obs_total, errs = 0.0, 0.0, []
        for o, mode in zip(obs, modes):
            row = o[o["well"] == w].iloc[0]
            p = predict_layer(o[o["well"] != w], float(row["dt"]), cfg, mode)
            pred_total += p
            obs_total += float(row["dz"])
            errs.append(p - float(row["dz"]))
        rows.append({"well": w, "z_start_m": float(obs[0][obs[0]["well"] == w]["z_top_m"].iloc[0]), "observed_dz_m": obs_total, "predicted_dz_m": pred_total, "error_m": pred_total - obs_total, "errors_by_layer": "; ".join(f"{e:+.1f}" for e in errs)})
    return pd.DataFrame(rows, columns=["well", "z_start_m", "observed_dz_m", "predicted_dz_m", "error_m", "errors_by_layer"])


def chain_summary(ev: pd.DataFrame) -> dict[str, Any]:
    """RMS, mean (bias), standard deviation and largest absolute error of a chain evaluation, and the number of wells.  A best k on the edge of the searched grid is no optimum (the profile is monotonic over the grid): `k_at_grid_edge`, and then k is not called identifiable."""
    if ev.empty:
        return {"wells": 0}
    e = ev["error_m"].to_numpy(float)
    lo, hi = bootstrap_rmse(e)
    return {"wells": int(len(e)), "rmse_m": float(np.sqrt(np.mean(e ** 2))), "rmse_p05_m": lo, "rmse_p95_m": hi, "bias_m": float(e.mean()), "sd_m": float(e.std()), "max_abs_error_m": float(np.abs(e).max())}


def horizon_twt_at_wells(picks: pd.DataFrame, scheme: dict[str, Any], load_horizon, max_distance_m: float) -> dict[str, pd.DataFrame]:
    """For each boundary with `time: horizon`: the two-way time of the horizon at the position of the well's pick of that boundary (the first available surface of the list), taken from the nearest nodes only within `max_distance_m` (no value is made up beyond it).
    `load_horizon(fragment)` returns the horizon nodes (`x`, `y`, `z` in ms); `picks` has `easting`, `northing`."""
    from ep_seismic import seismic_interp as sip

    out: dict[str, pd.DataFrame] = {}
    for b in scheme["boundaries"]:
        if b["time"] != "horizon":
            continue
        pos = []
        for w, g in picks.groupby("well"):
            for s in b["surfaces"]:
                h = g[(g["surface_name"] == s) & g["easting"].notna() & g["northing"].notna() & g["z_m"].notna()]
                if len(h):
                    pos.append((w, float(h["easting"].iloc[0]), float(h["northing"].iloc[0])))
                    break
        if not pos:
            continue
        nodes = load_horizon(b["horizon"])
        v = sip.horizon_at(nodes, np.array([p[1] for p in pos]), np.array([p[2] for p in pos]), max_distance_m)
        out[b["name"]] = pd.DataFrame({"well": [p[0] for p in pos], "twt_ms": v["z"].to_numpy(float), "nearest_node_m": v["nearest_node_m"].to_numpy(float)})
    return out


def run_scheme(name: str, picks: pd.DataFrame, scheme: dict[str, Any], cfg: dict[str, Any], horizon_twt: dict[str, pd.DataFrame] | None = None) -> dict[str, pd.DataFrame]:
    """One scheme end to end.  Returns the frames: `compare` (paired comparisons with a bootstrap interval over wells), `boundary`, `layer` (statistics per layer), `profile` (leave-one-well-out RMS against k per layer), `chain` (summary per chain variant) and `chain_well` (error per well and variant).
    Chain variants: ABOVE BCU (a single function from the seabed, `direct`, or the layer-cake, `cake`, each with the modes `above_modes`) to the BCU; BELOW BCU (a block or the layers, each with the modes `below_modes`) from the BCU to the top and the base of the Hugin, with the time at the BCU known;
    FULL: above plus below from the seabed to the Hugin top and base.  Every comparison is made on the SAME wells (those that have every boundary of the longest chain)."""
    bt = boundary_table(picks, scheme, float(cfg["water_velocity_m_s"]), horizon_twt)
    bt = bt.assign(scheme=name)
    # layers of the scheme: all that occur in a model
    pairs: list[tuple[str, str]] = []
    for layers in list(scheme["above_bcu"].values()) + [v for d in scheme["below_bcu"].values() for v in d.values()]:
        for p in layers:
            if tuple(p) not in pairs:
                pairs.append(tuple(p))
    lay_rows, prof_rows = [], []
    grp = bt.drop_duplicates("boundary").set_index("boundary")["group"].to_dict()
    for top, base in pairs:
        o = layer_observations(bt, top, base, float(cfg["min_thickness_m"]))
        if len(o) < int(cfg["min_wells"]):
            continue
        st, prof = fit_layer(o, cfg)
        lay_rows.append({"scheme": name, "layer": f"{top} -> {base}", "top": top, "base": base, "group_top": grp.get(top), "group_base": grp.get(base), **st})
        prof_rows.append(prof.assign(scheme=name, layer=f"{top} -> {base}"))
    chains, chain_w = [], []

    def add(kind, target, above, a_mode, below, b_mode, layers, modes, eval_wells):
        ev = chain_evaluation(bt, layers, cfg, modes, eval_wells)
        s = chain_summary(ev)
        chains.append({"scheme": name, "kind": kind, "target": target, "above_structure": above, "above_mode": a_mode, "below_structure": below, "below_mode": b_mode, **s})
        if len(ev):
            chain_w.append(ev.assign(scheme=name, kind=kind, target=target, above_structure=above, above_mode=a_mode, below_structure=below, below_mode=b_mode))

    def wells_of(layers):
        w = None
        for t, b in layers:
            s = set(layer_observations(bt, t, b, float(cfg["min_thickness_m"]))["well"])
            w = s if w is None else (w & s)
        return w or set()

    longest_above = scheme["above_bcu"]["cake"]
    for a_struct, a_layers in scheme["above_bcu"].items():            # above the BCU, on the wells that have every layer of the cake
        ew = wells_of(longest_above)
        for a_mode in cfg["above_modes"]:
            add("above_bcu", "bcu", a_struct, a_mode, "", "", a_layers, [a_mode] * len(a_layers), ew)
    for target, variants in scheme["below_bcu"].items():             # below the BCU: the time at the BCU is known, so the wells are those that have the BCU and the target
        ew = wells_of(variants["layered"])
        for b_struct, b_layers in variants.items():
            for b_mode in cfg["below_modes"]:
                add("below_bcu", target, "", "", b_struct, b_mode, b_layers, [b_mode] * len(b_layers), ew)
    for target, variants in scheme["below_bcu"].items():             # from the seabed to the Hugin
        ew = wells_of(longest_above) & wells_of(variants["layered"])
        for a_struct, a_layers in scheme["above_bcu"].items():
            for a_mode in cfg["above_modes"]:
                for b_struct, b_layers in variants.items():
                    for b_mode in cfg["below_modes"]:
                        add("full", target, a_struct, a_mode, b_struct, b_mode, list(a_layers) + list(b_layers), [a_mode] * len(a_layers) + [b_mode] * len(b_layers), ew)
    cols = ["scheme", "kind", "target", "above_structure", "above_mode", "below_structure", "below_mode"]
    cw = pd.concat(chain_w, ignore_index=True) if chain_w else pd.DataFrame(columns=cols)
    return {"compare": compare_chains(cw, comparison_specs(name, scheme)) if len(cw) else pd.DataFrame(), "boundary": bt, "layer": pd.DataFrame(lay_rows), "profile": pd.concat(prof_rows, ignore_index=True) if prof_rows else pd.DataFrame(), "chain": pd.DataFrame(chains), "chain_well": cw}

def bootstrap_rmse(err: np.ndarray, n: int = 1000, seed: int = 7) -> tuple[float, float]:
    """The 5th and 95th percentile of the RMS of the errors, resampling WELLS with replacement (a fixed seed): how far a root-mean-square error from a handful of wells can be trusted."""
    e = np.asarray(err, float)
    if len(e) < 3:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(e), size=(n, len(e)))
    r = np.sqrt((e[idx] ** 2).mean(axis=1))
    return float(np.percentile(r, 5)), float(np.percentile(r, 95))


def paired_rmse_difference(err_a: np.ndarray, err_b: np.ndarray, n: int = 2000, seed: int = 11) -> dict[str, float]:
    """RMS of model A minus RMS of model B on the same wells (negative: A is closer), with the 5th to 95th percentile of the difference from resampling wells (a paired bootstrap).  An interval that contains zero means the two models cannot be told apart on these wells."""
    a, b = np.asarray(err_a, float), np.asarray(err_b, float)
    if len(a) < 3 or len(a) != len(b):
        return {"diff_m": np.nan, "p05_m": np.nan, "p95_m": np.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = np.sqrt((a[idx] ** 2).mean(axis=1)) - np.sqrt((b[idx] ** 2).mean(axis=1))
    return {"diff_m": float(np.sqrt((a ** 2).mean()) - np.sqrt((b ** 2).mean())), "p05_m": float(np.percentile(d, 5)), "p95_m": float(np.percentile(d, 95))}


def compare_chains(chain_well: pd.DataFrame, specs: list[dict[str, Any]]) -> pd.DataFrame:
    """Paired comparisons of chain variants on their common wells.  A spec has `scheme`, `kind`, `target` and two variants `a` and `b`, each (above_structure, above_mode, below_structure, below_mode); the result has the RMS of each, the paired difference with its interval
    and a verdict: `a closer`, `b closer` or `not distinguishable` (the interval contains zero)."""
    rows = []
    key = ["above_structure", "above_mode", "below_structure", "below_mode"]
    for sp in specs:
        base = chain_well[(chain_well["scheme"] == sp["scheme"]) & (chain_well["kind"] == sp["kind"]) & (chain_well["target"] == sp["target"])]

        def pick(v):
            m = np.ones(len(base), dtype=bool)
            for kname, val in zip(key, v):
                m &= (base[kname] == val).to_numpy()
            return base[m].set_index("well")["error_m"]

        ea, eb = pick(sp["a"]), pick(sp["b"])
        common = ea.index.intersection(eb.index)
        if len(common) < 3:
            continue
        d = paired_rmse_difference(ea.loc[common].to_numpy(), eb.loc[common].to_numpy())
        verdict = "not distinguishable" if d["p05_m"] <= 0 <= d["p95_m"] else ("a closer" if d["p95_m"] < 0 else "b closer")
        rows.append({"scheme": sp["scheme"], "kind": sp["kind"], "target": sp["target"], "a": " / ".join(map(str, sp["a"])), "b": " / ".join(map(str, sp["b"])), "label": sp.get("label", ""), "wells": int(len(common)),
                     "rmse_a_m": float(np.sqrt((ea.loc[common] ** 2).mean())), "rmse_b_m": float(np.sqrt((eb.loc[common] ** 2).mean())), **d, "verdict": verdict})
    return pd.DataFrame(rows)


def comparison_specs(scheme_name: str, scheme: dict[str, Any]) -> list[dict[str, Any]]:
    """The comparisons asked of a scheme: above the BCU a single V0-k from the seabed against a constant velocity and against the layer-cake; below the BCU a constant velocity against V0-k and a block against the layers; and the same questions for the full depth to the Hugin
    with the other half of the chain held fixed (direct V0-k above, block constant below)."""
    sp = []
    for lab, a, b in (("direct V0-k against direct constant", ("direct", "v0k", "", ""), ("direct", "const", "", "")), ("direct V0-k against layer-cake constant", ("direct", "v0k", "", ""), ("cake", "const", "", "")),
                      ("direct V0-k against layer-cake V0-k", ("direct", "v0k", "", ""), ("cake", "v0k", "", ""))):
        sp.append({"scheme": scheme_name, "kind": "above_bcu", "target": "bcu", "a": a, "b": b, "label": lab})
    for target in scheme["below_bcu"]:
        for lab, a, b in (("block V0-k against block constant", ("", "", "block", "v0k"), ("", "", "block", "const")), ("block constant against layered constant", ("", "", "block", "const"), ("", "", "layered", "const")),
                          ("block auto against block constant", ("", "", "block", "auto"), ("", "", "block", "const"))):
            sp.append({"scheme": scheme_name, "kind": "below_bcu", "target": target, "a": a, "b": b, "label": lab})
        for lab, a, b in (("above: direct V0-k against layer-cake constant (below: block constant)", ("direct", "v0k", "block", "const"), ("cake", "const", "block", "const")), ("above: direct V0-k against direct constant (below: block constant)", ("direct", "v0k", "block", "const"), ("direct", "const", "block", "const")),
                          ("below: block V0-k against block constant (above: direct V0-k)", ("direct", "v0k", "block", "v0k"), ("direct", "v0k", "block", "const")), ("below: layered constant against block constant (above: direct V0-k)", ("direct", "v0k", "layered", "const"), ("direct", "v0k", "block", "const"))):
            sp.append({"scheme": scheme_name, "kind": "full", "target": target, "a": a, "b": b, "label": lab})
    return sp
"""Layer-cake depth models: boundaries, layer observations, leave-one-well-out k choice and chain errors (synthetic wells with a known V0-k)."""
import numpy as np
import pandas as pd

from ep_depth_velocity import depth_conversion as dc
from ep_depth_velocity import layer_cake as lc

CFG = {"k_grid": [-0.3, 1.5, 31], "min_improvement": 0.10, "min_thickness_m": 5.0}
SCHEME = {"boundaries": [
    {"name": "seabed", "group": "NORDLAND GP", "surfaces": ["Seabed"], "time": "water"},
    {"name": "a", "group": "G1", "surfaces": ["A top", "A alt"], "time": "pick"},
    {"name": "b", "group": "G2", "surfaces": ["B top"], "time": "horizon"}]}


def _wells(k_true, v0_true, n=12, thick=(400.0, 700.0), seed=1):
    """Wells whose layer has V0 = v0_true (plus a few per cent of noise) and gradient k_true; the top depth varies, so k is identifiable."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        top = 1000.0 + 150.0 * i
        z_top = dc.v0k_depth(np.array([0.0]), 1500.0, 0.0)[0] + top
        v0 = v0_true * (1 + rng.normal(0, 0.01)) + k_true * 0.0
        dz = float(rng.uniform(*thick))
        dt = float(dc.v0k_time(np.array([dz]), v0, k_true)[0])
        rows.append({"well": f"w{i}", "boundary": "top", "z_m": z_top, "t_ms": 1000.0}); rows.append({"well": f"w{i}", "boundary": "base", "z_m": z_top + dz, "t_ms": 1000.0 + 2000.0 * dt})
    return pd.DataFrame(rows)


def test_boundary_table_uses_the_first_available_surface_and_the_time_source():
    picks = pd.DataFrame({"well": ["w", "w", "w", "v"], "surface_name": ["Seabed", "A alt", "B top", "Seabed"], "z_m": [90.0, 1000.0, 2000.0, 90.0], "pick_twt_ms": [np.nan, 900.0, 1500.0, np.nan]})
    bt = lc.boundary_table(picks, SCHEME, 1480.0, {"b": pd.DataFrame({"well": ["w"], "twt_ms": [1480.0]})}).set_index(["well", "boundary"])
    assert abs(bt.loc[("w", "seabed"), "t_ms"] - 2000.0 * 90.0 / 1480.0) < 1e-9 and bt.loc[("w", "a"), "surface_used"] == "A alt" and bt.loc[("w", "b"), "t_ms"] == 1480.0 and bt.loc[("w", "b"), "time_basis"] == "horizon"
    assert ("v", "a") not in bt.index                                     # no pick of the group in the well: no boundary


def test_constant_velocity_layer_prefers_k_zero_and_a_gradient_layer_recovers_k():
    const = lc.layer_observations(_wells(0.0, 3000.0), "top", "base", 5.0)
    st0, prof0 = lc.fit_layer(const, CFG)
    assert st0["improvement"] < 0.10 and abs(st0["v0_const_median_m_s"] - 3000.0) < 100.0            # V0-k does not clearly beat a constant velocity here
    k_mode, info = lc.choose_k(const, CFG, "auto")
    assert k_mode == 0.0                                                  # auto stays with the constant velocity unless V0-k is clearly better
    grad = lc.layer_observations(_wells(0.8, 1800.0), "top", "base", 5.0)
    st1, _ = lc.fit_layer(grad, CFG)
    assert st1["improvement"] > 0.3 and st1["k_best"] > 0.4 and st1["rmse_best_m"] < st1["rmse_const_m"]


def test_chain_evaluation_predicts_the_held_out_well_without_using_it():
    bt = _wells(0.8, 1800.0, n=14)
    ev = lc.chain_evaluation(bt, [["top", "base"]], CFG, ["v0k"])
    s = lc.chain_summary(ev)
    assert s["wells"] == 14 and s["rmse_m"] < 40.0 and abs(s["bias_m"]) < 25.0
    ev_c = lc.chain_evaluation(bt, [["top", "base"]], CFG, ["const"])
    assert lc.chain_summary(ev_c)["rmse_m"] > s["rmse_m"]                  # a constant velocity is worse where the velocity increases with depth
    assert lc.chain_summary(lc.chain_evaluation(bt.iloc[:4], [["top", "base"]], CFG, ["const"]))["wells"] == 2


def test_bootstrap_and_paired_comparison_say_when_two_models_cannot_be_told_apart():
    rng = np.random.default_rng(3)
    a = rng.normal(0, 10, 30)
    b = a + rng.normal(0, 0.5, 30)                                          # almost the same errors
    d = lc.paired_rmse_difference(a, b)
    assert d["p05_m"] <= 0 <= d["p95_m"] and abs(d["diff_m"]) < 1.0
    far = lc.paired_rmse_difference(a, a * 3.0)                             # three times larger errors: clearly worse
    assert far["p95_m"] < 0
    lo, hi = lc.bootstrap_rmse(a)
    assert lo < float(np.sqrt((a ** 2).mean())) < hi and np.isnan(lc.bootstrap_rmse(a[:2])[0])
    cw = pd.DataFrame({"scheme": "s", "kind": "full", "target": "t", "well": [f"w{i}" for i in range(30)] * 2,
                       "above_structure": ["direct"] * 30 + ["cake"] * 30, "above_mode": "v0k", "below_structure": "block", "below_mode": "const", "error_m": np.r_[a, a * 3.0]})
    r = lc.compare_chains(cw, [{"scheme": "s", "kind": "full", "target": "t", "a": ("direct", "v0k", "block", "const"), "b": ("cake", "v0k", "block", "const"), "label": "x"}])
    assert r["verdict"].iloc[0] == "a closer" and r["wells"].iloc[0] == 30
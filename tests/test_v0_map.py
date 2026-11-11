"""V0 map: grouping of slot wells, the interpolators, leave-one-group-out evaluation (synthetic wells with a known lateral trend)."""
import numpy as np
import pandas as pd

from ep_depth_velocity import depth_conversion as dc
from ep_depth_velocity import v0_map as vm

CFG = {"group_distance_m": 300.0, "min_wells_map": 6, "clip_fraction": 0.5, "krige_ranges_m": [1500.0, 3000.0, 6000.0], "nugget_fraction": 0.2}
LCFG = {"k_grid": [-0.3, 1.5, 31], "min_improvement": 0.10, "min_thickness_m": 5.0}


def _obs(trend_per_km=0.0, n=24, k=0.0, seed=4, noise=0.005):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 8000, n)
    y = rng.uniform(0, 6000, n)
    v0 = 2500.0 * (1 + trend_per_km * (x / 1000.0 - 4.0)) * (1 + rng.normal(0, noise, n))
    dz = rng.uniform(150, 450, n)
    dt = dc.v0k_time(dz, v0, k)
    return pd.DataFrame({"well": [f"w{i}" for i in range(n)], "x": x, "y": y, "dz": dz, "dt": dt, "z_top_m": 2800.0, "z_base_m": 2800.0 + dz, "v_int_m_s": dz / dt})


def test_slot_wells_form_one_group_by_single_linkage():
    xy = np.array([[0.0, 0.0], [100.0, 0.0], [250.0, 50.0], [5000.0, 5000.0], [5100.0, 5050.0]])
    g = vm.groups_by_distance(xy, 300.0)
    assert g[0] == g[1] == g[2] and g[3] == g[4] and g[0] != g[3]


def test_surfaces_recover_a_plane_and_krige_ties_the_data():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 6000, (30, 2))
    v = 2000.0 + 0.1 * xy[:, 0] - 0.05 * xy[:, 1]
    p = vm.fit_surface("plane", xy, v, CFG)(np.array([[3000.0, 3000.0]]))[0]
    assert abs(p - (2000.0 + 300.0 - 150.0)) < 1.0
    k = vm.fit_surface("krige", xy, v + np.random.default_rng(2).normal(0, 1.0, 30), {**CFG, "nugget_fraction": 0.001})(xy[:3])
    assert np.allclose(k, v[:3], atol=20.0)
    assert np.allclose(vm.fit_surface("area", xy, v, CFG)(xy[:2]), np.median(v))
    assert np.allclose(vm.fit_surface("plane", xy[:4], v[:4], CFG)(xy[:2]), np.median(v[:4]))              # too few wells: the median


def test_a_lateral_trend_is_found_and_no_trend_leaves_the_area_value_recommended():
    wt, st = vm.evaluate_layer(_obs(trend_per_km=0.03), CFG, vm.METHODS, "const", LCFG)
    s = st.set_index("method")
    assert s.loc["plane", "rmse_m"] < 0.5 * s.loc["area", "rmse_m"] and s.loc["plane", "verdict"] == "closer than area" and vm.recommended_method(st) in ("plane", "plane_krige", "krige", "idw")
    wn, sn = vm.evaluate_layer(_obs(trend_per_km=0.0, noise=0.03), CFG, vm.METHODS, "const", LCFG)
    assert vm.recommended_method(sn) == "area" or sn.set_index("method").loc[vm.recommended_method(sn), "verdict"] == "closer than area"
    assert (wt["error_m"].abs() < 400).all() and {"v0_observed", "v0_predicted", "group"} <= set(wt.columns)


def test_the_surface_grid_covers_the_points():
    g = vm.surface_grid(lambda p: np.full(len(p), 2500.0), np.array([10.0, 20.0, 600.0]), np.array([10.0, 30.0, 600.0]), 250.0)
    assert len(g) == 2 and (g["v0"] == 2500.0).all()

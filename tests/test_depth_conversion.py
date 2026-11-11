"""Depth conversion: V0-k relations, the interval velocity, the fit, and the interval-velocity check of tops (synthetic)."""
import numpy as np
import pandas as pd

from ep_depth_velocity import depth_conversion as dc


def test_time_depth_relation_is_consistent_and_reduces_to_constant_velocity():
    z = np.array([0.0, 500.0, 1500.0, 3000.0])
    t = dc.v0k_time(z, 1500.0, 0.5)
    assert np.allclose(dc.v0k_depth(t, 1500.0, 0.5), z) and t[0] == 0.0
    assert np.allclose(dc.v0k_depth(np.array([1.0]), 2000.0, 0.0), 2000.0) and np.allclose(dc.v0k_time(np.array([1000.0]), 2000.0, 0.0), 0.5)
    # dz/dt at a depth equals the instantaneous velocity (numerical derivative of t(z))
    h = 0.01
    dzdt = h / (dc.v0k_time(np.array([1000.0 + h]), 1500.0, 0.5) - dc.v0k_time(np.array([1000.0]), 1500.0, 0.5))
    assert abs(dzdt[0] - dc.v0k_instantaneous(1000.0, 1500.0, 0.5)) < 0.01


def test_interval_velocity_is_the_log_mean_and_not_the_mid_depth_velocity():
    z1, z2 = 0.0, 1000.0
    v = float(dc.v0k_interval(z1, z2, 1500.0, 0.5))
    exact = (z2 - z1) / (dc.v0k_time(np.array([z2]), 1500.0, 0.5)[0] - dc.v0k_time(np.array([z1]), 1500.0, 0.5)[0])
    assert abs(v - exact) < 1e-6 and abs(v - float(dc.v0k_instantaneous(500.0, 1500.0, 0.5))) > 5.0       # 1738 against 1750 m/s at the mid depth
    assert abs(float(dc.v0k_average(np.array([1000.0]), 1500.0, 0.5)[0]) - v) < 1e-6                                   # from the datum the average equals the interval velocity


def test_fit_recovers_the_parameters_and_reports_residuals():
    t = np.linspace(0.2, 2.0, 12)
    z = dc.v0k_depth(t, 1600.0, 0.6)
    f = dc.fit_v0k(z, t)
    assert abs(f["v0"] - 1600.0) < 1.0 and abs(f["k"] - 0.6) < 0.01 and f["residual_sd_m"] < 0.1 and f["n"] == 12
    z2 = z.copy(); z2[3] += 60.0                                                       # one top 60 m too deep: a positive misfit shows as a negative residual (model shallower)
    f2 = dc.fit_v0k(z2, t)
    assert f2["residual"][3] < -30.0
    assert np.isnan(dc.fit_v0k(z[:2], t[:2])["v0"])


def test_zone_velocities_flag_an_unusual_top_in_one_well():
    rows = []
    for i in range(8):
        w = f"w{i}"
        base = 2000.0 + (60.0 if i == 7 else 0.0)                                    # well 7: the base top is 60 m too deep for its time
        rows += [{"well": w, "surface_name": "A", "z_m": 1000.0, "twt_ms": 1000.0}, {"well": w, "surface_name": "B", "z_m": base, "twt_ms": 1000.0 + 2000.0 * (1000.0 / 2500.0)}]
    iv = dc.zone_interval_velocities(pd.DataFrame(rows))
    fl = dc.flag_interval_outliers(iv, min_wells=5, z_limit=3.0, mad_floor_m_s=30.0)
    assert (fl["flag"] != "").sum() == 1 and fl.loc[fl["well"] == "w7", "flag"].iloc[0] == "high"
    assert dc.flag_interval_outliers(iv.iloc[:3], min_wells=5)["flag"].eq("").all()           # a zone seen in too few wells is not judged
    thin = dc.zone_interval_velocities(pd.DataFrame([{"well": "a", "surface_name": "A", "z_m": 1000.0, "twt_ms": 1000.0}, {"well": "a", "surface_name": "B", "z_m": 1005.0, "twt_ms": 1004.0}]))
    assert thin.empty                                                                  # a 5 m zone is skipped

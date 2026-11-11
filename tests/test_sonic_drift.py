"""Sonic drift: integration, drift against a check shot with a known offset, gaps, table comparison (synthetic)."""
import numpy as np
import pandas as pd

from ep_depth_velocity import sonic_drift as sd

CFG = {"max_gap_m": 10.0, "plausible_us_ft": [40.0, 200.0], "step_ms": 2.0}


def _survey():
    md = np.arange(0.0, 3001.0, 10.0)
    return pd.DataFrame({"md_m": md, "tvdss_m": md})          # a vertical well, depth below sea level = md


def test_constant_slowness_integrates_to_the_expected_two_way_time():
    c = pd.DataFrame({"md_m": np.arange(1000.0, 2001.0, 1.0), "value": 100.0})              # 100 us/ft = 328.084 us/m
    son, st = sd.sonic_twt(c, _survey(), CFG)
    assert abs(son["sonic_twt_ms"].iloc[-1] - 2 * 1000 * 328.084 / 1000) < 0.5 and st["outside_plausible"] == 0 and st["bridged_gap_m"] == 0.0


def test_drift_is_the_check_shot_minus_the_sonic_and_is_negative_when_the_sonic_is_slower():
    c = pd.DataFrame({"md_m": np.arange(1000.0, 2001.0, 1.0), "value": 100.0})
    son, st = sd.sonic_twt(c, _survey(), CFG)
    md = np.arange(1000.0, 2001.0, 50.0)
    cs = pd.DataFrame({"md_m": md, "twt_ms": 1500.0 + (md - 1000.0) * 2 * 0.31 * 1.0})      # one-way 0.31 ms/m against the sonic's 0.328 ms/m
    lv = sd.drift_levels(cs, son)
    assert lv["drift_ms"].iloc[0] == 0.0 and lv["drift_ms"].iloc[-1] < -30.0 and (lv["drift_ms"].diff().dropna() < 0).all()
    s = sd.drift_summary(lv, st, {"step_ms": 2.0})
    assert s["one_sign"] and s["drift_rate_ms_per_km"] < 0 and s["v_ratio_checkshot_over_sonic"] > 1.0
    assert sd.drift_levels(cs.assign(md_m=cs["md_m"] + 10.0), son, md_offset_m=-10.0)["drift_ms"].abs().max() == lv["drift_ms"].abs().max()      # an offset of the table's depth is applied as given


def test_a_short_gap_is_bridged_and_counted_and_a_long_gap_ends_the_segment():
    md = np.arange(1000.0, 1500.0, 1.0)
    short = pd.DataFrame({"md_m": md[(md < 1200) | (md > 1205)], "value": 90.0})
    son, st = sd.sonic_twt(short, _survey(), CFG)
    assert st["bridged_gap_m"] > 0 and st["segments"] == 1 and son["md_m"].max() == 1499.0
    long_ = pd.DataFrame({"md_m": md[(md < 1100) | (md > 1130)], "value": 90.0})
    son2, st2 = sd.sonic_twt(long_, _survey(), CFG)
    assert st2["segments"] == 2 and son2["md_m"].min() > 1130.0        # the longer segment (1131-1499) is kept


def test_tables_are_compared_at_shared_depths():
    a = pd.DataFrame({"md_m": [1000.0, 1015.0, 1030.0], "twt_ms": [1000.0, 1010.0, 1020.0]})
    b = pd.DataFrame({"md_m": [1015.0, 1030.0, 1045.0], "twt_ms": [1012.0, 1022.0, 1032.0]})
    r = sd.table_compare({"a": a, "b": b}).iloc[0]
    assert r["shared_levels"] == 2 and abs(r["median_diff_ms"] - 2.0) < 1e-9


def test_calibrated_table_equals_the_check_shot_at_the_levels_and_holds_the_last_drift():
    c = pd.DataFrame({"md_m": np.arange(1000.0, 2001.0, 1.0), "value": 100.0})
    son, _ = sd.sonic_twt(c, _survey(), CFG)
    md = np.arange(1000.0, 1901.0, 50.0)                                                      # the levels stop 100 m above the end of the sonic
    cs = pd.DataFrame({"md_m": md, "twt_ms": 1500.0 + (md - 1000.0) * 2 * 0.31})
    lv = sd.drift_levels(cs, son)
    cal, st = sd.calibrated_table(son, lv)
    assert st["max_level_residual_ms"] < 1e-6 and cal["md_m"].iloc[0] == 1000.0 and abs(cal["twt_ms"].iloc[0] - 1500.0) < 1e-9
    tail = cal[cal["extrapolated_below_last_level"]]
    assert len(tail) > 90 and tail["drift_applied_ms"].nunique() == 1 and st["share_extrapolated"] > 0.08
    assert abs((cal["twt_ms"] - cal["sonic_uncorrected_twt_ms"]).abs().max() - cal["drift_applied_ms"].abs().max()) < 1e-9
    # between the levels the calibrated time lies between the sonic time and the check-shot line
    mid = cal[(cal["md_m"] > 1020) & (cal["md_m"] < 1030)]
    assert (mid["twt_ms"] < mid["sonic_uncorrected_twt_ms"]).all()                           # the sonic is slower than this check shot, so the drift is negative
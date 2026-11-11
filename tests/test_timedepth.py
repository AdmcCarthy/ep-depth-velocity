"""VSP time-depth tables: the two layouts, the depth-reference test against a survey and the pick check (synthetic data only)."""
import numpy as np
import pandas as pd

from ep_seismic import vsp
from ep_depth_velocity import timedepth as td

PAIRS = """Vertical time-depth pairs from VSP survey
Curve name              : TZV_TIME_CHECKSHOT_NIVSP
Depth datum             : MSL

  Depth     Depth      Time
 MD RKB    TVD - MSL   TWT
    m         m         ms

 1000.0     945.0    1000.0
 1100.0    1045.0    1090.0
 1200.0    1145.0    1175.0
"""
PAIRS_MSL = PAIRS.replace("MD RKB", "MDMSL(m)")
VSURV = """RECNO    MD/RKB            TVD/RKB           TVD/MSL           S-R               T                 TV                TH-S              TC
    1    0.11000000D+03    0.11000000D+03    0.85000000D+02    0.00000000D+00    0.5D-01    0.5D-01    0.2D-02    0.575D-01
    2    0.11400000D+04    0.10799849D+04    0.10549849D+04    0.00000000D+00    0.5D+00    0.5D+00    0.2D-02    0.551D+00
"""


def test_parse_pairs_reads_the_depth_basis_from_the_header():
    d, meta = vsp.parse_pairs(PAIRS)
    assert len(d) == 3 and meta["md_basis"] == "RKB" and d["twt_ms"].iloc[1] == 1090.0
    assert vsp.parse_pairs(PAIRS_MSL)[1]["md_basis"] == "MSL"


def test_parse_vsurv_handles_fortran_exponents_and_doubles_the_corrected_time():
    d, meta = vsp.parse_vsurv(VSURV)
    assert d["md_m"].tolist() == [110.0, 1140.0] and abs(d["twt_ms"].iloc[0] - 115.0) < 1e-9          # 2000 x 0.0575 s
    assert abs(meta["rkb_from_table_m"] - 25.0) < 1e-9 and meta["md_basis"] == "RKB"


def test_read_timedepth_picks_the_layout_by_name_and_ignores_other_files():
    assert vsp.read_timedepth("x/TZV_TIME_CHECKSHOT_1.ASC", PAIRS, [r"TZV_TIME_CHECKSHOT_\d+\.ASC$"], [r"VSURV"]) is not None
    assert vsp.read_timedepth("x/VELOCITY_REPORT_1.ASC", PAIRS, [r"TZV_TIME_CHECKSHOT_\d+\.ASC$"], [r"VSURV"]) is None


def _td(md_basis="RKB", offset=0.0):
    d, meta = vsp.parse_pairs(PAIRS)
    d["md_m"] = d["md_m"] - offset                           # a table whose depth starts offset metres higher than the survey's
    meta["md_basis"] = md_basis
    return td.timedepth_table([("m1", "f1", meta, d)], {"f1": "W"})


def test_survey_check_tests_both_offsets_for_a_msl_table():
    survey = lambda name, md: np.asarray(md, float) - 55.0           # survey TVDSS = MD(RKB) - 55
    t = _td("MSL", offset=55.0)                                      # table MD is MD(RKB) - 55: the rotary-table elevation must be added
    r = td.survey_check(t, survey, {"W": 55.0}).iloc[0]
    assert r["md_offset_used_m"] == 55.0 and abs(r["residual_alt_median_m"]) > 5 and r["status"] == "agrees within 5 m"
    r2 = td.survey_check(_td("RKB"), lambda n, md: np.asarray(md, float) - 55.0, {"W": 55.0}).iloc[0]
    assert r2["md_offset_note"].startswith("none")


def test_pick_check_residual_and_range_and_candidates():
    t = td.timedepth_table([("m1", "f1", {"layout": "pairs", "md_basis": "RKB", "curve": ""}, vsp.parse_pairs(PAIRS)[0])], {})
    picks = pd.DataFrame({"well": ["A", "A", "A", "B"], "surface_name": ["top1", "top2", "deep", "top1"], "qualifier": ["", "", "", ""], "tvdss": [-1045.0, -500.0, np.nan, -1045.0], "twt": [1095.0, 400.0, 1200.0, 1090.0]})
    c = td.pick_check(t, picks, {"f1": ["A", "B"]})
    ok = c[c["not_checked_because"].eq("")].set_index(["wellbore_candidate", "surface_name"])
    assert abs(ok.loc[("A", "top1"), "residual_ms"] - 5.0) < 1e-9 and abs(ok.loc[("B", "top1"), "residual_ms"]) < 1e-9
    assert c.loc[c["surface_name"] == "top2", "not_checked_because"].iloc[0].startswith("outside") and c.loc[c["surface_name"] == "deep", "not_checked_because"].iloc[0].startswith("pick has no")
    s = td.pick_check_summary(c, 5.0).set_index("wellbore_candidate")
    assert s.loc["B", "residual_median_ms"] == 0.0 and s.loc["A", "picks_not_checked"] == 2


def test_summary_flags_a_time_that_does_not_increase_and_velocities_outside_the_band():
    d = pd.DataFrame({"md_m": [1000.0, 1010.0, 1020.0], "tvdss_m": [900.0, 905.0, 960.0], "twt_ms": [1000.0, 1000.0, 1001.0]})
    t = td.timedepth_table([("m", "f", {"layout": "pairs", "md_basis": "RKB", "curve": ""}, d)], {})
    s = td.timedepth_summary(t, (1400.0, 6000.0)).iloc[0]
    assert not s["time_monotonic"] or s["v_int_outside_band"] >= 1

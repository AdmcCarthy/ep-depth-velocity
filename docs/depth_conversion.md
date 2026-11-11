# Depth conversion: concepts, methods and a first look at the Volve data

Review of 2026-11-09 (`src/processing/depth_conversion.py`, `scripts/depth_conversion_review.py`, primer `depth_conversion`, requests 130-132). The web material is search summaries only, none of it read in full; the Volve numbers are computed by the script and are exploratory (no mart, no notebook yet).

## 1 What depth conversion is used for

* **The time-depth relationship (TDR)** at a well: a check shot or a sonic log (calibrated to it, `docs/timedepth.md`) ties the depth of a top, a perforation or a trajectory to the two-way time of the seismic. It is a measurement at one place.
* **Depth conversion of seismic data and surfaces**: a velocity model in x, y and time (or a function of depth) turns a time volume into a depth display, a time horizon into a depth map and a fault or any point from time to depth. The volume display stretches the traces (the wavelet changes), a horizon is converted node by node, and the same model must be used in both directions, or the well tops (converted from depth to time) will not tie.
* **The interval velocity of a zone** (a formation, a group) is a data point in itself: thickness in depth over thickness in time. It describes the rock (a chalk is fast, a Tertiary mudstone slow), shows the trend with depth that a V0-k function describes, and checks the tops: a zone with an abnormal velocity points to a wrong top, a wrong time or an unusual rock.

## 2 Definitions and the linear velocity function V0-k

Depth z is measured down from a datum and one-way time t from the same datum (two-way time is 2 t).

| Quantity | Definition |
| :--- | :--- |
| instantaneous velocity | v(z) = dz/dt = V0 + k z (V0 at the datum, k in 1/s) |
| time-depth relation | z(t) = (V0 / k) (exp(k t) - 1); t(z) = ln(1 + k z / V0) / k; for k = 0: z = V0 t |
| average velocity | v_av = z / t, from the datum |
| interval velocity | v_int = (z2 - z1) / (t2 - t1); for V0-k: k (z2 - z1) / ln((V0 + k z2) / (V0 + k z1)) |

The interval velocity of a V0-k medium is the logarithmic mean of the velocity at the two depths and is **not** the instantaneous velocity at the mid depth (1738 against 1750 m/s over 1000 m with V0 = 1500 and k = 0.5): the two are often confused. V0 is the velocity at the datum, so a water layer is treated separately (depth and time below the seabed) or V0 becomes a fitted parameter without a physical meaning at the surface. The relations are tested numerically (`tests/test_depth_conversion.py`).

## 3 Methods, from the search summaries

* A **layer-cake model** gives each layer its own interval velocity or function; the layer scheme follows the geology, more layers are not necessarily better, and a single layer can be enough.
* **V0-k within each layer** is common in the North Sea; k is often held constant over the area and V0 calibrated at the wells and then interpolated (kriged). A Danish Central Trough study found it effective when the interval velocity of the layer increases with depth, with the Chalk Group as a significant high-velocity layer; a Southern North Sea case combined good prestack depth-migration velocities in a small area with a V0-k analysis over a wider one.
* Residuals at the well tops (model depth minus well depth) are the yardstick: a mean near zero says no bias, a small spread a good fit; kriging the residuals gives an uncertainty that is smallest at the wells; a range of models with different methods shows the uncertainty from the choice. One practitioner guide advises against forcing the horizon picks to tie the wells (it hides the QC information) and calls regressions of V0 on the residuals error corrections in disguise.
* **Interval-velocity checks**: plot the check-shot interval velocity against depth with the tops overlaid; look for single points off the trend, non-physical instantaneous velocities, unwarranted inversions between layers and the 'deepest-interval bust' (the last interval of a borehole is unrepresentative of deeper rock); layer analysis can show misties caused by a wrong top or a spurious velocity.
* Not found: a quantitative comparison of layer-cake, V0-k and other functional forms on North Sea data, and any acceptance limit for an abnormal interval velocity (the limit used here is a statistical one, below).

## 4 Interval velocity as a check of the tops (implemented in `depth_conversion.py`)

For the zone between two consecutive tops of a well, `zone_interval_velocities` gives dz / (dTWT / 2) (zones thinner than 20 m or shorter than 8 ms are skipped: the pick time resolution dominates). `flag_interval_outliers` compares each value with the other wells' value of the same zone: median and median absolute deviation (floored at 30 m/s), a robust z-score, and a flag beyond 3 (zones seen in fewer than 5 wells are not judged); an inversion against the zone above is marked separately because it can be real (a shale below a sand). A flag says that a top, a time or the rock is unusual; it does not say which, and nothing is corrected. The thresholds are assumptions.

## 5 A first look at the Volve data (script output)

| Item | Result |
| :--- | :--- |
| Data | 340 interpretation picks with a two-way time in 32 wells (189 below the seabed in 20 wells with a seabed pick); time and depth horizons of the same survey on one grid for Ty, Shetland, Hugin top and base and BCU |
| One V0-k below the seabed | V0 1581 m/s, k 0.54 1/s, residual sd 61 m; **systematic by horizon**: -81 m (Hordaland), -76 (Utsira), -30 (Draupne), +65 (Ekofisk), +66 (Shetland), +80 (Ty): one function cannot follow the chalk |
| Constant velocity | the median average velocity of the picks is 2195 m/s |
| The team's own conversion | the time and depth horizons imply average velocities of 2072 to 2289 m/s; a single V0-k through them (V0 1259, k 0.80 from sea level) has a residual sd of 52 m with the same pattern (BCU -44 m, Shetland +40 m, Ty +26 m): the team did not use one function |
| Interval velocity of zones (wells, median, MAD) | Nordland-Utsira 1940 (10, 58); Utsira-Hordaland 2074 (19, 30); Hordaland-Ty 2219 (14, 31); Ty-Shetland 3131 (23, 125); Ekofisk-Hod 4422 (25, 101); Hod-Draupne 4082 (27, 231); Draupne-Heather 2766 (13, 92); Hugin top-base 3704 (11, 372); Skagerrak-Smith Bank 3869 (8, 205) m/s |
| Per node, between the team's horizons | Ty-Shetland 2921 (5-95 %: 2667-3289); Shetland-Hugin top 3880; Hugin top-base 3581 (2898-4663); Shetland-BCU 4453 m/s |
| Flags | 17 of 193 zone intervals beyond 3 robust standard deviations; the largest: Ty-Shetland of 15/9-F-11 B (5308 m/s) and F-15 D (5426) against 3131; Utsira-Hordaland of 15/9-F-7 (1707, z = -8) and Ekofisk-Hod low in four F-wells (3.7-4.0 km/s) and high in 15/9-4 and F-11 T2 |
| A data problem the check exposes | the Utsira top has the same depth (827.7 m) in the four wellbores of the F-1 slot but a two-way time of 792 to 905 ms: the pick times are not from one model or one datum (DQ-149) |

The picks' times have no stated origin (they agree to 0 ms with the check shot in F-15 A and differ from it by tens of ms elsewhere, notebook 29); the flags therefore mix velocity, pick-time and top problems. Thin zones (a 40 m Draupne-Heather) carry little time. The picks are not independent of the horizons (one interpretation team).

## 6 What is missing, and a proposed build

1. A **layer-cake V0-k** by stratigraphic group, V0 calibrated at the wells (k per layer from the check shots and sonic), with residual statistics per layer and well and a kriged uncertainty, compared with the team's depth horizons as a test of their model.
2. A **zone interval velocity mart** per well and zone with the flags, from picks and from the check-shot and sonic-calibrated tables (an independent velocity source for the same zones), and a notebook with the velocity-depth plot and the tops overlaid.
3. A **consistency check of the picks' two-way times** against the check shots and the calibrated tables per well, since the interval velocity cannot tell a wrong time from a wrong depth.
4. For a seismic depth display (a volume): a velocity cube or a TDR per trace is needed; none is delivered, so a display would rest on the modelled layer-cake and says so.

Decisions for Adam: the datum (sea level or seabed), the layers (Sodir groups or the picks' formations), whether the chalk is a layer of its own, and the flag limits.

## 7 Layer-cake models by Sodir group, seabed to BCU, and below the BCU (built 2026-11-09; `src/processing/layer_cake.py`, `config/depth_conversion.yaml`, marts `depth_model_*`, notebook 46)

**Method.** Each layer between two boundaries has a constant interval velocity or a V0-k function with V0 at the layer top. Boundaries are the shallowest available pick of a Sodir group in the well (Nordland from the seabed, Hordaland, Rogaland by the Ty Fm, Shetland by the Shetland GP or Ekofisk top, Viking by the Draupne or Heather top = the BCU proxy, Vestland by the Hugin); the group of each pick formation is the Sodir lexicon's. Two time sources: the picks' own two-way times and the two-way times of the seismic horizons at the wells; the seabed time is 2 z / 1480 m/s (assumed; the picks' seabed times are a median 25 ms off, DQ-150). For a trial k the V0 of each well is solved exactly from its own thickness and one-way time below the layer top; a held-out well is predicted with the median V0 of the others (one area value: no kriging) and its own time; k is the smallest leave-one-well-out RMS, chosen again inside every training set (nested). Modes: `const` (k = 0), `v0k`, `auto` (V0-k only if it lowers the RMS by 10 %). Chains: ABOVE the BCU (one function from the seabed, or the layer-cake), BELOW (a block from the BCU to the target, or one layer per group), and FULL to the top and base of the Hugin, always on the same wells; differences are paired bootstrap intervals over wells (a difference whose 90 % interval contains zero is 'not distinguishable'). A best k at the edge of the searched grid (-0.3 to 1.5 1/s) is no optimum and is not called identifiable.

**Results.**
* Layer velocities (picks' times, median over wells): Nordland 1967 m/s, Hordaland 2219, lower Rogaland 3131, Shetland chalk 4159, BCU to Hugin top 3070, Hugin 3704 m/s. V0-k beats a constant velocity clearly only for seabed to BCU (RMS 28 m against 68 m, k 0.78 1/s, V0 1366 m/s, 27 wells) and seabed to Hordaland (60 against 71 m); in the chalk and the thin layers the gradient is not identifiable.
* **Above the BCU** one V0-k function from the seabed gives an RMS error at the BCU of 24 m (picks' times, 12 common wells) and 30 m (horizon times, 23 wells), about 1 % of 2.8 km; the layer-cake by group is not distinguishable from it (+1.2 m, interval -4.6 to +9.4 m). The single function is the simpler model and no worse; layering is what absorbs the chalk step that one function from the seabed cannot follow only when it is not chosen as a layer of its own.
* **Below the BCU** a constant velocity and V0-k differ by 0.8 to 3.5 m RMS over a section of 75 m (the leave-one-well-out profile against k is nearly flat and its minimum lies on the edge of the grid); one block from the BCU is as good as or better than one layer per group. A constant velocity is enough for this data set.
* **To the Hugin** the simple chain (one V0-k function seabed to BCU, a constant velocity below) has an RMS error of 25 m (top, 9 wells, 90 % interval 12-38) and 16 m (base, 7 wells) with the picks' times, and 27 m (top, 18 wells, 16-35) and 25 m (base, 14 wells, 16-32) with horizon times; 26 of 34 paired comparisons cannot be told apart. **The time source matters more than the model form**: the picks' times are not seismic times, and a velocity calibrated with them (3.1 km/s from the BCU to the Hugin top) is wrong for horizon times (2.6 km/s).

**Limits.** One area V0 per layer (no map, no kriging, no uncertainty map); the BCU is a proxy at the wells (its horizon differs from the proxy by a median -0.2 m but up to -74 m); 7 to 27 wells; the depth horizons of the interpretation team are not compared; no seismic depth display is made (no velocity cube).

## 8 The team's depth conversion read from their horizons (built 2026-11-09; `src/processing/team_depth.py`, `config/team_depth.yaml`, marts `team_depth_*`, notebook 47)

**Can it be reverse engineered?** Partly. The archive holds the team's horizons in time AND in depth on one grid (Ty, Shetland, BCU, Hugin top and base, and a seabed), so every node has a time and a depth of the same surface and the depth conversion can be described from its result. The velocity cube, the well adjustment, any smoothing and a report are not delivered, so what is recovered is **a model that reproduces their depths, not their method**. The depth file names carry the words `unadj` (Shetland), `adj`, `adj2`, `updateF11wells`, `EasyDC`; they are listed and not interpreted. The Hugin time horizons are on the near stack, the others on the full stack.

**What reproduces the delivered depths** (interval velocity = thickness over one-way time at each node; readings use the thresholds of `config/team_depth.yaml`):

| Layer | Interval velocity (median) | Reading |
| :--- | ---: | :--- |
| seabed to Ty | 2092 m/s, thickness 2357 m (range 115 m) | velocity falls with thickness (R2 0.85) |
| Ty to Shetland | 2921 m/s, 138 m | velocity falls with thickness (R2 0.42) |
| Shetland to BCU (chalk) | 4453 m/s, 381 m | velocity falls with thickness (R2 0.72, -2.0 km/s per km) |
| **seabed to BCU** | 2286 m/s, 2873 m (range 642 m) | **V0 = 1757 m/s, k = 0.40 1/s reproduce the thickness to 12.7 m; a constant velocity leaves 32.7 m** |
| **BCU to Hugin top** | 2748 m/s | **a constant velocity: one value reproduces the thickness to 3.4 m, no dependence on thickness or depth** |
| Hugin top to base | 3583 m/s | no simple relation (9.3 m for a constant velocity) |

* No single function of time holds the depths: one common depth-time curve leaves 12-31 m per horizon, and a V0-k function fitted to each node from sea level leaves 6-19 m with a systematic pattern by horizon; V0 of that fit varies between nodes (inter-quartile range 1069-1240 m/s, k about 0.9), a lateral variation of about 15 %.
* A model calibrated to the WELLS (horizon times: V0 1818 m/s, k 0.36 from the seabed to the BCU, a constant 2602-2834 m/s below, notebook 46) applied to the team's time horizons differs from their depths by -1.6 m (BCU), -6.7 m (Hugin top) and -11.0 m (Hugin base) on average, RMS 13, 14 and 17 m; a plane in x and y explains 33-43 % of the difference (east +2.6 to +2.9, north -3.8 to -4.6 m per km): their depths carry a lateral velocity trend that one area V0 does not. The two routes agree to about 0.5 % of the depth.
* At the well picks the team's depth horizons leave an RMS of 26 m (Ty), 22 (Shetland), 23 (Hugin top) and 38 m (Hugin base), in-sample (adjusted); the layer-cake of the wells predicts each well without using it to 27 m (top) and 25 m (base).

**What the data suggest, and what they do not show.** The BCU to Hugin top interval is a constant velocity to 3 m and the seabed to BCU interval a V0-k function to 13 m, which matches the practice of one function above the BCU and a simple constant velocity below it. The intermediate layers (Ty, Shetland) have velocities that fall with thickness; one possible reading, which these data do not show, is that they carry the residual of one function applied from the seabed to the BCU. Two different methods can give the same depths over this area, and a model that reproduces them to 13 m is not their model.

## 9 A V0 map from the wells (built 2026-11-09; `src/processing/v0_map.py`, `config/depth_conversion.yaml` `v0_map`, marts `v0_map_*`, notebook 48)

**Method.** Each well gives V0 for a layer (solved exactly from its thickness and one-way time at a given k); a map is a function of easting and northing fitted to those values: `area` (the median, no map), `plane`, `idw`, `krige` (ordinary kriging, exponential covariance, range chosen on the training wells, nugget 20 %) and `plane_krige`. Wells of a slot share their velocity, so the evaluation **leaves out whole groups** of wells within 300 m (chained), fits the map on the rest and predicts the thickness with the held-out well's own time (k chosen inside each training set); a map is 'closer than area' only when the 90 % paired bootstrap interval over groups excludes zero. The map is also applied to the team's time horizons and compared with their depths.

**Results.**
* At one k (the best k of all wells) the V0 of the 23 wellbores with horizon times varies by only 1.0 % (1766-1840 m/s), about 27 m of depth at the BCU; it correlates with the easting (R = -0.77, -11.8 m/s per km) and with the depth of the BCU (R = +0.63), on a cluster of wells that spans about 1.7 km (20th to 80th percentile).
* **Above the BCU** a plane in V0 predicts left-out wells better than one area value (horizon times: RMS 27 against 38 m, difference -11.7 m, interval -16.3 to -6.8; 23 wells in 10 groups; picks' times 23 against 37 m); kriging is not better than the plane. **Below the BCU** no map helps: to the Hugin base every map is worse with horizon times, and with the picks' times none is closer to the Hugin top or base; two maps are closer to the Hugin top with horizon times but produce V0 from 2086 to 5304 m/s in a layer 53 m thick (a map of the noise of the times).
* **Applied to the team's time horizons** the plane map differs from their BCU by RMS 37 m with a gradient of -16 m per km east, against 13 m for one area value (gradient +2.6 m per km east, the opposite sign); the maps of the thin layers are worse still (60 m at the Hugin top). The wells' trend predicts other wells but moves the model away from the team's depth surface over the field.

**Reading and decision.** The two sources disagree on the lateral trend: the wells (pick depths with the horizon times at them) and the team's smooth, partly well-adjusted depth surface. The data do not decide which is right; the trend from the wells rests on a handful of wells in the east of a cluster. **Keep one area V0 above the BCU** (carrying a depth uncertainty of a few tens of metres at the BCU) **and one constant velocity per target below it**, until an independent depth control (a well not used by the team, a check shot at the BCU) or the team's velocity cube settles it. Not done: a map of k, faults as barriers, a fitted or anisotropic variogram, a map from the check-shot and sonic tables.

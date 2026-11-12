# ep-depth-velocity: From time to depth, and how sure the conversion is

V0-k and layer-cake depth conversion, V0 maps, time-depth tables, check-shot and sonic drift.

**Status: public (2026-11-11).** The library follows the rules of a public one (no host-repository imports, no data, no private paths, an automated publication gate in `scripts/publication_gate.py`); the gate runs in CI and must stay at zero blockers. Licence: MIT with acknowledgement (`LICENSE`).

## Install

```
uv add git+https://github.com/AdmcCarthy/ep-depth-velocity
```

Depends on: `ep-physics-base`, `ep-wellbore`, `ep-seismic`. Optional extras:

none

## What is in it

| Module | What it does |
| :--- | :--- |
| `depth_conversion` | Depth conversion: the linear velocity function V0-k, average and interval velocities, and the interval-velocity check of well tops.  Pure functions. |
| `layer_cake` | Layer-cake depth models: a velocity function per layer (a constant interval velocity or V0-k with V0 at the layer top), calibrated to well tops and ju |
| `v0_map` | A map of V0 from the wells: the velocity at the top of a layer (V0 of a V0-k function, or the constant interval velocity) interpolated between the wel |
| `timedepth` | Time-depth relations from VSP / check-shot tables and their checks against the deviation survey and the interpretation team's well picks.  Pure functi |
| `sonic_drift` | Sonic log against check shots: the integrated sonic two-way time, the drift (check-shot time minus integrated sonic time) at every check-shot level, a |

Functions are pure: they take frames and parameters and return `(result, events)` where they change or flag data; nothing reads or writes files except the configuration loader.

## Settings and assumptions

Every setting has a packaged default in `src/ep_depth_velocity/defaults/`. A project can override a file by providing `config/<same name>` in its working directory; the project file wins. A default is an assumption unless it says otherwise: its level (`placeholder`, `inferred`, `published`, `official`, `exact`) and a one-line basis travel with it, and a citation (author and year) is given where one exists. Libraries never import a knowledge store: a host that has one passes its values in as an overlay.

## Develop

```
uv sync --all-extras
uv run pytest            # 5 test files
uv run ruff check .
uv run python scripts/publication_gate.py
```

Tests use small synthetic data generated on the fly. No network, no real data.

## Licence and acknowledgement

MIT License with Acknowledgement ([LICENSE](LICENSE)): the MIT terms plus one condition. Anything built with this work says so,
visibly and in plain words, and does not imply endorsement: "Built with ep-depth-velocity by Adam McCarthy (https://github.com/AdmcCarthy/ep-depth-velocity)". The wording is custom, so
the licence is not OSI-approved. The library holds no third-party data; cited literature and vendored code keep their own terms.

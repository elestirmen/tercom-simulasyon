<a id="english"></a>

# TERCOM Terrain-Contour-Matching Localization Simulator for GNSS-Denied Navigation

[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.3.0-0A7BBB)](pyproject.toml)
[![UI](https://img.shields.io/badge/UI-PySide6-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-green)](#citation-data-and-license)

**English** · [Türkçe](#turkce)

This project is an experimental **Terrain Contour Matching (TERCOM)** simulator that estimates the position of an aircraft in GNSS-denied environments from a Digital Elevation Model (DEM), a laser altimeter, barometric altitude, and motion data. The software combines controlled synthetic experiments, GeoTIFF-based real-terrain studies, uncertainty analysis with quality gates, and reproducible parameter optimization into a single research infrastructure.

Rather than being an implementation of a single algorithm, the repository is a **measurement bench**: it lets you change a sensor model, a search parameter, or a quality threshold, and then read back the effect on position error, on the false-`FIX` rate, and on runtime, over the same routes and the same random seed.

> **Research software note:** This repository is not a flight-critical navigation system. The positions and performance metrics it produces should be evaluated for simulation and research purposes only.

## Table of Contents

- [Research Purpose and Scope](#research-purpose-and-scope)
- [How TERCOM Works](#how-tercom-works)
- [Methodology](#methodology)
  - [Localization Pipeline](#localization-pipeline)
  - [Altitude Models](#altitude-models)
  - [Motion Models](#motion-models)
  - [Coarse-to-Fine Search](#coarse-to-fine-search)
  - [Scoring and Quality Gates](#scoring-and-quality-gates)
  - [Ambiguity Detection](#ambiguity-detection)
  - [Localization States](#localization-states)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiment Modes](#experiment-modes)
- [Configuration Reference](#configuration-reference)
- [Command-Line Reference](#command-line-reference)
- [Outputs and Evaluation Metrics](#outputs-and-evaluation-metrics)
- [Reproducible Experiment Protocol](#reproducible-experiment-protocol)
- [Project Structure](#project-structure)
- [Validation](#validation)
- [Troubleshooting](#troubleshooting)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Citation, Data, and License](#citation-data-and-license)

## Research Purpose and Scope

The main research question of the project is **to what extent a time-dependent terrain-elevation profile can be matched on a reference DEM reliably and at a feasible computational cost**. Within this scope, the following sub-problems can be investigated, each one directly observable through a configuration switch:

| Sub-problem | How the simulator exposes it |
|---|---|
| Profile matching under known or unknown absolute flight altitude | `SensorConfig.altitude_mode`, three altitude models |
| The impact of ideal versus noisy sensor assumptions on localization success | `--realistic-noise` preset against the ideal baseline |
| Joint estimation of position and constant speed when the traveled distance is unknown | `--unknown-speed`, plus the `speed_error_m_s` metric |
| The accuracy–time trade-off between global search and local ROI tracking | `--search-roi-size`, plus the runtime columns |
| Detection of position and speed uncertainty in flat or repetitive topography | Ambiguity detection and the `AMBIGUOUS` state |
| The trade-off between quality thresholds and the false `FIX` rate | The three quality-gate thresholds, plus FIX precision |
| Multiprocessing execution of coarse map searches | `--parallel-workers`, plus the runtime breakdown |

The study can run either on a small, deterministic synthetic DEM or on a user-provided geo-referenced GeoTIFF DEM. External data is not included in this repository.

**Out of scope.** The project does not model flight dynamics, an inertial navigation system, a Kalman/particle filter fusion layer, terrain-following guidance, or any real-time avionics constraint. Localization is a *stateless* estimator per update: each update re-solves the position from the current measurement window, and past estimates only carry over through the ROI anchor.

## How TERCOM Works

The physical principle is a single subtraction. A laser altimeter measures the aircraft's height **above ground level** (AGL); an altitude model gives its height **above mean sea level** (MSL). The difference is the elevation of the terrain directly beneath the aircraft:

```text
terrain elevation  =  aircraft MSL altitude  −  laser AGL measurement
```

A single such value is not a position: on any real map, thousands of cells share the same elevation. What *is* nearly unique is a **sequence** of such values collected along a known relative path — the terrain profile, i.e. the elevation signature of the flight track. TERCOM localizes by sliding that measured signature over the reference DEM and asking where it fits best.

Three ingredients make the fit computable:

1. **The measurement window.** The last `profile_window_size` measurements (default `100`) are retained as a sliding window. Older measurements leave the window, so the profile follows the aircraft rather than growing without bound.
2. **The relative geometry.** Heading plus traveled distance turn the window into a rigid chain of offsets relative to its first sample. Because each sample carries its own heading, the chain reproduces turns, so L-shaped and zigzag routes keep their true shape.
3. **The candidate scan.** Only the *start* of the chain is unknown. The matcher places that start on every candidate DEM cell, samples the DEM along the chain with bilinear interpolation, and scores the resulting expected profile against the measured one. When heading or speed is also unknown, the chain is additionally rotated or scaled, which is why those modes cost more.

The consequence is that TERCOM performance is a property of the **terrain**, not only of the algorithm. Sharp, non-repeating relief produces a single deep score minimum; a flat plateau or a periodic ridge system produces many equally good minima. The simulator therefore does not force a solution when the minimum is not distinctive — it reports `AMBIGUOUS` instead.

## Methodology

### Localization Pipeline

```text
Real or synthetic DEM
        │
        ├──► flight and sensor simulation
        │              │
        │              ▼
        │    laser / barometer / motion measurements
        │              │
        │              ▼
        └──────► sliding terrain profile
                       │
                       ▼
         coarse → medium → fine search
                       │
                       ▼
         quality and uncertainty gate
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
     FIX           AMBIGUOUS  QUALITY INSUFFICIENT
```

Each update runs the following steps:

1. **Measure.** The sensor simulator produces one laser AGL reading (with noise, outliers, and dropouts), one barometric MSL reading (with bias and random walk), a heading, and — depending on the motion model — a traveled distance. The measurement is appended to the sliding window.
2. **Gate the window.** If the window is too short in samples (`min_profile_length`), in distance (`min_profile_distance_m`), or, in unknown-speed mode, in elapsed time (`min_profile_duration_s`), the update is abandoned with the reason `profile_incomplete`. This prevents a short, uninformative profile from locking onto a wrong place on the map.
3. **Build candidate profiles.** For every candidate start cell, the DEM is sampled along the offset chain by bilinear interpolation. Samples falling outside the DEM become `NaN` and are excluded from scoring.
4. **Score.** The expected AGL is derived from the altitude model, and the residual `measured − expected` is reduced to one number by the loss function (Huber by default). Lower is better.
5. **Filter.** The continuity gate, then the absolute quality gate, remove candidates that are physically implausible or simply not a good fit — see [Scoring and Quality Gates](#scoring-and-quality-gates).
6. **Decide.** The surviving candidates are checked for ambiguity. Only a distinctive, quality-passing best candidate becomes a `FIX` and updates the ROI anchor.

### Altitude Models

The altitude model determines how the *expected* laser reading is computed for a candidate cell, and therefore how much absolute altitude knowledge the localizer is allowed to assume. Selected via `SensorConfig.altitude_mode`.

| Mode | Information used by localization | Estimator | Research purpose |
|---|---|---|---|
| `known_msl_altitude` | Constant and known MSL altitude | `expected = constant_msl_m − DEM` | Ideal reference scenario |
| `unknown_constant_msl_altitude` | Constant but unknown MSL altitude along the profile | Altitude is solved per candidate as `median(laser + DEM)`, which cancels the unknown constant offset | Matching without absolute altitude |
| `barometric_altitude` | Time-dependent barometer measurement with bias and noise | A constant baro bias is solved per candidate as `median(laser + DEM − baro)`; the expected profile then follows the barometer's *shape* | More realistic sensor scenario |

The last two modes are robust to a constant altitude error because they estimate that offset from the data itself. What they cannot absorb is *drift within the window* — which is exactly why the barometer model has a random-walk term, and why the window has a maximum duration.

In the synthetic scenario, the flight altitude is not left to chance: if `constant_msl_m` sits below the highest terrain in the source map plus `min_safe_agl_m` (default `50 m`), it is automatically raised to a safe value rounded up to the next 10 m. This keeps the laser inside its measurement range instead of silently producing invalid samples.

### Motion Models

The motion model determines what the localizer knows about *how far the aircraft has moved* between two samples. Selected via `--motion-mode` or `LocalizationConfig.motion_mode`.

| Mode | Motion information provided to localization | Cost |
|---|---|---|
| `known_distance` | Perfect traveled distance; default ideal mode | Lowest — one candidate profile per cell |
| `measured_speed` | Distance derived from a noisy speed measurement (bias + noise + random walk) | Same as above, but odometry error accumulates along the window |
| `unknown_constant_speed` | Distance and speed are not provided; position and constant speed are searched jointly | Highest — the search is repeated per speed hypothesis |

In `unknown_constant_speed` mode, the relation `distance = speed × time` is established from the time difference for each speed hypothesis. Since each sample uses its own heading information, the geometry of turning, L-shaped, and zigzag routes is preserved. The search is itself coarse-to-fine: it sweeps `speed_search_min_m_s` to `speed_search_max_m_s` (default `5–30 m/s`) with a coarse step of `5 m/s`, keeps the best `speed_search_keep_hypotheses` (default `3`), then refines them at `1 m/s` and `0.2 m/s`. Once tracking is established, only a narrow band around the previous estimate is re-searched (`speed_tracking_half_range_m_s`, default `±1 m/s`).

Two behaviors are worth knowing before interpreting results from this mode:

- Because the profile can no longer be gated by distance (distance is precisely what is unknown), it is gated by **elapsed time** instead: `min_profile_duration_s` / `max_profile_duration_s`, default `30 s` / `120 s`.
- Selecting this mode also switches the altitude model to `barometric_altitude`, since assuming a known absolute altitude while claiming speed is unknown would not be a coherent scenario.

### Coarse-to-Fine Search

A full-resolution scan of every cell and every heading would be prohibitively expensive, so the spatial search runs in three narrowing passes. Each pass keeps only the best `top_k` candidates (default `5`) and hands them to the next.

| Stage | Stride | Search area | Headings evaluated |
|---|---|---|---|
| Coarse | `coarse_stride` = `10` px | Whole map, or the ROI when tracking | All candidate headings |
| Medium | `medium_stride` = `3` px | ±`refinement_radius_px` = `20` px around each surviving coarse candidate | The 5 headings closest to that candidate's heading |
| Fine | `fine_stride` = `1` px | ±`10` px (half the refinement radius) around each surviving medium candidate | ±`fine_heading_step_deg` = `0.5°` |

The coarse and medium passes compute only the score, not the full metric set. The final `top_k` candidates are then re-evaluated with every metric, because the quality gate needs inlier RMSE, correlation, and valid ratio, which are more expensive to compute. When only one heading is being searched, the coarse pass switches to a vectorized NumPy implementation instead of the generic candidate loop.

Only the coarse pass is parallelized (see [Parallel Coarse Search](#parallel-coarse-search)) — it is the stage whose cost scales with map area, whereas the refinement passes are already confined to small neighborhoods.

### Scoring and Quality Gates

**Loss function.** The residual between the measured and the expected profile is reduced to a single score by `loss_method`: `huber` (default), `rmse`, or `mae`. Huber is quadratic for residuals below `huber_delta` (default `10 m`) and linear above it, so a single laser outlier of `50 m` is penalized proportionally rather than dominating the entire window. **A lower score is better** — this is an error, not a similarity.

**Three filters, applied in order.** A candidate that survives all three becomes a position solution:

1. **Continuity gate** (`max_match_jump_m`, default `10 m`). Candidates farther from the previous accepted anchor than this distance are physically implausible between two updates and are dropped. If nothing survives, the update is rejected with the reason `continuity`. Setting the value to `0` disables the gate — which is what the fast unknown-speed preset does, since a scaled profile can legitimately move the anchor.
2. **Absolute quality gate.** Computed on the *inlier* profile, after discarding the worst `quality_trim_fraction` (default `5%`) of residuals, so one bad sample cannot poison the window. All three conditions must hold:
   - `inlier RMSE ≤ max_match_inlier_rmse_m` (default `3 m`)
   - `inlier correlation ≥ min_match_inlier_correlation` (default `0.80`)
   - `valid sample ratio ≥ min_match_valid_ratio` (default `0.80`)

   These are *absolute* thresholds, not relative ranking. This is the mechanism that prevents the classic TERCOM failure mode of confidently reporting the best of a set of uniformly bad matches. If nothing passes, the update is rejected with the reason `quality`.
3. **Ambiguity check.** Not a rejection — the best candidate is still reported, but it is flagged. See below.

When an update is rejected, the measured profile is **not** discarded. Instead the search area is widened: an ROI search grows, and a search that already covers the whole map drops its stale anchor and returns to global search. The accumulated measurement history survives the recovery.

### Ambiguity Detection

A low score alone does not mean a position is known. If several well-separated places on the map fit almost equally well, the correct engineering answer is "I do not know", not the arbitrary best of the set.

**Position ambiguity** requires both of the following to hold simultaneously:

- **Small score margin** — the relative score gap between the best and second-best candidate is below `0.05` (5%).
- **Large spatial spread** — the standard deviation of the top candidates' raster positions exceeds `10` px.

The conjunction matters. A small margin with a *small* spread simply means the score surface is flat around one correct minimum, which is normal and harmless; a small margin with a *large* spread means genuinely competing hypotheses in different parts of the map.

**Speed ambiguity** (unknown-speed mode only) applies the same idea to the speed axis. Candidates are first reduced to the best one per distinct speed, then flagged when the score margin is below `speed_ambiguity_score_margin` (default `0.05`) *and* the speed standard deviation across the top `speed_ambiguity_top_k` (default `5`) hypotheses exceeds `speed_ambiguity_std_threshold_m_s` (default `2 m/s`). The result is also reported as a qualitative confidence indicator:

| Indicator | Condition | UI label |
|---|---|---|
| `high` | Score margin ≥ `max(0.15, 3 × threshold)` | `YÜKSEK` |
| `medium` | Score margin ≥ threshold | `ORTA` |
| `low` | Score margin below threshold, but speeds are not scattered | `DÜŞÜK` |
| `ambiguous` | Small margin *and* scattered speeds | `BELİRSİZ` |

Either kind of ambiguity marks the update as `AMBIGUOUS`.

### Localization States

| State | Desktop UI label | Meaning | Recorded reason |
|---|---|---|---|
| `FIX` | `GÜVENLİ (FIX)` | Accepted solution that passed every quality gate | — |
| `AMBIGUOUS` | `BELİRSİZ (AMBIG)` | Candidates score similarly but are spatially scattered | — |
| `AMBIGUOUS` (speed) | `HIZ BELİRSİZ (AMBIG)` | Position is resolved, but the speed hypothesis is not separable | — |
| `QUALITY INSUFFICIENT` | `KALİTE YETERSİZ` | Best candidate was rejected by the absolute quality gate | `quality` |
| `RECOVERY` | `YENİDEN ARANIYOR` | Match was lost; the search area is being expanded | `continuity` / `no_candidates` |
| `NO MATCH` | `EŞLEŞME YOK` | Not enough profile data has accumulated yet | `profile_incomplete` |

For a research run, `QUALITY INSUFFICIENT` and `AMBIGUOUS` are **results, not failures**. A configuration that never produces them is usually a configuration whose thresholds are too loose, and its false-`FIX` rate will show it.

## Key Features

- PySide6-based manual task and telemetry interface, with live map, profile comparison, and per-step logging
- Small, deterministic synthetic terrain generation (`plane` and `valley` presets) that needs no external data
- GeoTIFF DEM reading with extent-preserving resampling and a nodata sanity check
- Laser, barometer, compass, and speed sensor error models including bias, noise, drift, outliers, and dropouts
- Known heading, or coarse-to-fine heading search down to `0.5°`
- Huber / RMSE / MAE based profile matching with outlier-trimmed quality metrics
- Global search plus an optional, progressively expanding ROI recovery flow that preserves the measurement history
- Absolute quality gates that prevent false local anchoring, and continuity gating against implausible jumps
- Joint estimation of a constant but unknown speed together with position, with a dedicated confidence indicator
- A persistent multiprocessing worker pool for large coarse searches, with verified serial–parallel equivalence
- Deterministic parameter optimization with validation/final route separation and Pareto analysis
- Experiment logging in CSV, JSON, JSONL, and XLSX formats
- A strict configuration boundary that keeps ground-truth route data out of the localizer

## Installation

### Requirements

- Windows, Linux, or macOS
- Python `3.10–3.13`
- A graphical session for the desktop interface (headless runs need no display)
- A GeoTIFF DEM for external-terrain experiments

### Setup

```powershell
git clone <repository-url>
cd tercom-simulasyon

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If only runtime dependencies are needed:

```powershell
python -m pip install -e .
```

Installation pulls in `matplotlib`, `numpy`, `PySide6`, and `rasterio`. The `dev` extra additionally installs `pytest` and `ruff`.

The package is installed in editable mode, so source edits take effect without reinstalling. A console entry point named `tercom-terrain-nav` is also registered and is equivalent to `python run_terrain_nav.py`.

### Verifying the Installation

```powershell
python run_terrain_nav.py --headless --fast
python -m pytest
```

The first command should finish in seconds and write `results/config.json` and `results/results.csv`. If the test suite passes, the numerical core, the CLI, and the interface configuration are all working.

## Quick Start

The entry point of the application is `run_terrain_nav.py`.

```powershell
# Fast desktop experiment on a small synthetic DEM
python run_terrain_nav.py --fast

# Headless, reproducible synthetic control run
python run_terrain_nav.py --headless --fast

# User-provided GeoTIFF DEM
python run_terrain_nav.py --dem "C:\data\terrain.tif"

# All options
python run_terrain_nav.py --help
```

When started without parameters, the program uses the local default DEM defined in the source code if it exists, and falls back to synthetic terrain otherwise. Because that behavior depends on a local file, **explicitly providing the DEM path with `--dem` on every run is recommended for academic reproducibility** — otherwise the same command can mean different things on two machines.

`--fast` is not merely a smaller map: it shrinks the synthetic terrain to `100 × 100` cells (`120 × 160` in unknown-speed mode), centers the route, and shortens it, so a full run takes seconds. It is intended for flow checks and development, not for performance claims.

### Manual Controls

In the desktop interface, the aircraft is flown by hand:

| Key | Function |
|---|---|
| `W` / `S` | Move forward / backward |
| `A` / `D` | Move left / right (lateral) |
| `Q` / `E` | Turn left / right |

The default manual movement command is `100 m`, the turn command is `15°`, and the profile sampling interval is `20 m`. These values can be changed through `RouteConfig`.

### Reading the Telemetry Panel

| Field | Meaning |
|---|---|
| `Adım (Step)` | Update index within the run |
| `CPU İşçileri` | Worker processes actually used in the last coarse search |
| `Gerçek Konum` / `Tahmin Konumu` | True and estimated position in map coordinates |
| `Gerçek Yön` / `Tahmin Yönü` | True and estimated heading |
| `Tahmini Hız` / `Hız Güveni` | Estimated speed and its confidence indicator (unknown-speed mode) |
| `Sensör MSL` / `Lazer AGL` | Current altitude and laser readings |
| `Konum Hatası` | Distance between the true and estimated position |
| `Arama Dağılımı` | Spatial spread of the top candidates |
| `Eşleşme Skoru` | Inlier RMSE of the accepted match |
| `Güven Durumu` | The localization state from the table above |

Two of these are easy to misread. `Eşleşme Skoru` is **not** the position error — it is the profile fit error, and a lower value is better; a run can show an excellent match score while being locked onto the wrong hill. `Arama Dağılımı` is derived from the spread of candidates in raster space, so to interpret it in physical meters it must be converted using the DEM pixel size.

## Experiment Modes

### Ideal Sensor Baseline

```powershell
python run_terrain_nav.py --headless --fast
```

This reference scenario uses known MSL altitude, known heading, and a perfect traveled distance. It serves as the control group for comparison against noisy scenarios: whatever error remains here is attributable to the matcher, the DEM resolution, and interpolation — not to sensors.

### Realistic Sensor Noise

```powershell
python run_terrain_nav.py --realistic-noise --dem "C:\data\terrain.tif"
```

This preset replaces absolute altitude with a biased, noisy barometer and switches the motion model to a noisy speed measurement. It also relaxes the parts of the configuration that would otherwise reject every update under realistic noise:

| Parameter | Default | With `--realistic-noise` |
|---|---|---|
| `altitude_mode` | `known_msl_altitude` | `barometric_altitude` |
| `baro_bias_m` | `0.0` | `75.0` |
| `baro_noise_std_m` | `1.0` | `2.0` |
| `baro_random_walk_std_m` | `0.1` | `0.03` |
| `speed_noise_std_m_s` | `0.0` | `0.25` |
| `speed_random_walk_std_m_s` | `0.0` | `0.03` |
| `min_profile_length` | `10` | `5` |
| `min_profile_distance_m` | `0.0` | `800.0` (external DEM) / `40.0` (fast synthetic) |
| `max_profile_distance_m` | `0.0` (uncapped) | `2000.0` (external DEM) |
| `max_match_inlier_rmse_m` | `3.0` | `5.0` |
| `max_match_jump_m` | `10.0` | `50.0` |
| `motion_mode` | `known_distance` | `measured_speed` |

The two profile-distance limits encode the central trade-off of this preset. The **minimum** of `800 m` exists because a short profile under noise is not distinctive enough and invites false global locking. The **maximum** of `2000 m` exists because the profile geometry is built from noisy odometry, so a very long window accumulates enough distance error to distort its own shape. Compass heading is treated as known here; heading noise can be introduced separately through `SensorConfig.heading_mode`.

### Localization Without Speed Information

```powershell
python run_terrain_nav.py --headless --fast --unknown-speed

# Equivalent explicit form
python run_terrain_nav.py --motion-mode unknown_constant_speed

# Restricting the physical speed range
python run_terrain_nav.py --unknown-speed --speed-search-min 8 --speed-search-max 24
```

This mode assumes the speed is constant along the sliding profile. Although the simulator knows the true speed in order to move the vehicle, the true route start, speed, and traveled distance are never passed to the localization layer. The true speed is used only at the final stage, to compute the `speed_error_m_s` metric.

Narrowing the search range with `--speed-search-min` / `--speed-search-max` is the single most effective way to make this mode both faster and more accurate, because every removed hypothesis is both a removed full search and a removed opportunity for a wrong speed to accidentally fit. A range that genuinely reflects the platform's flight envelope is a legitimate modeling assumption, not a shortcut — but it must be reported alongside the results.

### Global Search and ROI

The default `--search-roi-size 0` setting disables the ROI and searches the entire map at every update. The ROI must be enabled explicitly:

```powershell
python run_terrain_nav.py --dem "C:\data\terrain.tif" --search-roi-size 512
```

After a reliable match, the search is restricted to a window of this size centered on the last accepted position. If an update is then rejected, the window does not simply fail — it grows by roughly 50% per attempt, and when the expanded window still fails, the stale anchor is dropped and the search returns to global while the measured profile is preserved. An accepted match that lands near the ROI edge is also treated as suspicious, since the true optimum may lie just outside the window.

The ROI is not an accuracy method but a tracking optimization that reduces computational cost. On a large map it is the difference between scanning millions of cells and scanning a few hundred thousand; on a small map it can be slower than a global search, and it introduces a dependency on the previous fix that a strictly stateless evaluation may not want.

### Parallel Coarse Search

```powershell
# Default: min(4, CPU count) worker processes
python run_terrain_nav.py --parallel-workers 4

# Serial execution
python run_terrain_nav.py --parallel-workers 1
```

Large global searches are split into row bands and executed in persistent worker processes — persistent because the DEM is sent to each worker once at startup rather than per update. Small maps and ROI searches can stay serial to avoid inter-process communication overhead, and the engine falls back to serial execution when a search is too small to be worth distributing.

Parallelism affects **runtime only, not results**: the serial and parallel paths are required to produce identical candidates, and a regression test enforces this. The worker count should nevertheless be reported together with the experiment environment and the DEM size, because every timing figure in the output depends on it.

### Parameter Optimization

```powershell
# Default deterministic optimization plan
python run_terrain_nav.py --optimizer-benchmark --fast

# A short method check
python run_terrain_nav.py --optimizer-benchmark --fast `
  --optimizer-configs 8 `
  --optimizer-refined-configs 4 `
  --optimizer-final-configs 3 `
  --optimizer-routes 4 `
  --optimizer-max-updates-per-route 10
```

The optimizer is a deterministic funnel rather than a random search. It generates candidate configurations, then narrows them in three stages:

1. **Coarse sweep** — all candidates are run over a small subset of routes to eliminate the clearly unusable.
2. **Refined validation** — the survivors (`--optimizer-refined-configs`) are run over the full validation route set, and the Pareto frontier over the competing objectives is computed.
3. **Unseen final test** — the finalists (`--optimizer-final-configs`) are run over routes **held out** from every earlier stage.

The route library is built from eight templates (`duz_dogu`, `duz_kuzey`, `l_dogu_kuzey`, `l_kuzey_dogu`, `zikzak`, `merdiven`, `capraz_kesit`, `u_donus`), each instantiated at four orientations and scaled to the current map. Every fourth route is reserved for the final test and never used for selection, which is what makes the final numbers a genuine held-out measurement rather than a restatement of the tuning objective. Each route is additionally evaluated under three heading scenarios: `known_heading`, `noisy_heading_3deg`, and `noisy_heading_5deg`.

Four named configurations are reported, because "best" is not one thing:

| Selection | Optimized for |
|---|---|
| `safe` | Lowest false-`FIX` rate, then highest FIX precision |
| `fast` | Lowest P95 tracking time among configurations with FIX precision ≥ 95% |
| `accurate` | Lowest P95 position error |
| `balanced` | Best composite score across accuracy, precision, and runtime |

A solution counts as a *wrong* `FIX` when an accepted position exceeds `50 m` of error. A configuration selected in a small synthetic run should not be interpreted as a production setting unless it is additionally validated on an external DEM.

## Configuration Reference

The CLI exposes the most common switches, but every experiment parameter lives in the frozen dataclasses in [config.py](terrain_nav/config.py) and is serialized in full into `results/config.json`. The tables below list the fields most likely to matter for an experiment.

### `TerrainConfig` — map and reference data

| Field | Default | Meaning |
|---|---|---|
| `preset` | `valley` | Synthetic terrain shape: `valley` or `plane` |
| `seed` | `42` | Random seed for terrain texture and DEM noise |
| `rows` / `cols` | `1000` / `1000` | Synthetic DEM dimensions in cells |
| `dx` / `dy` | `1.0` / `1.0` | Cell size in meters |
| `base_elevation` | `1000.0` | Base elevation of the synthetic terrain (m) |
| `dem_noise_std_m` | `0.5` | Noise separating the reference DEM from truth |
| `dem_bias_m` | `0.0` | Constant offset between the reference DEM and truth |
| `dem_path` | `""` | External GeoTIFF path; empty means synthetic |
| `dem_target_size` | `2048` | Long-edge cell budget after resampling |

`dem_noise_std_m` and `dem_bias_m` model the fact that a reference map is never the terrain itself. The `plane` preset is deliberately degenerate — it is the control case for observing ambiguity, since a flat surface cannot localize anything.

### `RouteConfig` — flight path (ground truth only)

| Field | Default | Meaning |
|---|---|---|
| `start_row` / `start_col` | `500` / `500` | Route start cell |
| `heading_deg` | `0.0` | Initial heading (0° = North, 90° = East) |
| `speed_m_s` | `10.0` | True speed |
| `sample_spacing_m` | `10.0` | Distance between measurements |
| `route_length_m` | `1000.0` | Total route length |
| `manual_step_distance_m` | `100.0` | Distance per manual movement command |
| `manual_turn_step_deg` | `15.0` | Angle per manual turn command |

No field of this class is visible to the localizer. `localization_runtime_config()` copies only the sensor, algorithm, and motion-mode settings into the localization layer, which is the structural guarantee that a truth value cannot leak into an estimate.

### `SensorConfig` — measurement models

| Field | Default | Meaning |
|---|---|---|
| `altitude_mode` | `known_msl_altitude` | Altitude model; see [Altitude Models](#altitude-models) |
| `constant_msl_m` | `1500.0` | Flight altitude, auto-raised for terrain clearance if needed |
| `min_safe_agl_m` | `50.0` | Minimum clearance used for that automatic correction |
| `laser_noise_std_m` | `0.5` | Laser measurement noise |
| `laser_outlier_prob` | `0.01` | Probability of an outlier reading |
| `laser_outlier_magnitude_m` | `50.0` | Magnitude of that outlier |
| `laser_drop_prob` | `0.02` | Probability of a dropped (invalid) reading |
| `laser_min_range_m` / `laser_max_range_m` | `0.5` / `3000.0` | Measurement range; readings outside it are invalid |
| `baro_noise_std_m` | `1.0` | Barometer noise |
| `baro_bias_m` | `0.0` | Constant barometer offset |
| `baro_drift_rate_m_s` | `0.01` | Systematic drift per second |
| `baro_random_walk_std_m` | `0.1` | Random-walk component of drift |
| `heading_mode` | `known_heading` | `known_heading`, `noisy_heading`, or `unknown_heading` |
| `sensor_heading_noise_std_deg` | `1.0` | Compass noise when heading is not known |
| `speed_noise_std_m_s` | `0.0` | Speed measurement noise |
| `speed_bias_m_s` | `0.0` | Constant speed offset |

Laser dropouts and outliers are the reason the quality gate is computed on trimmed inliers rather than on all samples: at the defaults, roughly one sample in fifty is missing and one in a hundred is off by `50 m`.

### `AlgorithmConfig` — search, scoring, and gates

| Field | Default | Meaning |
|---|---|---|
| `profile_window_size` | `100` | Sliding window length in measurements |
| `min_profile_length` | `10` | Minimum valid samples before a match is attempted |
| `min_profile_distance_m` / `max_profile_distance_m` | `0.0` / `0.0` | Distance-based window limits; `0` disables |
| `min_profile_duration_s` / `max_profile_duration_s` | `30.0` / `120.0` | Time-based limits used in unknown-speed mode |
| `coarse_stride` / `medium_stride` / `fine_stride` | `10` / `3` / `1` | Search strides in pixels |
| `refinement_radius_px` | `20` | Neighborhood radius around each retained candidate |
| `top_k` | `5` | Candidates carried between search stages |
| `fine_heading_step_deg` | `0.5` | Angular resolution of the fine heading pass |
| `loss_method` | `huber` | `huber`, `rmse`, or `mae` |
| `huber_delta` | `10.0` | Huber transition point (m) |
| `quality_trim_fraction` | `0.05` | Share of worst residuals excluded from quality metrics |
| `max_match_inlier_rmse_m` | `3.0` | Quality gate: maximum inlier RMSE |
| `min_match_inlier_correlation` | `0.80` | Quality gate: minimum inlier correlation |
| `min_match_valid_ratio` | `0.80` | Quality gate: minimum valid sample ratio |
| `max_match_jump_m` | `10.0` | Continuity gate; `0` disables |
| `search_roi_size_px` | `0` | ROI edge length; `0` searches globally |
| `parallel_workers` | `1` | Worker processes for the coarse search |
| `speed_search_min_m_s` / `speed_search_max_m_s` | `5.0` / `30.0` | Speed hypothesis range |
| `speed_ambiguity_score_margin` | `0.05` | Speed ambiguity: score margin threshold |
| `speed_ambiguity_std_threshold_m_s` | `2.0` | Speed ambiguity: speed spread threshold |

The dataclass validates itself on construction: a non-positive search step, a maximum below its minimum, or an unknown resampling mode raises `ValueError` immediately rather than producing quietly wrong results.

## Command-Line Reference

| Option | Default | Description |
|---|---|---|
| `--headless` | off | Run without the desktop UI and write results to `results/` |
| `--fast` | off | Use a small, deterministic synthetic terrain |
| `--dem PATH` | source default, else synthetic | External GeoTIFF DEM |
| `--dem-target-size PX` | `2048` | Target cell count for the long edge of the DEM |
| `--search-roi-size PX` | `0` | ROI edge length in pixels; `0` searches the whole map |
| `--start-row ROW` / `--start-col COL` | `500` / `500` | Start cell of the route; centered in `--fast` synthetic mode |
| `--realistic-noise` | off | Barometric altitude and noisy speed preset |
| `--motion-mode MODE` | `known_distance` | `known_distance`, `measured_speed`, or `unknown_constant_speed` |
| `--unknown-speed` | off | Shorthand for `--motion-mode unknown_constant_speed` |
| `--speed-search-min M_S` / `--speed-search-max M_S` | `5` / `30` | Speed hypothesis range in m/s |
| `--parallel-workers N` | `min(4, CPU count)` | Persistent worker processes for large coarse searches |
| `--optimizer-benchmark` | off | Run the deterministic parameter optimization benchmark |
| `--optimizer-configs N` | `64` | Number of initial candidate configurations |
| `--optimizer-refined-configs N` | `12` | Configurations carried into the refinement stage |
| `--optimizer-final-configs N` | `10` | Configurations carried into the final evaluation |
| `--optimizer-routes N` | `12` | Number of routes evaluated; minimum `2` |
| `--optimizer-sample-spacing M` | route config | Profile sampling interval used during optimization |
| `--optimizer-max-updates-per-route N` | `0` | Update cap per route; `0` means unlimited |
| `--optimizer-output DIR` | `results/` | Output directory for optimization artifacts |

Note that `--parallel-workers` is the only option whose default depends on the machine. Pin it explicitly in any run whose timings will be reported.

## Outputs and Evaluation Metrics

### Simulation Run

A headless run produces two files under `results/`:

- `config.json` — the complete sensor, algorithm, terrain, and route configuration, serialized from the dataclasses. This is the authoritative record of the run; it contains every parameter, not only the ones passed on the command line.
- `results.csv` — one row per update, with 25 columns:

| Column group | Columns |
|---|---|
| Time and truth | `timestamp_s`, `true_x`, `true_y`, `true_heading` |
| Estimate | `est_x`, `est_y`, `est_heading`, `est_msl` |
| Error | `error_x`, `error_y`, `error_pos`, `error_heading` |
| Match quality | `is_ambiguous`, `score`, `inlier_rmse_m`, `correlation`, `valid_ratio` |
| Speed | `estimated_speed_m_s`, `second_best_speed_m_s`, `true_speed_m_s`, `speed_error_m_s`, `speed_is_ambiguous`, `speed_score_margin`, `speed_spread_m_s`, `speed_confidence` |

Rejected updates produce no row, so the row count is *smaller* than the update count. That difference is itself a measurement: the number of rejected solutions belongs in the report, and computing a mean error over accepted rows alone, without stating the acceptance rate, overstates performance.

### Optimization Run

An optimization run produces timestamped `optimizer_<stamp>_summary.csv`, `optimizer_<stamp>_details.jsonl`, and `optimizer_<stamp>.xlsx` files. The workbook contains one sheet per analysis axis:

| Sheet | Content |
|---|---|
| `Genel Ozet` | Run plan, route split, candidate counts, headline results |
| `Top Configurations` | The ten best configurations |
| `Pareto Frontier` | Non-dominated configurations across the competing objectives |
| `Final Test Results` | Held-out route results |
| `Quality Gate Analysis` | Effect of the correlation / RMSE / valid-ratio / trim thresholds |
| `Profile Analysis` | Effect of profile resampling mode and point count |
| `ROI Analysis` | Effect of ROI size |
| `Speed Search Analysis` | Effect of speed range, step sizes, and retained hypotheses |
| `Profile Duration Analysis` | Effect of the window duration limits |
| `Heading Analysis` | Breakdown by heading scenario |
| `Runtime Breakdown` | Per-stage timing |
| `Validation Results` | Full validation summaries |
| `Raw Details` | Per-update raw records |
| `Eliminated` | Configurations dropped, with the stage at which they were dropped |

### Key Metrics

| Metric | Interpretation |
|---|---|
| Position error (m) | Euclidean distance between true and estimated position; lower is better |
| Inlier RMSE (m) | Profile fit error after outlier samples are clipped; lower is better |
| Correlation | Shape agreement between the expected and measured profile; higher is better |
| Valid sample ratio | Share of the profile that can be evaluated within DEM bounds; higher is better |
| Correct FIX rate | Accepted solutions within the error threshold, over all updates |
| False FIX rate | Accepted solutions exceeding the error threshold, over all updates |
| FIX precision | Correct `FIX` count over all accepted `FIX` count |
| P95 position error | 95th percentile of position error across accepted solutions |
| Speed MAE (m/s) | Mean absolute speed error in unknown-speed experiments |
| Runtime (ms) | Computational cost of the global initial solution and of tracking updates |

The three `FIX` metrics answer different questions and should be reported together. **Correct FIX rate** asks how often the system produced a usable position at all; **false FIX rate** asks how often it produced a confident but wrong one; **FIX precision** asks how much a reported `FIX` can be trusted. A system that answers rarely but is never wrong and a system that always answers but is often wrong can share a correct-FIX rate while being entirely different systems.

Percentiles matter more than means here. A profile-matching failure is not a slightly larger error but a jump to a different part of the map, so a mean position error mixes two different populations. Report median and P95 alongside it.

## Reproducible Experiment Protocol

For academic comparisons, the following should be recorded alongside the results:

1. Git commit ID and Python version.
2. DEM source, coordinate reference system, cell size, extent, and file hash (e.g. SHA-256).
3. All sensor, route, and algorithm parameters, via `config.json`.
4. Randomness seed (`TerrainConfig.seed`; default `42`).
5. Run command, OS, CPU model, and the `--parallel-workers` value.
6. Success threshold, number of routes and updates evaluated, and number of rejected solutions.
7. Median and P95 error, false `FIX` rate, and runtime, alongside the mean.

Recommended minimum check:

```powershell
git rev-parse HEAD
python --version
python run_terrain_nav.py --headless --fast --parallel-workers 1
python -m pytest
```

Pinning `--parallel-workers 1` here is deliberate: it removes the machine-dependent default so that two people running this block get byte-comparable results.

Synthetic and external-DEM results should be reported in separate tables, because they answer different questions — the synthetic case measures the algorithm under known, controlled terrain statistics, while the external case measures it under a specific real landscape whose properties are not transferable to another region. Short smoke tests only validate the software flow; they do not count as scientific performance evidence.

## Project Structure

```text
run_terrain_nav.py           Entry point for the CLI and the desktop application
terrain_nav/
├── config.py                Experiment, sensor, and algorithm configurations
├── coordinates.py           World/raster transforms and heading math
├── synthetic.py             Deterministic synthetic terrain generation
├── terrain.py               Synthetic and GeoTIFF DEM management
├── sensors.py               Sensor simulation
├── profile.py               Route and terrain profile extraction
├── matcher.py               Coarse-to-fine profile matching
├── confidence.py            Position and speed uncertainty evaluation
├── metrics.py               Estimated-state and comparison records
├── simulation.py            Simulation and localization lifecycle
├── benchmark.py             Profile-variant benchmark infrastructure
├── optimizer.py             Deterministic parameter optimization
├── logging_io.py            JSON and CSV logging
├── rendering.py             Map and profile renderings
└── ui.py                    PySide6 desktop interface
tests/                       Regression tests for the active package
results/                     Runtime outputs
```

Two boundaries are worth understanding before modifying the code:

- **Truth versus working configuration.** `simulation.py` holds both the ground-truth route and the localization layer, but passes the localizer only a `LocalizationRuntimeConfig`, which by construction contains no route data. Any change that widens that interface risks a silent ground-truth leak — the optimizer even records a `ground_truth_leak_detected` flag for this reason.
- **Coordinate conventions.** `coordinates.py` is the single place where world coordinates (x = East, y = North) meet raster coordinates (col = East, row = **South**). The row axis is inverted with respect to North, and heading is measured clockwise from North. Sign errors here produce mirrored trajectories that still look plausible, so the conventions are covered by their own test module.

## Validation

Recommended validation surface after code changes:

```powershell
python -m pytest
python -m ruff check .
python -m compileall -q run_terrain_nav.py terrain_nav tests
python run_terrain_nav.py --help
python run_terrain_nav.py --headless --fast
```

The suite covers the numerical core, the interface, and the research workflows:

| Area | Test module |
|---|---|
| Coordinate and heading conventions | `test_coordinate_conventions.py` |
| Terrain generation and external DEM | `test_terrain_generation.py`, `test_external_dem.py` |
| Sensor models | `test_sensor_model.py` |
| Profile extraction | `test_profile_extraction.py` |
| Known / unknown altitude matching | `test_known_altitude_matching.py`, `test_unknown_altitude_matching.py` |
| Unknown-speed localization | `test_unknown_speed_localization.py` |
| Heading search | `test_heading_search.py` |
| Ambiguity detection | `test_ambiguity_detection.py` |
| ROI recovery and serial–parallel equivalence | `test_search_optimization.py` |
| End-to-end localization | `test_end_to_end_localization.py` |
| CLI and manual control | `test_headless_cli.py`, `test_manual_control.py` |
| Interface configuration | `test_ui.py` |
| Benchmark and optimization | `test_benchmark.py`, `test_optimizer_benchmark.py` |

## Troubleshooting

| Symptom | Likely cause | What to do |
|---|---|---|
| Every update reports `KALİTE YETERSİZ` | Quality thresholds are too strict for the noise level, or the DEM does not match the sensor datum | Compare `inlier_rmse_m` and `correlation` in `results.csv` against `max_match_inlier_rmse_m` and `min_match_inlier_correlation`; check that the DEM's vertical datum matches the assumed MSL |
| Every update reports `BELİRSİZ` | The terrain has no distinctive signature over the current window | Lengthen the window (`profile_window_size`, `min_profile_distance_m`) or use a route that crosses more relief; on a `plane` preset this is the correct result |
| `EŞLEŞME YOK` never clears | The window never satisfies the minimum gates | Check `min_profile_length`, `min_profile_distance_m`, and — in unknown-speed mode — `min_profile_duration_s` against the route length and sampling interval |
| Position locks onto a confidently wrong place | Profile too short for the map size, or gates too loose | Raise `min_profile_distance_m`, tighten `max_match_inlier_rmse_m`, and check the false-`FIX` rate rather than the mean error |
| External DEM rejected with a nodata error | The resampled window contains too many nodata cells | Crop or fill the DEM, or select a region with valid coverage |
| Runs are very slow | Global search over a large map, or unknown-speed mode with a wide speed range | Raise `--parallel-workers`, enable `--search-roi-size`, or narrow `--speed-search-min` / `--speed-search-max` |
| Estimated speed is stable but wrong | A wrong speed can fit a locally self-similar profile | Check `speed_confidence` and `speed_spread_m_s`; narrow the speed search range to the real flight envelope |
| The desktop interface will not start | No graphical session, or PySide6 not installed | Use `--headless`, or verify the `PySide6` installation |

## Assumptions and Limitations

- The system is a simulator; real flight hardware, time synchronization, and avionics safety requirements are not modeled.
- Profile matching is sensitive to DEM accuracy and resolution, and to the consistency of sensor and map datums. A vertical datum mismatch appears as a constant bias, which the unknown-altitude and barometric models will silently absorb — improving the apparent fit while telling you nothing about it.
- Flat or repetitive topography can reduce the joint observability of position and speed; in that case `AMBIGUOUS` or `QUALITY INSUFFICIENT` is an expected outcome.
- `unknown_constant_speed` assumes constant speed across the sliding profile window; the model must be extended for accelerated flight.
- Localization has no motion filter. Consecutive estimates are independent apart from the ROI anchor and the continuity gate, so the reported errors are raw matcher errors and would improve under a filtering layer that this project deliberately omits.
- The long edge of the DEM is sampled to `2048` cells by default. This preserves the physical extent but may reduce high-frequency topographic detail — the very detail that makes a profile distinctive. On a large map, `dem_target_size` is therefore an accuracy parameter, not only a performance one.
- The ROI setting and the parallel worker count should be recorded with the scientific parameters; timing results should be remeasured on different hardware and DEM sizes.
- Default thresholds are generalized across geographies and should not be treated as field-calibrated values.

## Citation, Data, and License

This repository does not yet contain a `CITATION.cff`, a DOI, or an author-approved bibliographic record. For academic use, at a minimum the repository name, the commit ID of the version used, and the access date should be stated. This section should be updated once official citation information is added.

External DEM files are not included in the repository. The license, producer, provenance, coordinate reference system, and preprocessing steps of the dataset used must be stated separately in the related publication.

This project is dual-licensed under the **MIT License** and the **Apache License 2.0**. You may choose either license when using, modifying, or distributing this code. See the [LICENSE](LICENSE) file for the complete terms of both licenses.

---

<a id="turkce"></a>

# GNSS-Yoksun Seyrüsefer için TERCOM Arazi Profili Lokalizasyon Simülatörü

[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Sürüm](https://img.shields.io/badge/s%C3%BCr%C3%BCm-0.3.0-0A7BBB)](pyproject.toml)
[![Arayüz](https://img.shields.io/badge/aray%C3%BCz-PySide6-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython/)
[![Lisans](https://img.shields.io/badge/lisans-MIT%2FApache--2.0-green)](#atıf-veri-ve-lisans)

[English](#english) · **Türkçe**

Bu proje, GNSS erişiminin bulunmadığı koşullarda bir hava aracının konumunu Sayısal Yükseklik Modeli (DEM), lazer altimetre, barometrik irtifa ve hareket bilgilerinden kestirmeyi amaçlayan deneysel bir **Terrain Contour Matching (TERCOM)** simülatörüdür. Yazılım; kontrollü sentetik deneyleri, GeoTIFF tabanlı gerçek arazi çalışmalarını, belirsizlik analizi ile kalite kapılarını ve tekrarlanabilir parametre optimizasyonunu tek bir araştırma altyapısında birleştirir.

Depo, tek bir algoritmanın uygulaması olmaktan çok bir **ölçüm tezgâhıdır**: bir sensör modelini, bir arama parametresini veya bir kalite eşiğini değiştirip bunun konum hatasına, yanlış `FIX` oranına ve çalışma süresine etkisini aynı rotalar ve aynı rastgelelik tohumu üzerinde geri okumanızı sağlar.

> **Araştırma yazılımı notu:** Bu depo bir uçuş-kritik seyrüsefer sistemi değildir. Üretilen konumlar ve performans ölçümleri yalnızca simülasyon ve araştırma amacıyla değerlendirilmelidir.

## İçindekiler

- [Araştırma amacı ve kapsam](#araştırma-amacı-ve-kapsam)
- [TERCOM nasıl çalışır](#tercom-nasıl-çalışır)
- [Yöntem](#yöntem)
  - [Lokalizasyon akışı](#lokalizasyon-akışı)
  - [Uçuş irtifası modelleri](#uçuş-irtifası-modelleri)
  - [Hareket bilgisi modelleri](#hareket-bilgisi-modelleri)
  - [Kabadan inceye arama](#kabadan-inceye-arama)
  - [Puanlama ve kalite kapıları](#puanlama-ve-kalite-kapıları)
  - [Belirsizlik saptama](#belirsizlik-saptama)
  - [Lokalizasyon durumları](#lokalizasyon-durumları)
- [Temel özellikler](#temel-özellikler)
- [Kurulum](#kurulum)
- [Hızlı başlangıç](#hızlı-başlangıç)
- [Deney modları](#deney-modları)
- [Konfigürasyon referansı](#konfigürasyon-referansı)
- [Komut satırı referansı](#komut-satırı-referansı)
- [Çıktılar ve değerlendirme ölçütleri](#çıktılar-ve-değerlendirme-ölçütleri)
- [Tekrarlanabilir deney protokolü](#tekrarlanabilir-deney-protokolü)
- [Proje yapısı](#proje-yapısı)
- [Doğrulama](#doğrulama)
- [Sorun giderme](#sorun-giderme)
- [Varsayımlar ve sınırlılıklar](#varsayımlar-ve-sınırlılıklar)
- [Atıf, veri ve lisans](#atıf-veri-ve-lisans)

## Araştırma amacı ve kapsam

Projenin temel araştırma sorusu, **zamana bağlı bir arazi-yükseklik profilinin referans DEM üzerinde ne ölçüde güvenilir ve hesaplama açısından uygulanabilir biçimde eşleştirilebileceğidir**. Bu kapsamda aşağıdaki alt problemler incelenebilir; her biri doğrudan bir konfigürasyon anahtarıyla gözlemlenebilir:

| Alt problem | Simülatörde karşılığı |
|---|---|
| Bilinen veya bilinmeyen mutlak uçuş irtifası altında profil eşleştirme | `SensorConfig.altitude_mode`, üç irtifa modeli |
| İdeal ve gürültülü sensör varsayımlarının lokalizasyon başarısına etkisi | `--realistic-noise` ön ayarının ideal referansla karşılaştırılması |
| Kat edilen mesafe bilinmediğinde konum ve sabit hızın birlikte kestirimi | `--unknown-speed` ve `speed_error_m_s` ölçütü |
| Global arama ile yerel ROI takibi arasındaki doğruluk–süre dengesi | `--search-roi-size` ve çalışma süresi sütunları |
| Düz ya da tekrarlayan topoğrafyada konum ve hız belirsizliğinin saptanması | Belirsizlik saptama ve `AMBIGUOUS` durumu |
| Kalite eşikleri ile yanlış `FIX` oranı arasındaki ödünleşim | Üç kalite kapısı eşiği ve FIX kesinliği |
| Kaba harita aramasının çok işlemli yürütülmesi | `--parallel-workers` ve çalışma süresi dökümü |

Çalışma, hem küçük ve deterministik bir sentetik DEM hem de kullanıcı tarafından sağlanan coğrafi referanslı GeoTIFF DEM üzerinde çalışabilir. Harici veri bu depoya dahil değildir.

**Kapsam dışı.** Proje; uçuş dinamiği, ataletsel seyrüsefer sistemi, Kalman/parçacık filtresi füzyon katmanı, arazi takipli güdüm veya gerçek zamanlı aviyonik kısıtlar modellemez. Lokalizasyon her güncellemede **durumsuz** bir kestiricidir: her güncelleme konumu mevcut ölçüm penceresinden yeniden çözer, geçmiş kestirimler yalnızca ROI ankrajı üzerinden taşınır.

## TERCOM nasıl çalışır

Fiziksel ilke tek bir çıkarma işlemidir. Lazer altimetre hava aracının **zeminden yüksekliğini** (AGL), irtifa modeli ise **deniz seviyesinden yüksekliğini** (MSL) verir. Aradaki fark, aracın tam altındaki arazinin yüksekliğidir:

```text
arazi yüksekliği  =  araç MSL irtifası  −  lazer AGL ölçümü
```

Tek bir değer konum bilgisi değildir: gerçek bir haritada binlerce hücre aynı yüksekliğe sahiptir. Neredeyse benzersiz olan şey, bilinen bir bağıl yol boyunca toplanmış bu değerlerin **dizisidir** — yani arazi profili, uçuş izinin yükseklik imzası. TERCOM, ölçülen bu imzayı referans DEM üzerinde kaydırarak en iyi oturduğu yeri sorar.

Bu oturtma işlemini hesaplanabilir kılan üç bileşen vardır:

1. **Ölçüm penceresi.** Son `profile_window_size` ölçüm (varsayılan `100`) kayan bir pencerede tutulur. Eski ölçümler pencereden çıkar; böylece profil sınırsız büyümek yerine aracı takip eder.
2. **Bağıl geometri.** Yön ve kat edilen mesafe, pencereyi ilk örneğe göre katı bir ofset zincirine dönüştürür. Her örnek kendi yön bilgisini taşıdığından zincir dönüşleri de üretir; L ve zikzak rotalar gerçek biçimlerini korur.
3. **Aday taraması.** Bilinmeyen yalnızca zincirin **başlangıcıdır**. Eşleştirici bu başlangıcı her aday DEM hücresine yerleştirir, zincir boyunca DEM'i iki doğrusal aradeğerlemeyle örnekler ve elde edilen beklenen profili ölçülenle karşılaştırarak puanlar. Yön veya hız da bilinmiyorsa zincir ayrıca döndürülür ya da ölçeklenir; bu modların pahalı olmasının nedeni budur.

Bunun sonucu şudur: TERCOM performansı yalnızca algoritmanın değil, **arazinin** bir özelliğidir. Keskin ve tekrarlamayan rölyef tek ve derin bir puan minimumu üretir; düz bir plato veya periyodik bir sırt dizisi ise birbirine denk çok sayıda minimum üretir. Bu nedenle simülatör, minimum ayırt edici değilken çözüm üretmeye zorlamaz — bunun yerine `AMBIGUOUS` bildirir.

## Yöntem

### Lokalizasyon akışı

```text
Gerçek veya sentetik DEM
        │
        ├──► uçuş ve sensör benzetimi
        │              │
        │              ▼
        │    lazer / barometre / hareket ölçümleri
        │              │
        │              ▼
        └──────► kayan arazi profili
                       │
                       ▼
           kaba → orta → ince arama
                       │
                       ▼
        kalite ve belirsizlik denetimi
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
     FIX           AMBIGUOUS  QUALITY INSUFFICIENT
```

Her güncelleme şu adımları yürütür:

1. **Ölç.** Sensör benzeticisi bir lazer AGL okuması (gürültü, aykırı değer ve kayıp okuma ile), bir barometrik MSL okuması (bias ve rastgele yürüyüş ile), bir yön ve — hareket modeline göre — kat edilen mesafeyi üretir. Ölçüm kayan pencereye eklenir.
2. **Pencereyi denetle.** Pencere örnek sayısı (`min_profile_length`), mesafe (`min_profile_distance_m`) veya bilinmeyen hız modunda geçen süre (`min_profile_duration_s`) bakımından yetersizse güncelleme `profile_incomplete` gerekçesiyle bırakılır. Bu, kısa ve ayırt edici olmayan bir profilin harita üzerinde yanlış bir yere kilitlenmesini önler.
3. **Aday profilleri kur.** Her aday başlangıç hücresi için DEM, ofset zinciri boyunca iki doğrusal aradeğerlemeyle örneklenir. DEM dışına düşen örnekler `NaN` olur ve puanlamaya girmez.
4. **Puanla.** Beklenen AGL irtifa modelinden türetilir; `ölçülen − beklenen` artığı kayıp fonksiyonuyla (varsayılan Huber) tek bir sayıya indirgenir. Düşük puan daha iyidir.
5. **Filtrele.** Önce süreklilik kapısı, sonra mutlak kalite kapısı; fiziksel olarak makul olmayan veya basitçe iyi oturmayan adaylar elenir — bkz. [Puanlama ve kalite kapıları](#puanlama-ve-kalite-kapıları).
6. **Karar ver.** Hayatta kalan adaylar belirsizlik açısından denetlenir. Yalnızca ayırt edici ve kaliteden geçen en iyi aday `FIX` olur ve ROI ankrajını günceller.

### Uçuş irtifası modelleri

İrtifa modeli, bir aday hücre için **beklenen** lazer okumasının nasıl hesaplanacağını, dolayısıyla lokalizasyonun ne kadar mutlak irtifa bilgisi varsayabileceğini belirler. `SensorConfig.altitude_mode` ile seçilir.

| Mod | Lokalizasyonun kullandığı bilgi | Kestirim | Araştırma amacı |
|---|---|---|---|
| `known_msl_altitude` | Sabit ve bilinen MSL irtifası | `beklenen = constant_msl_m − DEM` | İdeal referans senaryosu |
| `unknown_constant_msl_altitude` | Profil boyunca sabit fakat bilinmeyen MSL irtifası | İrtifa her aday için `medyan(lazer + DEM)` ile çözülür; bu, bilinmeyen sabit ofseti sadeleştirir | Mutlak irtifa bilgisiz eşleştirme |
| `barometric_altitude` | Bias ve gürültü içerebilen zamana bağlı barometre ölçümü | Sabit barometre bias'ı her aday için `medyan(lazer + DEM − baro)` ile çözülür; beklenen profil barometrenin **biçimini** izler | Daha gerçekçi sensör senaryosu |

Son iki mod sabit bir irtifa hatasına dayanıklıdır; çünkü bu ofseti verinin kendisinden kestirirler. Soğuramadıkları şey **pencere içindeki sürüklenmedir** — barometre modelinde rastgele yürüyüş teriminin ve pencerede bir üst süre sınırının bulunmasının nedeni tam olarak budur.

Sentetik senaryoda uçuş irtifası şansa bırakılmaz: `constant_msl_m`, kaynak haritadaki en yüksek arazi ile `min_safe_agl_m` (varsayılan `50 m`) toplamının altında kalıyorsa, bir üst 10 m'ye yuvarlanarak güvenli bir değere otomatik yükseltilir. Böylece lazer sessizce geçersiz örnekler üretmek yerine ölçüm menzili içinde kalır.

### Hareket bilgisi modelleri

Hareket modeli, lokalizasyonun iki örnek arasında **aracın ne kadar yol aldığı** hakkında ne bildiğini belirler. `--motion-mode` veya `LocalizationConfig.motion_mode` ile seçilir.

| Mod | Lokalizasyona verilen hareket bilgisi | Maliyet |
|---|---|---|
| `known_distance` | Kusursuz kat edilen mesafe; varsayılan ideal mod | En düşük — hücre başına tek aday profil |
| `measured_speed` | Gürültülü hız ölçümünden (bias + gürültü + rastgele yürüyüş) türetilen mesafe | Aynı, ancak pencere boyunca odometri hatası birikir |
| `unknown_constant_speed` | Mesafe ve hız verilmez; konum ve sabit hız birlikte aranır | En yüksek — arama her hız hipotezi için tekrarlanır |

`unknown_constant_speed` modunda her hız hipotezi için zaman farkından `mesafe = hız × zaman` ilişkisi kurulur. Her örneğin kendi yön bilgisi kullanıldığından dönüşlü, L ve zikzak rotaların geometrisi korunur. Aramanın kendisi de kabadan inceyedir: `speed_search_min_m_s` ile `speed_search_max_m_s` arası (varsayılan `5–30 m/s`) `5 m/s` kaba adımla taranır, en iyi `speed_search_keep_hypotheses` hipotez (varsayılan `3`) saklanır, ardından `1 m/s` ve `0.2 m/s` ile inceltilir. Takip kurulduktan sonra yalnızca önceki kestirimin çevresindeki dar bir bant yeniden aranır (`speed_tracking_half_range_m_s`, varsayılan `±1 m/s`).

Bu modun sonuçlarını yorumlamadan önce bilinmesi gereken iki davranış vardır:

- Profil artık mesafeyle denetlenemeyeceğinden (bilinmeyen tam olarak mesafedir), **geçen süreyle** denetlenir: `min_profile_duration_s` / `max_profile_duration_s`, varsayılan `30 s` / `120 s`.
- Bu modun seçilmesi irtifa modelini de `barometric_altitude`'a çevirir; çünkü hızın bilinmediğini savunurken mutlak irtifanın bilindiğini varsaymak tutarlı bir senaryo olmazdı.

### Kabadan inceye arama

Her hücrenin ve her yönün tam çözünürlükte taranması hesaplama açısından kabul edilemez olurdu; bu nedenle uzamsal arama giderek daralan üç geçişte yürütülür. Her geçiş yalnızca en iyi `top_k` adayı (varsayılan `5`) saklar ve bir sonrakine devreder.

| Aşama | Adım | Arama alanı | Denenen yönler |
|---|---|---|---|
| Kaba | `coarse_stride` = `10` px | Tüm harita veya takipteyken ROI | Tüm aday yönler |
| Orta | `medium_stride` = `3` px | Kalan her kaba adayın ±`refinement_radius_px` = `20` px komşuluğu | O adayın yönüne en yakın 5 yön |
| İnce | `fine_stride` = `1` px | Kalan her orta adayın ±`10` px (yarıçapın yarısı) komşuluğu | ±`fine_heading_step_deg` = `0.5°` |

Kaba ve orta geçişler yalnızca puanı hesaplar, tüm ölçüt kümesini değil. Son `top_k` aday daha sonra tüm ölçütlerle yeniden değerlendirilir; çünkü kalite kapısı, hesaplaması daha pahalı olan inlier RMSE, korelasyon ve geçerli örnek oranına ihtiyaç duyar. Yalnızca tek bir yön aranıyorsa kaba geçiş, genel aday döngüsü yerine vektörleştirilmiş bir NumPy uygulamasına geçer.

Yalnızca kaba geçiş paralelleştirilir (bkz. [Paralel kaba arama](#paralel-kaba-arama)) — maliyeti harita alanıyla ölçeklenen aşama odur; inceltme geçişleri zaten küçük komşuluklarla sınırlıdır.

### Puanlama ve kalite kapıları

**Kayıp fonksiyonu.** Ölçülen ve beklenen profil arasındaki artık, `loss_method` ile tek bir puana indirgenir: `huber` (varsayılan), `rmse` veya `mae`. Huber, `huber_delta` (varsayılan `10 m`) altındaki artıklar için karesel, üstünde ise doğrusaldır; böylece `50 m`'lik tek bir lazer aykırı değeri tüm pencereye hükmetmek yerine orantılı biçimde cezalandırılır. **Düşük puan daha iyidir** — bu bir benzerlik değil, bir hatadır.

**Sırayla uygulanan üç filtre.** Üçünü de geçen aday konum çözümü olur:

1. **Süreklilik kapısı** (`max_match_jump_m`, varsayılan `10 m`). Önceki kabul edilmiş ankrajdan bu mesafeden daha uzaktaki adaylar iki güncelleme arasında fiziksel olarak makul değildir ve elenir. Hiçbiri kalmazsa güncelleme `continuity` gerekçesiyle reddedilir. Değer `0` yapılırsa kapı devre dışı kalır — hızlı bilinmeyen hız ön ayarının yaptığı budur; çünkü ölçeklenen bir profil ankrajı meşru biçimde kaydırabilir.
2. **Mutlak kalite kapısı.** En kötü `quality_trim_fraction` (varsayılan `%5`) oranındaki artıklar atıldıktan sonra **inlier** profil üzerinde hesaplanır; böylece tek bir bozuk örnek tüm pencereyi zehirleyemez. Üç koşulun tamamı sağlanmalıdır:
   - `inlier RMSE ≤ max_match_inlier_rmse_m` (varsayılan `3 m`)
   - `inlier korelasyon ≥ min_match_inlier_correlation` (varsayılan `0.80`)
   - `geçerli örnek oranı ≥ min_match_valid_ratio` (varsayılan `0.80`)

   Bunlar göreli bir sıralama değil, **mutlak** eşiklerdir. Klasik TERCOM hata biçimini — hepsi kötü olan eşleşmelerin en iyisini kendinden emin biçimde bildirmeyi — önleyen mekanizma budur. Hiçbiri geçemezse güncelleme `quality` gerekçesiyle reddedilir.
3. **Belirsizlik denetimi.** Bir reddetme değildir; en iyi aday yine bildirilir ama işaretlenir. Aşağıya bakınız.

Bir güncelleme reddedildiğinde ölçülen profil **atılmaz**. Bunun yerine arama alanı genişletilir: ROI araması büyür, zaten tüm haritayı kapsayan bir arama ise eskimiş ankrajını bırakıp global aramaya döner. Biriken ölçüm geçmişi kurtarma boyunca korunur.

### Belirsizlik saptama

Düşük bir puan tek başına konumun bilindiği anlamına gelmez. Haritanın birbirinden uzak birkaç yeri neredeyse aynı ölçüde iyi oturuyorsa doğru mühendislik yanıtı, kümenin keyfî en iyisi değil, "bilmiyorum"dur.

**Konum belirsizliği** için aşağıdakilerin ikisinin de aynı anda sağlanması gerekir:

- **Küçük puan marjı** — en iyi ile ikinci en iyi aday arasındaki göreli puan farkı `0.05`'in (%5) altında.
- **Büyük uzamsal dağılım** — en iyi adayların raster konumlarının standart sapması `10` px'i aşıyor.

Bu birliktelik önemlidir. Küçük marj ile **küçük** dağılım, puan yüzeyinin tek bir doğru minimumun çevresinde düz olduğu anlamına gelir; bu normal ve zararsızdır. Küçük marj ile **büyük** dağılım ise haritanın farklı bölgelerinde gerçekten yarışan hipotezler demektir.

**Hız belirsizliği** (yalnızca bilinmeyen hız modunda) aynı fikri hız ekseninde uygular. Adaylar önce her ayrık hız için en iyisine indirgenir; ardından puan marjı `speed_ambiguity_score_margin` (varsayılan `0.05`) altında **ve** en iyi `speed_ambiguity_top_k` (varsayılan `5`) hipotezin hız standart sapması `speed_ambiguity_std_threshold_m_s` (varsayılan `2 m/s`) üzerindeyse işaretlenir. Sonuç ayrıca niteliksel bir güven göstergesi olarak bildirilir:

| Gösterge | Koşul | Arayüz etiketi |
|---|---|---|
| `high` | Puan marjı ≥ `max(0.15, 3 × eşik)` | `YÜKSEK` |
| `medium` | Puan marjı ≥ eşik | `ORTA` |
| `low` | Marj eşiğin altında, ancak hızlar dağınık değil | `DÜŞÜK` |
| `ambiguous` | Küçük marj **ve** dağınık hızlar | `BELİRSİZ` |

Her iki belirsizlik türü de güncellemeyi `AMBIGUOUS` olarak işaretler.

### Lokalizasyon durumları

| Durum | Masaüstü arayüz etiketi | Anlamı | Kayda geçen gerekçe |
|---|---|---|---|
| `FIX` | `GÜVENLİ (FIX)` | Tüm kalite kapılarını geçen, kabul edilmiş çözüm | — |
| `AMBIGUOUS` | `BELİRSİZ (AMBIG)` | Adaylar benzer puanlı fakat uzamsal olarak dağınık | — |
| `AMBIGUOUS` (hız) | `HIZ BELİRSİZ (AMBIG)` | Konum çözülür, ancak hız hipotezi ayrıştırılamaz | — |
| `QUALITY INSUFFICIENT` | `KALİTE YETERSİZ` | En iyi aday mutlak kalite kapısından reddedildi | `quality` |
| `RECOVERY` | `YENİDEN ARANIYOR` | Eşleşme kayboldu; arama alanı genişletiliyor | `continuity` / `no_candidates` |
| `NO MATCH` | `EŞLEŞME YOK` | Henüz yeterli profil verisi birikmedi | `profile_incomplete` |

Bir araştırma koşusunda `QUALITY INSUFFICIENT` ve `AMBIGUOUS` **birer sonuçtur, başarısızlık değil**. Bunları hiç üretmeyen bir konfigürasyon genellikle eşikleri fazla gevşek olan bir konfigürasyondur ve bunu yanlış `FIX` oranı ele verir.

## Temel özellikler

- PySide6 tabanlı manuel görev ve telemetri arayüzü; canlı harita, profil karşılaştırması ve adım bazlı günlük
- Harici veri gerektirmeyen, küçük ve deterministik sentetik arazi üretimi (`plane` ve `valley` ön ayarları)
- GeoTIFF DEM okuma; fiziksel kapsamı koruyan yeniden örnekleme ve nodata sağlık denetimi
- Bias, gürültü, sürüklenme, aykırı değer ve kayıp okuma içeren lazer, barometre, pusula ve hız sensörü hata modelleri
- Bilinen yön veya `0.5°`'ye kadar inen kabadan inceye yön araması
- Aykırı değer kırpmalı kalite ölçütleriyle Huber / RMSE / MAE temelli profil eşleştirme
- Global arama ve ölçüm geçmişini koruyan, kademeli genişleyen isteğe bağlı ROI kurtarma akışı
- Yanlış yerel ankrajı önleyen mutlak kalite kapıları ve makul olmayan sıçramalara karşı süreklilik denetimi
- Sabit fakat bilinmeyen hızın konumla birlikte kestirimi ve buna özel güven göstergesi
- Büyük kaba aramalar için kalıcı çok işlemli işçi havuzu; seri–paralel eşdeğerliği testle güvence altında
- Validasyon/final rota ayrımı ve Pareto analizi içeren deterministik parametre optimizasyonu
- CSV, JSON, JSONL ve XLSX biçimlerinde deney kayıtları
- Ground-truth rota verisini lokalizasyonun dışında tutan katı konfigürasyon sınırı

## Kurulum

### Gereksinimler

- Windows, Linux veya macOS
- Python `3.10–3.13`
- Masaüstü arayüzü için grafik oturumu (arayüzsüz koşular ekran gerektirmez)
- Harici arazi deneyi için GeoTIFF biçiminde bir DEM

### Kurulum adımları

```powershell
git clone <depo-adresi>
cd tercom-simulasyon

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Yalnızca çalışma zamanı bağımlılıkları gerekiyorsa:

```powershell
python -m pip install -e .
```

Kurulum `matplotlib`, `numpy`, `PySide6` ve `rasterio` paketlerini yükler. `dev` seçeneği ayrıca `pytest` ve `ruff` içerir.

Paket düzenlenebilir (editable) kipte kurulur; kaynak değişiklikleri yeniden kurulum gerektirmeden etkili olur. Ayrıca `tercom-terrain-nav` adlı bir konsol giriş noktası kaydedilir ve `python run_terrain_nav.py` ile eşdeğerdir.

### Kurulumun doğrulanması

```powershell
python run_terrain_nav.py --headless --fast
python -m pytest
```

İlk komut saniyeler içinde bitmeli ve `results/config.json` ile `results/results.csv` dosyalarını yazmalıdır. Test paketi geçiyorsa sayısal çekirdek, CLI ve arayüz konfigürasyonu çalışıyor demektir.

## Hızlı başlangıç

Uygulamanın giriş noktası `run_terrain_nav.py` dosyasıdır.

```powershell
# Küçük sentetik DEM ile hızlı masaüstü deneyi
python run_terrain_nav.py --fast

# Arayüzsüz, tekrarlanabilir sentetik kontrol koşusu
python run_terrain_nav.py --headless --fast

# Kullanıcı tarafından sağlanan GeoTIFF DEM
python run_terrain_nav.py --dem "C:\veri\arazi.tif"

# Tüm seçenekler
python run_terrain_nav.py --help
```

Program parametresiz başlatıldığında, kaynak kodda tanımlı yerel varsayılan DEM mevcutsa onu kullanır; dosya bulunamazsa sentetik araziye geri döner. Bu davranış yerel bir dosyaya bağlı olduğundan, **akademik tekrarlanabilirlik için DEM yolunun her koşuda `--dem` ile açıkça verilmesi önerilir** — aksi hâlde aynı komut iki makinede farklı anlamlara gelebilir.

`--fast` yalnızca daha küçük bir harita demek değildir: sentetik araziyi `100 × 100` hücreye (bilinmeyen hız modunda `120 × 160`) küçültür, rotayı ortalar ve kısaltır; böylece tam bir koşu saniyeler sürer. Akış kontrolü ve geliştirme içindir, performans iddiası için değil.

### Manuel denetimler

Masaüstü arayüzünde araç elle uçurulur:

| Tuş | İşlev |
|---|---|
| `W` / `S` | İleri / geri hareket |
| `A` / `D` | Sola / sağa yanal hareket |
| `Q` / `E` | Sola / sağa dönüş |

Varsayılan manuel hareket komutu `100 m`, dönüş komutu `15°` ve profil örnekleme aralığı `20 m`'dir. Bu değerler `RouteConfig` üzerinden değiştirilebilir.

### Telemetri panelinin okunması

| Alan | Anlamı |
|---|---|
| `Adım (Step)` | Koşu içindeki güncelleme sırası |
| `CPU İşçileri` | Son kaba aramada fiilen kullanılan işçi süreç sayısı |
| `Gerçek Konum` / `Tahmin Konumu` | Harita koordinatlarında gerçek ve kestirilen konum |
| `Gerçek Yön` / `Tahmin Yönü` | Gerçek ve kestirilen yön |
| `Tahmini Hız` / `Hız Güveni` | Kestirilen hız ve güven göstergesi (bilinmeyen hız modu) |
| `Sensör MSL` / `Lazer AGL` | Güncel irtifa ve lazer okumaları |
| `Konum Hatası` | Gerçek ve kestirilen konum arasındaki uzaklık |
| `Arama Dağılımı` | En iyi adayların uzamsal yayılımı |
| `Eşleşme Skoru` | Kabul edilen eşleşmenin inlier RMSE değeri |
| `Güven Durumu` | Yukarıdaki tablodaki lokalizasyon durumu |

Bunlardan ikisi kolayca yanlış okunur. `Eşleşme Skoru` konum hatası **değildir**; profil uyum hatasıdır ve düşük değer daha iyidir — bir koşu, yanlış tepeye kilitlenmişken bile mükemmel bir eşleşme skoru gösterebilir. `Arama Dağılımı` adayların raster uzayındaki yayılımından türetilir; fiziksel metre olarak yorumlanacaksa DEM piksel boyutuyla dönüştürülmelidir.

## Deney modları

### İdeal sensör modu

```powershell
python run_terrain_nav.py --headless --fast
```

Bu referans senaryosu bilinen MSL irtifası, bilinen yön ve kusursuz hareket mesafesi kullanır. Gürültülü senaryolarla karşılaştırma için kontrol grubu niteliğindedir: burada kalan hata sensörlere değil, eşleştiriciye, DEM çözünürlüğüne ve aradeğerlemeye atfedilebilir.

### Gerçekçi sensör gürültüsü

```powershell
python run_terrain_nav.py --realistic-noise --dem "C:\veri\arazi.tif"
```

Bu ön ayar mutlak irtifayı bias ve gürültü içeren bir barometreyle değiştirir ve hareket modelini gürültülü hız ölçümüne çevirir. Ayrıca, gerçekçi gürültü altında her güncellemeyi reddedecek olan konfigürasyon kısımlarını gevşetir:

| Parametre | Varsayılan | `--realistic-noise` ile |
|---|---|---|
| `altitude_mode` | `known_msl_altitude` | `barometric_altitude` |
| `baro_bias_m` | `0.0` | `75.0` |
| `baro_noise_std_m` | `1.0` | `2.0` |
| `baro_random_walk_std_m` | `0.1` | `0.03` |
| `speed_noise_std_m_s` | `0.0` | `0.25` |
| `speed_random_walk_std_m_s` | `0.0` | `0.03` |
| `min_profile_length` | `10` | `5` |
| `min_profile_distance_m` | `0.0` | `800.0` (harici DEM) / `40.0` (hızlı sentetik) |
| `max_profile_distance_m` | `0.0` (sınırsız) | `2000.0` (harici DEM) |
| `max_match_inlier_rmse_m` | `3.0` | `5.0` |
| `max_match_jump_m` | `10.0` | `50.0` |
| `motion_mode` | `known_distance` | `measured_speed` |

İki profil-mesafe sınırı bu ön ayarın temel ödünleşimini kodlar. `800 m`'lik **alt** sınır vardır; çünkü gürültü altında kısa bir profil yeterince ayırt edici değildir ve yanlış global kilitlenmeye davetiye çıkarır. `2000 m`'lik **üst** sınır vardır; çünkü profil geometrisi gürültülü odometriden kurulur ve çok uzun bir pencere kendi biçimini bozacak kadar mesafe hatası biriktirir. Pusula yönü burada bilinen kabul edilir; yön gürültüsü ayrıca `SensorConfig.heading_mode` ile eklenebilir.

### Hız bilgisi olmadan lokalizasyon

```powershell
python run_terrain_nav.py --headless --fast --unknown-speed

# Eşdeğer açık gösterim
python run_terrain_nav.py --motion-mode unknown_constant_speed

# Fiziksel hız aralığını sınırlandırma
python run_terrain_nav.py --unknown-speed --speed-search-min 8 --speed-search-max 24
```

Bu mod kayan profil boyunca hızın sabit olduğunu varsayar. Simülatör aracı hareket ettirmek için gerçek hızı bilse de lokalizasyon katmanına gerçek rota başlangıcı, hızı veya kat edilen mesafe aktarılmaz. Gerçek hız yalnızca sonuç aşamasında `speed_error_m_s` metriğini hesaplamak için kullanılır.

Arama aralığını `--speed-search-min` / `--speed-search-max` ile daraltmak, bu modu hem hızlandırmanın hem de doğrulaştırmanın en etkili tek yoludur; çünkü elenen her hipotez hem bir tam aramanın hem de yanlış bir hızın tesadüfen oturma fırsatının ortadan kalkması demektir. Platformun gerçek uçuş zarfını yansıtan bir aralık meşru bir modelleme varsayımıdır, kestirme yol değil — ancak sonuçlarla birlikte raporlanmalıdır.

### Global arama ve ROI

Varsayılan `--search-roi-size 0` ayarı ROI'yi kapatır ve her güncellemede tüm haritayı arar. ROI yalnızca açıkça etkinleştirilmelidir:

```powershell
python run_terrain_nav.py --dem "C:\veri\arazi.tif" --search-roi-size 512
```

Güvenilir bir eşleşmeden sonra arama, son kabul edilen konum merkezli bu boyutta bir pencereyle sınırlanır. Sonraki bir güncelleme reddedilirse pencere yalnızca başarısız olmaz — her denemede yaklaşık %50 büyür; genişletilmiş pencere de başarısız olduğunda eskimiş ankraj bırakılır ve ölçülen profil korunarak global aramaya dönülür. ROI kenarına yakın düşen kabul edilmiş bir eşleşme de şüpheli sayılır; çünkü gerçek optimum pencerenin hemen dışında olabilir.

ROI bir doğruluk yöntemi değil, hesaplama maliyetini azaltmayı amaçlayan izleme optimizasyonudur. Büyük bir haritada milyonlarca hücre yerine birkaç yüz bin hücre taramak anlamına gelir; küçük bir haritada global aramadan yavaş olabilir ve katı biçimde durumsuz bir değerlendirmenin istemeyebileceği bir "önceki fix" bağımlılığı getirir.

### Paralel kaba arama

```powershell
# Varsayılan: min(4, CPU çekirdek sayısı) işçi süreç
python run_terrain_nav.py --parallel-workers 4

# Seri yürütme
python run_terrain_nav.py --parallel-workers 1
```

Büyük global aramalar satır bantlarına ayrılarak kalıcı işçi süreçlerinde yürütülür — kalıcıdır, çünkü DEM her güncellemede değil, başlangıçta bir kez her işçiye aktarılır. Küçük harita ve ROI aramaları süreçler arası iletişim maliyetinden kaçınmak için seri kalabilir; motor, dağıtmaya değmeyecek kadar küçük aramalarda seri yürütmeye geri döner.

Paralellik **yalnızca süreyi etkiler, sonuçları değil**: seri ve paralel yolların özdeş adaylar üretmesi gerekir ve bunu bir regresyon testi denetler. Yine de işçi sayısı deney ortamı ve DEM boyutuyla birlikte raporlanmalıdır; çünkü çıktıdaki her süre değeri buna bağlıdır.

### Parametre optimizasyonu

```powershell
# Varsayılan deterministik optimizasyon planı
python run_terrain_nav.py --optimizer-benchmark --fast

# Kısa bir yöntem kontrolü
python run_terrain_nav.py --optimizer-benchmark --fast `
  --optimizer-configs 8 `
  --optimizer-refined-configs 4 `
  --optimizer-final-configs 3 `
  --optimizer-routes 4 `
  --optimizer-max-updates-per-route 10
```

Optimizasyon rastgele bir arama değil, deterministik bir hunidir. Aday konfigürasyonlar üretilir ve üç aşamada daraltılır:

1. **Kaba tarama** — tüm adaylar küçük bir rota alt kümesinde koşturularak açıkça kullanışsız olanlar elenir.
2. **İnceltilmiş validasyon** — hayatta kalanlar (`--optimizer-refined-configs`) tüm validasyon rota kümesinde koşturulur ve yarışan amaçlar üzerinde Pareto sınırı hesaplanır.
3. **Görülmemiş final test** — finalistler (`--optimizer-final-configs`), önceki tüm aşamalardan **ayrı tutulmuş** rotalarda koşturulur.

Rota kütüphanesi sekiz şablondan kurulur (`duz_dogu`, `duz_kuzey`, `l_dogu_kuzey`, `l_kuzey_dogu`, `zikzak`, `merdiven`, `capraz_kesit`, `u_donus`); her biri dört yönelimde örneklenir ve güncel haritaya göre ölçeklenir. Her dördüncü rota final teste ayrılır ve seçim için hiç kullanılmaz; final sayılarını ayarlama amacının tekrarı olmaktan çıkarıp gerçek bir dışarıda-bırakılmış ölçüme dönüştüren de budur. Her rota ayrıca üç yön senaryosunda değerlendirilir: `known_heading`, `noisy_heading_3deg`, `noisy_heading_5deg`.

Dört adlandırılmış konfigürasyon raporlanır; çünkü "en iyi" tek bir şey değildir:

| Seçim | Neye göre optimize |
|---|---|
| `safe` | En düşük yanlış `FIX` oranı, sonra en yüksek FIX kesinliği |
| `fast` | FIX kesinliği ≥ %95 olan konfigürasyonlar içinde en düşük P95 takip süresi |
| `accurate` | En düşük P95 konum hatası |
| `balanced` | Doğruluk, kesinlik ve süre arasında en iyi bileşik skor |

Kabul edilen bir konumun hatası `50 m`'yi aşarsa çözüm **yanlış** `FIX` sayılır. Küçük sentetik koşuda seçilen bir konfigürasyon, harici DEM üzerinde ayrıca doğrulanmadan üretim ayarı olarak yorumlanmamalıdır.

## Konfigürasyon referansı

CLI en yaygın anahtarları sunar; ancak her deney parametresi [config.py](terrain_nav/config.py) içindeki dondurulmuş dataclass'larda yaşar ve `results/config.json` içine eksiksiz yazılır. Aşağıdaki tablolar bir deney için en çok önem taşıyan alanları listeler.

### `TerrainConfig` — harita ve referans veri

| Alan | Varsayılan | Anlamı |
|---|---|---|
| `preset` | `valley` | Sentetik arazi biçimi: `valley` veya `plane` |
| `seed` | `42` | Arazi dokusu ve DEM gürültüsü için rastgelelik tohumu |
| `rows` / `cols` | `1000` / `1000` | Hücre cinsinden sentetik DEM boyutları |
| `dx` / `dy` | `1.0` / `1.0` | Metre cinsinden hücre boyutu |
| `base_elevation` | `1000.0` | Sentetik arazinin taban yüksekliği (m) |
| `dem_noise_std_m` | `0.5` | Referans DEM'i gerçeklikten ayıran gürültü |
| `dem_bias_m` | `0.0` | Referans DEM ile gerçeklik arasındaki sabit ofset |
| `dem_path` | `""` | Harici GeoTIFF yolu; boş ise sentetik |
| `dem_target_size` | `2048` | Yeniden örnekleme sonrası uzun kenar hücre bütçesi |

`dem_noise_std_m` ve `dem_bias_m`, bir referans haritanın hiçbir zaman arazinin kendisi olmadığı gerçeğini modeller. `plane` ön ayarı bilinçli olarak yozdur — düz bir yüzey hiçbir şeyi konumlandıramayacağından belirsizliği gözlemlemek için kontrol durumudur.

### `RouteConfig` — uçuş yolu (yalnızca ground truth)

| Alan | Varsayılan | Anlamı |
|---|---|---|
| `start_row` / `start_col` | `500` / `500` | Rotanın başlangıç hücresi |
| `heading_deg` | `0.0` | Başlangıç yönü (0° = Kuzey, 90° = Doğu) |
| `speed_m_s` | `10.0` | Gerçek hız |
| `sample_spacing_m` | `10.0` | Ölçümler arası mesafe |
| `route_length_m` | `1000.0` | Toplam rota uzunluğu |
| `manual_step_distance_m` | `100.0` | Manuel hareket komutu başına mesafe |
| `manual_turn_step_deg` | `15.0` | Manuel dönüş komutu başına açı |

Bu sınıfın hiçbir alanı lokalizasyona görünmez. `localization_runtime_config()` yalnızca sensör, algoritma ve hareket modu ayarlarını lokalizasyon katmanına kopyalar; bir gerçeklik değerinin kestirime sızamayacağının yapısal güvencesi budur.

### `SensorConfig` — ölçüm modelleri

| Alan | Varsayılan | Anlamı |
|---|---|---|
| `altitude_mode` | `known_msl_altitude` | İrtifa modeli; bkz. [Uçuş irtifası modelleri](#uçuş-irtifası-modelleri) |
| `constant_msl_m` | `1500.0` | Uçuş irtifası; arazi emniyeti için gerekirse otomatik yükseltilir |
| `min_safe_agl_m` | `50.0` | Bu otomatik düzeltmede kullanılan asgari emniyet payı |
| `laser_noise_std_m` | `0.5` | Lazer ölçüm gürültüsü |
| `laser_outlier_prob` | `0.01` | Aykırı okuma olasılığı |
| `laser_outlier_magnitude_m` | `50.0` | Bu aykırı okumanın büyüklüğü |
| `laser_drop_prob` | `0.02` | Kayıp (geçersiz) okuma olasılığı |
| `laser_min_range_m` / `laser_max_range_m` | `0.5` / `3000.0` | Ölçüm menzili; dışındaki okumalar geçersizdir |
| `baro_noise_std_m` | `1.0` | Barometre gürültüsü |
| `baro_bias_m` | `0.0` | Sabit barometre ofseti |
| `baro_drift_rate_m_s` | `0.01` | Saniye başına sistematik sürüklenme |
| `baro_random_walk_std_m` | `0.1` | Sürüklenmenin rastgele yürüyüş bileşeni |
| `heading_mode` | `known_heading` | `known_heading`, `noisy_heading` veya `unknown_heading` |
| `sensor_heading_noise_std_deg` | `1.0` | Yön bilinmiyorken pusula gürültüsü |
| `speed_noise_std_m_s` | `0.0` | Hız ölçüm gürültüsü |
| `speed_bias_m_s` | `0.0` | Sabit hız ofseti |

Kalite kapısının tüm örnekler yerine kırpılmış inlier'lar üzerinde hesaplanmasının nedeni lazer kayıpları ve aykırı değerleridir: varsayılan ayarlarda yaklaşık her elli örnekten biri eksik, her yüz örnekten biri `50 m` sapmalıdır.

### `AlgorithmConfig` — arama, puanlama ve kapılar

| Alan | Varsayılan | Anlamı |
|---|---|---|
| `profile_window_size` | `100` | Ölçüm cinsinden kayan pencere uzunluğu |
| `min_profile_length` | `10` | Eşleştirme denenmeden önceki asgari geçerli örnek sayısı |
| `min_profile_distance_m` / `max_profile_distance_m` | `0.0` / `0.0` | Mesafe temelli pencere sınırları; `0` kapatır |
| `min_profile_duration_s` / `max_profile_duration_s` | `30.0` / `120.0` | Bilinmeyen hız modunda kullanılan süre sınırları |
| `coarse_stride` / `medium_stride` / `fine_stride` | `10` / `3` / `1` | Piksel cinsinden arama adımları |
| `refinement_radius_px` | `20` | Saklanan her aday çevresindeki komşuluk yarıçapı |
| `top_k` | `5` | Arama aşamaları arasında taşınan aday sayısı |
| `fine_heading_step_deg` | `0.5` | İnce yön geçişinin açısal çözünürlüğü |
| `loss_method` | `huber` | `huber`, `rmse` veya `mae` |
| `huber_delta` | `10.0` | Huber geçiş noktası (m) |
| `quality_trim_fraction` | `0.05` | Kalite ölçütlerinden dışlanan en kötü artık oranı |
| `max_match_inlier_rmse_m` | `3.0` | Kalite kapısı: azami inlier RMSE |
| `min_match_inlier_correlation` | `0.80` | Kalite kapısı: asgari inlier korelasyon |
| `min_match_valid_ratio` | `0.80` | Kalite kapısı: asgari geçerli örnek oranı |
| `max_match_jump_m` | `10.0` | Süreklilik kapısı; `0` kapatır |
| `search_roi_size_px` | `0` | ROI kenar uzunluğu; `0` global arar |
| `parallel_workers` | `1` | Kaba arama için işçi süreç sayısı |
| `speed_search_min_m_s` / `speed_search_max_m_s` | `5.0` / `30.0` | Hız hipotezi aralığı |
| `speed_ambiguity_score_margin` | `0.05` | Hız belirsizliği: puan marjı eşiği |
| `speed_ambiguity_std_threshold_m_s` | `2.0` | Hız belirsizliği: hız dağılımı eşiği |

Dataclass kendini kuruluşta doğrular: pozitif olmayan bir arama adımı, alt sınırından küçük bir üst sınır veya tanınmayan bir yeniden örnekleme modu, sessizce yanlış sonuç üretmek yerine anında `ValueError` yükseltir.

## Komut satırı referansı

| Seçenek | Varsayılan | Açıklama |
|---|---|---|
| `--headless` | kapalı | Masaüstü arayüzü olmadan çalışır, sonuçları `results/` altına yazar |
| `--fast` | kapalı | Küçük ve deterministik sentetik arazi kullanır |
| `--dem PATH` | kaynak koddaki varsayılan, yoksa sentetik | Harici GeoTIFF DEM |
| `--dem-target-size PX` | `2048` | DEM'in uzun kenarı için hedef hücre sayısı |
| `--search-roi-size PX` | `0` | Piksel cinsinden ROI kenar uzunluğu; `0` tüm haritayı arar |
| `--start-row ROW` / `--start-col COL` | `500` / `500` | Rotanın başlangıç hücresi; `--fast` sentetik modda ortalanır |
| `--realistic-noise` | kapalı | Barometrik irtifa ve gürültülü hız ön ayarı |
| `--motion-mode MODE` | `known_distance` | `known_distance`, `measured_speed` veya `unknown_constant_speed` |
| `--unknown-speed` | kapalı | `--motion-mode unknown_constant_speed` kısayolu |
| `--speed-search-min M_S` / `--speed-search-max M_S` | `5` / `30` | m/s cinsinden hız hipotezi aralığı |
| `--parallel-workers N` | `min(4, CPU çekirdek sayısı)` | Büyük kaba aramalar için kalıcı işçi süreç sayısı |
| `--optimizer-benchmark` | kapalı | Deterministik parametre optimizasyonu koşusunu başlatır |
| `--optimizer-configs N` | `64` | Başlangıç aday konfigürasyon sayısı |
| `--optimizer-refined-configs N` | `12` | İyileştirme aşamasına taşınan konfigürasyon sayısı |
| `--optimizer-final-configs N` | `10` | Final değerlendirmesine taşınan konfigürasyon sayısı |
| `--optimizer-routes N` | `12` | Değerlendirilen rota sayısı; en az `2` |
| `--optimizer-sample-spacing M` | rota konfigürasyonu | Optimizasyon sırasında kullanılan profil örnekleme aralığı |
| `--optimizer-max-updates-per-route N` | `0` | Rota başına güncelleme sınırı; `0` sınırsız demektir |
| `--optimizer-output DIR` | `results/` | Optimizasyon çıktıları için hedef dizin |

Varsayılanı makineye bağlı olan tek seçeneğin `--parallel-workers` olduğuna dikkat edin. Süre raporlanacak her koşuda bunu açıkça sabitleyin.

## Çıktılar ve değerlendirme ölçütleri

### Simülasyon koşusu

Arayüzsüz koşu, `results/` altında iki dosya üretir:

- `config.json` — dataclass'lardan serileştirilmiş, eksiksiz sensör, algoritma, arazi ve rota konfigürasyonu. Koşunun asıl kaydı budur; yalnızca komut satırında verilenleri değil, tüm parametreleri içerir.
- `results.csv` — güncelleme başına bir satır, 25 sütun:

| Sütun grubu | Sütunlar |
|---|---|
| Zaman ve gerçeklik | `timestamp_s`, `true_x`, `true_y`, `true_heading` |
| Kestirim | `est_x`, `est_y`, `est_heading`, `est_msl` |
| Hata | `error_x`, `error_y`, `error_pos`, `error_heading` |
| Eşleşme kalitesi | `is_ambiguous`, `score`, `inlier_rmse_m`, `correlation`, `valid_ratio` |
| Hız | `estimated_speed_m_s`, `second_best_speed_m_s`, `true_speed_m_s`, `speed_error_m_s`, `speed_is_ambiguous`, `speed_score_margin`, `speed_spread_m_s`, `speed_confidence` |

Reddedilen güncellemeler satır üretmez; bu nedenle satır sayısı güncelleme sayısından **azdır**. Bu fark başlı başına bir ölçümdür: reddedilen çözüm sayısı rapora girmelidir ve kabul oranını belirtmeden yalnızca kabul edilen satırlar üzerinden ortalama hata hesaplamak performansı olduğundan iyi gösterir.

### Optimizasyon koşusu

Optimizasyon çalışması zaman damgalı `optimizer_<damga>_summary.csv`, `optimizer_<damga>_details.jsonl` ve `optimizer_<damga>.xlsx` dosyaları üretir. Çalışma kitabı her analiz ekseni için bir sayfa içerir:

| Sayfa | İçerik |
|---|---|
| `Genel Ozet` | Koşu planı, rota ayrımı, aday sayıları, başlıca sonuçlar |
| `Top Configurations` | En iyi on konfigürasyon |
| `Pareto Frontier` | Yarışan amaçlar üzerinde domine edilmeyen konfigürasyonlar |
| `Final Test Results` | Dışarıda bırakılmış rota sonuçları |
| `Quality Gate Analysis` | Korelasyon / RMSE / geçerli oran / kırpma eşiklerinin etkisi |
| `Profile Analysis` | Profil yeniden örnekleme modu ve nokta sayısının etkisi |
| `ROI Analysis` | ROI boyutunun etkisi |
| `Speed Search Analysis` | Hız aralığı, adım boyutları ve saklanan hipotezlerin etkisi |
| `Profile Duration Analysis` | Pencere süre sınırlarının etkisi |
| `Heading Analysis` | Yön senaryosuna göre döküm |
| `Runtime Breakdown` | Aşama bazında süre |
| `Validation Results` | Tam validasyon özetleri |
| `Raw Details` | Güncelleme bazında ham kayıtlar |
| `Eliminated` | Elenen konfigürasyonlar ve elendikleri aşama |

### Başlıca ölçütler

| Ölçüt | Yorum |
|---|---|
| Konum hatası (m) | Gerçek ve kestirilen konum arasındaki Öklid uzaklığı; düşük iyidir |
| Inlier RMSE (m) | Aykırı örnekler kırpıldıktan sonraki profil uyum hatası; düşük iyidir |
| Korelasyon | Beklenen ve ölçülen profil biçimi arasındaki uyum; yüksek iyidir |
| Geçerli örnek oranı | DEM sınırları içinde değerlendirilebilen profil payı; yüksek iyidir |
| Doğru `FIX` oranı | Kabul edilen ve hata eşiği içinde kalan çözümlerin tüm güncellemelere oranı |
| Yanlış `FIX` oranı | Kabul edildiği halde hata eşiğini aşan çözümlerin tüm güncellemelere oranı |
| `FIX` kesinliği | Doğru `FIX` sayısının kabul edilen tüm `FIX` sayısına oranı |
| P95 konum hatası | Kabul edilen çözümlerde konum hatasının 95. yüzdelik değeri |
| Hız MAE (m/s) | Bilinmeyen hız deneylerinde mutlak hız hatası ortalaması |
| Çalışma süresi (ms) | Global ilk çözüm ve takip güncellemelerinin hesaplama maliyeti |

Üç `FIX` ölçütü farklı sorulara yanıt verir ve birlikte raporlanmalıdır. **Doğru FIX oranı** sistemin ne sıklıkla kullanılabilir bir konum ürettiğini; **yanlış FIX oranı** ne sıklıkla kendinden emin ama yanlış bir konum ürettiğini; **FIX kesinliği** ise bildirilen bir `FIX`'e ne kadar güvenilebileceğini sorar. Nadiren yanıt verip hiç yanılmayan bir sistem ile her zaman yanıt verip sık yanılan bir sistem aynı doğru-FIX oranını paylaşabilir, ama bunlar tamamen farklı sistemlerdir.

Burada yüzdelikler ortalamalardan daha anlamlıdır. Profil eşleştirme hatası biraz büyük bir hata değil, haritanın başka bir bölgesine sıçramadır; bu nedenle ortalama konum hatası iki farklı popülasyonu karıştırır. Ortalamanın yanında medyan ve P95 raporlayın.

## Tekrarlanabilir deney protokolü

Akademik karşılaştırmalarda aşağıdaki bilgiler sonuçlarla birlikte kaydedilmelidir:

1. Git commit kimliği ve Python sürümü.
2. DEM kaynağı, koordinat referans sistemi, hücre boyutu, kapsamı ve dosya özeti (örneğin SHA-256).
3. `config.json` ile tüm sensör, rota ve algoritma parametreleri.
4. Rastgelelik tohumu (`TerrainConfig.seed`; varsayılan `42`).
5. Çalıştırma komutu, işletim sistemi, CPU modeli ve `--parallel-workers` değeri.
6. Başarı eşiği, değerlendirilen rota ve güncelleme sayısı ile reddedilen çözüm sayısı.
7. Ortalama yanında medyan ve P95 hata, yanlış `FIX` oranı ve çalışma süresi.

Önerilen asgari kontrol:

```powershell
git rev-parse HEAD
python --version
python run_terrain_nav.py --headless --fast --parallel-workers 1
python -m pytest
```

Buradaki `--parallel-workers 1` bilinçlidir: makineye bağlı varsayılanı devre dışı bırakır; böylece bu bloğu çalıştıran iki kişi bayt düzeyinde karşılaştırılabilir sonuç alır.

Sentetik ve harici DEM sonuçları ayrı tablolar halinde raporlanmalıdır; çünkü farklı sorulara yanıt verirler — sentetik durum algoritmayı bilinen ve denetimli arazi istatistikleri altında ölçer, harici durum ise özellikleri başka bir bölgeye aktarılamayacak belirli bir gerçek coğrafyada ölçer. Kısa smoke testleri yalnızca yazılım akışını doğrular; bilimsel performans kanıtı sayılmaz.

## Proje yapısı

```text
run_terrain_nav.py           CLI ve masaüstü uygulamasının giriş noktası
terrain_nav/
├── config.py                Deney, sensör ve algoritma konfigürasyonları
├── coordinates.py           Dünya/raster dönüşümleri ve yön matematiği
├── synthetic.py             Deterministik sentetik arazi üretimi
├── terrain.py               Sentetik ve GeoTIFF DEM yönetimi
├── sensors.py               Sensör benzetimi
├── profile.py               Rota ve arazi profili çıkarımı
├── matcher.py               Kabadan inceye profil eşleştirme
├── confidence.py            Konum ve hız belirsizliği değerlendirmesi
├── metrics.py               Kestirilen durum ve karşılaştırma kayıtları
├── simulation.py            Simülasyon ve lokalizasyon yaşam döngüsü
├── benchmark.py             Profil varyantı benchmark altyapısı
├── optimizer.py             Deterministik parametre optimizasyonu
├── logging_io.py            JSON ve CSV kayıtları
├── rendering.py             Harita ve profil çizimleri
└── ui.py                    PySide6 masaüstü arayüzü
tests/                       Aktif paket için regresyon testleri
results/                     Çalışma zamanı çıktıları
```

Kodu değiştirmeden önce anlaşılması gereken iki sınır vardır:

- **Gerçeklik ile çalışma konfigürasyonu.** `simulation.py` hem ground-truth rotayı hem de lokalizasyon katmanını barındırır; ancak lokalizasyona yalnızca, yapısı gereği rota verisi içermeyen bir `LocalizationRuntimeConfig` aktarır. Bu arayüzü genişleten her değişiklik sessiz bir gerçeklik sızıntısı riski taşır — optimizatör bu nedenle bir `ground_truth_leak_detected` bayrağı bile kaydeder.
- **Koordinat sözleşmeleri.** `coordinates.py`, dünya koordinatlarının (x = Doğu, y = Kuzey) raster koordinatlarıyla (sütun = Doğu, satır = **Güney**) buluştuğu tek yerdir. Satır ekseni Kuzey'e göre terstir ve yön, Kuzey'den saat yönünde ölçülür. Buradaki işaret hataları hâlâ makul görünen aynalanmış yörüngeler üretir; bu nedenle sözleşmelerin kendi test modülü vardır.

## Doğrulama

Kod değişikliklerinden sonra önerilen doğrulama yüzeyi:

```powershell
python -m pytest
python -m ruff check .
python -m compileall -q run_terrain_nav.py terrain_nav tests
python run_terrain_nav.py --help
python run_terrain_nav.py --headless --fast
```

Test paketi sayısal çekirdeği, arayüzü ve araştırma akışlarını kapsar:

| Alan | Test modülü |
|---|---|
| Koordinat ve yön sözleşmeleri | `test_coordinate_conventions.py` |
| Arazi üretimi ve harici DEM | `test_terrain_generation.py`, `test_external_dem.py` |
| Sensör modelleri | `test_sensor_model.py` |
| Profil çıkarımı | `test_profile_extraction.py` |
| Bilinen / bilinmeyen irtifa eşleştirme | `test_known_altitude_matching.py`, `test_unknown_altitude_matching.py` |
| Bilinmeyen hızla lokalizasyon | `test_unknown_speed_localization.py` |
| Yön araması | `test_heading_search.py` |
| Belirsizlik saptama | `test_ambiguity_detection.py` |
| ROI kurtarma ve seri–paralel eşdeğerliği | `test_search_optimization.py` |
| Uçtan uca lokalizasyon | `test_end_to_end_localization.py` |
| CLI ve manuel denetim | `test_headless_cli.py`, `test_manual_control.py` |
| Arayüz konfigürasyonu | `test_ui.py` |
| Benchmark ve optimizasyon | `test_benchmark.py`, `test_optimizer_benchmark.py` |

## Sorun giderme

| Belirti | Olası neden | Ne yapmalı |
|---|---|---|
| Her güncelleme `KALİTE YETERSİZ` veriyor | Kalite eşikleri gürültü düzeyi için fazla katı ya da DEM sensör datumuyla uyuşmuyor | `results.csv` içindeki `inlier_rmse_m` ve `correlation` değerlerini `max_match_inlier_rmse_m` ve `min_match_inlier_correlation` ile karşılaştırın; DEM'in düşey datumunun varsayılan MSL ile uyuştuğunu doğrulayın |
| Her güncelleme `BELİRSİZ` veriyor | Arazinin mevcut pencerede ayırt edici bir imzası yok | Pencereyi uzatın (`profile_window_size`, `min_profile_distance_m`) veya daha çok rölyef kesen bir rota kullanın; `plane` ön ayarında bu zaten doğru sonuçtur |
| `EŞLEŞME YOK` hiç geçmiyor | Pencere asgari kapıları hiç sağlamıyor | `min_profile_length`, `min_profile_distance_m` ve bilinmeyen hız modunda `min_profile_duration_s` değerlerini rota uzunluğu ve örnekleme aralığıyla karşılaştırın |
| Konum kendinden emin biçimde yanlış yere kilitleniyor | Harita boyutuna göre profil çok kısa veya kapılar fazla gevşek | `min_profile_distance_m` değerini yükseltin, `max_match_inlier_rmse_m` değerini sıkın ve ortalama hata yerine yanlış `FIX` oranına bakın |
| Harici DEM nodata hatasıyla reddediliyor | Yeniden örneklenen pencerede çok fazla nodata hücresi var | DEM'i kırpın veya doldurun, ya da geçerli kapsama sahip bir bölge seçin |
| Koşular çok yavaş | Büyük haritada global arama veya geniş hız aralıklı bilinmeyen hız modu | `--parallel-workers` değerini artırın, `--search-roi-size` etkinleştirin veya `--speed-search-min` / `--speed-search-max` aralığını daraltın |
| Kestirilen hız kararlı ama yanlış | Yerel olarak kendine benzeyen bir profile yanlış bir hız da oturabilir | `speed_confidence` ve `speed_spread_m_s` değerlerine bakın; hız arama aralığını gerçek uçuş zarfına daraltın |
| Masaüstü arayüzü açılmıyor | Grafik oturumu yok veya PySide6 kurulu değil | `--headless` kullanın ya da `PySide6` kurulumunu doğrulayın |

## Varsayımlar ve sınırlılıklar

- Sistem bir simülatördür; gerçek uçuş donanımı, zaman senkronizasyonu ve aviyonik emniyet gereksinimleri modellenmez.
- Profil eşleştirme, DEM doğruluğu ve çözünürlüğü ile sensör/harita datumlarının tutarlılığına duyarlıdır. Düşey datum uyuşmazlığı sabit bir bias olarak görünür; bilinmeyen irtifa ve barometrik modeller bunu sessizce soğurur — görünen uyumu iyileştirir ama size bu konuda hiçbir şey söylemez.
- Düz veya tekrarlayan topoğrafya, konum ve hızın birlikte gözlemlenebilirliğini azaltabilir; bu durumda `AMBIGUOUS` ya da `QUALITY INSUFFICIENT` beklenen bir sonuçtur.
- `unknown_constant_speed`, kayan profil penceresi boyunca sabit hız varsayar; ivmeli uçuşlar için model genişletilmelidir.
- Lokalizasyonda hareket filtresi yoktur. Ardışık kestirimler ROI ankrajı ve süreklilik kapısı dışında bağımsızdır; dolayısıyla raporlanan hatalar ham eşleştirici hatalarıdır ve bu projenin bilinçli olarak dışarıda bıraktığı bir filtreleme katmanıyla iyileşirdi.
- DEM'in uzun kenarı varsayılan olarak `2048` hücreye örneklenir. Bu işlem fiziksel kapsamı korur ancak yüksek frekanslı topoğrafik ayrıntıyı — yani bir profili ayırt edici kılan ayrıntıyı — azaltabilir. Büyük bir haritada `dem_target_size` bu nedenle yalnızca bir performans değil, bir doğruluk parametresidir.
- ROI ayarı ve paralel işçi sayısı bilimsel parametrelerle birlikte kaydedilmeli; farklı donanım ve DEM boyutlarında süre sonuçları yeniden ölçülmelidir.
- Varsayılan eşikler tüm coğrafyalara genellenmiştir ve saha kalibrasyonlu değerler olarak değerlendirilmemelidir.

## Atıf, veri ve lisans

Bu depoda henüz `CITATION.cff`, DOI veya yazarlar tarafından onaylanmış bir kaynakça kaydı bulunmamaktadır. Akademik kullanımda en azından depo adı, kullanılan sürümün commit kimliği ve erişim tarihi belirtilmelidir. Resmî atıf bilgisi eklendiğinde bu bölüm güncellenmelidir.

Harici DEM dosyaları depoya dahil değildir. Kullanılan veri kümesinin lisansı, üreticisi, tarihçesi, koordinat referans sistemi ve ön işleme adımları ilgili yayında ayrıca belirtilmelidir.

Bu proje **MIT Lisansı** ve **Apache Lisansı 2.0** altında çift lisanslanmıştır. Bu kodu kullanırken, değiştirirken veya dağıtırken iki lisanstan birini seçebilirsiniz. Tüm lisans koşulları için [LICENSE](LICENSE) dosyasına bakınız.

<a id="english"></a>

# TERCOM Terrain-Contour-Matching Localization Simulator for GNSS-Denied Navigation

[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.3.0-0A7BBB)](pyproject.toml)
[![UI](https://img.shields.io/badge/UI-PySide6-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython/)
[![License](https://img.shields.io/badge/license-not%20specified-lightgrey)](#citation-data-and-license)

**English** · [Türkçe](#turkce)

This project is an experimental **Terrain Contour Matching (TERCOM)** simulator that estimates the position of an aircraft in GNSS-denied environments from a Digital Elevation Model (DEM), a laser altimeter, barometric altitude, and motion data. The software combines controlled synthetic experiments, GeoTIFF-based real-terrain studies, uncertainty analysis with quality gates, and reproducible parameter optimization into a single research infrastructure.

> **Research software note:** This repository is not a flight-critical navigation system. The positions and performance metrics it produces should be evaluated for simulation and research purposes only.

## Table of Contents

- [Research Purpose and Scope](#research-purpose-and-scope)
- [Methodology](#methodology)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiment Modes](#experiment-modes)
- [Command-Line Reference](#command-line-reference)
- [Outputs and Evaluation Metrics](#outputs-and-evaluation-metrics)
- [Reproducible Experiment Protocol](#reproducible-experiment-protocol)
- [Project Structure](#project-structure)
- [Validation](#validation)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Citation, Data, and License](#citation-data-and-license)

## Research Purpose and Scope

The main research question of the project is **to what extent a time-dependent terrain-elevation profile can be matched on a reference DEM reliably and at a feasible computational cost**. Within this scope, the following problems can be investigated:

- Profile matching under known or unknown absolute flight altitude
- The impact of ideal versus noisy sensor assumptions on localization success
- Joint estimation of position and constant speed when the traveled distance is unknown
- The accuracy–time trade-off between global search and local Region of Interest (ROI) tracking
- Detection of position and speed uncertainty in flat or repetitive topography
- The trade-off between quality thresholds and the false `FIX` rate
- Multiprocessing execution of coarse map searches, and the resulting runtime analysis

The study can run either on a small, deterministic synthetic DEM or on a user-provided geo-referenced GeoTIFF DEM. External data is not included in this repository.

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

For each measurement, the laser altitude and the candidate DEM elevation are converted into an expected terrain profile according to the selected altitude model. Candidate positions are scored with the Huber loss function by default; a lower score indicates a better fit. After matching, inlier RMSE, correlation, and the valid-sample ratio are checked. Candidates with similar scores but scattered in space are treated as uncertain, and a position solution is not forced.

### Altitude Models

| Mode | Information used by localization | Research purpose |
|---|---|---|
| `known_msl_altitude` | Constant and known MSL altitude | Ideal reference scenario |
| `unknown_constant_msl_altitude` | Constant but unknown MSL altitude along the profile | Matching without absolute altitude |
| `barometric_altitude` | Time-dependent barometer measurement with bias and noise | More realistic sensor scenario |

### Motion Models

| Mode | Motion information provided to localization |
|---|---|
| `known_distance` | Perfect traveled distance; default ideal mode |
| `measured_speed` | Distance derived from a noisy speed measurement |
| `unknown_constant_speed` | Distance and speed are not provided; position and constant speed are searched jointly |

In `unknown_constant_speed` mode, the relation `distance = speed × time` is established from the time difference for each speed hypothesis. Since each sample uses its own heading information, the geometry of turning, L-shaped, and zigzag routes is preserved. The default speed search range is `5–30 m/s`; the coarse, medium, and fine steps are `5`, `1`, and `0.2 m/s` respectively.

### Localization States

| State | Desktop UI label | Meaning |
|---|---|---|
| `FIX` | `GÜVENLİ (FIX)` | Accepted solution that passed every quality gate |
| `AMBIGUOUS` | `BELİRSİZ (AMBIG)` | Candidates score similarly but are spatially scattered |
| `AMBIGUOUS` (speed) | `HIZ BELİRSİZ (AMBIG)` | Position is resolved, but the speed hypothesis is not separable |
| `QUALITY INSUFFICIENT` | `KALİTE YETERSİZ` | Best candidate was rejected by the absolute quality gate |
| `RECOVERY` | `YENİDEN ARANIYOR` | Match was lost; the search area is being expanded |
| `NO MATCH` | `EŞLEŞME YOK` | Not enough profile data has accumulated yet |

## Key Features

- PySide6-based manual task and telemetry interface
- Small, deterministic synthetic terrain generation
- GeoTIFF DEM reading with extent-preserving resampling
- Laser, barometer, compass, and speed sensor error models
- Known heading, or coarse-to-fine heading search
- Huber / RMSE / MAE based profile matching
- Global search plus an optional, progressively expanding ROI recovery flow
- Absolute quality gates that prevent false local anchoring
- Joint estimation of a constant but unknown speed together with position
- A persistent multiprocessing worker pool for large coarse searches
- Deterministic parameter optimization with validation/final route separation
- Experiment logging in CSV, JSON, JSONL, and XLSX formats

## Installation

### Requirements

- Windows, Linux, or macOS
- Python `3.10–3.13`
- A graphical session for the desktop interface
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

When started without parameters, the program uses the local default DEM defined in the source code if it exists, and falls back to synthetic terrain otherwise. For academic reproducibility, providing the DEM path explicitly with `--dem` on every run is recommended.

### Manual Controls

| Key | Function |
|---|---|
| `W` / `S` | Move forward / backward |
| `A` / `D` | Move left / right (lateral) |
| `Q` / `E` | Turn left / right |

The default manual movement command is `100 m`, the turn command is `15°`, and the profile sampling interval is `20 m`. These values can be changed through `RouteConfig`.

## Experiment Modes

### Ideal Sensor Baseline

```powershell
python run_terrain_nav.py --headless --fast
```

This reference scenario uses known MSL altitude, known heading, and a perfect traveled distance. It serves as the control group for comparison against noisy scenarios.

### Realistic Sensor Noise

```powershell
python run_terrain_nav.py --realistic-noise --dem "C:\data\terrain.tif"
```

This preset replaces absolute altitude with a barometer that carries bias and noise, and uses a noisy speed measurement. In external-DEM runs a measured profile of at least `800 m` is expected, so that short profiles do not cause false global locking; the sliding profile is capped at roughly `2000 m` to bound accumulated odometry error. Compass heading is treated as known in this preset; heading noise can be configured separately through `SensorConfig.heading_mode`.

### Localization Without Speed Information

```powershell
python run_terrain_nav.py --headless --fast --unknown-speed

# Equivalent explicit form
python run_terrain_nav.py --motion-mode unknown_constant_speed

# Restricting the physical speed range
python run_terrain_nav.py --unknown-speed --speed-search-min 8 --speed-search-max 24
```

This mode assumes the speed is constant along the sliding profile. Although the simulator knows the true speed in order to move the vehicle, the true route start, speed, and traveled distance are never passed to the localization layer. The true speed is used only at the final stage, to compute the `speed_error_m_s` metric.

### Global Search and ROI

The default `--search-roi-size 0` setting disables the ROI and searches the entire map at every update. The ROI must be enabled explicitly:

```powershell
python run_terrain_nav.py --dem "C:\data\terrain.tif" --search-roi-size 512
```

After a reliable match, the local ROI is used. If the match is lost, the search area is expanded progressively; when the old anchor becomes invalid, the measurement profile is preserved and the search reverts to global. The ROI is not an accuracy method but a tracking optimization that reduces computational cost.

### Parallel Coarse Search

```powershell
# Default: min(4, CPU count) worker processes
python run_terrain_nav.py --parallel-workers 4

# Serial execution
python run_terrain_nav.py --parallel-workers 1
```

Large global searches are split into row bands and executed in persistent worker processes. Small maps and ROI searches can stay serial to avoid inter-process communication overhead. The worker count should be reported together with the experiment environment and the DEM size.

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

Optimization covers candidate configuration generation, validation/final route separation, Pareto analysis, and safe/fast/accurate/balanced selections. A configuration selected in a small synthetic run should not be interpreted as a production setting unless it is additionally validated on an external DEM.

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

## Outputs and Evaluation Metrics

A headless run produces the following files under `results/`:

- `config.json` — the sensor, algorithm, terrain, and route configuration of the run
- `results.csv` — true and estimated state per step, together with quality metrics

An optimization run produces timestamped `*_summary.csv`, `*_details.jsonl`, and `*.xlsx` files. The XLSX workbook contains an overall summary, the best configurations, and the final evaluation tables.

Key metrics:

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

The "Match Score" shown in the interface is not the actual position error; it reports the profile fit error, where a lower value is better. "Search Dispersion" is derived from the spread of candidates in raster space; to interpret it in physical meters, it must be converted using the DEM pixel size.

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

Synthetic and external-DEM results should be reported in separate tables. Short smoke tests only validate the software flow; they do not count as scientific performance evidence.

## Project Structure

```text
run_terrain_nav.py           Entry point for the CLI and the desktop application
terrain_nav/
├── config.py                Experiment, sensor, and algorithm configurations
├── terrain.py               Synthetic and GeoTIFF DEM management
├── sensors.py               Sensor simulation
├── profile.py               Route and terrain profile extraction
├── matcher.py               Coarse-to-fine profile matching
├── confidence.py            Position and speed uncertainty evaluation
├── simulation.py            Simulation and localization lifecycle
├── benchmark.py             Profile-variant benchmark infrastructure
├── optimizer.py             Deterministic parameter optimization
├── logging_io.py            JSON and CSV logging
├── rendering.py             Map and profile renderings
└── ui.py                    PySide6 desktop interface
tests/                       Regression tests for the active package
results/                     Runtime outputs
```

The working configuration visible to localization is kept separate from the ground-truth route fields. This boundary is meant to prevent the algorithm from inadvertently using simulation truth.

## Validation

Recommended validation surface after code changes:

```powershell
python -m pytest
python -m ruff check .
python -m compileall -q run_terrain_nav.py terrain_nav tests
python run_terrain_nav.py --help
python run_terrain_nav.py --headless --fast
```

Tests cover coordinate transformations, sensor models, known and unknown altitude, unknown speed, heading search, ROI recovery, serial–parallel equivalence, external DEM handling, interface configuration, benchmark, and optimization flows.

## Assumptions and Limitations

- The system is a simulator; real flight hardware, time synchronization, and avionics safety requirements are not modeled.
- Profile matching is sensitive to DEM accuracy and resolution, and to the consistency of sensor and map datums.
- Flat or repetitive topography can reduce the joint observability of position and speed; in that case `AMBIGUOUS` or `QUALITY INSUFFICIENT` is an expected outcome.
- `unknown_constant_speed` assumes constant speed across the sliding profile window; the model must be extended for accelerated flight.
- The long edge of the DEM is sampled to `2048` cells by default. This preserves the physical extent but may reduce high-frequency topographic detail.
- The ROI setting and the parallel worker count should be recorded with the scientific parameters; timing results should be remeasured on different hardware and DEM sizes.
- Default thresholds are generalized across geographies and should not be treated as field-calibrated values.

## Citation, Data, and License

This repository does not yet contain a `CITATION.cff`, a DOI, or an author-approved bibliographic record. For academic use, at a minimum the repository name, the commit ID of the version used, and the access date should be stated. This section should be updated once official citation information is added.

External DEM files are not included in the repository. The license, producer, provenance, coordinate reference system, and preprocessing steps of the dataset used must be stated separately in the related publication.

Since the repository contains no explicit license file, it should not be assumed that reuse or redistribution of the code is automatically permitted. Adding a proper `LICENSE` file before publication is recommended.

---

<a id="turkce"></a>

# GNSS-Yoksun Seyrüsefer için TERCOM Arazi Profili Lokalizasyon Simülatörü

[![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Sürüm](https://img.shields.io/badge/s%C3%BCr%C3%BCm-0.3.0-0A7BBB)](pyproject.toml)
[![Arayüz](https://img.shields.io/badge/aray%C3%BCz-PySide6-41CD52?logo=qt&logoColor=white)](https://doc.qt.io/qtforpython/)
[![Lisans](https://img.shields.io/badge/lisans-belirtilmemi%C5%9F-lightgrey)](#atıf-veri-ve-lisans)

[English](#english) · **Türkçe**

Bu proje, GNSS erişiminin bulunmadığı koşullarda bir hava aracının konumunu Sayısal Yükseklik Modeli (DEM), lazer altimetre, barometrik irtifa ve hareket bilgilerinden kestirmeyi amaçlayan deneysel bir **Terrain Contour Matching (TERCOM)** simülatörüdür. Yazılım; kontrollü sentetik deneyleri, GeoTIFF tabanlı gerçek arazi çalışmalarını, belirsizlik analizi ile kalite kapılarını ve tekrarlanabilir parametre optimizasyonunu tek bir araştırma altyapısında birleştirir.

> **Araştırma yazılımı notu:** Bu depo bir uçuş-kritik seyrüsefer sistemi değildir. Üretilen konumlar ve performans ölçümleri yalnızca simülasyon ve araştırma amacıyla değerlendirilmelidir.

## İçindekiler

- [Araştırma amacı ve kapsam](#araştırma-amacı-ve-kapsam)
- [Yöntem](#yöntem)
- [Temel özellikler](#temel-özellikler)
- [Kurulum](#kurulum)
- [Hızlı başlangıç](#hızlı-başlangıç)
- [Deney modları](#deney-modları)
- [Komut satırı referansı](#komut-satırı-referansı)
- [Çıktılar ve değerlendirme ölçütleri](#çıktılar-ve-değerlendirme-ölçütleri)
- [Tekrarlanabilir deney protokolü](#tekrarlanabilir-deney-protokolü)
- [Proje yapısı](#proje-yapısı)
- [Doğrulama](#doğrulama)
- [Varsayımlar ve sınırlılıklar](#varsayımlar-ve-sınırlılıklar)
- [Atıf, veri ve lisans](#atıf-veri-ve-lisans)

## Araştırma amacı ve kapsam

Projenin temel araştırma sorusu, **zamana bağlı bir arazi-yükseklik profilinin referans DEM üzerinde ne ölçüde güvenilir ve hesaplama açısından uygulanabilir biçimde eşleştirilebileceğidir**. Bu kapsamda aşağıdaki problemler incelenebilir:

- Bilinen veya bilinmeyen mutlak uçuş irtifası altında profil eşleştirme
- İdeal ve gürültülü sensör varsayımlarının lokalizasyon başarısına etkisi
- Kat edilen mesafe bilinmediğinde konum ve sabit hızın birlikte kestirimi
- Global arama ile yerel ilgi bölgesi (ROI) takibi arasındaki doğruluk–süre dengesi
- Düz ya da tekrarlayan topoğrafyada konum ve hız belirsizliğinin saptanması
- Kalite eşikleri ile yanlış `FIX` oranı arasındaki ödünleşim
- Kaba harita aramasının çok işlemli yürütülmesi ve çalışma süresi analizi

Çalışma, hem küçük ve deterministik bir sentetik DEM hem de kullanıcı tarafından sağlanan coğrafi referanslı GeoTIFF DEM üzerinde çalışabilir. Harici veri bu depoya dahil değildir.

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

Her ölçüm için lazer irtifası ile aday DEM yüksekliği, seçilen irtifa modeline göre beklenen arazi profiline dönüştürülür. Aday konumlar varsayılan olarak Huber kayıp fonksiyonu ile puanlanır; düşük puan daha iyi uyumu belirtir. Eşleştirme sonrası inlier RMSE, korelasyon ve geçerli örnek oranı denetlenir. Birbirine yakın puanlı fakat uzamsal olarak dağınık adaylar belirsiz kabul edilir ve zorla konum çözümü üretilmez.

### İrtifa modelleri

| Mod | Lokalizasyonun kullandığı bilgi | Araştırma amacı |
|---|---|---|
| `known_msl_altitude` | Sabit ve bilinen MSL irtifası | İdeal referans senaryosu |
| `unknown_constant_msl_altitude` | Profil boyunca sabit fakat bilinmeyen MSL irtifası | Mutlak irtifa bilgisiz eşleştirme |
| `barometric_altitude` | Bias ve gürültü içerebilen zamana bağlı barometre ölçümü | Daha gerçekçi sensör senaryosu |

### Hareket bilgisi modelleri

| Mod | Lokalizasyona verilen hareket bilgisi |
|---|---|
| `known_distance` | Kusursuz kat edilen mesafe; varsayılan ideal mod |
| `measured_speed` | Gürültülü hız ölçümünden türetilen mesafe |
| `unknown_constant_speed` | Mesafe ve hız verilmez; konum ve sabit hız birlikte aranır |

`unknown_constant_speed` modunda her hız hipotezi için zaman farkından `mesafe = hız × zaman` ilişkisi kurulur. Her örneğin kendi yön bilgisi kullanıldığından dönüşlü, L ve zikzak rotaların geometrisi korunur. Varsayılan hız arama aralığı `5–30 m/s`; kaba, orta ve ince adımlar sırasıyla `5`, `1` ve `0.2 m/s`'dir.

### Lokalizasyon durumları

| Durum | Masaüstü arayüz etiketi | Anlamı |
|---|---|---|
| `FIX` | `GÜVENLİ (FIX)` | Tüm kalite kapılarını geçen, kabul edilmiş çözüm |
| `AMBIGUOUS` | `BELİRSİZ (AMBIG)` | Adaylar benzer puanlı fakat uzamsal olarak dağınık |
| `AMBIGUOUS` (hız) | `HIZ BELİRSİZ (AMBIG)` | Konum çözülür, ancak hız hipotezi ayrıştırılamaz |
| `QUALITY INSUFFICIENT` | `KALİTE YETERSİZ` | En iyi aday mutlak kalite kapısından reddedildi |
| `RECOVERY` | `YENİDEN ARANIYOR` | Eşleşme kayboldu; arama alanı genişletiliyor |
| `NO MATCH` | `EŞLEŞME YOK` | Henüz yeterli profil verisi birikmedi |

## Temel özellikler

- PySide6 tabanlı manuel görev ve telemetri arayüzü
- Küçük ve deterministik sentetik arazi üretimi
- GeoTIFF DEM okuma ve fiziksel kapsamı koruyan yeniden örnekleme
- Lazer, barometre, pusula ve hız sensörü hata modelleri
- Bilinen yön veya coarse-to-fine yön araması
- Huber / RMSE / MAE temelli profil eşleştirme
- Global arama ve isteğe bağlı, kademeli genişleyen ROI kurtarma akışı
- Yanlış yerel ankrajı önleyen mutlak kalite kapıları
- Sabit fakat bilinmeyen hızın konumla birlikte kestirimi
- Büyük kaba aramalar için kalıcı çok işlemli işçi havuzu
- Validasyon/final rota ayrımlı deterministik parametre optimizasyonu
- CSV, JSON, JSONL ve XLSX biçimlerinde deney kayıtları

## Kurulum

### Gereksinimler

- Windows, Linux veya macOS
- Python `3.10–3.13`
- Masaüstü arayüzü için grafik oturumu
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

Program parametresiz başlatıldığında, kaynak kodda tanımlı yerel varsayılan DEM mevcutsa onu kullanır; dosya bulunamazsa sentetik araziye geri döner. Akademik tekrarlanabilirlik için DEM yolunun her koşuda `--dem` ile açıkça verilmesi önerilir.

### Manuel denetimler

| Tuş | İşlev |
|---|---|
| `W` / `S` | İleri / geri hareket |
| `A` / `D` | Sola / sağa yanal hareket |
| `Q` / `E` | Sola / sağa dönüş |

Varsayılan manuel hareket komutu `100 m`, dönüş komutu `15°` ve profil örnekleme aralığı `20 m`'dir. Bu değerler `RouteConfig` üzerinden değiştirilebilir.

## Deney modları

### İdeal sensör modu

```powershell
python run_terrain_nav.py --headless --fast
```

Bu referans senaryosu bilinen MSL irtifası, bilinen yön ve kusursuz hareket mesafesi kullanır. Gürültülü senaryolarla karşılaştırma için kontrol grubu niteliğindedir.

### Gerçekçi sensör gürültüsü

```powershell
python run_terrain_nav.py --realistic-noise --dem "C:\veri\arazi.tif"
```

Bu ön ayar mutlak irtifa yerine bias/gürültü içeren barometreyi ve gürültülü hız ölçümünü kullanır. Harici DEM koşularında kısa profille yanlış global kilitlenmeyi azaltmak için en az `800 m` ölçülmüş profil beklenir; biriken odometri hatasını sınırlamak için kayan profil yaklaşık `2000 m` ile sınırlandırılır. Pusula yönü bu ön ayarda bilinen kabul edilir; yön gürültüsü ayrıca `SensorConfig.heading_mode` ile yapılandırılabilir.

### Hız bilgisi olmadan lokalizasyon

```powershell
python run_terrain_nav.py --headless --fast --unknown-speed

# Eşdeğer açık gösterim
python run_terrain_nav.py --motion-mode unknown_constant_speed

# Fiziksel hız aralığını sınırlandırma
python run_terrain_nav.py --unknown-speed --speed-search-min 8 --speed-search-max 24
```

Bu mod kayan profil boyunca hızın sabit olduğunu varsayar. Simülatör aracı hareket ettirmek için gerçek hızı bilse de lokalizasyon katmanına gerçek rota başlangıcı, hızı veya kat edilen mesafe aktarılmaz. Gerçek hız yalnızca sonuç aşamasında `speed_error_m_s` metriğini hesaplamak için kullanılır.

### Global arama ve ROI

Varsayılan `--search-roi-size 0` ayarı ROI'yi kapatır ve her güncellemede tüm haritayı arar. ROI yalnızca açıkça etkinleştirilmelidir:

```powershell
python run_terrain_nav.py --dem "C:\veri\arazi.tif" --search-roi-size 512
```

Güvenilir bir eşleşmeden sonra yerel ROI kullanılır. Eşleşme kaybolursa arama alanı kademeli genişletilir; eski ankraj geçersiz olduğunda ölçüm profili korunarak global aramaya dönülür. ROI bir doğruluk yöntemi değil, hesaplama maliyetini azaltmayı amaçlayan izleme optimizasyonudur.

### Paralel kaba arama

```powershell
# Varsayılan: min(4, CPU çekirdek sayısı) işçi süreç
python run_terrain_nav.py --parallel-workers 4

# Seri yürütme
python run_terrain_nav.py --parallel-workers 1
```

Büyük global aramalar satır bantlarına ayrılarak kalıcı işçi süreçlerinde yürütülür. Küçük harita ve ROI aramaları süreçler arası iletişim maliyetinden kaçınmak için seri kalabilir. İşçi sayısı deney ortamı ve DEM boyutuyla birlikte raporlanmalıdır.

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

Optimizasyon; aday konfigürasyon üretimi, validasyon/final rota ayrımı, Pareto analizi ve güvenli/hızlı/doğru/dengeli seçimleri içerir. Küçük sentetik koşuda seçilen bir konfigürasyon, harici DEM üzerinde ayrıca doğrulanmadan üretim ayarı olarak yorumlanmamalıdır.

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

## Çıktılar ve değerlendirme ölçütleri

Arayüzsüz koşu, `results/` altında aşağıdaki dosyaları üretir:

- `config.json` — koşunun sensör, algoritma, arazi ve rota konfigürasyonu
- `results.csv` — adım bazında gerçek ve kestirilen durum ile kalite ölçütleri

Optimizasyon çalışması zaman damgalı `*_summary.csv`, `*_details.jsonl` ve `*.xlsx` dosyaları üretir. XLSX çalışma kitabı genel özet, en iyi konfigürasyonlar ve final değerlendirme tablolarını içerir.

Başlıca ölçütler:

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

Arayüzdeki "Eşleşme Skoru" gerçek konum hatası değildir; profil uyum hatasını gösterir ve düşük değer daha iyidir. "Arama Dağılımı" adayların raster uzayındaki yayılımından türetilir; fiziksel metre olarak yorumlanacaksa DEM piksel boyutuyla dönüştürülmelidir.

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

Sentetik ve harici DEM sonuçları ayrı tablolar halinde raporlanmalıdır. Kısa smoke testleri yalnızca yazılım akışını doğrular; bilimsel performans kanıtı sayılmaz.

## Proje yapısı

```text
run_terrain_nav.py           CLI ve masaüstü uygulamasının giriş noktası
terrain_nav/
├── config.py                Deney, sensör ve algoritma konfigürasyonları
├── terrain.py               Sentetik ve GeoTIFF DEM yönetimi
├── sensors.py               Sensör benzetimi
├── profile.py               Rota ve arazi profili çıkarımı
├── matcher.py               Coarse-to-fine profil eşleştirme
├── confidence.py            Konum ve hız belirsizliği değerlendirmesi
├── simulation.py            Simülasyon ve lokalizasyon yaşam döngüsü
├── benchmark.py             Profil varyantı benchmark altyapısı
├── optimizer.py             Deterministik parametre optimizasyonu
├── logging_io.py            JSON ve CSV kayıtları
├── rendering.py             Harita ve profil çizimleri
└── ui.py                    PySide6 masaüstü arayüzü
tests/                       Aktif paket için regresyon testleri
results/                     Çalışma zamanı çıktıları
```

Lokalizasyonun görebildiği çalışma konfigürasyonu, ground-truth rota alanlarından ayrılmıştır. Bu sınır, algoritmanın simülasyon gerçeğini yanlışlıkla kullanmasını önlemeyi amaçlar.

## Doğrulama

Kod değişikliklerinden sonra önerilen doğrulama yüzeyi:

```powershell
python -m pytest
python -m ruff check .
python -m compileall -q run_terrain_nav.py terrain_nav tests
python run_terrain_nav.py --help
python run_terrain_nav.py --headless --fast
```

Testler; koordinat dönüşümleri, sensör modelleri, bilinen ve bilinmeyen irtifa, bilinmeyen hız, yön araması, ROI kurtarma, seri–paralel eşdeğerliği, harici DEM, arayüz konfigürasyonu, benchmark ve optimizasyon akışlarını kapsar.

## Varsayımlar ve sınırlılıklar

- Sistem bir simülatördür; gerçek uçuş donanımı, zaman senkronizasyonu ve aviyonik emniyet gereksinimleri modellenmez.
- Profil eşleştirme, DEM doğruluğu ve çözünürlüğü ile sensör/harita datumlarının tutarlılığına duyarlıdır.
- Düz veya tekrarlayan topoğrafya, konum ve hızın birlikte gözlemlenebilirliğini azaltabilir; bu durumda `AMBIGUOUS` ya da `QUALITY INSUFFICIENT` beklenen bir sonuçtur.
- `unknown_constant_speed`, kayan profil penceresi boyunca sabit hız varsayar; ivmeli uçuşlar için model genişletilmelidir.
- DEM'in uzun kenarı varsayılan olarak `2048` hücreye örneklenir. Bu işlem fiziksel kapsamı korur ancak yüksek frekanslı topoğrafik ayrıntıyı azaltabilir.
- ROI ayarı ve paralel işçi sayısı bilimsel parametrelerle birlikte kaydedilmeli; farklı donanım ve DEM boyutlarında süre sonuçları yeniden ölçülmelidir.
- Varsayılan eşikler tüm coğrafyalara genellenmiştir ve saha kalibrasyonlu değerler olarak değerlendirilmemelidir.

## Atıf, veri ve lisans

Bu depoda henüz `CITATION.cff`, DOI veya yazarlar tarafından onaylanmış bir kaynakça kaydı bulunmamaktadır. Akademik kullanımda en azından depo adı, kullanılan sürümün commit kimliği ve erişim tarihi belirtilmelidir. Resmî atıf bilgisi eklendiğinde bu bölüm güncellenmelidir.

Harici DEM dosyaları depoya dahil değildir. Kullanılan veri kümesinin lisansı, üreticisi, tarihçesi, koordinat referans sistemi ve ön işleme adımları ilgili yayında ayrıca belirtilmelidir.

Depoda açık bir lisans dosyası bulunmadığından, kodun yeniden kullanımı veya dağıtımı için otomatik olarak izin verildiği varsayılmamalıdır. Yayınlamadan önce uygun bir `LICENSE` dosyasının proje sahipleri tarafından eklenmesi önerilir.

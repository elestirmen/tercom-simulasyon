# TERCOM Terrain Contour Matching Localization Simulator for GNSS-Denied Navigation

[🇹🇷 Türkçe versiyonu için tıklayınız](#türkçe)

This project is an experimental **Terrain Contour Matching (TERCOM)** simulator aimed at estimating the position of an aircraft in environments without GNSS access using a Digital Elevation Model (DEM), laser altimeter, barometric altitude, and motion data. The software combines controlled synthetic experiments, GeoTIFF-based real terrain studies, uncertainty analysis with quality gates, and reproducible parameter optimization into a single research infrastructure.

> **Research software note:** This repository is not a flight-critical navigation system. The generated positions and performance metrics should only be evaluated for simulation and research purposes.

## Table of Contents

- [Research Purpose and Scope](#research-purpose-and-scope)
- [Methodology](#methodology)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Experiment Modes](#experiment-modes)
- [Outputs and Evaluation Metrics](#outputs-and-evaluation-metrics)
- [Reproducible Experiment Protocol](#reproducible-experiment-protocol)
- [Project Structure](#project-structure)
- [Validation](#validation)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [Citation, Data, and License](#citation-data-and-license)

## Research Purpose and Scope

The main research question of the project is **to what extent a time-dependent terrain-elevation profile can be reliably and computationally feasibly matched on a reference DEM**. In this context, the following problems can be investigated:

- Profile matching under known or unknown absolute flight altitude,
- The impact of ideal and noisy sensor assumptions on localization success,
- Joint estimation of position and constant speed when the traveled distance is unknown,
- The accuracy-time trade-off between global search and local Region of Interest (ROI) tracking,
- Detection of position/speed uncertainty in flat or repetitive topography,
- The trade-off between quality thresholds and false `FIX` rate,
- Multiprocessing execution of coarse map searches and runtime analysis.

The study can run on both a small and deterministic synthetic DEM or a user-provided geo-referenced GeoTIFF DEM. External data is not included in this repository.

## Methodology

### Localization Flow

```text
Real/Synthetic DEM
        │
        ├──► flight and sensor simulation ──► laser/barometer/motion measurements
        │                                      │
        │                                      ▼
        └──────────────────────────────► sliding terrain profile
                                               │
                                               ▼
                                      coarse → medium → fine search
                                               │
                                               ▼
                                  quality and uncertainty check
                                               │
                         ┌─────────────────────┼─────────────────────┐
                         ▼                     ▼                     ▼
                       FIX                 AMBIGUOUS       QUALITY INSUFFICIENT
```

For each measurement, the laser altitude and candidate DEM elevation are converted into an expected terrain profile according to the selected altitude model. Candidate positions are scored by default with the Huber loss function; a lower score indicates a better fit. After matching, inlier RMSE, correlation, and valid sample ratio are checked. Candidates with close scores but spatially scattered are considered uncertain, and a position solution is not forced.

### Altitude Models

| Mode | Information used by localization | Research purpose |
|---|---|---|
| `known_msl_altitude` | Constant and known MSL altitude | Ideal reference scenario |
| `unknown_constant_msl_altitude` | Constant but unknown MSL altitude along the profile | Matching without absolute altitude |
| `barometric_altitude` | Time-dependent barometer measurement with bias and noise | More realistic sensor scenario |

### Motion Information Models

| Mode | Motion information provided to localization |
|---|---|
| `known_distance` | Perfect traveled distance; default ideal mode |
| `measured_speed` | Distance derived from noisy speed measurement |
| `unknown_constant_speed` | Distance/speed not provided; position and constant speed are searched jointly |

In `unknown_constant_speed` mode, the relationship `distance = speed × time` is established from the time difference for each speed hypothesis. Since each sample uses its own heading information, the geometry of turning, L-shaped, and zigzag routes is preserved. The default speed search range is `5–30 m/s`; coarse, medium, and fine steps are `5`, `1`, and `0.2 m/s`, respectively.

## Key Features

- PySide6-based manual task and telemetry interface,
- Small and deterministic synthetic terrain generation,
- GeoTIFF DEM reading and physical extent-preserving resampling,
- Laser, barometer, compass, and speed sensor error models,
- Known heading or coarse-to-fine heading search,
- Huber/RMSE/MAE based profile matching,
- Global search and optional, progressively expanding ROI recovery flow,
- Absolute quality gates preventing false local anchoring,
- Joint estimation of constant but unknown speed with position,
- Persistent multiprocessing worker pool for large coarse searches,
- Deterministic parameter optimization with validation/final route separation,
- Experiment logging in CSV, JSON, JSONL, and XLSX formats.

## Installation

### Requirements

- Windows, Linux, or macOS,
- Python `3.10–3.13`,
- Graphical session for desktop interface,
- A DEM in GeoTIFF format for external terrain experiments.

Recommended installation with PowerShell:

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

Installation will install `matplotlib`, `numpy`, `PySide6`, and `rasterio` packages. The development option also includes `pytest` and `ruff`.

## Quick Start

The entry point of the active application is the `run_terrain_nav.py` file.

```powershell
# Fast desktop experiment with small synthetic DEM
python run_terrain_nav.py --fast

# Headless, reproducible synthetic control
python run_terrain_nav.py --headless --fast

# User-provided GeoTIFF DEM
python run_terrain_nav.py --dem "C:\data\terrain.tif"

# All options
python run_terrain_nav.py --help
```

When started without parameters, the program uses the local default DEM defined in the source code if available; if not found, it falls back to the synthetic terrain. For academic reproducibility, it is recommended to explicitly provide the DEM path with `--dem` on every run.

Manual controls in the desktop interface:

| Key | Function |
|---|---|
| `W` / `S` | Move forward / backward |
| `A` / `D` | Move left / right (lateral) |
| `Q` / `E` | Turn left / right |

The default manual movement command is `100 m`, turn command is `15°`, and profile sampling interval is `20 m`. These values can be modified via `RouteConfig`.

## Experiment Modes

### Ideal Sensor Mode

```powershell
python run_terrain_nav.py --headless --fast
```

This reference scenario uses known MSL altitude, heading, and perfect traveled distance. It serves as a control group for comparison with noisy scenarios.

### Realistic Sensor Noise

```powershell
python run_terrain_nav.py --realistic-noise --dem "C:\data\terrain.tif"
```

This preset uses a barometer containing bias/noise instead of absolute altitude and a noisy speed measurement. In external DEM runs, at least an `800 m` measured profile is expected to reduce false global locking with short profiles; the sliding profile is limited to approximately `2000 m` to bound accumulated odometry error. Compass heading is considered known in this preset; heading noise can be further configured with `SensorConfig.heading_mode`.

### Localization Without Speed Information

```powershell
python run_terrain_nav.py --headless --fast --unknown-speed

# Equivalent explicit notation
python run_terrain_nav.py --motion-mode unknown_constant_speed

# Restricting the physical speed range
python run_terrain_nav.py --unknown-speed --speed-search-min 8 --speed-search-max 24
```

This mode assumes the speed is constant along the sliding profile. Although the simulator knows the true speed to move the vehicle, the true route start, speed, or traveled distance are not passed to the localization layer. The true speed is only used at the final stage to calculate the `speed_error_m_s` metric.

### Global Search and ROI

The default `--search-roi-size 0` setting disables ROI and searches the entire map at every update. ROI must be explicitly enabled:

```powershell
python run_terrain_nav.py --dem "C:\data\terrain.tif" --search-roi-size 512
```

After a reliable match, the local ROI is used. If the match is lost, the search area is progressively expanded; when the old anchor becomes invalid, the measurement profile is preserved and it reverts to global search. ROI is not an accuracy method but a tracking optimization aimed at reducing computational cost.

### Parallel Coarse Search

```powershell
# Default upper limit: maximum 4 worker processes
python run_terrain_nav.py --parallel-workers 4

# Serial execution
python run_terrain_nav.py --parallel-workers 1
```

Large global searches are divided into row bands and executed in persistent worker processes. Small map and ROI searches can remain serial to avoid inter-process communication overhead. The number of workers should be reported along with the experiment environment and DEM size.

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

Optimization involves candidate configuration generation, validation/final route separation, Pareto analysis, and safe/fast/accurate/balanced selections. A configuration selected in a small synthetic run should not be interpreted as a production setting without being additionally validated on an external DEM.

## Outputs and Evaluation Metrics

The headless run generates the following files under `results/`:

- `config.json`: sensor, algorithm, terrain, and route configuration of the run,
- `results.csv`: true and estimated state along with quality metrics per step.

The optimization run produces timestamped `*_summary.csv`, `*_details.jsonl`, and `*.xlsx` files. The XLSX workbook contains an overall summary, best configurations, and final evaluation tables.

Key metrics:

| Metric | Comment |
|---|---|
| Position error (m) | Euclidean distance between true and estimated position; lower is better |
| Inlier RMSE (m) | Profile fit error after clipping outlier samples; lower is better |
| Correlation | Expected and measured profile shape match; higher is better |
| Valid sample ratio | Proportion of the profile that can be evaluated within DEM bounds; higher is better |
| Correct FIX rate | Ratio of accepted solutions that remain within the error threshold to all updates |
| False FIX rate | Ratio of accepted solutions that exceed the error threshold to all updates |
| FIX precision | Ratio of true `FIX` count to all accepted `FIX` count |
| P95 position error | 95th percentile value of position error in accepted solutions |
| Speed MAE (m/s) | Mean absolute speed error in unknown speed experiments |
| Runtime (ms) | Computational cost of global initial solution and tracking updates |

The "Match Score" in the interface is not the actual position error; it indicates the profile fit error, and a lower value is better. "Search Dispersion" is derived from the spread of candidates in raster space; if it is to be interpreted in physical meters, it must be converted with the DEM pixel size.

## Reproducible Experiment Protocol

For academic comparisons, the following information should be recorded alongside the results:

1. Git commit ID and Python version.
2. DEM source, coordinate reference system, cell size, extent, and file hash (e.g., SHA-256).
3. All sensor, route, and algorithm parameters via `config.json`.
4. Randomness seed (`TerrainConfig.seed`; default `42`).
5. Run command, OS, CPU model, and `--parallel-workers` value.
6. Success threshold, number of routes/updates evaluated, and number of rejected solutions.
7. Median/P95 error, false `FIX` rate, and runtime alongside the mean.

Recommended minimum check:

```powershell
git rev-parse HEAD
python --version
python run_terrain_nav.py --headless --fast --parallel-workers 1
python -m pytest
```

Synthetic and external DEM results should be reported in separate tables. Short smoke tests only validate the software flow; they do not count as scientific performance evidence.

## Project Structure

```text
run_terrain_nav.py           Entry point for CLI and desktop application
terrain_nav/
├── config.py                Experiment, sensor, and algorithm configurations
├── terrain.py               Synthetic/GeoTIFF DEM management
├── sensors.py               Sensor simulation
├── profile.py               Route and terrain profile extraction
├── matcher.py               Coarse-to-fine profile matching
├── confidence.py            Position and speed uncertainty evaluation
├── simulation.py            Simulation and localization lifecycle
├── benchmark.py             Profile variant benchmark infrastructure
├── optimizer.py             Deterministic parameter optimization
├── logging_io.py            JSON and CSV logging
├── rendering.py             Map/profile renderings
└── ui.py                    PySide6 desktop interface
tests/                       Regression tests for the active package
results/                     Runtime outputs
```

The working configuration visible to the localization is separated from ground-truth route fields. This boundary aims to prevent the algorithm from inadvertently using simulation truth.

## Validation

Recommended validation surface after code changes:

```powershell
python -m pytest
python -m ruff check .
python -m compileall -q run_terrain_nav.py terrain_nav tests
python run_terrain_nav.py --help
python run_terrain_nav.py --headless --fast
```

Tests cover coordinate transformations, sensor models, known/unknown altitude, unknown speed, heading search, ROI recovery, serial–parallel equivalence, external DEM, interface configuration, benchmark, and optimization flows.

## Assumptions and Limitations

- The system is a simulator; real flight hardware, time synchronization, and avionics safety requirements are not modeled.
- Profile matching is sensitive to DEM accuracy and resolution, and consistency of sensor/map datums.
- Flat or repetitive topography can reduce the joint observability of position and speed; in this case, `AMBIGUOUS` or `QUALITY INSUFFICIENT` is an expected outcome.
- `unknown_constant_speed` assumes constant speed across the sliding profile window; the model must be extended for accelerated flights.
- The long edge of the DEM is sampled to `2048` cells by default. This preserves physical extent but may reduce high-frequency topographic detail.
- The ROI and parallel worker count should be recorded with scientific parameters; timing results should be remeasured on different hardware and DEM sizes.
- Default thresholds are generalized across geographies and should not be considered field-calibrated values.

## Citation, Data, and License

This repository does not yet contain a `CITATION.cff`, DOI, or author-approved bibliographic record. For academic use, at a minimum, the repository name, commit ID of the used version, and access date should be stated. This section should be updated once official citation information is added.

External DEM files are not included in the repository. The license, producer, history, coordinate reference system, and preprocessing steps of the dataset used must be additionally stated in the related publication.

Since there is no explicit license file in the repository, it should not be assumed that reuse or distribution of the code is automatically permitted. It is recommended that a proper `LICENSE` file be added by the project owners before publication.

---

<a id="türkçe"></a>
# GNSS-Yoksun Seyrüsefer için TERCOM Arazi Profili Lokalizasyon Simülatörü

*[Read this in English](#tercom-terrain-contour-matching-localization-simulator-for-gnss-denied-navigation)*


Bu proje, GNSS erişiminin bulunmadığı koşullarda bir hava aracının konumunu
Sayısal Yükseklik Modeli (DEM), lazer altimetre, barometrik irtifa ve hareket
bilgilerinden kestirmeyi amaçlayan deneysel bir **Terrain Contour Matching
(TERCOM)** simülatörüdür. Yazılım; kontrollü sentetik deneyleri, GeoTIFF tabanlı
gerçek arazi çalışmalarını, belirsizlik analizi ile kalite kapılarını ve
tekrarlanabilir parametre optimizasyonunu tek bir araştırma altyapısında birleştirir.

> **Araştırma yazılımı notu:** Bu depo bir uçuş-kritik seyrüsefer sistemi değildir.
> Üretilen konumlar ve performans ölçümleri yalnızca simülasyon ve araştırma
> amacıyla değerlendirilmelidir.

## İçindekiler

- [Araştırma amacı ve kapsam](#araştırma-amacı-ve-kapsam)
- [Yöntem](#yöntem)
- [Temel özellikler](#temel-özellikler)
- [Kurulum](#kurulum)
- [Hızlı başlangıç](#hızlı-başlangıç)
- [Deney modları](#deney-modları)
- [Çıktılar ve değerlendirme ölçütleri](#çıktılar-ve-değerlendirme-ölçütleri)
- [Tekrarlanabilir deney protokolü](#tekrarlanabilir-deney-protokolü)
- [Proje yapısı](#proje-yapısı)
- [Doğrulama](#doğrulama)
- [Varsayımlar ve sınırlılıklar](#varsayımlar-ve-sınırlılıklar)
- [Atıf, veri ve lisans](#atıf-veri-ve-lisans)

## Araştırma amacı ve kapsam

Projenin temel araştırma sorusu, **zamana bağlı bir arazi-yükseklik profilinin
referans DEM üzerinde ne ölçüde güvenilir ve hesaplama açısından uygulanabilir
biçimde eşleştirilebileceğidir**. Bu kapsamda aşağıdaki problemler incelenebilir:

- bilinen veya bilinmeyen mutlak uçuş irtifası altında profil eşleştirme,
- ideal ve gürültülü sensör varsayımlarının lokalizasyon başarısına etkisi,
- kat edilen mesafe bilinmediğinde konum ve sabit hızın birlikte kestirimi,
- global arama ile yerel ilgi bölgesi (ROI) takibi arasındaki doğruluk–süre dengesi,
- düz ya da tekrarlayan topoğrafyada konum/hız belirsizliğinin saptanması,
- kalite eşikleri ile yanlış `FIX` oranı arasındaki ödünleşim,
- kaba harita aramasının çok işlemli yürütülmesi ve çalışma süresi analizi.

Çalışma, hem küçük ve deterministik bir sentetik DEM hem de kullanıcı tarafından
sağlanan coğrafi referanslı GeoTIFF DEM üzerinde çalışabilir. Harici veri bu depoya
dahil değildir.

## Yöntem

### Lokalizasyon akışı

```text
Gerçek/sentetik DEM
        │
        ├──► uçuş ve sensör benzetimi ──► lazer/barometre/hareket ölçümleri
        │                                      │
        │                                      ▼
        └──────────────────────────────► kayan arazi profili
                                               │
                                               ▼
                                      kaba → orta → ince arama
                                               │
                                               ▼
                                  kalite ve belirsizlik denetimi
                                               │
                         ┌─────────────────────┼─────────────────────┐
                         ▼                     ▼                     ▼
                       FIX                 AMBIGUOUS       QUALITY INSUFFICIENT
```

Her ölçüm için lazer irtifası ile aday DEM yüksekliği, seçilen irtifa modeline
göre beklenen arazi profiline dönüştürülür. Aday konumlar varsayılan olarak Huber
kayıp fonksiyonu ile puanlanır; düşük puan daha iyi uyumu belirtir. Eşleştirme
sonrası inlier RMSE, korelasyon ve geçerli örnek oranı denetlenir. Birbirine yakın
puanlı fakat uzamsal olarak dağınık adaylar belirsiz kabul edilir ve zorla konum
çözümü üretilmez.

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
| `unknown_constant_speed` | Mesafe/hız verilmez; konum ve sabit hız birlikte aranır |

`unknown_constant_speed` modunda her hız hipotezi için zaman farkından
`mesafe = hız × zaman` ilişkisi kurulur. Her örneğin kendi yön bilgisi
kullanıldığından dönüşlü, L ve zikzak rotaların geometrisi korunur. Varsayılan hız
arama aralığı `5–30 m/s`; kaba, orta ve ince adımlar sırasıyla `5`, `1` ve
`0.2 m/s`'dir.

## Temel özellikler

- PySide6 tabanlı manuel görev ve telemetri arayüzü,
- küçük ve deterministik sentetik arazi üretimi,
- GeoTIFF DEM okuma ve fiziksel kapsamı koruyan yeniden örnekleme,
- lazer, barometre, pusula ve hız sensörü hata modelleri,
- bilinen yön veya coarse-to-fine yön araması,
- Huber/RMSE/MAE temelli profil eşleştirme,
- global arama ve isteğe bağlı, kademeli genişleyen ROI kurtarma akışı,
- yanlış yerel ankrajı önleyen mutlak kalite kapıları,
- sabit fakat bilinmeyen hızın konumla birlikte kestirimi,
- büyük kaba aramalar için kalıcı çok işlemli işçi havuzu,
- validasyon/final rota ayrımlı deterministik parametre optimizasyonu,
- CSV, JSON, JSONL ve XLSX biçimlerinde deney kayıtları.

## Kurulum

### Gereksinimler

- Windows, Linux veya macOS,
- Python `3.10–3.13`,
- masaüstü arayüzü için grafik oturumu,
- harici arazi deneyi için GeoTIFF biçiminde bir DEM.

PowerShell ile önerilen kurulum:

```powershell
git clone <depo-adresi>
Set-Location "tercom-simulasyon"

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Yalnızca çalışma zamanı bağımlılıkları gerekiyorsa:

```powershell
python -m pip install -e .
```

Kurulum `matplotlib`, `numpy`, `PySide6` ve `rasterio` paketlerini yükler.
Geliştirme seçeneği ayrıca `pytest` ve `ruff` içerir.

## Hızlı başlangıç

Aktif uygulamanın giriş noktası `run_terrain_nav.py` dosyasıdır.

```powershell
# Küçük sentetik DEM ile hızlı masaüstü deneyi
python run_terrain_nav.py --fast

# Arayüzsüz, tekrarlanabilir sentetik kontrol
python run_terrain_nav.py --headless --fast

# Kullanıcı tarafından sağlanan GeoTIFF DEM
python run_terrain_nav.py --dem "C:\veri\arazi.tif"

# Tüm seçenekler
python run_terrain_nav.py --help
```

Program parametresiz başlatıldığında, kaynak kodda tanımlı yerel varsayılan DEM
mevcutsa onu kullanır; dosya bulunamazsa sentetik araziye geri döner. Akademik
tekrarlanabilirlik için DEM yolunun her koşuda `--dem` ile açıkça verilmesi önerilir.

Masaüstü arayüzündeki manuel denetimler:

| Tuş | İşlev |
|---|---|
| `W` / `S` | İleri / geri hareket |
| `A` / `D` | Sola / sağa yanal hareket |
| `Q` / `E` | Sola / sağa dönüş |

Varsayılan manuel hareket komutu `100 m`, dönüş komutu `15°` ve profil örnekleme
aralığı `20 m`'dir. Bu değerler `RouteConfig` üzerinden değiştirilebilir.

## Deney modları

### İdeal sensör modu

```powershell
python run_terrain_nav.py --headless --fast
```

Bu referans senaryosu bilinen MSL irtifası, yön ve kusursuz hareket mesafesi
kullanır. Gürültülü senaryolarla karşılaştırma için kontrol grubu niteliğindedir.

### Gerçekçi sensör gürültüsü

```powershell
python run_terrain_nav.py --realistic-noise --dem "C:\veri\arazi.tif"
```

Bu ön ayar mutlak irtifa yerine bias/gürültü içeren barometreyi ve gürültülü hız
ölçümünü kullanır. Harici DEM koşularında kısa profille yanlış global kilitlenmeyi
azaltmak için en az `800 m` ölçülmüş profil beklenir; biriken odometri hatasını
sınırlamak için kayan profil yaklaşık `2000 m` ile sınırlandırılır. Pusula yönü bu
ön ayarda bilinen kabul edilir; yön gürültüsü ayrıca `SensorConfig.heading_mode`
ile yapılandırılabilir.

### Hız bilgisi olmadan lokalizasyon

```powershell
python run_terrain_nav.py --headless --fast --unknown-speed

# Eşdeğer açık gösterim
python run_terrain_nav.py --motion-mode unknown_constant_speed

# Fiziksel hız aralığını sınırlandırma
python run_terrain_nav.py --unknown-speed --speed-search-min 8 --speed-search-max 24
```

Bu mod kayan profil boyunca hızın sabit olduğunu varsayar. Simülatör aracı hareket
ettirmek için gerçek hızı bilse de lokalizasyon katmanına gerçek rota başlangıcı,
hızı veya kat edilen mesafe aktarılmaz. Gerçek hız yalnızca sonuç aşamasında
`speed_error_m_s` metriğini hesaplamak için kullanılır.

### Global arama ve ROI

Varsayılan `--search-roi-size 0` ayarı ROI'yi kapatır ve her güncellemede tüm
haritayı arar. ROI yalnızca açıkça etkinleştirilmelidir:

```powershell
python run_terrain_nav.py --dem "C:\veri\arazi.tif" --search-roi-size 512
```

Güvenilir bir eşleşmeden sonra yerel ROI kullanılır. Eşleşme kaybolursa arama alanı
kademeli genişletilir; eski ankraj geçersiz olduğunda ölçüm profili korunarak
global aramaya dönülür. ROI bir doğruluk yöntemi değil, hesaplama maliyetini
azaltmayı amaçlayan izleme optimizasyonudur.

### Paralel kaba arama

```powershell
# Varsayılan üst sınır: en fazla 4 işçi süreç
python run_terrain_nav.py --parallel-workers 4

# Seri yürütme
python run_terrain_nav.py --parallel-workers 1
```

Büyük global aramalar satır bantlarına ayrılarak kalıcı işçi süreçlerinde
yürütülür. Küçük harita ve ROI aramaları süreçler arası iletişim maliyetinden
kaçınmak için seri kalabilir. İşçi sayısı deney ortamı ve DEM boyutuyla birlikte
raporlanmalıdır.

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

Optimizasyon; aday konfigürasyon üretimi, validasyon/final rota ayrımı, Pareto
analizi ve güvenli/hızlı/doğru/dengeli seçimleri içerir. Küçük sentetik koşuda
seçilen bir konfigürasyon, harici DEM üzerinde ayrıca doğrulanmadan üretim ayarı
olarak yorumlanmamalıdır.

## Çıktılar ve değerlendirme ölçütleri

Arayüzsüz koşu, `results/` altında aşağıdaki dosyaları üretir:

- `config.json`: koşunun sensör, algoritma, arazi ve rota konfigürasyonu,
- `results.csv`: adım bazında gerçek ve kestirilen durum ile kalite ölçütleri.

Optimizasyon çalışması zaman damgalı `*_summary.csv`, `*_details.jsonl` ve
`*.xlsx` dosyaları üretir. XLSX çalışma kitabı genel özet, en iyi
konfigürasyonlar ve final değerlendirme tablolarını içerir.

Başlıca ölçütler:

| Ölçüt | Yorum |
|---|---|
| Konum hatası (m) | Gerçek ve kestirilen konum arasındaki Öklid uzaklığı; düşük iyidir |
| Inlier RMSE (m) | Aykırı örnekler kırpıldıktan sonraki profil uyum hatası; düşük iyidir |
| Korelasyon | Ölçülen ve beklenen profil biçimi uyumu; yüksek iyidir |
| Geçerli örnek oranı | DEM sınırları içinde değerlendirilebilen profil payı; yüksek iyidir |
| Correct FIX oranı | Kabul edilen ve hata eşiği içinde kalan çözümlerin tüm güncellemelere oranı |
| False FIX oranı | Kabul edildiği halde hata eşiğini aşan çözümlerin tüm güncellemelere oranı |
| FIX precision | Doğru `FIX` sayısının kabul edilen tüm `FIX` sayısına oranı |
| P95 konum hatası | Kabul edilen çözümlerde konum hatasının 95. yüzdelik değeri |
| Hız MAE (m/s) | Bilinmeyen hız deneylerinde mutlak hız hatası ortalaması |
| Çalışma süresi (ms) | Global ilk çözüm ve takip güncellemelerinin hesaplama maliyeti |

Arayüzdeki “Eşleşme Skoru” gerçek konum hatası değildir; profil uyum hatasını
gösterir ve düşük değer daha iyidir. “Arama Dağılımı” adayların raster uzayındaki
yayılımından türetilir; fiziksel metre olarak yorumlanacaksa DEM piksel boyutuyla
dönüştürülmelidir.

## Tekrarlanabilir deney protokolü

Akademik karşılaştırmalarda aşağıdaki bilgiler sonuçlarla birlikte kaydedilmelidir:

1. Git commit kimliği ve Python sürümü.
2. DEM kaynağı, koordinat referans sistemi, hücre boyutu, kapsamı ve dosya özeti
   (örneğin SHA-256).
3. `config.json` ile tüm sensör, rota ve algoritma parametreleri.
4. Rastgelelik tohumu (`TerrainConfig.seed`; varsayılan `42`).
5. Çalıştırma komutu, işletim sistemi, CPU modeli ve `--parallel-workers` değeri.
6. Başarı eşiği, değerlendirilen rota/güncelleme sayısı ve reddedilen çözüm sayısı.
7. Ortalama yanında medyan/P95 hata, yanlış `FIX` oranı ve çalışma süresi.

Önerilen asgari kontrol:

```powershell
git rev-parse HEAD
python --version
python run_terrain_nav.py --headless --fast --parallel-workers 1
python -m pytest
```

Sentetik ve harici DEM sonuçları ayrı tablolar halinde raporlanmalıdır. Kısa smoke
testleri yalnızca yazılım akışını doğrular; bilimsel performans kanıtı sayılmaz.

## Proje yapısı

```text
run_terrain_nav.py           CLI ve masaüstü uygulamasının giriş noktası
terrain_nav/
├── config.py                Deney, sensör ve algoritma konfigürasyonları
├── terrain.py               Sentetik/GeoTIFF DEM yönetimi
├── sensors.py               Sensör benzetimi
├── profile.py               Rota ve arazi profili çıkarımı
├── matcher.py               Coarse-to-fine profil eşleştirme
├── confidence.py            Konum ve hız belirsizliği değerlendirmesi
├── simulation.py            Simülasyon ve lokalizasyon yaşam döngüsü
├── benchmark.py             Profil varyantı benchmark altyapısı
├── optimizer.py             Deterministik parametre optimizasyonu
├── logging_io.py            JSON ve CSV kayıtları
├── rendering.py             Harita/profil çizimleri
└── ui.py                    PySide6 masaüstü arayüzü
tests/                       Aktif paket için regresyon testleri
results/                     Çalışma zamanı çıktıları
```

Lokalizasyonun görebildiği çalışma konfigürasyonu, ground-truth rota alanlarından
ayrılmıştır. Bu sınır, algoritmanın simülasyon gerçeğini yanlışlıkla kullanmasını
önlemeyi amaçlar.

## Doğrulama

Kod değişikliklerinden sonra önerilen doğrulama yüzeyi:

```powershell
python -m pytest
python -m ruff check .
python -m compileall -q run_terrain_nav.py terrain_nav tests
python run_terrain_nav.py --help
python run_terrain_nav.py --headless --fast
```

Testler; koordinat dönüşümleri, sensör modelleri, bilinen/bilinmeyen irtifa,
bilinmeyen hız, yön araması, ROI kurtarma, seri–paralel eşdeğerliği, harici DEM,
arayüz konfigürasyonu, benchmark ve optimizasyon akışlarını kapsar.

## Varsayımlar ve sınırlılıklar

- Sistem bir simülatördür; gerçek uçuş donanımı, zaman senkronizasyonu ve aviyonik
  emniyet gereksinimleri modellenmez.
- Profil eşleştirme, DEM doğruluğu ve çözünürlüğü ile sensör/harita datumlarının
  tutarlılığına duyarlıdır.
- Düz veya tekrarlayan topoğrafya, konum ve hızın birlikte gözlemlenebilirliğini
  azaltabilir; bu durumda `AMBIGUOUS` ya da `QUALITY INSUFFICIENT` beklenen bir
  sonuçtur.
- `unknown_constant_speed`, kayan profil penceresi boyunca sabit hız varsayar;
  ivmeli uçuşlar için model genişletilmelidir.
- DEM'in uzun kenarı varsayılan olarak `2048` hücreye örneklenir. Bu işlem fiziksel
  kapsamı korur ancak yüksek frekanslı topoğrafik ayrıntıyı azaltabilir.
- ROI ve paralel işçi sayısı bilimsel parametrelerle birlikte kaydedilmeli; farklı
  donanım ve DEM boyutlarında süre sonuçları yeniden ölçülmelidir.
- Varsayılan eşikler tüm coğrafyalara genellenmiş, saha kalibrasyonlu değerler
  olarak değerlendirilmemelidir.

## Atıf, veri ve lisans

Bu depoda henüz `CITATION.cff`, DOI veya yazarlar tarafından onaylanmış bir kaynakça
kaydı bulunmamaktadır. Akademik kullanımda en azından depo adı, kullanılan sürümün
commit kimliği ve erişim tarihi belirtilmelidir. Resmî atıf bilgisi eklendiğinde bu
bölüm güncellenmelidir.

Harici DEM dosyaları depoya dahil değildir. Kullanılan veri kümesinin lisansı,
üreticisi, tarihçesi, koordinat referans sistemi ve ön işleme adımları ilgili
yayında ayrıca belirtilmelidir.

Depoda açık bir lisans dosyası bulunmadığından, kodun yeniden kullanımı veya
dağıtımı için otomatik olarak izin verildiği varsayılmamalıdır. Yayınlamadan önce
uygun bir `LICENSE` dosyasının proje sahipleri tarafından eklenmesi önerilir.

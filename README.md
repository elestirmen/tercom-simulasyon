# GNSS-Yoksun Seyrüsefer için TERCOM Arazi Profili Lokalizasyon Simülatörü

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

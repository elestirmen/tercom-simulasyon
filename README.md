# TERCOM Terrain Navigation

Bu depo, GNSS olmadan çalışan DEM ve lazer-altimetre tabanlı terrain-profile
lokalizasyon simülatörünü içerir. Aktif uygulamanın tek giriş noktası
`run_terrain_nav.py` dosyasıdır.

## Kurulum

```powershell
python -m pip install -e ".[dev]"
```

## Kullanım

```powershell
# Masaüstü arayüzü ve varsayılan Karlık GeoTIFF'i
python run_terrain_nav.py

# Küçük sentetik DEM ile hızlı UI
python run_terrain_nav.py --fast

# Headless doğrulama
python run_terrain_nav.py --headless --fast

# Gerçekçi sensör gürültüsü modu
python run_terrain_nav.py --realistic-noise

# Başka bir GeoTIFF
python run_terrain_nav.py --dem "C:\path\terrain.tif"
```

Arayüz manuel çalışır: `W/S/A/D` 100 metre hareket, `Q/E` 15 derece dönüş
uygular. Manuel hareket sırasında sensör/lokalizasyon profili varsayılan olarak
20 metrede bir örneklenir; yani tek bir 100 metrelik komut profil grafiğine
yaklaşık 5 yeni kayıt ekler. Değerler `RouteConfig` üzerinden değiştirilebilir.
Arayüzdeki `Gerçekçi sensör gürültüsü` seçeneği, simülasyon başlamadan
`--realistic-noise` ile aynı sensör modunu açıp kapatır.

## Harita kapsamları

- Tam kaynak harita; İHA uçuşunun, sensör örneklemesinin ve lokalizasyonun ortak
  kapsamıdır. İHA yalnızca gerçek kaynak harita sınırında durdurulur.
- Varsayılan olarak profil her adımda bütün haritada aranır; yerel ROI yalnızca
  `--search-roi-size` sıfırdan büyük verildiğinde hız optimizasyonu olarak açılır.
- Turuncu ROI, bu seçenek açıkken güvenilir eşleşmeden sonra kullanılan yerel
  arama alanıdır.
- ROI açıkken eşleşme kaybolursa arama alanı kademeli büyür; eski ankraj geçersiz
  olduğunda sistem ölçüm profilini koruyarak yeniden bütün haritada arama yapar.
- Manuel dönüşlerde her ölçüm, kendisine ulaşan hareket vektörüyle profile
  eklenir; böylece L ve zikzak rotaların geometrisi korunur.
- Inlier RMSE, inlier korelasyonu veya geçerli örnek oranı yetersiz bir aday
  konum `FIX` olarak kabul edilmez ve yanlış bir yerel ROI ankrajı oluşturamaz.
  Kalite kapısı, profilin en kötü küçük yüzdesini dışarıda bırakarak tekil lazer
  outlier'larının uzun süreli `KALİTE YETERSİZ` üretmesini engeller.

Dış DEM'in tamamı, uzun kenarı varsayılan olarak 2048 hücre olacak şekilde
yeniden örneklenir. Böylece bellek kullanımı sınırlı kalırken haritanın hiçbir
bölümü lokalizasyon dışında bırakılmaz. `--dem-target-size` ve
`--search-roi-size` seçenekleri çözünürlük ile yerel ROI boyutunu kontrol eder.
`--search-roi-size 0` ve varsayılan ayar ROI'yi kapatır.

## Sensör modları

Varsayılan mod, mevcut regresyonları korumak için idealize edilmiştir:
`known_msl_altitude`, bilinen pusula yönü, lazer altimetre ve kusursuz hız/mesafe
bilgisiyle çalışır.

`--realistic-noise` modu, aynı GPS'siz eşleştirme akışını daha gerçekçi ölçüm
varsayımlarıyla çalıştırır: mutlak MSL irtifa doğrudan bilinmez, barometrede
sabit bilinmeyen bias ve gürültü vardır, lokalizasyona verilen hareket mesafesi
ise gürültülü hız ölçümünden türetilir. Bu mod, kısa profille yanlış global
eşleşmeye kilitlenmemek için dış DEM koşularında varsayılan olarak en az
800 metre ölçülmüş profil uzunluğu bekler ve hız gürültüsünün çok uzun profilde
birikmesini azaltmak için kayan profili yaklaşık 2 km ile sınırlar. Manuel UI'da
20 metrelik örnekleme aralığıyla bu, aramada ve profil grafiğinde son yaklaşık
100 ölçüm kaydının tutulması anlamına gelir. Pusula bu modda da bilinen yön
olarak kalır; heading gürültüsü ayrıca
`SensorConfig.heading_mode` üzerinden ayarlanabilir.

## Speed-free localization

Hareket bilgisi üç moddan biriyle seçilir:

- `known_distance`: lokalizasyon kusursuz kat edilen mesafeyi kullanır ve
  geriye uyumlu varsayılan moddur.
- `measured_speed`: sensör hızından türetilen, gürültülü mesafeyi kullanır.
- `unknown_constant_speed`: hız veya kat edilen mesafe almaz; barometre, lazer,
  pusula, ölçüm zamanları, DEM ve fiziksel hız sınırlarıyla çalışır.

Hızsız mod komut satırından iki eşdeğer biçimde açılabilir:

```powershell
python run_terrain_nav.py --unknown-speed
python run_terrain_nav.py --motion-mode unknown_constant_speed

# Tekrarlanabilir küçük sentetik/headless koşu
python run_terrain_nav.py --headless --fast --unknown-speed

# Fiziksel hız aralığını sınırla
python run_terrain_nav.py --unknown-speed --speed-search-min 8 --speed-search-max 24
```

Masaüstü arayüzündeki `Hareket bilgisi` menüsünden de `Hız bilinmiyor` seçilebilir.
Canlı telemetri, tahmini hızı ve hız güvenini gösterir.

Bu mod, kayan profil penceresi boyunca hızın sabit fakat bilinmeyen olduğunu
varsayar. Her aday hız için ardışık zaman farkı `dt` ile `v * dt` hareketi
hesaplanır; her ölçümün kendi pusula yönü kullanıldığı için L, zikzak ve dönüşlü
rotalar korunur. Konum ve hız hipotezi birlikte DEM üzerinde aranır. Hız gridinin
varsayılanları:

- aralık: `5.0 .. 30.0 m/s`
- kaba adım: `5.0 m/s`
- orta adım: `1.0 m/s`
- ince adım: `0.2 m/s`
- minimum profil süresi: `30 s`
- maksimum profil süresi: `120 s`

Simülatör uçağı hareket ettirmek için gerçek hızı bilir; ancak
`unknown_constant_speed` ölçümlerinde `traveled_distance_m` ve
`measured_speed_m_s` alanları boş bırakılır. Lokalizasyon çalışma konfigürasyonu
rota başlangıcını, rota hızını veya diğer ground-truth rota alanlarını içermez.
Gerçek hız yalnızca sonuç CSV'sindeki `speed_error_m_s` değerlendirme metriği
için logger katmanında kullanılır.

Sonuçlar; tahmini konum ve hızın yanında eşleşme skoru, inlier RMSE,
korelasyon, geçerli örnek oranı, konumsal belirsizlik, ikinci en iyi hız,
hız skor marjı, Top-K hız yayılımı ve hız güvenini içerir.

> Düz veya tekrarlayan topoğrafyada konum ve hız birlikte gözlemlenebilir
> olmayabilir. Bu durumda sistemin zorla `FIX` üretmesi yerine `AMBIGUOUS` veya
> `QUALITY INSUFFICIENT` sonucu vermesi beklenir. Çok kısa profiller de hız
> tahmini için yeterli arazi bilgisi taşımayabilir.

## Yapı

```text
run_terrain_nav.py       CLI ve UI giriş noktası
terrain_nav/             aktif simülasyon ve lokalizasyon paketi
tests/                   yalnızca aktif paketin regresyon testleri
```

`terrain_nav` paketi; konfigürasyon, koordinat dönüşümleri, DEM yönetimi,
sensör modeli, profil çıkarımı, coarse-to-fine eşleştirme, simülasyon, çizim ve
PySide6 arayüzünden oluşur. Sentetik test DEM'i paket içinde üretilir; başka bir
depoya çalışma zamanı bağımlılığı yoktur.

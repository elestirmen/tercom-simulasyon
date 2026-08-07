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
uygular. Değerler `RouteConfig` üzerinden değiştirilebilir.
Arayüzdeki `Gerçekçi sensör gürültüsü` seçeneği, simülasyon başlamadan
`--realistic-noise` ile aynı sensör modunu açıp kapatır.

## Harita kapsamları

- Tam kaynak harita; İHA uçuşunun, sensör örneklemesinin ve lokalizasyonun ortak
  kapsamıdır. İHA yalnızca gerçek kaynak harita sınırında durdurulur.
- İlk güvenilir eşleşme bulunana kadar profil bütün haritada aranır.
- Turuncu ROI, güvenilir eşleşmeden sonra kullanılan hızlı yerel arama alanıdır.
- Eşleşme kaybolursa ROI kademeli büyür; eski ankraj geçersiz olduğunda sistem
  ölçüm profilini koruyarak yeniden bütün haritada arama yapar.
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

## Sensör modları

Varsayılan mod, mevcut regresyonları korumak için idealize edilmiştir:
`known_msl_altitude`, bilinen pusula yönü, lazer altimetre ve kusursuz hız/mesafe
bilgisiyle çalışır.

`--realistic-noise` modu, aynı GPS'siz eşleştirme akışını daha gerçekçi ölçüm
varsayımlarıyla çalıştırır: mutlak MSL irtifa doğrudan bilinmez, barometrede
sabit bilinmeyen bias ve gürültü vardır, lokalizasyona verilen hareket mesafesi
ise gürültülü hız ölçümünden türetilir. Bu mod, kısa profille yanlış global
eşleşmeye kilitlenmemek için varsayılan olarak en az 400 metre ölçülmüş profil
uzunluğu bekler. Pusula bu modda da bilinen yön olarak kalır; heading gürültüsü
ayrıca `SensorConfig.heading_mode` üzerinden ayarlanabilir.

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

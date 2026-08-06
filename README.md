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

# Başka bir GeoTIFF
python run_terrain_nav.py --dem "C:\path\terrain.tif"
```

Arayüz manuel çalışır: `W/S/A/D` 100 metre hareket, `Q/E` 15 derece dönüş
uygular. Değerler `RouteConfig` üzerinden değiştirilebilir.

## Harita kapsamları

- Tam kaynak harita, İHA uçuşunun ve kaynak DEM sensör örneklemesinin sınırıdır.
- Kesikli çerçeve, bellekteki yüksek ayrıntılı lokalizasyon kapsamasıdır.
- Turuncu ROI, profil eşleştiricinin o adımda aradığı dinamik alt bölgedir.
- İHA lokalizasyon kapsamasının dışına uçabilir; bu durumda sensör ve uçuş sürer,
  eşleştirme kapsama dönene kadar bekler.

Varsayılan dış DEM penceresi 4096 kaynak pikselden okunup 2048 hücreye
yeniden örneklenir. `--dem-window-size`, `--dem-target-size`, `--dem-row`,
`--dem-col` ve `--search-roi-size` seçenekleri bu davranışı kontrol eder.

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

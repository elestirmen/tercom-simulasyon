# Terrain Navigation Simulation (GNSS-Denied)

Bu modül, DEM (Sayısal Yükseklik Modeli) ve Lazer Altimetre kullanarak GNSS'siz ortamlarda İHA lokalizasyonu simülasyonunu gerçekleştirir.

## Özellikler
- **Ground-Truth İzolasyonu:** Algoritma kesinlikle gerçek koordinatları ve temiz DEM verisini göremez. Yalnızca gürültülü `nav_dem`'e erişir.
- **Sensör Modelleri:** Barometre (drift/bias) ve Lazer Altimetre (outlier, drop_prob, bias, gürültü) tam modellenmiştir.
- **Rota Modları:** `straight_heading`, `heading_sequence`, `waypoint_route`.
- **Eşleştirme:** Coarse-to-fine ve Exhaustive arama. RMSE, MAE, Huber loss metrikleri.
- **Partikül Filtresi:** Yüksek belirsizlik durumları için opsiyonel parçacık filtresi `terrain_nav/particle_filter.py`.

## Kurulum ve Kullanım

```bash
# Headless (CLI) modu çalıştırmak için
python terrain_profile_localization_dashboard.py --headless

# Küçük DEM ile hızlı test
python terrain_profile_localization_dashboard.py --headless --fast

# Varsayılan Karlık DEM'i normal UI ile kullanmak için
python terrain_profile_localization_dashboard.py

# Başka bir dış DEM veya pencere konumu vermek için
python terrain_profile_localization_dashboard.py --dem "C:\\path\\terrain.tif" --dem-window-size 4096 --dem-target-size 2048

# PySide6 Arayüzünü Başlatmak için
python terrain_profile_localization_dashboard.py
```

## DEM performans ayarlari

`--dem-window-size` fiziksel kaynak pencereyi, `--dem-target-size` ise arama icin bellekte tutulacak raster boyutunu belirler.
Varsayilan `4096` kaynak pencere `2048` hedef hucreye yeniden orneklenir; fiziksel alan korunur. Daha hizli calisma icin
`--dem-target-size 1024`, daha yuksek ayrinti icin `--dem-target-size 4096` kullanilabilir.
Ilk guvenilir eslesmeden sonra arama varsayilan olarak `512 x 512` pencerede surdurulur;
global aramayi zorlamak icin `--search-roi-size 0` verilebilir.
Arama adaylari bellekte topluca tutulmaz; yalnizca en iyi `top_k` kayit saklanir. Dis DEM'de
ucus MSL degeri gerekiyorsa en yuksek navigasyon DEM hucresi + guvenli AGL seviyesine otomatik
yukseltilir. Duz rota, ayni raster penceresinde DEM disina cikmadan kalacak sekilde ortalanir.
Bilinen irtifa ve yon senaryosunda kaba skor haritasi NumPy ile toplu hesaplanir; bu yol gecici
RAM dizileri kullanarak Python tabanli hucre hucre taramayi azaltir.

## Mimari
- `config.py`: Konfigürasyon sınıfları (`dataclass`).
- `coordinates.py`: Açı ve raster-koordinat dönüşümleri.
- `terrain.py`: Sentetik DEM üretimi veya GeoTIFF'ten sınırlı pencere okuyan DEM adaptörü.
- `simulation.py`: Çevrimiçi simülasyon döngüsü.
- `matcher.py`: Profil eşleştirme (TERCOM tarzı) algoritması.
- `ui.py` & `rendering.py`: PySide6 ve Matplotlib görselleştirme arayüzü.

Varsayılan normal çalıştırmada `C:\\d_surucusu\\visual_navigation\\template-matching\\karlik_30_cm_bingmap_utm_elevation.tif`
varsa kullanılır. Büyük GeoTIFF dosyaları tamamen RAM'e alınmaz; varsayılan olarak merkezden `4096 x 4096`
piksel pencere okunur ve gerçek raster piksel boyutu arama geometrisine aktarılır. `--dem-row`, `--dem-col`,
`--start-row` ve `--start-col` ile pencere ve yerel başlangıç değiştirilebilir.

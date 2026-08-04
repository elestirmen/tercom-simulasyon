## GPS-Denied Review

### Mevcut yaklaşım
Proje, gözlem haritasından üç pencere çıkarıp model ile üç template üretir ve bunları referans haritada `matchTemplate` ile arar. Eşleşen kutuların kesişimi konum tahmini olarak kullanılır.

### Temel zayıflıklar
- Güven skoru olmadan tek-frame karar verildiğinde hatalı eşleşme sonrası ROI kolayca yanlış yere kilitlenebilir.
- Görev mantığı operatör komutuna bağlıydı; waypoint takibi ve düşük güvende yeniden kazanım davranışı eksikti.
- Tek-adım görsel sıçramalar (yanlış eşleşme) takip merkezini ani biçimde kaydırabiliyordu.

### Eklenen iyileştirmeler
Aşağıdakiler `gps_denied_autonomy.py` içinde tanımlı ve dashboard ana döngüsünde aktif kullanılır:
- **Lokalizasyon kalitesi** (`compute_localization_quality`): normalize skorlar, `score_floor` / `score_mean`, merkez yayılımı (`center_spread_px`), birleşik `confidence` ve `is_reliable` bayrağı (eşikler `localization_*_threshold`).
- **Dejenerasyon ve belirsizlik koruması**: düşük şablon standart sapması, yetersiz bağımsız tepe marjı ve katı üçlü geometriyi sağlamayan sonuçlar Kalman'a girmeden reddedilir.
- **Düşük güvene bağlı ROI büyütme** (`update_search_window_size`): geometri doğru görünse bile güven düşükse pencere büyür; ardışık düşük güven sonrasında tam-harita yeniden kazanımı tetiklenir.
- **Açık başlangıç varsayımı**: senaryo yalnız ilk başlangıç konumunu bilinen öncül kabul eder; ilk eşleme bu konum merkezli ROI'de yapılır. Sonraki gerçek konumlar algoritmaya verilmez ve konum kaybında kontrollü global yeniden kazanım kullanılır.
- **İrtifa ölçeği**: DEM/AGL tabanlı yama ölçeği gözlem çıkarımına ek olarak gerçek ve eşleşen şablon kutularına uygulanır.
- **Sensör füzyonu** (`fuse_measurement_with_prior`): takip merkezi ölçüm güvenine göre yumuşatılır; `max_visual_jump_px` eşiğini aşan sıçramalar reddedilir.
- **Kalman filtresi** (`simulation_core.filters.ConstantVelocityKalmanFilter`, K tuşu / `kalman_enabled`): `x, y, vx, vy` durumlu gerçek sabit-hız filtresi; yalnızca güvenilir ölçümlerde güncellenir ve bilinen hareket komutunu kontrol girdisi olarak kullanır.
- **Korelasyon güven düzeltmesi**: normalize korelasyon yöntemlerinde ham `0.0` artık `%50` olarak yeniden ölçeklenmez. Negatif korelasyon sıfır kanıt sayılır; `localization_score_threshold` doğrudan pozitif korelasyon tabanında anlam kazanır.
- **Otonom waypoint modu** (`choose_autonomous_action`, `update_waypoint_progress`): P tuşu ile açılır, fare ile harita üzerinde hedef seçilir; gövde-ekseni hizalama, ardışık kabul ve takılma (stuck) kurtarma içerir.

> Not: Üçlü örnekleme hâlâ **diagonal** geometridedir (`get_observation_boxes`); offset vektörü başlık açısıyla döndürülür ama üç pencere eş-doğrusal kalır.

### Tanılama (diagnostic) toplu çalıştırma
Dashboard, üçlü şablon kalitesini ölçen bir tanılama modu içerir (`run_template_diagnostics`):
- `SimulationConfig.diagnostic_benchmark_enabled = True` → başlangıçta çalışır.
- `SimulationConfig.diagnostic_benchmark_only = True` → çıktı yazıldıktan sonra dashboard açılmadan çıkar.
- Tohum noktaları: `SimulationConfig.diagnostic_benchmark_points`.

Çıktılar `diagnostics/template_diag_YYYYMMDD_HHMMSS/` altına yazılır: her vaka için `case_XX_..._triptych.png`, `case_XX_..._meta.json` ve `summary.json`.

### Mühendislik yorumu
- Yazılım mühendisi gözüyle: algı (`localize_template_triplet`), kalite/füzyon (`gps_denied_autonomy`) ve görev mantığı (otonom döngü) ayrı katmanlara ayrılmış durumda.
- İHA mühendisi gözüyle: düşük güvende agresif ilerleme yerine dönüş/yeniden kazanım tercih ediliyor; Kalman açıkken arama çerçevesi filtre konumuna odaklanarak tek-adım hatalarına dayanıklılık artıyor.
- Bilimsel gözle: her adım CSV'ye (`log_simulasyon_*.csv`) skor, güven, yayılım, tepe marjı, şablon varyansı, geometri kararı, ham/Kalman hata (px ve m) ve çekirdek işlem süresi (`islem_ms`) olarak yazılır; tanılama vakaları PNG/JSON olarak dışa aktarılır.

### Çalıştırma
- Manuel dashboard: `python simulasyon_yonlendirme_uclu_dashboard.py`
- Otonom waypoint modu: `SimulationConfig.autonomous_mode_enabled = True` (veya çalışırken **P**).
- Tanılama: `SimulationConfig.diagnostic_benchmark_enabled = True` (yalnız tanılama için ayrıca `diagnostic_benchmark_only = True`).

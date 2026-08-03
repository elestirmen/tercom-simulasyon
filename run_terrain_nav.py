import sys
from pathlib import Path

# 'simulasyon' dizinini Python yollarına ekleyerek içindeki modülleri erişilebilir kılıyoruz.
root_dir = Path(__file__).parent
simulasyon_dir = root_dir / "simulasyon"
if str(simulasyon_dir) not in sys.path:
    sys.path.insert(0, str(simulasyon_dir))

# Ana başlatıcıyı içe aktarıp çalıştırıyoruz.
from terrain_profile_localization_dashboard import main

if __name__ == "__main__":
    main()

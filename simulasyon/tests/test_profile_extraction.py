"""Tests for profile extraction."""

import numpy as np
import math
from terrain_nav.profile import extract_profile, rotate_offsets
from terrain_nav.coordinates import CoordinateTransform

def test_extract_profile_bilinear():
    # 6. 37.5° yönünde bilinear örnekleme doğru çalışır.
    # 3. Yalnız tam piksel satırı veya sütunu üzerinden profil çıkarılması kabul edilemez. (Bilinear applied)
    dem = np.array([
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
        [70.0, 80.0, 90.0]
    ])
    ct = CoordinateTransform(dx=1.0, dy=1.0)
    
    # Let's extract at (r=0.5, c=0.5)
    # dx=0, dy=0 from start (0.5, 0.5) -> sample at 0.5, 0.5
    # which is center of the top-left cell 2x2.
    # Cell 0,0: 10, 0,1: 20, 1,0: 40, 1,1: 50
    # Center should be (10+20+40+50)/4 = 30.0
    
    offsets = [(0.0, 0.0, 0.0)]
    prof = extract_profile(dem, 0.5, 0.5, offsets, ct)
    assert math.isclose(prof[0], 30.0)

def test_harita_disi_eleme():
    # 15. Harita dışı örnekleme güvenli biçimde ele alınır.
    # 41. Harita sınırını aşan adaylar elenir.
    dem = np.ones((10, 10))
    ct = CoordinateTransform(dx=1.0, dy=1.0)
    
    offsets = [
        (0.0, 0.0, 0.0), # inside
        (100.0, 0.0, 0.0), # outside (col + 100)
        (0.0, 100.0, 0.0)  # outside (row - 100)
    ]
    prof = extract_profile(dem, 5.0, 5.0, offsets, ct)
    
    assert not np.isnan(prof[0])
    assert np.isnan(prof[1])
    assert np.isnan(prof[2])

def test_rotate_offsets():
    # offsets: dx=10, dy=0 (East)
    offsets = [(10.0, 0.0, 90.0)]
    
    # Rotate by 90 degrees.
    # new_dx = 10*cos(90) - 0*sin(90) = 0
    # new_dy = 10*sin(90) + 0*cos(90) = 10
    rotated = rotate_offsets(offsets, 90.0)
    
    assert math.isclose(rotated[0][0], 0.0, abs_tol=1e-9)
    assert math.isclose(rotated[0][1], 10.0, abs_tol=1e-9)
    assert math.isclose(rotated[0][2], 180.0)

"""Tests for known altitude matching."""

import numpy as np
import math
from terrain_nav.config import LocalizationConfig, SensorConfig
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.matcher import ProfileMatcher

def test_four_measurements():
    # 22. [450,480,500,540] örneği 1680 metre üretir.
    # 23. Bilinen MSL irtifasında doğru profil bulunur.
    laser = np.array([450.0, 480.0, 500.0, 540.0])
    valid = np.array([True, True, True, True])
    baro = np.zeros(4)
    dem = np.array([
        [1230.0, 1200.0, 1180.0, 1140.0],
        [1230.0, 1200.0, 1180.0, 1140.0]
    ])
    # The profile in dem is at row=0, cols=0,1,2,3
    # With start row=0, col=0, moving East (+col)
    
    ct = CoordinateTransform(dx=1.0, dy=1.0)
    offsets = [
        (0.0, 0.0, 90.0),
        (1.0, 0.0, 90.0),
        (2.0, 0.0, 90.0),
        (3.0, 0.0, 90.0)
    ]
    
    config = LocalizationConfig()
    # Replace altitude_mode in a hacky way since it's frozen
    object.__setattr__(config.sensor, 'altitude_mode', 'unknown_constant_msl_altitude')
    object.__setattr__(config.algorithm, 'min_profile_length', 3)
    
    matcher = ProfileMatcher(config, dem, ct)
    cands = matcher.exhaustive_search(laser, valid, baro, offsets, search_headings=[90.0])
    
    assert len(cands) > 0
    best = cands[0]
    assert math.isclose(best.row, 0.0)
    assert math.isclose(best.col, 0.0)
    assert math.isclose(best.estimated_msl_m, 1680.0)
    assert math.isclose(best.score, 0.0, abs_tol=1e-5)

def test_first_difference_profile():
    # 25. Birinci fark profili doğru hesaplanır.
    laser = np.array([100.0, 110.0, 130.0])
    valid = np.array([True, True, True])
    baro = np.zeros(3)
    # diff of laser: [10.0, 20.0]
    # expected diff is diff of (H - DEM) = -diff(DEM)
    # let DEM be [900, 890, 870]. diff(DEM) = [-10, -20]. -diff(DEM) = [10, 20]. Match!
    dem = np.array([
        [900.0, 890.0, 870.0, 850.0],
        [900.0, 890.0, 870.0, 850.0]
    ])
    ct = CoordinateTransform(dx=1.0, dy=1.0)
    offsets = [(0.0,0.0,90.0), (1.0,0.0,90.0), (2.0,0.0,90.0)]
    
    config = LocalizationConfig()
    object.__setattr__(config.algorithm, 'min_profile_length', 3)
    
    matcher = ProfileMatcher(config, dem, ct)
    cands = matcher.exhaustive_search(laser, valid, baro, offsets, search_headings=[90.0])
    assert len(cands) > 0
    assert math.isclose(cands[0].metrics["diff_rmse"], 0.0, abs_tol=1e-5)


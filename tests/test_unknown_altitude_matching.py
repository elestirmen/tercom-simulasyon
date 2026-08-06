"""Tests for unknown altitude matching."""

import math

import numpy as np

from terrain_nav.config import LocalizationConfig
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.matcher import ProfileMatcher


def test_unknown_altitude_median():
    # 24. Bilinmeyen MSL irtifasında median() tabanlı kaydırma doğru tahmin yapar.
    laser = np.array([450.0, 480.0, 500.0, 540.0])
    valid = np.array([True, True, True, True])
    baro = np.zeros(4)
    dem = np.array(
        [[1230.0, 1200.0, 1180.0, 1140.0, 1100.0], [1230.0, 1200.0, 1180.0, 1140.0, 1100.0]]
    )
    ct = CoordinateTransform(dx=1.0, dy=1.0)
    offsets = [(0.0, 0.0, 90.0), (1.0, 0.0, 90.0), (2.0, 0.0, 90.0), (3.0, 0.0, 90.0)]

    config = LocalizationConfig()
    object.__setattr__(config.sensor, "altitude_mode", "unknown_constant_msl_altitude")
    object.__setattr__(config.algorithm, "min_profile_length", 3)

    matcher = ProfileMatcher(config, dem, ct)
    cands = matcher.exhaustive_search(laser, valid, baro, offsets, search_headings=[90.0])

    assert len(cands) > 0
    best = cands[0]
    # In row=0, col=0, laser + dem = [450+1230, 480+1200, 500+1180, 540+1140] = [1680, 1680, 1680, 1680]
    # Median is 1680.0
    assert math.isclose(best.estimated_msl_m, 1680.0)
    assert math.isclose(best.score, 0.0, abs_tol=1e-5)


def test_huber_loss_outliers():
    # 36. Lazer outlier ölçümleri, Huber veya Median hatasıyla tolere edilebilir.
    # We will introduce an outlier to the laser measurements.
    laser = np.array([450.0, 480.0, 1000.0, 540.0])  # 1000.0 is an outlier
    valid = np.array([True, True, True, True])
    baro = np.zeros(4)
    dem = np.array(
        [[1230.0, 1200.0, 1180.0, 1140.0, 1100.0], [1230.0, 1200.0, 1180.0, 1140.0, 1100.0]]
    )
    ct = CoordinateTransform(dx=1.0, dy=1.0)
    offsets = [(0.0, 0.0, 90.0), (1.0, 0.0, 90.0), (2.0, 0.0, 90.0), (3.0, 0.0, 90.0)]

    config = LocalizationConfig()
    object.__setattr__(config.sensor, "altitude_mode", "unknown_constant_msl_altitude")
    object.__setattr__(config.algorithm, "min_profile_length", 3)
    object.__setattr__(config.algorithm, "loss_method", "huber")

    matcher = ProfileMatcher(config, dem, ct)
    cands = matcher.exhaustive_search(laser, valid, baro, offsets, search_headings=[90.0])

    best = cands[0]
    # median of [1680, 1680, 2180, 1680] -> 1680.0
    assert math.isclose(best.estimated_msl_m, 1680.0)
    # For error=500, delta=10: loss = 0.5*100 + 10*490 = 4950. Mean over 4 is 1237.5.
    assert math.isclose(best.score, 1237.5)

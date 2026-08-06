"""Tests for heading search."""

import math

import numpy as np

from terrain_nav.config import LocalizationConfig
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.matcher import ProfileMatcher


def test_heading_search_coarse_to_fine_returns_candidates():
    dem = np.zeros((20, 20))
    # Create a diagonal trench (heading 45 degrees)
    for i in range(15):
        dem[i, i] = -10.0
        dem[i, i + 1] = -10.0
        dem[i + 1, i] = -10.0

    ct = CoordinateTransform(dx=1.0, dy=1.0)
    # We sampled at 45 degrees
    offsets = [(0.0, 0.0, 45.0), (1.414, 1.414, 45.0), (2.828, 2.828, 45.0)]
    laser = np.array([10.0, 10.0, 10.0])
    valid = np.array([True, True, True])
    baro = np.zeros(3)

    config = LocalizationConfig()
    object.__setattr__(config.sensor, "altitude_mode", "known_msl_altitude")
    object.__setattr__(config.sensor, "constant_msl_m", 0.0)
    object.__setattr__(config.algorithm, "min_profile_length", 3)
    object.__setattr__(config.algorithm, "coarse_stride", 2)
    object.__setattr__(config.algorithm, "medium_stride", 1)
    object.__setattr__(config.algorithm, "fine_stride", 1)

    matcher = ProfileMatcher(config, dem, ct)
    search_headings = [0.0, 45.0, 90.0, 135.0, 180.0]

    # Coarse to fine
    cands_cf = matcher.coarse_to_fine_search(laser, valid, baro, offsets, search_headings)
    assert len(cands_cf) > 0


def test_heading_unknown_match():
    # Create a horizontal trench
    dem = np.zeros((10, 10))
    dem[5, 2:8] = -50.0
    dem[6, 2:8] = -50.0

    ct = CoordinateTransform(dx=1.0, dy=1.0)
    # The true path goes East (heading 90) along row 5
    # Base offsets are generated assuming we think we're going North (heading 0)
    offsets = [
        (0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, -2.0, 0.0),
    ]  # Moves North (dy increases negatively)
    laser = np.array([50.0, 50.0, 50.0])  # we see trench
    valid = np.array([True, True, True])
    baro = np.zeros(3)

    config = LocalizationConfig()
    object.__setattr__(config.sensor, "altitude_mode", "known_msl_altitude")
    object.__setattr__(config.sensor, "constant_msl_m", 0.0)
    object.__setattr__(config.algorithm, "min_profile_length", 3)

    matcher = ProfileMatcher(config, dem, ct)
    search_headings = [0.0, 90.0, 180.0, 270.0]

    cands = matcher.exhaustive_search(laser, valid, baro, offsets, search_headings, stride=1)

    assert len(cands) > 0
    best = cands[0]
    assert math.isclose(best.score, 0.0, abs_tol=1e-5)
    # the best heading candidate should be 90.0
    assert math.isclose(best.heading_deg, 90.0)

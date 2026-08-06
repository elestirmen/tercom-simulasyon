"""Tests for terrain generation and ground-truth isolation."""

import numpy as np

from terrain_nav.config import TerrainConfig
from terrain_nav.terrain import TerrainManager


def test_terrain_deterministic_seed():
    # 11. Aynı seed aynı DEM'i üretir.
    c1 = TerrainConfig(seed=42, rows=100, cols=100)
    tm1 = TerrainManager(c1)

    c2 = TerrainConfig(seed=42, rows=100, cols=100)
    tm2 = TerrainManager(c2)

    np.testing.assert_array_equal(tm1.get_truth_dem(), tm2.get_truth_dem())
    np.testing.assert_array_equal(tm1.get_navigation_dem(), tm2.get_navigation_dem())


def test_terrain_different_seed():
    # 12. Farklı seed farklı DEM üretir.
    c1 = TerrainConfig(seed=42, rows=100, cols=100)
    tm1 = TerrainManager(c1)

    c2 = TerrainConfig(seed=99, rows=100, cols=100)
    tm2 = TerrainManager(c2)

    assert not np.allclose(tm1.get_truth_dem(), tm2.get_truth_dem())


def test_truth_vs_navigation_dem():
    # 13. Truth DEM ve navigation DEM birbirinden ayrıdır.
    # 14. Lokalizasyon algoritması truth DEM'e erişmez (TerrainManager üzerinden izole).
    c = TerrainConfig(seed=42, rows=100, cols=100, dem_bias_m=10.0, dem_noise_std_m=0.5)
    tm = TerrainManager(c)

    truth = tm.get_truth_dem()
    nav = tm.get_navigation_dem()

    # Nav should be shifted by ~10m bias + noise
    diff = nav - truth
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)

    assert np.isclose(mean_diff, 10.0, atol=0.1)
    assert np.isclose(std_diff, 0.5, atol=0.1)


def test_terrain_extent():
    c = TerrainConfig(rows=200, cols=300, dx=2.0, dy=3.0)
    tm = TerrainManager(c)
    max_x, max_y = tm.get_extent()
    assert np.isclose(max_x, 600.0)  # 300 * 2.0
    assert np.isclose(max_y, 600.0)  # 200 * 3.0

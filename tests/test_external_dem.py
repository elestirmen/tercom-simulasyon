"""Tests for bounded-resolution complete GeoTIFF DEM loading."""

import numpy as np
import rasterio
from rasterio.transform import from_origin

from terrain_nav.config import LocalizationConfig, RouteConfig, SensorConfig, TerrainConfig
from terrain_nav.simulation import SimulationEngine
from terrain_nav.terrain import TerrainManager


def test_external_dem_covers_complete_source(tmp_path):
    dem_path = tmp_path / "external_dem.tif"
    source = np.arange(100, dtype=np.float32).reshape(10, 10)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=source.shape[0],
        width=source.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32636",
        transform=from_origin(500000.0, 4200000.0, 0.3, 0.3),
    ) as dataset:
        dataset.write(source, 1)

    config = TerrainConfig(
        dem_path=str(dem_path),
        dem_target_size=10,
        dem_noise_std_m=0.0,
        dem_bias_m=0.0,
    )
    terrain = TerrainManager(config)

    assert terrain.get_truth_dem().dtype == np.float32
    assert terrain.get_navigation_dem().dtype == np.float32
    np.testing.assert_array_equal(terrain.get_truth_dem(), source)
    np.testing.assert_array_equal(terrain.get_navigation_dem(), source)
    assert np.isclose(terrain.dx, 0.3)
    assert np.isclose(terrain.dy, 0.3)
    assert np.allclose(terrain.get_extent(), (3.0, 3.0))
    np.testing.assert_array_equal(terrain.get_display_dem(max_edge=20), source)
    assert np.allclose(terrain.get_display_extent(), (3.0, 3.0))
    assert np.allclose(
        terrain.get_display_bounds(),
        (500000.0, 500003.0, 4199997.0, 4200000.0),
    )
    assert np.allclose(terrain.get_display_offset(), (500000.0, 4200000.0))


def test_external_dem_downsampling_preserves_physical_extent(tmp_path):
    dem_path = tmp_path / "external_dem_large.tif"
    source = np.arange(64, dtype=np.float32).reshape(8, 8)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=source.shape[0],
        width=source.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32636",
        transform=from_origin(500000.0, 4200000.0, 0.3, 0.3),
    ) as dataset:
        dataset.write(source, 1)

    terrain = TerrainManager(
        TerrainConfig(
            dem_path=str(dem_path),
            dem_target_size=4,
            dem_noise_std_m=0.0,
            dem_bias_m=0.0,
        )
    )

    assert terrain.get_truth_dem().shape == (4, 4)
    assert np.isclose(terrain.dx, 0.6)
    assert np.isclose(terrain.dy, 0.6)
    assert np.allclose(terrain.get_extent(), (2.4, 2.4))


def test_external_dem_raises_flight_altitude_to_safe_clearance(tmp_path):
    dem_path = tmp_path / "high_external_dem.tif"
    source = np.full((8, 8), 1500.0, dtype=np.float32)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float32",
        crs="EPSG:32636",
        transform=from_origin(500000.0, 4200000.0, 1.0, 1.0),
    ) as dataset:
        dataset.write(source, 1)

    config = LocalizationConfig(
        terrain=TerrainConfig(
            dem_path=str(dem_path),
            dem_target_size=8,
            dem_noise_std_m=0.0,
        ),
        route=RouteConfig(route_length_m=2.0, sample_spacing_m=1.0),
        sensor=SensorConfig(constant_msl_m=1400.0, min_safe_agl_m=50.0),
    )

    simulation = SimulationEngine(config)

    assert simulation.config.sensor.constant_msl_m == 1550.0

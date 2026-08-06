"""Tests for keyboard-style manual simulation commands."""

from dataclasses import replace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from terrain_nav.config import LocalizationConfig, TerrainConfig
from terrain_nav.simulation import MotionOutOfBoundsError, SimulationEngine


def _manual_config() -> LocalizationConfig:
    config = LocalizationConfig()
    terrain = replace(
        config.terrain,
        preset="plane",
        rows=320,
        cols=320,
        dem_noise_std_m=0.0,
        relief=0.0,
        roughness=0.0,
    )
    route = replace(
        config.route,
        start_row=160,
        start_col=160,
        route_length_m=300.0,
        manual_step_distance_m=100.0,
        heading_deg=0.0,
    )
    algorithm = replace(config.algorithm, min_profile_length=100)
    return replace(config, terrain=terrain, route=route, algorithm=algorithm)


def test_manual_forward_and_rotation_are_relative_to_vehicle_heading():
    sim = SimulationEngine(_manual_config(), manual_control=True)
    x0, y0, h0 = sim.get_current_state()

    true_state, estimate, _measurement = sim.execute_motion(100.0, 0.0)
    assert true_state == sim.get_current_state()
    assert abs(true_state[0] - x0) < 1e-6
    assert abs(true_state[1] - (y0 + 100.0)) < 1e-6
    assert true_state[2] == h0
    assert estimate is None

    sim.turn_vehicle(90.0)
    x1, y1, h1 = sim.get_current_state()
    assert h1 == 90.0
    sim.execute_motion(100.0, 0.0)
    x2, y2, _h2 = sim.get_current_state()
    assert abs(x2 - (x1 + 100.0)) < 1e-6
    assert abs(y2 - y1) < 1e-6


def test_manual_wasd_relative_directions_move_100_meters():
    expected_offsets = {
        0.0: (0.0, 100.0),
        180.0: (0.0, -100.0),
        -90.0: (-100.0, 0.0),
        90.0: (100.0, 0.0),
    }

    for relative_heading, (expected_dx, expected_dy) in expected_offsets.items():
        sim = SimulationEngine(_manual_config(), manual_control=True)
        x0, y0, _heading = sim.get_current_state()
        sim.execute_motion(100.0, relative_heading)
        x1, y1, _heading = sim.get_current_state()
        assert abs((x1 - x0) - expected_dx) < 1e-6
        assert abs((y1 - y0) - expected_dy) < 1e-6


def test_manual_command_budget_uses_manual_step_distance():
    sim = SimulationEngine(_manual_config(), manual_control=True)
    assert sim.get_total_steps() == 3
    for relative_heading in (0.0, 180.0, 0.0, 180.0):
        sim.execute_motion(100.0, relative_heading)
    assert sim.step_idx == 4


def test_manual_motion_cannot_leave_loaded_dem_window():
    sim = SimulationEngine(_manual_config(), manual_control=True)
    sim.execute_motion(100.0, 0.0)
    state_before_rejected_command = sim.get_current_state()
    step_before_rejected_command = sim.step_idx

    with pytest.raises(MotionOutOfBoundsError, match="kaynak haritanın dışına"):
        sim.execute_motion(100.0, 0.0)

    assert sim.get_current_state() == state_before_rejected_command
    assert sim.step_idx == step_before_rejected_command


def test_manual_motion_can_leave_localization_coverage_inside_source_map(tmp_path):
    dem_path = tmp_path / "wide_source_dem.tif"
    source = np.full((20, 20), 1200.0, dtype=np.float32)
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype="float32",
        crs="EPSG:32636",
        transform=from_origin(500000.0, 4200000.0, 10.0, 10.0),
    ) as dataset:
        dataset.write(source, 1)

    config = LocalizationConfig(
        terrain=TerrainConfig(
            dem_path=str(dem_path),
            dem_window_size=4,
            dem_target_size=4,
            dem_noise_std_m=0.0,
        )
    )
    sim = SimulationEngine(config, manual_control=True)

    true_state, estimate, measurement = sim.execute_motion(50.0, 90.0)

    assert sim.terrain.is_inside_source_map(true_state[0], true_state[1])
    assert not sim.terrain.is_inside_navigation_window(true_state[0], true_state[1])
    assert estimate is None
    assert measurement.laser_agl_m > 0.0
    assert sim.get_localization_status()["mode"] == "outside_loaded_window"

    _true_state, estimate, _measurement = sim.execute_motion(50.0, 270.0)

    assert estimate is None
    assert sim.get_localization_status()["mode"] == "full_loaded_window"
    assert len(sim.localization.measurements) == 1

"""Tests for the parameter optimization benchmark."""

import zipfile
from dataclasses import replace

from terrain_nav.benchmark import build_benchmark_routes
from terrain_nav.config import MOTION_MODE_UNKNOWN_CONSTANT_SPEED, LocalizationConfig
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.optimizer import (
    COARSE_STRIDES,
    MAX_PROFILE_DURATIONS,
    MIN_PROFILE_DURATIONS,
    PROFILE_MODES,
    PROFILE_POINTS,
    OptimizerRunConfig,
    format_optimizer_summary,
    generate_optimizer_candidates,
    run_optimizer_benchmark,
    split_optimizer_routes,
)
from terrain_nav.sensors import Measurement
from terrain_nav.simulation import LocalizationEngine
from terrain_nav.terrain import TerrainManager


def _optimizer_config() -> LocalizationConfig:
    config = LocalizationConfig()
    terrain = replace(
        config.terrain,
        rows=96,
        cols=112,
        dx=10.0,
        dy=10.0,
        dem_noise_std_m=0.0,
        dem_bias_m=0.0,
    )
    route = replace(
        config.route,
        speed_m_s=10.0,
        sample_spacing_m=20.0,
        manual_sample_spacing_m=20.0,
    )
    sensor = replace(
        config.sensor,
        laser_noise_std_m=0.0,
        laser_outlier_prob=0.0,
        laser_drop_prob=0.0,
    )
    algorithm = replace(
        config.algorithm,
        min_profile_length=3,
        max_match_jump_m=0.0,
        refinement_radius_px=4,
    )
    return replace(config, terrain=terrain, route=route, sensor=sensor, algorithm=algorithm)


def test_optimizer_candidate_plan_covers_requested_parameter_values() -> None:
    candidates = generate_optimizer_candidates(_optimizer_config(), limit=80)

    assert set(PROFILE_MODES) <= {candidate.profile_mode for candidate in candidates}
    assert set(PROFILE_POINTS) <= {candidate.profile_points for candidate in candidates}
    assert set(COARSE_STRIDES) <= {candidate.coarse_stride for candidate in candidates}
    assert set(MIN_PROFILE_DURATIONS) <= {
        candidate.min_profile_duration_s for candidate in candidates
    }
    assert set(MAX_PROFILE_DURATIONS) <= {
        candidate.max_profile_duration_s for candidate in candidates
    }
    assert len({candidate.config_id for candidate in candidates}) == len(candidates)


def test_optimizer_splits_routes_deterministically() -> None:
    terrain = TerrainManager(_optimizer_config().terrain)
    try:
        routes = build_benchmark_routes(terrain, max_routes=8)
    finally:
        terrain.close()

    validation, final = split_optimizer_routes(routes)

    assert validation
    assert final
    assert not ({route.name for route in validation} & {route.name for route in final})
    assert [route.name for route in final] == [routes[3].name, routes[7].name]


def test_optimizer_benchmark_writes_excel_and_preserves_unknown_speed_isolation(tmp_path) -> None:
    run_config = OptimizerRunConfig(
        initial_config_limit=4,
        refined_config_limit=2,
        final_config_limit=1,
        max_routes=4,
        stage1_route_limit=1,
        stage2_route_limit=2,
        final_route_limit=1,
        heading_scenarios=("known_heading",),
    )

    result = run_optimizer_benchmark(
        _optimizer_config(),
        run_config=run_config,
        output_dir=tmp_path,
    )

    assert result.validation_summaries
    assert result.final_summaries
    assert result.default_final is not None
    assert result.optimized_final is not None
    assert result.excel_path is not None
    assert all(not row["ground_truth_leak_detected"] for row in result.raw_details)
    assert "Onerilen" in format_optimizer_summary(result)
    with zipfile.ZipFile(result.excel_path) as workbook:
        names = set(workbook.namelist())
    assert "xl/workbook.xml" in names
    assert "xl/worksheets/sheet13.xml" in names


def test_unknown_speed_interpolated_profile_does_not_reintroduce_motion_fields() -> None:
    config = replace(
        _optimizer_config(),
        motion_mode=MOTION_MODE_UNKNOWN_CONSTANT_SPEED,
        algorithm=replace(
            _optimizer_config().algorithm,
            profile_resampling_mode="interpolated",
            profile_points=5,
            min_profile_length=5,
            min_profile_duration_s=1.0,
            max_profile_duration_s=20.0,
        ),
    )
    terrain = TerrainManager(config.terrain)
    try:
        dem = terrain.get_navigation_dem()
        engine = LocalizationEngine(config, dem, CoordinateTransform(config.terrain.dx, config.terrain.dy))
        for index in range(10):
            engine.add_measurement(
                Measurement(
                    laser_agl_m=100.0 + index,
                    laser_valid=True,
                    baro_msl_m=1500.0,
                    sensor_heading_deg=90.0,
                    traveled_distance_m=None,
                    measured_speed_m_s=None,
                    timestamp_s=float(index),
                )
            )

        profile = engine._measurements_for_localization()
    finally:
        terrain.close()

    assert len(profile) == 5
    assert all(measurement.traveled_distance_m is None for measurement in profile)
    assert all(measurement.measured_speed_m_s is None for measurement in profile)

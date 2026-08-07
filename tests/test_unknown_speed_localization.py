"""Tests for localization without speed or traveled-distance input."""

from dataclasses import replace

import numpy as np
import pytest

from run_terrain_nav import build_config
from terrain_nav.config import (
    MOTION_MODE_UNKNOWN_CONSTANT_SPEED,
    AlgorithmConfig,
    LocalizationConfig,
)
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.matcher import evaluate_candidate
from terrain_nav.profile import build_offsets_for_candidate_speed, extract_profile
from terrain_nav.sensors import Measurement
from terrain_nav.simulation import LocalizationEngine, SimulationEngine


def _distinctive_dem(rows: int = 72, cols: int = 132) -> np.ndarray:
    y, x = np.mgrid[0:rows, 0:cols]
    return np.asarray(
        900.0
        + 0.012 * x**2
        + 0.009 * y**2
        + 0.0025 * x * y
        + 13.0 * np.sin(x / 4.1)
        + 6.0 * np.cos((x + 1.7 * y) / 6.3)
        + 9.0 * np.sin((2.1 * x - y) / 9.7)
        + 5.0 * np.cos((x**2 + 3.0 * y) / 38.0),
        dtype=np.float32,
    )


def _unknown_speed_config(
    *,
    heading_mode: str = "known_heading",
    speed_min: float = 5.0,
    speed_max: float = 26.0,
    coarse_speed_step: float = 5.0,
    fine_speed_step: float = 0.25,
) -> LocalizationConfig:
    base = LocalizationConfig()
    return replace(
        base,
        motion_mode=MOTION_MODE_UNKNOWN_CONSTANT_SPEED,
        sensor=replace(
            base.sensor,
            altitude_mode="barometric_altitude",
            heading_mode=heading_mode,
            heading_uncertainty_deg=5.0,
        ),
        algorithm=replace(
            base.algorithm,
            min_profile_length=14,
            min_profile_duration_s=3.0,
            max_profile_duration_s=20.0,
            speed_search_min_m_s=speed_min,
            speed_search_max_m_s=speed_max,
            speed_search_coarse_step_m_s=coarse_speed_step,
            speed_search_medium_step_m_s=1.0,
            speed_search_fine_step_m_s=fine_speed_step,
            speed_search_keep_hypotheses=3,
            coarse_stride=4,
            medium_stride=2,
            fine_stride=1,
            refinement_radius_px=6,
            top_k=7,
            max_match_inlier_rmse_m=0.15,
            min_match_inlier_correlation=0.90,
            max_match_jump_m=0.0,
        ),
    )


def _profile_measurements(
    dem: np.ndarray,
    true_speed_m_s: float,
    *,
    sensor_heading_deg: float = 90.0,
) -> tuple[list[Measurement], tuple[float, float]]:
    ct = CoordinateTransform(1.0, 1.0)
    timestamps = [index * 0.25 for index in range(14)]
    true_measurements = [
        Measurement(0.0, True, 2075.0, 90.0, None, None, timestamp)
        for timestamp in timestamps
    ]
    true_offsets = build_offsets_for_candidate_speed(true_measurements, true_speed_m_s, ct)
    terrain_profile = extract_profile(dem, 38.0, 20.0, true_offsets, ct)
    measurements = [
        Measurement(
            laser_agl_m=2000.0 - float(elevation),
            laser_valid=True,
            baro_msl_m=2075.0,
            sensor_heading_deg=sensor_heading_deg,
            traveled_distance_m=None,
            measured_speed_m_s=None,
            timestamp_s=timestamp,
        )
        for timestamp, elevation in zip(timestamps, terrain_profile, strict=True)
    ]
    final_position = (20.0 + true_speed_m_s * timestamps[-1], -38.0)
    return measurements, final_position


@pytest.mark.parametrize("true_speed_m_s", [8.0, 12.0, 17.0, 23.0])
def test_unknown_speed_jointly_recovers_speed_and_position(true_speed_m_s: float) -> None:
    dem = _distinctive_dem()
    config = _unknown_speed_config()
    measurements, true_position = _profile_measurements(dem, true_speed_m_s)
    engine = LocalizationEngine(config, dem, CoordinateTransform(1.0, 1.0))
    for measurement in measurements:
        engine.add_measurement(measurement)

    estimate = engine.localize(measurements[-1].timestamp_s)

    assert estimate is not None
    assert estimate.estimated_speed_m_s == pytest.approx(true_speed_m_s, abs=0.25)
    position_error = np.hypot(
        estimate.estimated_x - true_position[0],
        estimate.estimated_y - true_position[1],
    )
    assert position_error < 1.0
    assert estimate.quality_score is not None and estimate.quality_score < 0.15
    assert estimate.quality_correlation is not None
    assert estimate.quality_correlation > 0.99
    assert estimate.quality_valid_ratio == 1.0
    assert not estimate.is_ambiguous
    assert not estimate.speed_is_ambiguous
    assert estimate.second_best_speed_m_s is not None


def test_wrong_speed_hypotheses_score_worse_at_true_position() -> None:
    dem = _distinctive_dem()
    config = _unknown_speed_config()
    measurements, _true_position = _profile_measurements(dem, 17.0)
    laser = np.array([measurement.laser_agl_m for measurement in measurements])
    valid = np.ones(len(measurements), dtype=bool)
    baro = np.array([measurement.baro_msl_m for measurement in measurements])
    ct = CoordinateTransform(1.0, 1.0)

    scores = {}
    for speed in (10.0, 17.0, 25.0):
        offsets = build_offsets_for_candidate_speed(measurements, speed, ct)
        candidate = evaluate_candidate(
            38.0,
            20.0,
            90.0,
            dem,
            laser,
            valid,
            baro,
            offsets,
            ct,
            config,
        )
        assert candidate is not None
        scores[speed] = candidate.score

    assert scores[17.0] < scores[10.0]
    assert scores[17.0] < scores[25.0]


def test_candidate_speed_offsets_follow_turning_headings() -> None:
    measurements = [
        Measurement(0.0, True, 0.0, 0.0, None, None, 0.0),
        Measurement(0.0, True, 0.0, 90.0, None, None, 1.0),
        Measurement(0.0, True, 0.0, 0.0, None, None, 2.0),
        Measurement(0.0, True, 0.0, 270.0, None, None, 3.0),
    ]

    offsets = build_offsets_for_candidate_speed(
        measurements,
        10.0,
        CoordinateTransform(1.0, 1.0),
    )

    assert np.allclose([offset[:2] for offset in offsets], [(0, 0), (10, 0), (10, 10), (0, 10)])


def test_unknown_speed_measurement_does_not_leak_truth_motion() -> None:
    config = build_config(
        fast_mode=True,
        motion_mode=MOTION_MODE_UNKNOWN_CONSTANT_SPEED,
        speed_search_min_m_s=8.0,
        speed_search_max_m_s=22.0,
    )
    simulation = SimulationEngine(config)
    try:
        _truth, _estimate, measurement = simulation.step()

        assert measurement.traveled_distance_m is None
        assert measurement.measured_speed_m_s is None
        assert measurement.timestamp_s == 0.0
        assert not hasattr(simulation.localization.config, "route")
        assert not hasattr(simulation.localization.config, "terrain")
        assert simulation.localization.config.motion_mode == MOTION_MODE_UNKNOWN_CONSTANT_SPEED
    finally:
        simulation.close()


def test_unknown_speed_engine_rejects_motion_fields() -> None:
    config = _unknown_speed_config()
    engine = LocalizationEngine(
        config,
        _distinctive_dem(),
        CoordinateTransform(1.0, 1.0),
    )

    with pytest.raises(ValueError, match="must not contain speed or distance"):
        engine.add_measurement(
            Measurement(100.0, True, 1000.0, 90.0, 10.0, 10.0, 0.0)
        )


def test_unknown_speed_uses_duration_window_instead_of_distance() -> None:
    base = _unknown_speed_config()
    config = replace(
        base,
        algorithm=replace(
            base.algorithm,
            min_profile_length=2,
            min_profile_duration_s=1.0,
            max_profile_duration_s=2.0,
            profile_window_size=100,
        ),
    )
    engine = LocalizationEngine(
        config,
        _distinctive_dem(),
        CoordinateTransform(1.0, 1.0),
    )
    for timestamp in (0.0, 1.0, 2.0, 3.0, 4.0):
        engine.add_measurement(
            Measurement(100.0, True, 1000.0, 90.0, None, None, timestamp)
        )

    assert [measurement.timestamp_s for measurement in engine.measurements] == [2.0, 3.0, 4.0]
    assert engine._profile_duration_s() == 2.0


def test_flat_terrain_does_not_force_unknown_speed_fix() -> None:
    base = _unknown_speed_config(
        speed_min=5.0,
        speed_max=10.0,
        coarse_speed_step=5.0,
        fine_speed_step=0.5,
    )
    config = replace(
        base,
        algorithm=replace(
            base.algorithm,
            min_profile_length=5,
            min_profile_duration_s=4.0,
            max_profile_duration_s=10.0,
            coarse_stride=5,
            top_k=3,
        ),
    )
    dem = np.full((40, 40), 900.0, dtype=np.float32)
    engine = LocalizationEngine(config, dem, CoordinateTransform(1.0, 1.0))
    for timestamp in range(5):
        engine.add_measurement(
            Measurement(100.0, True, 1075.0, 90.0, None, None, float(timestamp))
        )

    estimate = engine.localize(4.0)

    assert estimate is None or estimate.is_ambiguous
    assert engine.get_search_status()["rejection_reason"] in {None, "quality"}


def test_unknown_speed_with_noisy_heading_recovers_route() -> None:
    dem = _distinctive_dem(rows=60, cols=110)
    config = _unknown_speed_config(
        heading_mode="noisy_heading",
        speed_min=14.0,
        speed_max=20.0,
        coarse_speed_step=3.0,
        fine_speed_step=0.5,
    )
    config = replace(
        config,
        algorithm=replace(
            config.algorithm,
            speed_search_keep_hypotheses=2,
            coarse_stride=5,
            top_k=5,
            max_match_inlier_rmse_m=0.20,
        ),
    )
    measurements, true_position = _profile_measurements(
        dem,
        17.0,
        sensor_heading_deg=92.0,
    )
    engine = LocalizationEngine(config, dem, CoordinateTransform(1.0, 1.0))
    for measurement in measurements:
        engine.add_measurement(measurement)

    estimate = engine.localize(measurements[-1].timestamp_s)

    assert estimate is not None
    assert estimate.estimated_speed_m_s == pytest.approx(17.0, abs=0.5)
    assert estimate.estimated_heading_deg == pytest.approx(90.0, abs=1.0)
    assert np.hypot(
        estimate.estimated_x - true_position[0],
        estimate.estimated_y - true_position[1],
    ) < 1.0


def test_unknown_speed_config_validation_and_cli_builder() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        AlgorithmConfig(speed_search_min_m_s=0.0)
    with pytest.raises(ValueError, match="greater than"):
        AlgorithmConfig(speed_search_min_m_s=20.0, speed_search_max_m_s=10.0)
    with pytest.raises(ValueError, match="motion_mode"):
        LocalizationConfig(motion_mode="truth_speed")

    config = build_config(
        fast_mode=True,
        motion_mode=MOTION_MODE_UNKNOWN_CONSTANT_SPEED,
        speed_search_min_m_s=8.0,
        speed_search_max_m_s=24.0,
    )
    assert config.motion_mode == MOTION_MODE_UNKNOWN_CONSTANT_SPEED
    assert config.sensor.altitude_mode == "barometric_altitude"
    assert config.algorithm.speed_search_min_m_s == 8.0
    assert config.algorithm.speed_search_max_m_s == 24.0
    assert config.algorithm.min_profile_duration_s == 4.0

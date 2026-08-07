"""Regression tests for bounded, low-memory profile search."""

import numpy as np

from terrain_nav.config import LocalizationConfig
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.matcher import Candidate, ProfileMatcher, evaluate_candidate
from terrain_nav.sensors import Measurement
from terrain_nav.simulation import LocalizationEngine


def test_roi_uses_same_stride_lattice_as_global_search():
    dem = np.zeros((12, 12), dtype=np.float32)
    dem[6, 6] = 900.0
    laser = np.array([100.0])
    valid = np.array([True])
    baro = np.array([0.0])
    offsets = [(0.0, 0.0, 0.0)]

    config = LocalizationConfig()
    object.__setattr__(config.sensor, "constant_msl_m", 1000.0)
    object.__setattr__(config.algorithm, "min_profile_length", 1)
    object.__setattr__(config.algorithm, "top_k", 1)
    object.__setattr__(config.algorithm, "loss_method", "rmse")
    matcher = ProfileMatcher(config, dem, CoordinateTransform(1.0, 1.0))

    global_best = matcher.exhaustive_search(laser, valid, baro, offsets, [0.0], stride=3)[0]
    roi_best = matcher.exhaustive_search(
        laser,
        valid,
        baro,
        offsets,
        [0.0],
        stride=3,
        search_bounds=(4, 10, 4, 10),
    )[0]
    vectorized_best = matcher._vectorized_known_altitude_search(
        laser,
        valid,
        baro,
        offsets,
        0.0,
        stride=3,
        search_bounds=None,
    )[0]

    assert (global_best.row, global_best.col) == (6, 6)
    assert (roi_best.row, roi_best.col) == (6, 6)
    assert (vectorized_best.row, vectorized_best.col) == (6, 6)
    assert vectorized_best.score == global_best.score
    assert roi_best.metrics["rmse"] == 0.0


def test_continuity_gate_rejects_lower_score_impossible_jump():
    config = LocalizationConfig()
    object.__setattr__(config.algorithm, "min_profile_length", 1)
    object.__setattr__(config.algorithm, "max_match_jump_m", 10.0)
    engine = LocalizationEngine(
        config,
        np.zeros((100, 100), dtype=np.float32),
        CoordinateTransform(1.0, 1.0),
    )
    engine.last_match_pixel = (50.0, 50.0)
    engine.add_measurement(Measurement(100.0, True, 1000.0, 0.0, 0.0))

    impossible = Candidate(50.0, 70.0, 0.0, 1000.0, 1.0, 1.0, {})
    plausible = Candidate(50.0, 52.0, 0.0, 1000.0, 2.0, 1.0, {})
    engine.matcher.coarse_to_fine_search = lambda *args, **kwargs: [
        impossible,
        plausible,
    ]

    estimate = engine.localize(0.0)

    assert estimate is not None
    assert estimate.estimated_x == 52.0
    assert engine.last_match_pixel == (50.0, 52.0)


def test_vectorized_coarse_scores_match_scalar_losses():
    rng = np.random.default_rng(7)
    dem = rng.normal(900.0, 20.0, size=(16, 16)).astype(np.float32)
    offsets = [(0.0, 0.0, 90.0), (1.0, 0.0, 90.0), (2.0, 0.0, 90.0)]
    laser = 1000.0 - dem[4, 4:7].astype(np.float64)
    valid = np.ones(3, dtype=bool)
    baro = np.full(3, 1075.0)

    for altitude_mode in (
        "known_msl_altitude",
        "unknown_constant_msl_altitude",
        "barometric_altitude",
    ):
        for loss_method in ("rmse", "mae", "huber"):
            config = LocalizationConfig()
            object.__setattr__(config.sensor, "altitude_mode", altitude_mode)
            object.__setattr__(config.sensor, "constant_msl_m", 1000.0)
            object.__setattr__(config.algorithm, "min_profile_length", 3)
            object.__setattr__(config.algorithm, "loss_method", loss_method)
            matcher = ProfileMatcher(config, dem, CoordinateTransform(1.0, 1.0))

            scalar = matcher.exhaustive_search(laser, valid, baro, offsets, [90.0], stride=2)[0]
            vectorized = matcher._vectorized_known_altitude_search(
                laser,
                valid,
                baro,
                offsets,
                90.0,
                stride=2,
                search_bounds=None,
            )[0]

            assert (vectorized.row, vectorized.col) == (scalar.row, scalar.col)
            assert np.isclose(vectorized.score, scalar.score)


def test_realistic_noise_config_uses_barometric_altitude_and_noisy_speed():
    from run_terrain_nav import build_config

    config = build_config(fast_mode=True, realistic_noise=True)

    assert config.sensor.altitude_mode == "barometric_altitude"
    assert config.sensor.baro_bias_m != 0.0
    assert config.sensor.speed_noise_std_m_s > 0.0
    assert config.algorithm.min_profile_distance_m == 40.0
    assert config.algorithm.min_profile_length == 5
    assert config.sensor.heading_mode == "known_heading"


def test_default_config_keeps_idealized_known_altitude_mode():
    from run_terrain_nav import build_config

    config = build_config(fast_mode=True)

    assert config.sensor.altitude_mode == "known_msl_altitude"
    assert config.sensor.speed_noise_std_m_s == 0.0


def test_measured_speed_distance_is_used_for_profile_offsets():
    config = LocalizationConfig()
    engine = LocalizationEngine(
        config,
        np.zeros((100, 100), dtype=np.float32),
        CoordinateTransform(1.0, 1.0),
    )

    engine.add_measurement(Measurement(100.0, True, 1000.0, 90.0, 10.0, 10.0))
    engine.add_measurement(Measurement(100.0, True, 1000.0, 90.0, 12.0, 12.0))

    assert np.allclose(engine.relative_offsets[1][:2], (12.0, 0.0))


def test_min_profile_distance_delays_localization_until_profile_is_long_enough():
    config = LocalizationConfig()
    object.__setattr__(config.algorithm, "min_profile_length", 1)
    object.__setattr__(config.algorithm, "min_profile_distance_m", 20.0)
    engine = LocalizationEngine(
        config,
        np.zeros((100, 100), dtype=np.float32),
        CoordinateTransform(1.0, 1.0),
    )
    engine.matcher.coarse_to_fine_search = lambda *args, **kwargs: [
        Candidate(50.0, 50.0, 0.0, 1000.0, 0.0, 1.0, {"inlier_rmse": 0.0})
    ]

    engine.add_measurement(Measurement(100.0, True, 1000.0, 0.0, 10.0))
    engine.add_measurement(Measurement(100.0, True, 1000.0, 0.0, 9.0))
    assert engine.localize(0.0) is None

    engine.add_measurement(Measurement(100.0, True, 1000.0, 0.0, 11.0))
    assert engine.localize(1.0) is not None


def test_old_vectorized_known_altitude_name_stays_compatible():
    dem = np.arange(25, dtype=np.float32).reshape(5, 5)
    offsets = [(0.0, 0.0, 90.0)]
    laser = np.array([1000.0 - dem[2, 2]])
    valid = np.array([True])
    baro = np.zeros(1)

    config = LocalizationConfig()
    object.__setattr__(config.algorithm, "min_profile_length", 1)
    matcher = ProfileMatcher(config, dem, CoordinateTransform(1.0, 1.0))

    cands = matcher._vectorized_known_altitude_search(
        laser,
        valid,
        baro,
        offsets,
        90.0,
        stride=1,
        search_bounds=None,
    )

    assert cands


def test_profile_anchor_follows_removed_measurement_motion():
    config = LocalizationConfig()
    object.__setattr__(config.algorithm, "profile_window_size", 1)
    engine = LocalizationEngine(
        config,
        np.zeros((100, 100), dtype=np.float32),
        CoordinateTransform(1.0, 1.0),
    )
    engine.last_match_pixel = (50.0, 50.0)
    engine.add_measurement(Measurement(100.0, True, 1000.0, 90.0, 10.0))
    engine.add_measurement(Measurement(100.0, True, 1000.0, 90.0, 10.0))

    assert np.allclose(engine.last_match_pixel, (50.0, 60.0))


def test_profile_offsets_use_motion_into_each_current_measurement():
    config = LocalizationConfig()
    engine = LocalizationEngine(
        config,
        np.zeros((100, 100), dtype=np.float32),
        CoordinateTransform(1.0, 1.0),
    )

    engine.add_measurement(Measurement(100.0, True, 1000.0, 90.0, 10.0))
    engine.add_measurement(Measurement(100.0, True, 1000.0, 0.0, 20.0))

    assert np.allclose(engine.relative_offsets[0][:2], (0.0, 0.0))
    assert np.allclose(engine.relative_offsets[1][:2], (0.0, 20.0))


def test_sliding_profile_anchor_uses_new_oldest_measurement_motion():
    config = LocalizationConfig()
    object.__setattr__(config.algorithm, "profile_window_size", 1)
    engine = LocalizationEngine(
        config,
        np.zeros((100, 100), dtype=np.float32),
        CoordinateTransform(1.0, 1.0),
    )
    engine.last_match_pixel = (50.0, 50.0)
    engine.add_measurement(Measurement(100.0, True, 1000.0, 90.0, 10.0))
    engine.add_measurement(Measurement(100.0, True, 1000.0, 0.0, 20.0))

    assert np.allclose(engine.last_match_pixel, (30.0, 50.0))


def test_absolute_quality_gate_rejects_confident_but_bad_candidate():
    config = LocalizationConfig()
    object.__setattr__(config.algorithm, "min_profile_length", 1)
    engine = LocalizationEngine(
        config,
        np.zeros((100, 100), dtype=np.float32),
        CoordinateTransform(1.0, 1.0),
    )
    engine.add_measurement(Measurement(100.0, True, 1000.0, 0.0, 0.0))
    bad_candidate = Candidate(
        80.0,
        80.0,
        0.0,
        1000.0,
        39.0,
        1.0,
        {"inlier_rmse": 8.0, "inlier_correlation": 0.35},
    )
    engine.matcher.coarse_to_fine_search = lambda *args, **kwargs: [bad_candidate]

    assert engine.localize(0.0) is None
    assert engine.last_match_pixel is None
    status = engine.get_search_status()
    assert status["mode"] == "global_search"
    assert status["rejection_reason"] == "quality"
    assert status["rejected_score"] == 8.0


def test_quality_gate_accepts_single_laser_outlier_in_long_profile():
    config = LocalizationConfig()
    object.__setattr__(config.sensor, "constant_msl_m", 1000.0)
    object.__setattr__(config.algorithm, "min_profile_length", 10)

    terrain_line = np.linspace(900.0, 999.0, 120, dtype=np.float64)
    dem = np.vstack([terrain_line, terrain_line]).astype(np.float32)
    offsets = [(float(index), 0.0, 90.0) for index in range(100)]
    laser = 1000.0 - terrain_line[:100]
    laser[45] += 50.0
    valid = np.ones(100, dtype=bool)
    baro = np.zeros(100)

    candidate = evaluate_candidate(
        0.0,
        0.0,
        90.0,
        dem,
        laser,
        valid,
        baro,
        offsets,
        CoordinateTransform(1.0, 1.0),
        config,
    )
    assert candidate is not None
    assert candidate.score > config.algorithm.max_match_inlier_rmse_m
    assert candidate.metrics["inlier_rmse"] < 1e-4
    assert candidate.metrics["inlier_correlation"] > 0.99

    engine = LocalizationEngine(config, dem, CoordinateTransform(1.0, 1.0))
    assert engine._candidate_passes_quality_gate(candidate)


def test_search_status_distinguishes_global_search_from_local_roi():
    config = LocalizationConfig()
    object.__setattr__(config.algorithm, "search_roi_size_px", 20)
    engine = LocalizationEngine(
        config,
        np.zeros((100, 120), dtype=np.float32),
        CoordinateTransform(1.0, 1.0),
    )

    initial = engine.get_search_status()
    assert initial["mode"] == "global_search"
    assert initial["phase"] == "initial"
    assert initial["bounds"] == (0, 100, 0, 120)
    assert initial["draw_bounds"] is None

    engine.last_match_pixel = (50.0, 60.0)
    local = engine.get_search_status()
    assert local["mode"] == "local_roi"
    assert local["phase"] == "tracking"
    assert local["bounds"] == (40, 60, 50, 70)
    assert local["draw_bounds"] == local["bounds"]


def test_full_roi_failure_drops_stale_anchor_and_reacquires_globally():
    config = LocalizationConfig()
    object.__setattr__(config.algorithm, "min_profile_length", 1)
    object.__setattr__(config.algorithm, "search_roi_size_px", 20)
    engine = LocalizationEngine(
        config,
        np.zeros((100, 120), dtype=np.float32),
        CoordinateTransform(1.0, 1.0),
    )
    engine.last_match_pixel = (50.0, 50.0)
    engine.current_search_roi_size = 120
    engine.add_measurement(Measurement(100.0, True, 1000.0, 0.0, 0.0))
    engine.matcher.coarse_to_fine_search = lambda *args, **kwargs: []

    assert engine.localize(0.0) is None
    recovery = engine.get_search_status()
    assert recovery["mode"] == "global_search"
    assert recovery["phase"] == "recovery"
    assert not recovery["has_anchor"]
    assert len(engine.measurements) == 1

    recovered = Candidate(80.0, 80.0, 0.0, 1000.0, 1.0, 1.0, {})
    engine.matcher.coarse_to_fine_search = lambda *args, **kwargs: [recovered]

    assert engine.localize(1.0) is not None
    assert engine.last_match_pixel == (80.0, 80.0)
    assert engine.get_search_status()["phase"] == "tracking"


def test_ambiguous_global_result_does_not_create_a_false_local_lock():
    config = LocalizationConfig()
    object.__setattr__(config.algorithm, "min_profile_length", 1)
    engine = LocalizationEngine(
        config,
        np.zeros((100, 100), dtype=np.float32),
        CoordinateTransform(1.0, 1.0),
    )
    engine.add_measurement(Measurement(100.0, True, 1000.0, 0.0, 0.0))
    engine.matcher.coarse_to_fine_search = lambda *args, **kwargs: [
        Candidate(20.0, 20.0, 0.0, 1000.0, 1.0, 1.0, {}),
        Candidate(80.0, 80.0, 0.0, 1000.0, 1.0, 1.0, {}),
    ]

    estimate = engine.localize(0.0)

    assert estimate is not None
    assert estimate.is_ambiguous
    assert engine.last_match_pixel is None
    assert engine.get_search_status()["mode"] == "global_search"

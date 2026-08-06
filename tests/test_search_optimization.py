"""Regression tests for bounded, low-memory profile search."""

import numpy as np

from terrain_nav.config import LocalizationConfig
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.matcher import Candidate, ProfileMatcher
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
    baro = np.zeros(3)

    for loss_method in ("rmse", "mae", "huber"):
        config = LocalizationConfig()
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


def test_search_status_distinguishes_loaded_window_from_local_roi():
    config = LocalizationConfig()
    object.__setattr__(config.algorithm, "search_roi_size_px", 20)
    engine = LocalizationEngine(
        config,
        np.zeros((100, 120), dtype=np.float32),
        CoordinateTransform(1.0, 1.0),
    )

    initial = engine.get_search_status()
    assert initial["mode"] == "full_loaded_window"
    assert initial["bounds"] == (0, 100, 0, 120)
    assert initial["draw_bounds"] is None

    engine.last_match_pixel = (50.0, 60.0)
    local = engine.get_search_status()
    assert local["mode"] == "local_roi"
    assert local["bounds"] == (40, 60, 50, 70)
    assert local["draw_bounds"] == local["bounds"]

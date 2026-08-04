import numpy as np

from gps_denied_autonomy import compute_localization_quality
from simulasyon_yonlendirme_uclu_dashboard import (
    SimulationConfig,
    compute_intersection_box,
    extract_search_region,
    get_box_center,
    get_template_boxes_from_observation_boxes,
    is_strict_triplet_alignment,
    localize_template_triplet,
    resize_templates_to_effective_size,
    seed_known_initial_position,
    should_force_global_recovery,
    update_search_window_size,
)


def test_flat_templates_are_rejected_as_degenerate() -> None:
    rng = np.random.default_rng(7)
    search = rng.integers(0, 256, (320, 320), dtype=np.uint8)
    templates = [np.zeros((64, 64), dtype=np.uint8) for _ in range(3)]
    config = SimulationConfig(
        use_parallel_matching=False,
        use_pyramid_matching=False,
    )
    scores, boxes, predicted, mode, _backend, evidence = localize_template_triplet(
        search,
        (0, 0),
        templates,
        config,
    )
    quality = compute_localization_quality(
        scores,
        boxes,
        predicted,
        mode,
        False,
        config.localization_score_threshold,
        config.localization_confidence_threshold,
        config.localization_spread_threshold_px,
        peak_margins=tuple(item.peak_margin for item in evidence),
        peak_margin_threshold=config.localization_peak_margin_threshold,
        template_stddevs=tuple(item.template_stddev for item in evidence),
        template_std_threshold=config.localization_template_std_threshold,
        strict_alignment=False,
    )
    assert quality.is_reliable is False
    assert quality.reason == "template_variance"
    assert quality.template_std_floor == 0.0


def test_quality_requires_geometry_and_unique_peak() -> None:
    boxes = ((0, 0, 64, 64), (20, 20, 64, 64), (40, 40, 64, 64))
    predicted, mode = compute_intersection_box(list(boxes))
    config = SimulationConfig()
    geometry_failure = compute_localization_quality(
        (0.8, 0.8, 0.8),
        boxes,
        predicted,
        mode,
        False,
        config.localization_score_threshold,
        config.localization_confidence_threshold,
        config.localization_spread_threshold_px,
        peak_margins=(0.2, 0.2, 0.2),
        peak_margin_threshold=config.localization_peak_margin_threshold,
        template_stddevs=(20.0, 20.0, 20.0),
        template_std_threshold=config.localization_template_std_threshold,
        strict_alignment=False,
    )
    assert geometry_failure.reason == "geometry"

    ambiguity_failure = compute_localization_quality(
        (0.8, 0.8, 0.8),
        boxes,
        predicted,
        mode,
        False,
        config.localization_score_threshold,
        config.localization_confidence_threshold,
        config.localization_spread_threshold_px,
        peak_margins=(0.2, 0.005, 0.2),
        peak_margin_threshold=config.localization_peak_margin_threshold,
        template_stddevs=(20.0, 20.0, 20.0),
        template_std_threshold=config.localization_template_std_threshold,
        strict_alignment=True,
    )
    assert ambiguity_failure.reason == "ambiguity"


def test_real_data_calibrated_positive_score_floor_is_accepted() -> None:
    config = SimulationConfig()
    boxes = ((0, 0, 512, 512), (100, 100, 512, 512), (200, 200, 512, 512))
    predicted, mode = compute_intersection_box(list(boxes))
    quality = compute_localization_quality(
        (0.358, 0.301, 0.276),
        boxes,
        predicted,
        mode,
        False,
        config.localization_score_threshold,
        config.localization_confidence_threshold,
        config.localization_spread_threshold_px,
        peak_margins=(0.204, 0.163, 0.141),
        peak_margin_threshold=config.localization_peak_margin_threshold,
        template_stddevs=(26.4, 25.8, 24.9),
        template_std_threshold=config.localization_template_std_threshold,
        strict_alignment=True,
    )
    assert quality.is_reliable is True
    assert quality.reason == "ok"


def test_real_data_tentative_roi_match_is_accepted() -> None:
    """The formerly rejected step after a global cold start is a valid match."""
    config = SimulationConfig()
    boxes = ((0, 0, 512, 512), (100, 100, 512, 512), (200, 200, 512, 512))
    predicted, mode = compute_intersection_box(list(boxes))
    quality = compute_localization_quality(
        (0.2717, 0.2415, 0.2536),
        boxes,
        predicted,
        mode,
        False,
        config.localization_score_threshold,
        config.localization_confidence_threshold,
        config.localization_spread_threshold_px,
        peak_margins=(0.1123, 0.1033, 0.0641),
        peak_margin_threshold=config.localization_peak_margin_threshold,
        template_stddevs=(20.0, 20.0, 20.0),
        template_std_threshold=config.localization_template_std_threshold,
        strict_alignment=True,
    )
    assert quality.is_reliable is True
    assert quality.reason == "ok"


def test_missing_prior_and_forced_recovery_use_global_search() -> None:
    reference = np.arange(120 * 160, dtype=np.uint8).reshape(120, 160)
    config = SimulationConfig()
    region, origin, box, mode = extract_search_region(
        reference,
        None,
        40,
        0,
        config,
    )
    assert mode == "global"
    assert origin == (0, 0)
    assert box == (0, 0, 160, 120)
    assert region.shape == reference.shape

    region, origin, _box, mode = extract_search_region(
        reference,
        (80, 60),
        40,
        4,
        config,
        force_global=True,
    )
    assert mode == "global"
    assert origin == (0, 0)
    assert region.shape == reference.shape


def test_declared_initial_position_seeds_first_roi_only() -> None:
    config = SimulationConfig(initial_position_known=True)
    start = (10604, 8814)
    assert seed_known_initial_position(None, start, 0, config) == start
    assert seed_known_initial_position(None, (999, 888), 1, config) is None
    assert seed_known_initial_position((10, 20), start, 0, config) == (10, 20)

    unknown_config = SimulationConfig(initial_position_known=False)
    assert seed_known_initial_position(None, start, 0, unknown_config) is None


def test_low_quality_expands_search_even_with_strict_geometry() -> None:
    config = SimulationConfig()
    boxes = [(0, 0, 512, 512), (100, 100, 512, 512), (200, 200, 512, 512)]
    assert is_strict_triplet_alignment(boxes, "abc", config)
    grown = update_search_window_size(
        config.base_search_window_size,
        boxes,
        "abc",
        config,
        quality_reliable=False,
    )
    assert grown == config.base_search_window_size + config.search_window_failure_growth


def test_global_recovery_waits_until_orange_roi_has_expanded() -> None:
    config = SimulationConfig(
        global_recovery_after_low_confidence_steps=3,
        global_recovery_min_window_size=6000,
    )
    assert not should_force_global_recovery((1000, 1000), 8, 5999, config)
    assert should_force_global_recovery((1000, 1000), 3, 6000, config)
    assert not should_force_global_recovery(None, 20, 15000, config)


def test_progressive_recovery_reaches_map_size_before_global_search() -> None:
    config = SimulationConfig(
        max_search_window_size=15000,
        global_recovery_min_window_size=6000,
        progressive_global_recovery=True,
    )
    map_shape = (22987, 30720)
    assert not should_force_global_recovery(
        (1000, 1000), 50, 15000, config, map_shape,
    )
    assert not should_force_global_recovery(
        (1000, 1000), 50, 30719, config, map_shape,
    )
    assert should_force_global_recovery(
        (1000, 1000), 50, 30720, config, map_shape,
    )

    grown = update_search_window_size(
        15000,
        [],
        "none",
        config,
        quality_reliable=False,
        reference_map_shape=map_shape,
    )
    assert grown == 15500


def test_altitude_scales_templates_without_moving_their_centers() -> None:
    config = SimulationConfig()
    templates = [np.ones((512, 512), dtype=np.uint8) for _ in range(3)]
    factors = (0.5, 1.0, 2.0)
    resized = resize_templates_to_effective_size(templates, config, factors)
    assert [template.shape for template in resized] == [
        (256, 256),
        (512, 512),
        (1024, 1024),
    ]

    observation_boxes = [
        (0, 0, 544, 544),
        (100, 100, 544, 544),
        (200, 200, 544, 544),
    ]
    scaled_boxes = get_template_boxes_from_observation_boxes(
        observation_boxes,
        config,
        factors,
    )
    assert [get_box_center(box) for box in scaled_boxes] == [
        get_box_center(box) for box in observation_boxes
    ]
    assert [box[2] for box in scaled_boxes] == [256, 512, 1024]


def test_272_mode_uses_scaled_triplet_offset_for_geometry() -> None:
    config = SimulationConfig(sample_window_size=272)
    boxes = [(0, 0, 256, 256), (50, 50, 256, 256), (100, 100, 256, 256)]
    assert is_strict_triplet_alignment(boxes, "abc", config)


def test_altitude_georeference_defaults_to_observation_raster() -> None:
    config = SimulationConfig()
    assert config.observation_grid_georef_path is None
    assert config.observation_georef_path == config.observation_map_path

from gps_denied_autonomy import compute_localization_quality, normalize_match_score


def test_normalized_correlation_keeps_raw_positive_meaning() -> None:
    assert normalize_match_score(-0.4, False) == 0.0
    assert normalize_match_score(0.0, False) == 0.0
    assert normalize_match_score(0.4, False) == 0.4
    assert normalize_match_score(1.0, False) == 1.0


def test_sqdiff_score_is_inverted() -> None:
    assert normalize_match_score(0.0, True) == 1.0
    assert normalize_match_score(0.25, True) == 0.75
    assert normalize_match_score(1.0, True) == 0.0


def test_weak_raw_correlations_are_not_reported_reliable() -> None:
    boxes = ((0, 0, 20, 20), (10, 10, 20, 20), (20, 20, 20, 20))
    quality = compute_localization_quality(
        (0.24, 0.23, 0.25),
        boxes,
        (10, 10, 20, 20),
        "abc",
        False,
        score_threshold=0.35,
        confidence_threshold=0.40,
        spread_threshold_px=120.0,
    )
    assert quality.is_reliable is False
    assert quality.reason == "score_floor"
    assert quality.score_floor == 0.23


def test_consistent_strong_correlations_are_reliable() -> None:
    boxes = ((0, 0, 20, 20), (10, 10, 20, 20), (20, 20, 20, 20))
    quality = compute_localization_quality(
        (0.72, 0.68, 0.75),
        boxes,
        (10, 10, 20, 20),
        "abc",
        False,
        score_threshold=0.35,
        confidence_threshold=0.40,
        spread_threshold_px=120.0,
    )
    assert quality.is_reliable is True
    assert quality.reason == "ok"

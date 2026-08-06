"""Tests for ambiguity detection."""

from terrain_nav.confidence import detect_ambiguity
from terrain_nav.matcher import Candidate


def test_flat_terrain_ambiguity():
    # 30. Düz arazide algoritma Ambiguous durumunu tespit eder.
    # If the terrain is flat, many locations will have the same score.

    c1 = Candidate(
        row=10, col=10, heading_deg=0, estimated_msl_m=1000, score=1.0, valid_ratio=1.0, metrics={}
    )
    c2 = Candidate(
        row=50, col=50, heading_deg=0, estimated_msl_m=1000, score=1.05, valid_ratio=1.0, metrics={}
    )
    c3 = Candidate(
        row=100, col=10, heading_deg=0, estimated_msl_m=1000, score=1.1, valid_ratio=1.0, metrics={}
    )

    cands = [c1, c2, c3]

    is_ambiguous, margin, spread = detect_ambiguity(
        cands, score_margin_threshold=0.10, spatial_spread_threshold_px=5.0
    )

    assert is_ambiguous
    assert margin < 0.10
    assert spread > 20.0  # Standard deviation of (10, 50, 100) is large


def test_valley_clear_match():
    # A clear valley match where one location is much better than the rest.
    c1 = Candidate(
        row=10, col=10, heading_deg=0, estimated_msl_m=1000, score=1.0, valid_ratio=1.0, metrics={}
    )
    c2 = Candidate(
        row=50, col=50, heading_deg=0, estimated_msl_m=1000, score=5.0, valid_ratio=1.0, metrics={}
    )

    cands = [c1, c2]

    is_ambiguous, margin, spread = detect_ambiguity(
        cands, score_margin_threshold=0.10, spatial_spread_threshold_px=5.0
    )

    assert not is_ambiguous
    assert margin > 0.5


def test_local_minimum_cluster():
    # Many good matches but they are all in the exact same spot (e.g. sub-pixel search)
    c1 = Candidate(
        row=10.0,
        col=10.0,
        heading_deg=0,
        estimated_msl_m=1000,
        score=1.0,
        valid_ratio=1.0,
        metrics={},
    )
    c2 = Candidate(
        row=10.1,
        col=10.0,
        heading_deg=0,
        estimated_msl_m=1000,
        score=1.01,
        valid_ratio=1.0,
        metrics={},
    )
    c3 = Candidate(
        row=10.0,
        col=10.1,
        heading_deg=0,
        estimated_msl_m=1000,
        score=1.02,
        valid_ratio=1.0,
        metrics={},
    )

    cands = [c1, c2, c3]

    is_ambiguous, margin, spread = detect_ambiguity(
        cands, score_margin_threshold=0.10, spatial_spread_threshold_px=5.0
    )

    assert not is_ambiguous  # Margin is small, but they are tightly clustered!
    assert margin < 0.10
    assert spread < 1.0  # Very small spread

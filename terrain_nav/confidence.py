"""Confidence and ambiguity detection for profile matching."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from terrain_nav.matcher import Candidate


@dataclass(frozen=True)
class SpeedConfidence:
    """Summary of how distinctly the terrain profile identifies speed."""

    is_ambiguous: bool
    best_speed_m_s: Optional[float]
    second_best_speed_m_s: Optional[float]
    score_margin: Optional[float]
    top_speed_std_m_s: Optional[float]
    indicator: str


def detect_ambiguity(
    candidates: List[Candidate],
    score_margin_threshold: float = 0.05,
    spatial_spread_threshold_px: float = 10.0,
    score_getter: Optional[Callable[[Candidate], float]] = None,
) -> Tuple[bool, float, float]:
    """
    Detects if the match is ambiguous based on top candidates.
    Returns (is_ambiguous, score_margin, spatial_spread).
    """
    if len(candidates) < 2:
        return False, 1.0, 0.0

    c1 = candidates[0]
    c2 = candidates[1]
    get_score = score_getter or (lambda candidate: candidate.score)

    # 1. Score Margin
    # If the second best is very close in score to the best, it might be ambiguous.
    epsilon = 1e-6
    score_1 = get_score(c1)
    score_2 = get_score(c2)
    margin = abs(score_2 - score_1) / max(abs(score_2), epsilon)

    # 2. Spatial spread of Top K
    # If top candidates are all clustered together, it's just a local minimum plateau.
    # If they are far apart, it's true ambiguity (e.g. flat terrain or repeating patterns).
    rows = np.array([c.row for c in candidates])
    cols = np.array([c.col for c in candidates])

    # Use standard deviation of positions as spread
    spread_r = np.std(rows)
    spread_c = np.std(cols)
    spread = np.sqrt(spread_r**2 + spread_c**2)

    # It is ambiguous if the margin is small AND the spatial spread is large
    is_ambiguous = (margin < score_margin_threshold) and (spread > spatial_spread_threshold_px)

    return is_ambiguous, float(margin), float(spread)


def assess_speed_confidence(
    candidates: List[Candidate],
    *,
    score_margin_threshold: float,
    speed_std_threshold_m_s: float,
    top_k: int,
    score_getter: Optional[Callable[[Candidate], float]] = None,
) -> SpeedConfidence:
    """Detect similarly scoring hypotheses that imply substantially different speeds."""
    get_score = score_getter or (lambda candidate: candidate.score)
    best_by_speed: dict[float, Candidate] = {}
    for candidate in candidates:
        speed = candidate.estimated_speed_m_s
        if speed is None or not np.isfinite(speed):
            continue
        key = round(float(speed), 9)
        previous = best_by_speed.get(key)
        if previous is None or get_score(candidate) < get_score(previous):
            best_by_speed[key] = candidate

    ranked = sorted(
        best_by_speed.values(),
        key=lambda candidate: (get_score(candidate), candidate.estimated_speed_m_s or 0.0),
    )
    if not ranked:
        return SpeedConfidence(False, None, None, None, None, "unavailable")
    if len(ranked) == 1:
        speed = float(ranked[0].estimated_speed_m_s)
        return SpeedConfidence(False, speed, None, None, 0.0, "low")

    best = ranked[0]
    second = ranked[1]
    score_1 = float(get_score(best))
    score_2 = float(get_score(second))
    epsilon = 1e-6
    margin = abs(score_2 - score_1) / max(abs(score_2), epsilon)
    top = ranked[: max(2, int(top_k))]
    speed_spread = float(
        np.std([float(candidate.estimated_speed_m_s) for candidate in top])
    )
    is_ambiguous = (
        margin < score_margin_threshold and speed_spread > speed_std_threshold_m_s
    )

    if is_ambiguous:
        indicator = "ambiguous"
    elif margin >= max(0.15, score_margin_threshold * 3.0):
        indicator = "high"
    elif margin >= score_margin_threshold:
        indicator = "medium"
    else:
        indicator = "low"

    return SpeedConfidence(
        is_ambiguous=is_ambiguous,
        best_speed_m_s=float(best.estimated_speed_m_s),
        second_best_speed_m_s=float(second.estimated_speed_m_s),
        score_margin=float(margin),
        top_speed_std_m_s=speed_spread,
        indicator=indicator,
    )

"""Confidence and ambiguity detection for profile matching."""

from typing import List, Tuple

import numpy as np

from terrain_nav.matcher import Candidate


def detect_ambiguity(
    candidates: List[Candidate],
    score_margin_threshold: float = 0.05,
    spatial_spread_threshold_px: float = 10.0,
) -> Tuple[bool, float, float]:
    """
    Detects if the match is ambiguous based on top candidates.
    Returns (is_ambiguous, score_margin, spatial_spread).
    """
    if len(candidates) < 2:
        return False, 1.0, 0.0

    c1 = candidates[0]
    c2 = candidates[1]

    # 1. Score Margin
    # If the second best is very close in score to the best, it might be ambiguous.
    epsilon = 1e-6
    margin = abs(c2.score - c1.score) / max(abs(c2.score), epsilon)

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

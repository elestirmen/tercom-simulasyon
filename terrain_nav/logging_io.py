"""Logging and I/O for headless execution."""

import csv
import json
from dataclasses import asdict
from typing import List, Optional, Tuple

from terrain_nav.config import LocalizationConfig
from terrain_nav.metrics import EstimatedState


def save_config(config: LocalizationConfig, filepath: str):
    """Save config to JSON."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=4)


def save_results(
    results: List[Tuple[Tuple[float, float, float], Optional[EstimatedState]]],
    filepath: str,
    *,
    true_speed_m_s: float | None = None,
):
    """Save step-by-step results to CSV."""
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp_s",
                "true_x",
                "true_y",
                "true_heading",
                "est_x",
                "est_y",
                "est_heading",
                "est_msl",
                "error_x",
                "error_y",
                "error_pos",
                "error_heading",
                "is_ambiguous",
                "score",
                "inlier_rmse_m",
                "correlation",
                "valid_ratio",
                "estimated_speed_m_s",
                "second_best_speed_m_s",
                "true_speed_m_s",
                "speed_error_m_s",
                "speed_is_ambiguous",
                "speed_score_margin",
                "speed_spread_m_s",
                "speed_confidence",
            ]
        )

        for true_s, est_s in results:
            if est_s is None:
                continue
            err_x = est_s.estimated_x - true_s[0]
            err_y = est_s.estimated_y - true_s[1]
            err_pos = (err_x**2 + err_y**2) ** 0.5
            err_h = est_s.estimated_heading_deg - true_s[2]
            speed_error = (
                abs(est_s.estimated_speed_m_s - true_speed_m_s)
                if est_s.estimated_speed_m_s is not None and true_speed_m_s is not None
                else None
            )

            writer.writerow(
                [
                    est_s.timestamp_s,
                    true_s[0],
                    true_s[1],
                    true_s[2],
                    est_s.estimated_x,
                    est_s.estimated_y,
                    est_s.estimated_heading_deg,
                    est_s.estimated_msl_m,
                    err_x,
                    err_y,
                    err_pos,
                    err_h,
                    int(est_s.is_ambiguous),
                    est_s.score,
                    est_s.quality_score,
                    est_s.quality_correlation,
                    est_s.quality_valid_ratio,
                    est_s.estimated_speed_m_s,
                    est_s.second_best_speed_m_s,
                    true_speed_m_s,
                    speed_error,
                    int(est_s.speed_is_ambiguous),
                    est_s.speed_score_margin,
                    est_s.speed_spread_m_s,
                    est_s.speed_confidence,
                ]
            )

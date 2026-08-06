"""Logging and I/O for headless execution."""

import csv
import json
from dataclasses import asdict
from typing import List, Tuple

from terrain_nav.config import LocalizationConfig
from terrain_nav.metrics import EstimatedState


def save_config(config: LocalizationConfig, filepath: str):
    """Save config to JSON."""
    with open(filepath, "w") as f:
        json.dump(asdict(config), f, indent=4)


def save_results(results: List[Tuple[Tuple[float, float, float], EstimatedState]], filepath: str):
    """Save step-by-step results to CSV."""
    with open(filepath, "w", newline="") as f:
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
            ]
        )

        for true_s, est_s in results:
            if est_s is None:
                continue
            err_x = est_s.estimated_x - true_s[0]
            err_y = est_s.estimated_y - true_s[1]
            err_pos = (err_x**2 + err_y**2) ** 0.5
            err_h = est_s.estimated_heading_deg - true_s[2]

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
                ]
            )

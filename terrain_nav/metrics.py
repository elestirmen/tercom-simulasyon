"""Metrics definitions."""

from dataclasses import dataclass


@dataclass
class EstimatedState:
    timestamp_s: float
    is_ambiguous: bool
    estimated_x: float
    estimated_y: float
    estimated_heading_deg: float
    estimated_msl_m: float
    score: float
    spatial_spread: float
    score_margin: float

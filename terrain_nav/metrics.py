"""Metrics definitions."""

from dataclasses import dataclass
from typing import Optional


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
    quality_score: Optional[float] = None
    quality_correlation: Optional[float] = None


@dataclass
class ProfileComparison:
    distances_m: list[float]
    measured_elevation_m: list[float]
    matched_elevation_m: list[float]
    status: str
    candidate_score: Optional[float] = None
    quality_score: Optional[float] = None
    quality_correlation: Optional[float] = None

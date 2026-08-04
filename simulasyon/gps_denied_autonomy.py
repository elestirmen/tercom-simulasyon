from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

MapBox = Tuple[int, int, int, int]
MapPoint = Tuple[int, int]


@dataclass(frozen=True)
class LocalizationQuality:
    normalized_scores: Tuple[float, ...]
    score_floor: float
    score_mean: float
    center_spread_px: float
    confidence: float
    is_reliable: bool
    reason: str
    peak_margin_floor: float = 0.0
    template_std_floor: float = 0.0
    strict_alignment: bool = True


@dataclass(frozen=True)
class MissionScenario:
    name: str
    start_row: int
    start_col: int
    initial_heading_degrees: float
    initial_altitude_agl_m: float
    waypoints: Tuple[MapPoint, ...]
    max_steps: int
    forced_low_confidence_steps: Tuple[int, ...] = ()
    forced_global_search_steps: Tuple[int, ...] = ()
    heading_drift_per_step_degrees: float = 0.0
    altitude_step_events: Tuple[Tuple[int, float], ...] = ()


def normalize_heading_degrees(heading_degrees: float) -> float:
    return float(heading_degrees % 360.0)


def rotate_offset(
    delta_x: float,
    delta_y: float,
    angle_degrees: float,
) -> Tuple[float, float]:
    angle_radians = math.radians(normalize_heading_degrees(angle_degrees))
    cos_angle = math.cos(angle_radians)
    sin_angle = math.sin(angle_radians)
    return (
        (delta_x * cos_angle) - (delta_y * sin_angle),
        (delta_x * sin_angle) + (delta_y * cos_angle),
    )


def shortest_heading_error_degrees(
    current_heading_degrees: float,
    desired_heading_degrees: float,
) -> float:
    difference = (
        normalize_heading_degrees(desired_heading_degrees)
        - normalize_heading_degrees(current_heading_degrees)
        + 180.0
    ) % 360.0 - 180.0
    return float(difference)


def distance_between_points(point_a: MapPoint, point_b: MapPoint) -> float:
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def compute_box_center(box: MapBox) -> MapPoint:
    return (box[0] + (box[2] // 2), box[1] + (box[3] // 2))


def build_box_from_center(center: MapPoint, width: int, height: int) -> MapBox:
    return (
        int(round(center[0] - (width / 2.0))),
        int(round(center[1] - (height / 2.0))),
        int(width),
        int(height),
    )


def normalize_match_score(score_value: float, is_sqdiff_method: bool) -> float:
    """Normalize a template-match score to a quality value in ``[0, 1]``.

    ``TM_CCOEFF_NORMED`` and the other normalized correlation methods already
    use a meaningful positive-correlation range.  Mapping ``[-1, 1]`` to
    ``[0, 1]`` made a raw score of ``0`` look like 50% confidence and allowed
    weak matches to pass the configured floor.  Negative correlation is now
    treated as zero evidence instead.
    """
    score_value = float(score_value)
    if is_sqdiff_method:
        if 0.0 <= score_value <= 1.0:
            return max(0.0, min(1.0, 1.0 - score_value))
        return 1.0 / (1.0 + max(0.0, score_value))

    if -1.0 <= score_value <= 1.0:
        return max(0.0, min(1.0, score_value))

    positive_score = max(0.0, score_value)
    return positive_score / (1.0 + positive_score)


def compute_localization_quality(
    score_values: Sequence[float],
    matched_boxes: Sequence[MapBox],
    predicted_intersection_box: MapBox,
    intersection_mode: str,
    is_sqdiff_method: bool,
    score_threshold: float,
    confidence_threshold: float,
    spread_threshold_px: float,
    peak_margins: Sequence[float] = (),
    peak_margin_threshold: float = 0.0,
    template_stddevs: Sequence[float] = (),
    template_std_threshold: float = 0.0,
    strict_alignment: Optional[bool] = None,
) -> LocalizationQuality:
    normalized_scores = tuple(
        normalize_match_score(score_value, is_sqdiff_method) for score_value in score_values
    )
    score_floor = min(normalized_scores) if normalized_scores else 0.0
    score_mean = (
        sum(normalized_scores) / float(len(normalized_scores))
        if normalized_scores
        else 0.0
    )

    predicted_center = compute_box_center(predicted_intersection_box)
    if matched_boxes:
        center_spread_px = sum(
            distance_between_points(compute_box_center(box), predicted_center)
            for box in matched_boxes
        ) / float(len(matched_boxes))
    else:
        center_spread_px = float("inf")

    intersection_weight = {
        "abc": 1.00,
        "ab": 0.72,
        "bc": 0.72,
        "ac": 0.72,
        "center_fallback": 0.20,
    }.get(intersection_mode, 0.20)
    spatial_consistency = max(
        0.0,
        1.0 - (center_spread_px / max(1.0, float(spread_threshold_px))),
    )
    confidence = max(
        0.0,
        min(
            1.0,
            (0.45 * score_mean)
            + (0.25 * score_floor)
            + (0.20 * spatial_consistency)
            + (0.10 * intersection_weight),
        ),
    )

    peak_margin_floor = min(peak_margins) if peak_margins else float("inf")
    template_std_floor = min(template_stddevs) if template_stddevs else float("inf")
    geometry_ok = True if strict_alignment is None else bool(strict_alignment)

    if template_std_floor < float(template_std_threshold):
        reason = "template_variance"
        is_reliable = False
    elif score_floor < score_threshold:
        reason = "score_floor"
        is_reliable = False
    elif not geometry_ok:
        reason = "geometry"
        is_reliable = False
    elif peak_margin_floor < float(peak_margin_threshold):
        reason = "ambiguity"
        is_reliable = False
    elif center_spread_px > spread_threshold_px:
        reason = "spread"
        is_reliable = False
    elif confidence < confidence_threshold:
        reason = "confidence"
        is_reliable = False
    else:
        reason = "ok"
        is_reliable = True

    return LocalizationQuality(
        normalized_scores=normalized_scores,
        score_floor=float(score_floor),
        score_mean=float(score_mean),
        center_spread_px=float(center_spread_px),
        confidence=float(confidence),
        is_reliable=is_reliable,
        reason=reason,
        peak_margin_floor=float(peak_margin_floor),
        template_std_floor=float(template_std_floor),
        strict_alignment=geometry_ok,
    )


def fuse_measurement_with_prior(
    prior_center: Optional[MapPoint],
    measured_center: MapPoint,
    quality: LocalizationQuality,
    max_visual_jump_px: float,
    blend_gain: float,
) -> Tuple[MapPoint, bool, float]:
    if prior_center is None:
        return measured_center, quality.is_reliable, 0.0

    prior_error_px = distance_between_points(prior_center, measured_center)
    if prior_error_px > (float(max_visual_jump_px) * 1.75):
        return prior_center, False, float(prior_error_px)
    if not quality.is_reliable:
        return prior_center, False, float(prior_error_px)

    effective_gain = max(0.0, min(1.0, float(blend_gain) * quality.confidence))
    fused_center = (
        int(round(prior_center[0] + ((measured_center[0] - prior_center[0]) * effective_gain))),
        int(round(prior_center[1] + ((measured_center[1] - prior_center[1]) * effective_gain))),
    )
    return fused_center, True, float(prior_error_px)


def heading_to_target(
    current_center: MapPoint,
    target_center: MapPoint,
) -> float:
    delta_x = target_center[0] - current_center[0]
    delta_y = target_center[1] - current_center[1]
    return normalize_heading_degrees(math.degrees(math.atan2(delta_y, delta_x)) + 90.0)


def propagate_center_with_action(
    center: Optional[MapPoint],
    action: str,
    heading_degrees: float,
    step_size_px: float,
) -> Optional[MapPoint]:
    if center is None:
        return None

    move_x = 0.0
    move_y = 0.0
    if action == "forward":
        move_x, move_y = rotate_offset(0.0, -float(step_size_px), heading_degrees)
    elif action == "backward":
        move_x, move_y = rotate_offset(0.0, float(step_size_px), heading_degrees)
    elif action == "strafe_right":
        move_x, move_y = rotate_offset(float(step_size_px), 0.0, heading_degrees)
    elif action == "strafe_left":
        move_x, move_y = rotate_offset(-float(step_size_px), 0.0, heading_degrees)

    return (
        int(round(center[0] + move_x)),
        int(round(center[1] + move_y)),
    )


def advance_waypoint_index(
    current_index: int,
    position: MapPoint,
    waypoints: Sequence[MapPoint],
    acceptance_radius_px: float,
) -> int:
    next_index = int(current_index)
    while next_index < len(waypoints):
        distance_px = distance_between_points(position, waypoints[next_index])
        if distance_px > float(acceptance_radius_px):
            break
        next_index += 1
    return next_index


def update_waypoint_progress(
    current_index: int,
    consecutive_hits: int,
    position: Optional[MapPoint],
    waypoints: Sequence[MapPoint],
    acceptance_radius_px: float,
    localization_confidence: float,
    acceptance_confidence_threshold: float,
    required_consecutive_hits: int,
) -> Tuple[int, int]:
    if position is None or current_index >= len(waypoints):
        return int(current_index), 0

    if localization_confidence < float(acceptance_confidence_threshold):
        return int(current_index), 0

    distance_px = distance_between_points(position, waypoints[current_index])
    if distance_px > float(acceptance_radius_px):
        return int(current_index), 0

    consecutive_hits = int(consecutive_hits) + 1
    if consecutive_hits < int(max(1, required_consecutive_hits)):
        return int(current_index), consecutive_hits
    return int(current_index) + 1, 0


def choose_autonomous_action(
    estimated_center: Optional[MapPoint],
    target_center: Optional[MapPoint],
    heading_degrees: float,
    low_confidence_steps: int,
    waypoint_acceptance_radius_px: float,
    rotation_tolerance_deg: float,
    body_axis_deadband_px: float,
) -> str:
    if target_center is None:
        return "hold"
    if estimated_center is None:
        return "rotate_right" if (low_confidence_steps % 2) else "rotate_left"

    if low_confidence_steps >= 2:
        return "rotate_right" if (low_confidence_steps % 2) else "rotate_left"

    distance_px = distance_between_points(estimated_center, target_center)
    if distance_px <= float(waypoint_acceptance_radius_px):
        return "hold"

    desired_heading = heading_to_target(estimated_center, target_center)
    heading_error = shortest_heading_error_degrees(heading_degrees, desired_heading)
    if abs(heading_error) > float(rotation_tolerance_deg):
        return "rotate_right" if heading_error > 0.0 else "rotate_left"

    delta_x = target_center[0] - estimated_center[0]
    delta_y = target_center[1] - estimated_center[1]
    body_x, body_y = rotate_offset(delta_x, delta_y, -heading_degrees)

    if abs(body_x) > float(body_axis_deadband_px) and abs(body_x) > (abs(body_y) * 0.60):
        return "strafe_right" if body_x > 0.0 else "strafe_left"
    if body_y < (-float(body_axis_deadband_px) * 0.35):
        return "forward"
    if body_y > float(body_axis_deadband_px):
        return "backward"
    if abs(body_x) > (float(body_axis_deadband_px) * 0.35):
        return "strafe_right" if body_x > 0.0 else "strafe_left"
    return "forward" if body_y <= 0.0 else "backward"

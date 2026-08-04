"""Localization simulation between an observation map and a reference map.

Observation map:
    Simulates the live image source seen by the UAV. A moving crop is taken
    from this map.

Reference map:
    The map searched to estimate where that crop belongs.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

import cv2
import numpy as np
from tensorflow.keras.models import load_model

UP_KEYS = (ord("w"), ord("W"), 82, 2490368, 65362)
DOWN_KEYS = (ord("s"), ord("S"), 84, 2621440, 65364)
LEFT_KEYS = (ord("a"), ord("A"), 81, 2424832, 65361)
RIGHT_KEYS = (ord("d"), ord("D"), 83, 2555904, 65363)


@dataclass(frozen=True)
class SimulationConfig:
    reference_map_path: Path = Path("haritalar/ana_harita_urgup_30_cm__GPU_model_f32_k3_epoch_00001_sigmoid_(1_ 1)_06_10_2022_.h5.jpg_resized.jpg_geo.tif_geo.tif_r.tif")
    observation_map_path: Path = Path("parcalar/urgup_bingmap_utm_30_cm.tif")
    model_path: Path = Path("GPU_model_f32_k3_epoch_00001_sigmoid_(1_ 1)_06_10_2022_.h5")
    initial_row: int = 2000
    initial_col: int = 2000
    sample_window_size: int = 576
    model_input_size: int = 544
    crop_margin: int = 16
    step_size: int = 250
    display_size: Tuple[int, int] = (1000, 1000)
    match_method: int = cv2.TM_CCOEFF_NORMED
    prediction_color: Tuple[int, int, int] = (30, 30, 255)
    reference_color: Tuple[int, int, int] = (0, 204, 0)
    rectangle_thickness: int = 35
    status_text_color: Tuple[int, int, int] = (0, 255, 255)
    status_text_scale: float = 2.0
    status_text_thickness: int = 4
    dashboard_background_color: Tuple[int, int, int] = (18, 18, 24)
    panel_background_color: Tuple[int, int, int] = (32, 36, 42)
    panel_border_color: Tuple[int, int, int] = (78, 84, 92)
    panel_title_color: Tuple[int, int, int] = (230, 230, 230)
    predicted_path_color: Tuple[int, int, int] = (0, 170, 255)
    actual_path_color: Tuple[int, int, int] = (0, 255, 120)
    error_line_color: Tuple[int, int, int] = (255, 255, 0)
    dashboard_window_name: str = "Dashboard"
    observation_panel_title: str = "Gozlem Penceresi"
    autoencoder_panel_title: str = "Autoencoder Ciktisi"
    reference_panel_title: str = "Referans Harita"
    panel_padding: int = 20
    panel_gap: int = 20
    panel_inner_padding: int = 12
    panel_title_height: int = 38
    hud_font_scale: float = 0.72
    hud_font_thickness: int = 2
    path_history_limit: int = 120
    enable_local_search: bool = True
    local_search_margin: int = 1400
    global_search_interval: int = 10
    local_search_score_threshold: float = 0.35


@dataclass(frozen=True)
class ReferencePreviewState:
    panel_rect: Tuple[int, int, int, int]
    paste_x: int
    paste_y: int
    preview_width: int
    preview_height: int
    scale_x: float
    scale_y: float
    base_preview: np.ndarray


def load_grayscale_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), 0)
    if image is None:
        raise FileNotFoundError(f"Image could not be loaded: {path}")
    return image


def validate_config(config: SimulationConfig) -> None:
    output_size = get_output_template_size(config)
    if output_size <= 0:
        raise ValueError("crop_margin is too large for the selected model_input_size.")
    if config.sample_window_size <= 0 or config.model_input_size <= 0:
        raise ValueError("Window sizes must be positive.")


def load_assets(config: SimulationConfig) -> Tuple[np.ndarray, np.ndarray, object]:
    validate_config(config)

    reference_map = load_grayscale_image(config.reference_map_path)
    observation_map = load_grayscale_image(config.observation_map_path)
    if observation_map.shape != reference_map.shape:
        observation_map = cv2.resize(
            observation_map,
            (reference_map.shape[1], reference_map.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    model = load_model(str(config.model_path))
    return reference_map, observation_map, model


def get_output_template_size(config: SimulationConfig) -> int:
    return config.model_input_size - (2 * config.crop_margin)


def clamp_bottom_right(
    row: int,
    col: int,
    image_shape: Tuple[int, int],
    window_size: int,
) -> Tuple[int, int]:
    height, width = image_shape
    clamped_row = min(max(row, window_size), height)
    clamped_col = min(max(col, window_size), width)
    return clamped_row, clamped_col


def extract_observation_window(
    image: np.ndarray,
    row: int,
    col: int,
    window_size: int,
) -> Tuple[np.ndarray, int, int]:
    row, col = clamp_bottom_right(row, col, image.shape, window_size)
    window = image[row - window_size : row, col - window_size : col]
    return window, row, col


def prepare_observation_for_model(
    observation_map: np.ndarray,
    row: int,
    col: int,
    config: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    observation_window, row, col = extract_observation_window(
        observation_map,
        row=row,
        col=col,
        window_size=config.sample_window_size,
    )

    resized_window = cv2.resize(
        observation_window,
        (config.model_input_size, config.model_input_size),
        interpolation=cv2.INTER_NEAREST,
    )
    normalized_window = (resized_window.astype(np.float32) - 127.5) / 127.5
    model_input = normalized_window.reshape(
        -1,
        config.model_input_size,
        config.model_input_size,
        1,
    )
    return model_input, observation_window, row, col


def predict_with_autoencoder(
    model: object,
    model_input: np.ndarray,
    config: SimulationConfig,
) -> np.ndarray:
    prediction = model.predict(model_input, verbose=0)
    prediction = prediction.reshape(config.model_input_size, config.model_input_size)
    cropped_prediction = prediction[
        config.crop_margin : config.model_input_size - config.crop_margin,
        config.crop_margin : config.model_input_size - config.crop_margin,
    ]
    normalized_prediction = cv2.normalize(
        cropped_prediction,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    return normalized_prediction


def run_template_match(
    reference_map: np.ndarray,
    template: np.ndarray,
    match_method: int,
) -> Tuple[float, Tuple[int, int]]:
    result = cv2.matchTemplate(reference_map, template, match_method, None)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if match_method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        return float(min_val), min_loc
    return float(max_val), max_loc


def build_local_search_roi(
    reference_shape: Tuple[int, int],
    template_shape: Tuple[int, int],
    previous_top_left: Tuple[int, int],
    margin: int,
) -> Tuple[int, int, int, int]:
    reference_height, reference_width = reference_shape
    template_height, template_width = template_shape
    previous_x, previous_y = previous_top_left

    left = max(previous_x - margin, 0)
    top = max(previous_y - margin, 0)
    right = min(previous_x + template_width + margin, reference_width)
    bottom = min(previous_y + template_height + margin, reference_height)

    if (right - left) < template_width:
        left = max(0, min(left, reference_width - template_width))
        right = min(reference_width, left + template_width)
    if (bottom - top) < template_height:
        top = max(0, min(top, reference_height - template_height))
        bottom = min(reference_height, top + template_height)

    return left, top, right, bottom


def should_fallback_to_global_search(
    score: float,
    match_method: int,
    config: SimulationConfig,
) -> bool:
    if match_method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        return False
    return score < config.local_search_score_threshold


def localize_on_reference_map(
    reference_map: np.ndarray,
    template: np.ndarray,
    match_method: int,
    config: SimulationConfig,
    step_count: int,
    previous_top_left: Optional[Tuple[int, int]] = None,
) -> Tuple[float, Tuple[int, int], str]:
    should_run_global_search = (
        previous_top_left is None
        or not config.enable_local_search
        or config.global_search_interval <= 0
        or (step_count % config.global_search_interval) == 0
    )

    if should_run_global_search:
        score, top_left = run_template_match(reference_map, template, match_method)
        return score, top_left, "global"

    left, top, right, bottom = build_local_search_roi(
        reference_map.shape,
        template.shape,
        previous_top_left,
        config.local_search_margin,
    )
    local_region = reference_map[top:bottom, left:right]
    local_score, local_top_left = run_template_match(local_region, template, match_method)
    localized_top_left = (local_top_left[0] + left, local_top_left[1] + top)

    if should_fallback_to_global_search(local_score, match_method, config):
        score, top_left = run_template_match(reference_map, template, match_method)
        return score, top_left, "global-fallback"

    return local_score, localized_top_left, "local"


def extract_matched_patch(
    reference_map: np.ndarray,
    top_left: Tuple[int, int],
    template_shape: Tuple[int, int],
) -> np.ndarray:
    template_height, template_width = template_shape
    x1, y1 = top_left
    x2 = min(x1 + template_width, reference_map.shape[1])
    y2 = min(y1 + template_height, reference_map.shape[0])
    return reference_map[y1:y2, x1:x2]


def compute_reference_box(
    row: int,
    col: int,
    reference_map_shape: Tuple[int, int],
    config: SimulationConfig,
) -> Tuple[int, int, int, int]:
    output_size = get_output_template_size(config)
    legacy_scale = config.crop_margin / float(config.sample_window_size)

    scaled_col = int(col + (col * legacy_scale))
    scaled_row = int(row + (row * legacy_scale))

    left = scaled_col - output_size
    top = scaled_row - output_size
    right = scaled_col
    bottom = scaled_row

    height, width = reference_map_shape
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, width)
    bottom = min(bottom, height)

    return left, top, right, bottom


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def get_dashboard_layout(
    config: SimulationConfig,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    dashboard_width, dashboard_height = config.display_size
    left_panel_width = int(dashboard_width * 0.30)
    map_width = (
        dashboard_width
        - (3 * config.panel_padding)
        - config.panel_gap
        - left_panel_width
    )
    stacked_panel_height = (
        dashboard_height
        - (2 * config.panel_padding)
        - config.panel_gap
    ) // 2

    observation_rect = (
        config.panel_padding,
        config.panel_padding,
        left_panel_width,
        stacked_panel_height,
    )
    autoencoder_rect = (
        config.panel_padding,
        config.panel_padding + stacked_panel_height + config.panel_gap,
        left_panel_width,
        stacked_panel_height,
    )
    map_rect = (
        config.panel_padding + left_panel_width + config.panel_gap,
        config.panel_padding,
        map_width,
        dashboard_height - (2 * config.panel_padding),
    )
    return observation_rect, autoencoder_rect, map_rect


def get_panel_content_rect(
    panel_rect: Tuple[int, int, int, int],
    config: SimulationConfig,
) -> Tuple[int, int, int, int]:
    x, y, width, height = panel_rect
    return (
        x + config.panel_inner_padding,
        y + config.panel_title_height + config.panel_inner_padding,
        width - (2 * config.panel_inner_padding),
        height - config.panel_title_height - (2 * config.panel_inner_padding),
    )


def calculate_fit_size(
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
) -> Tuple[int, int]:
    scale = min(float(target_width) / image_width, float(target_height) / image_height)
    fitted_width = max(1, int(round(image_width * scale)))
    fitted_height = max(1, int(round(image_height * scale)))
    return fitted_width, fitted_height


def resize_to_fit(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    if image_height == 0 or image_width == 0:
        return np.zeros((target_height, target_width, 3), dtype=np.uint8)

    resized_width, resized_height = calculate_fit_size(
        image_width,
        image_height,
        target_width,
        target_height,
    )

    scale = min(float(target_width) / image_width, float(target_height) / image_height)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_NEAREST
    return cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)


def draw_panel_frame(
    canvas: np.ndarray,
    panel_rect: Tuple[int, int, int, int],
    title: str,
    config: SimulationConfig,
) -> None:
    x, y, width, height = panel_rect
    cv2.rectangle(
        canvas,
        (x, y),
        (x + width, y + height),
        config.panel_background_color,
        -1,
    )
    cv2.rectangle(
        canvas,
        (x, y),
        (x + width, y + height),
        config.panel_border_color,
        2,
    )

    cv2.putText(
        canvas,
        title,
        (x + 12, y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        config.panel_title_color,
        2,
    )

 

def draw_panel(
    canvas: np.ndarray,
    image: np.ndarray,
    panel_rect: Tuple[int, int, int, int],
    title: str,
    config: SimulationConfig,
) -> None:
    draw_panel_frame(canvas, panel_rect, title, config)

    content_x, content_y, content_width, content_height = get_panel_content_rect(
        panel_rect,
        config,
    )

    fitted = resize_to_fit(ensure_bgr(image), content_width, content_height)
    fitted_height, fitted_width = fitted.shape[:2]
    paste_x = content_x + (content_width - fitted_width) // 2
    paste_y = content_y + (content_height - fitted_height) // 2
    canvas[paste_y : paste_y + fitted_height, paste_x : paste_x + fitted_width] = fitted


def get_template_center(
    top_left: Tuple[int, int],
    template_shape: Tuple[int, int],
) -> Tuple[int, int]:
    template_height, template_width = template_shape
    return (
        top_left[0] + (template_width // 2),
        top_left[1] + (template_height // 2),
    )


def get_reference_box_center(reference_box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    return (
        (reference_box[0] + reference_box[2]) // 2,
        (reference_box[1] + reference_box[3]) // 2,
    )


def compute_error_pixels(
    predicted_center: Tuple[int, int],
    actual_center: Tuple[int, int],
) -> float:
    return float(
        np.hypot(
            predicted_center[0] - actual_center[0],
            predicted_center[1] - actual_center[1],
        )
    )


def create_reference_preview_state(
    reference_map: np.ndarray,
    panel_rect: Tuple[int, int, int, int],
    config: SimulationConfig,
) -> ReferencePreviewState:
    content_x, content_y, content_width, content_height = get_panel_content_rect(
        panel_rect,
        config,
    )
    preview_width, preview_height = calculate_fit_size(
        reference_map.shape[1],
        reference_map.shape[0],
        content_width,
        content_height,
    )
    preview_image = cv2.resize(
        reference_map,
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )
    preview_image = ensure_bgr(preview_image)

    paste_x = content_x + (content_width - preview_width) // 2
    paste_y = content_y + (content_height - preview_height) // 2

    return ReferencePreviewState(
        panel_rect=panel_rect,
        paste_x=paste_x,
        paste_y=paste_y,
        preview_width=preview_width,
        preview_height=preview_height,
        scale_x=preview_width / float(reference_map.shape[1]),
        scale_y=preview_height / float(reference_map.shape[0]),
        base_preview=preview_image,
    )


def draw_path(
    image: np.ndarray,
    points: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    if len(points) < 2:
        return
    cv2.polylines(image, [np.array(points, dtype=np.int32)], False, color, thickness)


def scale_point_to_preview(
    point: Tuple[int, int],
    preview_state: ReferencePreviewState,
) -> Tuple[int, int]:
    x = int(round(point[0] * preview_state.scale_x))
    y = int(round(point[1] * preview_state.scale_y))
    x = min(max(x, 0), preview_state.preview_width - 1)
    y = min(max(y, 0), preview_state.preview_height - 1)
    return x, y


def scale_box_to_preview(
    box: Tuple[int, int, int, int],
    preview_state: ReferencePreviewState,
) -> Tuple[int, int, int, int]:
    left = int(round(box[0] * preview_state.scale_x))
    top = int(round(box[1] * preview_state.scale_y))
    right = int(round(box[2] * preview_state.scale_x))
    bottom = int(round(box[3] * preview_state.scale_y))
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    return left, top, right, bottom


def draw_scaled_path(
    preview_image: np.ndarray,
    points: List[Tuple[int, int]],
    preview_state: ReferencePreviewState,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    if len(points) < 2:
        return
    scaled_points = [scale_point_to_preview(point, preview_state) for point in points]
    cv2.polylines(
        preview_image,
        [np.array(scaled_points, dtype=np.int32)],
        False,
        color,
        thickness,
    )


def draw_hud(
    canvas: np.ndarray,
    map_rect: Tuple[int, int, int, int],
    score: float,
    observation_cursor: Tuple[int, int],
    predicted_top_left: Tuple[int, int],
    actual_top_left: Tuple[int, int],
    error_pixels: float,
    step_count: int,
    last_direction: str,
    search_mode: str,
    config: SimulationConfig,
) -> None:
    x, y, width, height = map_rect
    hud_width = min(340, width - 24)
    hud_height = 200
    hud_x = x + 12
    hud_y = y + 50

    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (hud_x, hud_y),
        (hud_x + hud_width, hud_y + hud_height),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.42, canvas, 0.58, 0, canvas)
    cv2.rectangle(
        canvas,
        (hud_x, hud_y),
        (hud_x + hud_width, hud_y + hud_height),
        config.panel_border_color,
        2,
    )

    direction_label = last_direction if last_direction else "idle"
    hud_lines = [
        f"Skor: {score:.4f}",
        f"Imlec: row={observation_cursor[0]} col={observation_cursor[1]}",
        f"Tahmin: x={predicted_top_left[0]} y={predicted_top_left[1]}",
        f"Gercek: x={actual_top_left[0]} y={actual_top_left[1]}",
        f"Hata: {error_pixels:.1f} px",
        f"Adim: {step_count}",
        f"Yon: {direction_label}",
        f"Arama: {search_mode}",
    ]

    for index, line in enumerate(hud_lines):
        cv2.putText(
            canvas,
            line,
            (hud_x + 14, hud_y + 30 + (index * 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            config.hud_font_scale,
            config.panel_title_color,
            config.hud_font_thickness,
        )

    help_text = "Ok tuslari/WASD: hareket | Q veya ESC: cikis"
    cv2.putText(
        canvas,
        help_text,
        (x + 12, y + height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        config.panel_title_color,
        2,
    )


def draw_localization_dashboard(
    observation_rect: Tuple[int, int, int, int],
    autoencoder_rect: Tuple[int, int, int, int],
    reference_preview_state: ReferencePreviewState,
    observation_window: np.ndarray,
    autoencoder_output: np.ndarray,
    predicted_top_left: Tuple[int, int],
    reference_box: Tuple[int, int, int, int],
    score: float,
    observation_cursor: Tuple[int, int],
    step_count: int,
    last_direction: str,
    search_mode: str,
    predicted_history: List[Tuple[int, int]],
    actual_history: List[Tuple[int, int]],
    config: SimulationConfig,
) -> np.ndarray:
    dashboard_width, dashboard_height = config.display_size
    canvas = np.full(
        (dashboard_height, dashboard_width, 3),
        config.dashboard_background_color,
        dtype=np.uint8,
    )
    map_rect = reference_preview_state.panel_rect
    annotated_reference_preview = reference_preview_state.base_preview.copy()

    predicted_center = get_template_center(predicted_top_left, autoencoder_output.shape)
    actual_center = get_reference_box_center(reference_box)
    scaled_predicted_top_left = scale_point_to_preview(
        predicted_top_left,
        reference_preview_state,
    )
    scaled_predicted_bottom_right = scale_point_to_preview(
        (
            predicted_top_left[0] + autoencoder_output.shape[1],
            predicted_top_left[1] + autoencoder_output.shape[0],
        ),
        reference_preview_state,
    )
    scaled_reference_box = scale_box_to_preview(reference_box, reference_preview_state)
    scaled_predicted_center = scale_point_to_preview(predicted_center, reference_preview_state)
    scaled_actual_center = scale_point_to_preview(actual_center, reference_preview_state)

    overlay_thickness = max(1, int(round(6 * reference_preview_state.scale_x)))
    marker_radius = max(4, int(round(14 * reference_preview_state.scale_x)))

    draw_scaled_path(
        annotated_reference_preview,
        actual_history,
        reference_preview_state,
        config.actual_path_color,
        max(2, overlay_thickness),
    )
    draw_scaled_path(
        annotated_reference_preview,
        predicted_history,
        reference_preview_state,
        config.predicted_path_color,
        max(2, overlay_thickness),
    )

    cv2.rectangle(
        annotated_reference_preview,
        scaled_predicted_top_left,
        scaled_predicted_bottom_right,
        config.prediction_color,
        max(2, overlay_thickness),
    )
    cv2.rectangle(
        annotated_reference_preview,
        (scaled_reference_box[0], scaled_reference_box[1]),
        (scaled_reference_box[2], scaled_reference_box[3]),
        config.reference_color,
        max(2, overlay_thickness),
    )
    cv2.circle(
        annotated_reference_preview,
        scaled_predicted_center,
        marker_radius,
        config.prediction_color,
        -1,
    )
    cv2.circle(
        annotated_reference_preview,
        scaled_actual_center,
        marker_radius,
        config.reference_color,
        -1,
    )
    cv2.line(
        annotated_reference_preview,
        scaled_predicted_center,
        scaled_actual_center,
        config.error_line_color,
        max(2, overlay_thickness),
    )

    draw_panel(
        canvas,
        observation_window,
        observation_rect,
        config.observation_panel_title,
        config,
    )
    draw_panel(
        canvas,
        autoencoder_output,
        autoencoder_rect,
        config.autoencoder_panel_title,
        config,
    )
    draw_panel_frame(canvas, map_rect, config.reference_panel_title, config)
    canvas[
        reference_preview_state.paste_y : reference_preview_state.paste_y
        + reference_preview_state.preview_height,
        reference_preview_state.paste_x : reference_preview_state.paste_x
        + reference_preview_state.preview_width,
    ] = annotated_reference_preview

    draw_hud(
        canvas=canvas,
        map_rect=map_rect,
        score=score,
        observation_cursor=observation_cursor,
        predicted_top_left=predicted_top_left,
        actual_top_left=(reference_box[0], reference_box[1]),
        error_pixels=compute_error_pixels(predicted_center, actual_center),
        step_count=step_count,
        last_direction=last_direction,
        search_mode=search_mode,
        config=config,
    )

    return canvas


def move_observation_cursor(
    row: int,
    col: int,
    direction: str,
    image_shape: Tuple[int, int],
    config: SimulationConfig,
) -> Tuple[int, int]:
    if direction == "down":
        row += config.step_size
    elif direction == "up":
        row -= config.step_size
    elif direction == "right":
        col += config.step_size
    elif direction == "left":
        col -= config.step_size

    return clamp_bottom_right(row, col, image_shape, config.sample_window_size)


def get_direction_from_key(key: int) -> str:
    if key in UP_KEYS:
        return "up"
    if key in DOWN_KEYS:
        return "down"
    if key in LEFT_KEYS:
        return "left"
    if key in RIGHT_KEYS:
        return "right"
    return ""


def print_localization_status(
    score: float,
    predicted_top_left: Tuple[int, int],
    reference_box: Tuple[int, int, int, int],
    row: int,
    col: int,
    error_pixels: float,
    step_count: int,
    last_direction: str,
    search_mode: str,
) -> None:
    print(f"cursor=(row={row}, col={col})")
    print(f"predicted_top_left={predicted_top_left} score={score:.4f}")
    print(
        f"actual_top_left=({reference_box[0]}, {reference_box[1]}) "
        f"error={error_pixels:.1f}px step={step_count} "
        f"direction={last_direction} search={search_mode}"
    )
    print(
        "reference_box="
        f"({reference_box[0]}, {reference_box[1]}, {reference_box[2]}, {reference_box[3]})"
    )


def main() -> None:
    config = SimulationConfig()
    reference_map, observation_map, model = load_assets(config)
    predicted_history = []
    actual_history = []
    step_count = 0
    last_direction = "idle"
    last_predicted_top_left = None

    observation_rect, autoencoder_rect, map_rect = get_dashboard_layout(config)
    reference_preview_state = create_reference_preview_state(reference_map, map_rect, config)

    row, col = clamp_bottom_right(
        config.initial_row,
        config.initial_col,
        observation_map.shape,
        config.sample_window_size,
    )

    cv2.namedWindow(config.dashboard_window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        config.dashboard_window_name,
        config.display_size[0],
        config.display_size[1],
    )

    while True:
        model_input, observation_window, row, col = prepare_observation_for_model(
            observation_map,
            row,
            col,
            config,
        )
        template = predict_with_autoencoder(model, model_input, config)
        score, predicted_top_left, search_mode = localize_on_reference_map(
            reference_map,
            template,
            config.match_method,
            config,
            step_count,
            last_predicted_top_left,
        )
        reference_box = compute_reference_box(
            row,
            col,
            reference_map.shape,
            config,
        )
        predicted_center = get_template_center(predicted_top_left, template.shape)
        actual_center = get_reference_box_center(reference_box)
        error_pixels = compute_error_pixels(predicted_center, actual_center)

        predicted_history.append(predicted_center)
        actual_history.append(actual_center)
        predicted_history = predicted_history[-config.path_history_limit :]
        actual_history = actual_history[-config.path_history_limit :]

        print_localization_status(
            score,
            predicted_top_left,
            reference_box,
            row,
            col,
            error_pixels,
            step_count,
            last_direction,
            search_mode,
        )
        dashboard = draw_localization_dashboard(
            observation_rect=observation_rect,
            autoencoder_rect=autoencoder_rect,
            reference_preview_state=reference_preview_state,
            observation_window=observation_window,
            autoencoder_output=template,
            predicted_top_left=predicted_top_left,
            reference_box=reference_box,
            score=score,
            observation_cursor=(row, col),
            step_count=step_count,
            last_direction=last_direction,
            search_mode=search_mode,
            predicted_history=predicted_history,
            actual_history=actual_history,
            config=config,
        )
        cv2.imshow(config.dashboard_window_name, dashboard)
        last_predicted_top_left = predicted_top_left

        key = cv2.waitKeyEx(0)
        if key in (27, ord("q"), ord("Q")):
            break

        direction = get_direction_from_key(key)
        if direction:
            row, col = move_observation_cursor(
                row,
                col,
                direction,
                observation_map.shape,
                config,
            )
            last_direction = direction
            step_count += 1
        else:
            print(f"Unrecognized key code: {key}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

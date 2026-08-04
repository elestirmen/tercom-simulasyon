"""Triple-template localization simulation.

Observation map:
    Simulates the live image source seen by the UAV. Three neighboring crops
    are extracted from this map.

Reference map:
    The map searched independently by each crop. The final predicted position
    is estimated from the intersection of the matched boxes.
"""

import argparse
import concurrent.futures
import csv
import dataclasses
import importlib.util
import json
import math
import os
import random
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import IO, List, Optional, Tuple

_HAS_QT = bool(
    importlib.util.find_spec("PySide6") or importlib.util.find_spec("PyQt5")
)

try:
    from PIL import ImageFont as _PILFont, ImageDraw as _PILDraw, Image as _PILImg
    _HAS_PIL = True
    _PIL_FONT_PATHS = [
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    _pil_font_cache: dict = {}

    def _get_pil_font(size_px: int):
        if size_px not in _pil_font_cache:
            for _fp in _PIL_FONT_PATHS:
                if os.path.exists(_fp):
                    try:
                        _pil_font_cache[size_px] = _PILFont.truetype(_fp, size_px)
                        break
                    except Exception:
                        continue
            else:
                _pil_font_cache[size_px] = _PILFont.load_default()
        return _pil_font_cache[size_px]

except ImportError:
    _HAS_PIL = False

_TR_ASCII = str.maketrans(
    "şıığüöçŞİĞÜÖÇ",
    "siiguocSIGUOC",
)


def _put_text_tr(canvas, text: str, baseline_xy: tuple, size_px: int, color: tuple, thickness: int = 1) -> None:
    """cv2.putText wrapper with Unicode support via PIL (Turkish chars)."""
    needs_unicode = any(ord(c) > 127 for c in text)
    if _HAS_PIL and needs_unicode:
        x0, y0 = baseline_xy
        font = _get_pil_font(size_px)
        try:
            bbox = font.getbbox(text)
            off_x, off_y = -bbox[0], -bbox[1]
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except AttributeError:
            tw, th = font.getsize(text)
            off_x, off_y = 0, 0
        try:
            ascent = font.getmetrics()[0]
        except Exception:
            ascent = size_px
        draw_y = y0 - ascent
        cx0 = max(0, x0 - 2)
        cy0 = max(0, draw_y - 2)
        cx1 = min(canvas.shape[1], x0 + tw + 4)
        cy1 = min(canvas.shape[0], draw_y + th + 4)
        if cx0 >= cx1 or cy0 >= cy1:
            return
        patch = canvas[cy0:cy1, cx0:cx1]
        pil = _PILImg.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        _PILDraw.Draw(pil).text(
            (x0 - cx0 + off_x, draw_y - cy0 + off_y),
            text,
            fill=(color[2], color[1], color[0]),
            font=font,
        )
        canvas[cy0:cy1, cx0:cx1] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return
    cv2.putText(
        canvas,
        text.translate(_TR_ASCII),
        baseline_xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        size_px / 28.0,
        color,
        thickness,
        cv2.LINE_AA,
    )

from gps_denied_autonomy import (
    LocalizationQuality,
    choose_autonomous_action,
    compute_localization_quality,
    fuse_measurement_with_prior,
    propagate_center_with_action,
    update_waypoint_progress,
)
from simulation_core import (
    ConstantVelocityKalmanFilter,
    RasterioGraySource,
    close_raster_source,
)

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

import cv2
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pyproj import Transformer

UP_KEYS = (ord("w"), ord("W"), 82, 2490368, 65362)
DOWN_KEYS = (ord("s"), ord("S"), 84, 2621440, 65364)
LEFT_KEYS = (ord("a"), ord("A"), 81, 2424832, 65361)
RIGHT_KEYS = (ord("d"), ord("D"), 83, 2555904, 65363)
ROTATE_LEFT_KEYS = (ord("q"), ord("Q"))
ROTATE_RIGHT_KEYS = (ord("e"), ord("E"))
ALTITUDE_UP_KEYS = (ord("+"), ord("="), 43, 61, 107)
ALTITUDE_DOWN_KEYS = (ord("-"), ord("_"), 45, 95, 109)
EXIT_KEYS = (27, ord("x"), ord("X"))
AUTONOMOUS_TOGGLE_KEYS = (ord("p"), ord("P"))
KALMAN_TOGGLE_KEYS = (ord("k"), ord("K"))
NORM_CYCLE_KEYS = (ord("n"), ord("N"))
OBS_WINDOW_CYCLE_KEYS = (ord("v"), ord("V"))
REF_PATCH_TOGGLE_KEYS = (ord("m"), ord("M"))
TRANSLATION_ACTIONS = ("forward", "backward", "strafe_left", "strafe_right")

_NORM_MODES = ("HAM", "CLAHE", "HISTEQ", "EDGE")
_OBS_WINDOW_SMALL = 272
MAP_DIRECTORY = Path("haritalar")
MAP_FILE_SUFFIXES = (".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp")
MODEL_DIRECTORY = Path("model")
MODEL_FILE_SUFFIXES = (".h5", ".hdf5", ".keras")


def apply_observation_norm(image: np.ndarray, mode: str) -> np.ndarray:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    if mode == "HISTEQ":
        return cv2.equalizeHist(gray)
    if mode == "EDGE":
        edges = cv2.Laplacian(gray, cv2.CV_32F)
        edges = np.abs(edges)
        max_v = edges.max()
        if max_v > 0:
            edges = (edges / max_v * 255.0).astype(np.uint8)
        else:
            edges = np.zeros_like(gray)
        return edges
    return gray
COMPASS_LABELS = ("K", "KD", "D", "GD", "G", "GB", "B", "KB")
_QT_KEY_MAP = {
    16777235: 65362,  # Qt.Key_Up
    16777237: 65364,  # Qt.Key_Down
    16777234: 65361,  # Qt.Key_Left
    16777236: 65363,  # Qt.Key_Right
    16777216: 27,     # Qt.Key_Escape
}
TEMPLATE_COLORS = (
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
)
UI_COLORS = {
    "panel_bg": (30, 33, 42),
    "panel_border": (90, 96, 112),
    "btn_on": (56, 124, 245),
    "btn_off": (58, 60, 70),
    "btn_hover_on": (78, 146, 255),
    "btn_hover_off": (80, 84, 96),
    "toggle_on": (76, 175, 80),
    "toggle_off": (120, 120, 130),
    "toggle_knob": (255, 255, 255),
    "text_primary": (245, 247, 252),
    "text_shadow": (0, 0, 0),
    "accent": (66, 133, 244),
    "header_bg": (45, 48, 65),
    "collapse_btn": (55, 60, 80),
    "collapse_hover": (80, 85, 110),
}


@dataclass(frozen=True)
class SimulationConfig:
    scenario_mode: str = "normal"  # "normal" veya "irtifa"
    reference_map_path: Path = MAP_DIRECTORY
    observation_map_path: Path = Path("parcalar/urgup_bingmap_30cm_utm.tif")
    observation_georef_path: Path = Path("parcalar/urgup_bingmap_30cm_utm.tif")
    # None ise observation_georef_path kullanılır. Eski sabitlenmiş türetilmiş
    # raster adı artık veri paketinde bulunmadığından taşınabilir varsayılan budur.
    observation_grid_georef_path: Optional[Path] = None
    dem_path: Path = Path("ana_harita_urgup_30_cm_utm_elevation.tif")
    model_path: Path = MODEL_DIRECTORY
    sample_window_size: int = 544
    model_input_size: int = 544
    crop_margin: int = 16
    template_size: int = 512
    template_offset: int = 100
    initial_row: int = 2500
    initial_col: int = 2500
    initial_position_known: bool = True
    random_start: bool = True
    random_start_middle_band_ratio: float = 0.50
    step_size: int = 250
    initial_heading_degrees: float = 0.0
    rotation_step_degrees: float = 15.0
    initial_altitude_agl_m: float = 110.0
    altitude_step_m: float = 10.0
    min_altitude_agl_m: float = 30.0
    max_altitude_agl_m: float = 250.0
    minimum_patch_agl_m: float = 5.0
    reference_map_gsd_cm_per_px: float = 29.85
    camera_sensor_width_mm: float = 13.2
    camera_focal_length_mm: float = 8.8
    virtual_camera_width_px: int = 544
    align_observation_to_reference_grid: bool = True
    stream_rasters: bool = True
    display_size: Tuple[int, int] = (1000, 1000)
    mission_control_canvas_size: Tuple[int, int] = (1600, 900)
    left_panel_width_ratio: float = 0.20
    right_info_panel_width: int = 180
    match_method: int = cv2.TM_CCOEFF_NORMED
    use_parallel_matching: bool = True
    use_pyramid_matching: bool = True
    coarse_scale: float = 0.5
    roi_pad_factor: float = 0.4
    base_search_window_size: int = 2048
    max_search_window_size: int = 15000
    search_window_growth_step: int = 100
    search_window_failure_growth: int = 500
    kalman_window_growth_factor: float = 0.4
    triplet_alignment_tolerance_px: float = 45.0
    global_refresh_interval: int = 0
    dashboard_background_color: Tuple[int, int, int] = (18, 18, 24)
    panel_background_color: Tuple[int, int, int] = (32, 36, 42)
    panel_border_color: Tuple[int, int, int] = (78, 84, 92)
    panel_title_color: Tuple[int, int, int] = (230, 230, 230)
    actual_path_color: Tuple[int, int, int] = (0, 255, 120)
    predicted_path_color: Tuple[int, int, int] = (0, 170, 255)
    actual_intersection_color: Tuple[int, int, int] = (0, 204, 0)
    predicted_intersection_color: Tuple[int, int, int] = (0, 215, 255)
    error_line_color: Tuple[int, int, int] = (255, 255, 0)
    dashboard_window_name: str = "Dashboard"
    observation_panel_title: str = "Anlık Görüntü"
    template_panel_title: str = "Model Çıktısı"
    reference_panel_title: str = "Referans Harita"
    panel_padding: int = 20
    panel_gap: int = 20
    panel_inner_padding: int = 12
    panel_title_height: int = 38
    hud_font_scale: float = 0.70
    hud_font_thickness: int = 2
    path_history_limit: int = 120
    observation_context_margin: int = 120
    template_strip_tile_size: int = 180
    template_strip_gap: int = 12
    rectangle_thickness: int = 3
    search_window_color: Tuple[int, int, int] = (0, 165, 255)
    heading_indicator_color: Tuple[int, int, int] = (255, 220, 0)
    ui_buttons_enabled: bool = True
    ui_button_font_scale: float = 0.82
    ui_button_thickness: int = 2
    ui_button_scale: float = 0.40
    show_info_panel: bool = True
    show_trajectory: bool = True
    show_roi_frame: bool = True
    show_tm_boxes: bool = True
    show_heading_arrow: bool = True
    show_observation_boxes: bool = True
    reference_viewport_base_size: int = 6000
    reference_viewport_padding: int = 600
    reference_viewport_search_padding: int = 320
    reference_viewport_search_min_size: int = 4200
    diagnostic_benchmark_enabled: bool = False
    diagnostic_benchmark_only: bool = False
    diagnostic_output_dir: Path = Path("diagnostics")
    diagnostic_tile_size: int = 256
    diagnostic_benchmark_points: Tuple[Tuple[int, int], ...] = (
        (12000, 15000),
        (8000, 10000),
        (16000, 22000),
        (6000, 24000),
        (18000, 12000),
    )
    # --- Lokalizasyon kalitesi eşikleri ---
    localization_score_threshold: float = 0.24
    localization_confidence_threshold: float = 0.31
    localization_spread_threshold_px: float = 120.0
    localization_peak_margin_threshold: float = 0.03
    localization_template_std_threshold: float = 2.0
    localization_require_strict_triplet: bool = True
    global_recovery_after_low_confidence_steps: int = 3
    global_recovery_min_window_size: int = 6000
    progressive_global_recovery: bool = True
    # --- Sensör füzyonu ---
    sensor_fusion_blend_gain: float = 0.75
    max_visual_jump_px: float = 600.0
    # --- Kalman filtresi ---
    kalman_enabled: bool = True
    kalman_process_noise: float = 50.0
    kalman_measurement_noise: float = 80.0
    # --- CSV adım loglama ---
    log_csv_enabled: bool = True
    log_csv_path: Optional[Path] = None
    # --- Otonom mod ---
    autonomous_mode_enabled: bool = False
    autonomous_step_interval_ms: int = 400
    autonomous_min_step_size_px: float = 40.0
    autonomous_low_confidence_recovery_steps: int = 4
    autonomous_stuck_max_steps: int = 4
    autonomous_stuck_distance_epsilon_px: float = 20.0
    waypoint_acceptance_radius_px: float = 150.0
    waypoint_rotation_tolerance_deg: float = 15.0
    waypoint_body_axis_deadband_px: float = 50.0
    waypoint_required_consecutive_hits: int = 2
    waypoint_acceptance_confidence_threshold: float = 0.40


@dataclass(frozen=True)
class ReferencePreviewState:
    panel_rect: Tuple[int, int, int, int]
    paste_x: int
    paste_y: int
    preview_width: int
    preview_height: int
    scale_x: float
    scale_y: float
    viewport_left: int
    viewport_top: int
    viewport_width: int
    viewport_height: int
    base_preview: np.ndarray


@dataclass(frozen=True)
class TemplateMatchEvidence:
    score: float
    top_left: Tuple[int, int]
    peak_margin: float
    template_stddev: float


@dataclass
class TerrainContext:
    dem_dataset: object
    observation_dataset: object
    observation_to_dem_transformer: object
    resized_observation_shape: Tuple[int, int]


@dataclass(frozen=True)
class AltitudeSimulationState:
    altitude_agl_m: float
    altitude_msl_m: float
    center_ground_elevation_m: float
    patch_ground_elevations_m: Tuple[float, float, float]
    patch_agl_m: Tuple[float, float, float]
    patch_scale_factors: Tuple[float, float, float]
    center_gsd_cm_per_px: float


PositionKalmanFilter = ConstantVelocityKalmanFilter


# ---------------------------------------------------------------------------
# CSV loglama
# ---------------------------------------------------------------------------

_CSV_FIELDNAMES = (
    "adim", "zaman", "row", "col", "baslik_deg", "irtifa_m",
    "aksiyon", "skor_a", "skor_b", "skor_c",
    "kesisim_modu", "arama_modu", "match_backend",
    "gercek_x", "gercek_y", "ham_tahmin_x", "ham_tahmin_y",
    "kalman_x", "kalman_y",
    "hata_px", "kalman_hata_px",
    "hata_m", "kalman_hata_m",
    "guven", "skor_min", "skor_ort", "yayilma_px",
    "tepe_marji_min", "sablon_std_min", "geometri_siki",
    "guvenilir", "guvenilirlik_neden", "arama_pencere_px", "islem_ms",
)


def _imshow_keepratio(
    window_name: str,
    image: np.ndarray,
    lb_state: dict,
) -> None:
    """Görüntüyü letterbox ile göster; pencere yeniden boyutlandırılırsa oran korunur.
    lb_state sözlüğüne scale/x_off/y_off yazar — fare callback ters dönüşüm için kullanır."""
    img_h, img_w = image.shape[:2]
    try:
        _, _, win_w, win_h = cv2.getWindowImageRect(window_name)
    except Exception:
        win_w, win_h = 0, 0

    if win_w <= 0 or win_h <= 0:
        lb_state.update(scale=1.0, x_off=0, y_off=0)
        cv2.imshow(window_name, image)
        return

    scale = min(win_w / img_w, win_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    x_off = (win_w - new_w) // 2
    y_off = (win_h - new_h) // 2
    lb_state.update(scale=scale, x_off=x_off, y_off=y_off)

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    padded = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    padded[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    cv2.imshow(window_name, padded)


def _open_csv_log(
    config: "SimulationConfig",
) -> Tuple[Optional[csv.DictWriter], Optional[IO]]:
    if not config.log_csv_enabled:
        return None, None
    log_path = config.log_csv_path
    if log_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path("log_simulasyon_%s.csv" % timestamp)
    csv_file = open(str(log_path), "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=_CSV_FIELDNAMES)
    writer.writeheader()
    return writer, csv_file


def _write_csv_row(
    writer: csv.DictWriter,
    step_count: int,
    row: int,
    col: int,
    heading_degrees: float,
    altitude_state: "AltitudeSimulationState",
    last_action: str,
    score_values: List[float],
    intersection_mode: str,
    search_mode: str,
    match_backend: str,
    actual_center: Tuple[int, int],
    raw_predicted_center: Tuple[int, int],
    kalman_center: Optional[Tuple[int, int]],
    actual_center_ref: Tuple[int, int],
    quality: "LocalizationQuality",
    search_window_size: int,
    processing_ms: float,
) -> None:
    kalman_x = kalman_center[0] if kalman_center is not None else ""
    kalman_y = kalman_center[1] if kalman_center is not None else ""
    kalman_err = (
        math.hypot(kalman_center[0] - actual_center_ref[0], kalman_center[1] - actual_center_ref[1])
        if kalman_center is not None else ""
    )
    raw_err = math.hypot(
        raw_predicted_center[0] - actual_center_ref[0],
        raw_predicted_center[1] - actual_center_ref[1],
    )
    gsd_cm = (
        altitude_state.center_gsd_cm_per_px
        if altitude_state.center_gsd_cm_per_px > 0.0
        else 0.0
    )
    raw_err_m = round(raw_err * gsd_cm / 100.0, 2) if gsd_cm > 0.0 else ""
    kalman_err_m = (
        round(float(kalman_err) * gsd_cm / 100.0, 2)
        if (kalman_err != "" and gsd_cm > 0.0) else ""
    )
    scores = list(score_values) + [0.0] * max(0, 3 - len(score_values))
    writer.writerow({
        "adim": step_count,
        "zaman": datetime.now().strftime("%H:%M:%S.%f")[:-3],
        "row": row,
        "col": col,
        "baslik_deg": round(heading_degrees, 2),
        "irtifa_m": round(altitude_state.altitude_agl_m, 1),
        "aksiyon": last_action,
        "skor_a": round(scores[0], 5),
        "skor_b": round(scores[1], 5),
        "skor_c": round(scores[2], 5),
        "kesisim_modu": intersection_mode,
        "arama_modu": search_mode,
        "match_backend": match_backend,
        "gercek_x": actual_center[0],
        "gercek_y": actual_center[1],
        "ham_tahmin_x": raw_predicted_center[0],
        "ham_tahmin_y": raw_predicted_center[1],
        "kalman_x": kalman_x,
        "kalman_y": kalman_y,
        "hata_px": round(raw_err, 2),
        "kalman_hata_px": round(kalman_err, 2) if kalman_err != "" else "",
        "hata_m": raw_err_m,
        "kalman_hata_m": kalman_err_m,
        "guven": round(quality.confidence, 4),
        "skor_min": round(quality.score_floor, 4),
        "skor_ort": round(quality.score_mean, 4),
        "yayilma_px": round(quality.center_spread_px, 2),
        "tepe_marji_min": round(quality.peak_margin_floor, 5),
        "sablon_std_min": round(quality.template_std_floor, 3),
        "geometri_siki": int(quality.strict_alignment),
        "guvenilir": int(quality.is_reliable),
        "guvenilirlik_neden": quality.reason,
        "arama_pencere_px": search_window_size,
        "islem_ms": round(float(processing_ms), 2),
    })


# ---------------------------------------------------------------------------
# Argparse & config override
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Üçlü şablon eşleme lokalizasyon simülasyonu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--senaryo", default=None, choices=["normal", "irtifa"],
                        help="Simülasyon senaryosu")
    parser.add_argument("--referans", default=None, metavar="YOL",
                        help="Referans harita dosyasi veya haritalar klasoru")
    parser.add_argument("--gozlem", default=None, metavar="YOL",
                        help="Gözlem harita dosyası")
    parser.add_argument("--model", default=None, metavar="YOL",
                        help="Model dosyasi veya model klasoru")
    parser.add_argument("--adim-px", type=int, default=None, metavar="N",
                        help="Hareket adım büyüklüğü (piksel)")
    parser.add_argument("--arama-penceresi", type=int, default=None, metavar="N",
                        help="Başlangıç arama penceresi (piksel)")
    parser.add_argument("--raster-stream", dest="stream_rasters", action="store_true", default=None,
                        help="GeoTIFF dosyalarını düşük bellekli pencere erişimiyle kullan")
    parser.add_argument("--raster-bellek", dest="stream_rasters", action="store_false",
                        help="GeoTIFF dosyalarını eski davranışla tamamen belleğe al")
    parser.add_argument("--kalman", action="store_true", default=None,
                        help="Kalman filtresini aktif et")
    parser.add_argument("--kalman-yok", dest="kalman", action="store_false",
                        help="Kalman filtresini devre dışı bırak")
    parser.add_argument("--csv-yok", action="store_true",
                        help="CSV loglamayı devre dışı bırak")
    parser.add_argument("--csv-dosya", default=None, metavar="YOL",
                        help="CSV log dosyası yolu")
    parser.add_argument("--otonom-aralik-ms", type=int, default=None, metavar="MS",
                        help="Otonom mod adım aralığı (ms)")
    parser.add_argument("--rastgele-baslangic", action="store_true", default=None,
                        help="Rastgele başlangıç konumu kullan")
    parser.add_argument("--sabit-baslangic", dest="rastgele_baslangic", action="store_false",
                        help="Sabit başlangıç konumu kullan")
    return parser.parse_args()


def _apply_args_to_config(
    config: "SimulationConfig",
    args: argparse.Namespace,
) -> "SimulationConfig":
    overrides = {}
    if args.senaryo is not None:
        overrides["scenario_mode"] = args.senaryo
    if args.referans is not None:
        overrides["reference_map_path"] = Path(args.referans)
    if args.gozlem is not None:
        overrides["observation_map_path"] = Path(args.gozlem)
    if args.model is not None:
        overrides["model_path"] = Path(args.model)
    if args.adim_px is not None:
        overrides["step_size"] = args.adim_px
    if args.arama_penceresi is not None:
        overrides["base_search_window_size"] = args.arama_penceresi
    if args.stream_rasters is not None:
        overrides["stream_rasters"] = args.stream_rasters
    if args.kalman is not None:
        overrides["kalman_enabled"] = args.kalman
    if args.csv_yok:
        overrides["log_csv_enabled"] = False
    if args.csv_dosya is not None:
        overrides["log_csv_path"] = Path(args.csv_dosya)
    if args.otonom_aralik_ms is not None:
        overrides["autonomous_step_interval_ms"] = args.otonom_aralik_ms
    if args.rastgele_baslangic is not None:
        overrides["random_start"] = args.rastgele_baslangic
    if not overrides:
        return config
    return dataclasses.replace(config, **overrides)


def load_grayscale_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), 0)
    if image is None:
        raise FileNotFoundError("Image could not be loaded: %s" % path)
    return image


def is_georaster_path(path: Path) -> bool:
    return path.suffix.lower() in (".tif", ".tiff")


def read_raster_dataset_as_grayscale(dataset: object) -> np.ndarray:
    if dataset.count <= 0:
        raise ValueError("Raster has no bands: %s" % getattr(dataset, "name", "<unknown>"))

    if dataset.count == 1:
        grayscale = dataset.read(1)
    else:
        rgb_band_count = min(3, dataset.count)
        rgb = np.moveaxis(dataset.read(list(range(1, rgb_band_count + 1))), 0, -1)
        if rgb.shape[2] == 1:
            grayscale = rgb[:, :, 0]
        else:
            grayscale = cv2.cvtColor(rgb[:, :, :3], cv2.COLOR_RGB2GRAY)

    if grayscale.dtype == np.uint8:
        return grayscale
    return cv2.normalize(
        grayscale,
        None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )


def load_grayscale_raster(path: Path) -> np.ndarray:
    with rio.open(str(path)) as dataset:
        return read_raster_dataset_as_grayscale(dataset)


def load_observation_aligned_to_reference_grid(
    observation_path: Path,
    reference_path: Path,
) -> np.ndarray:
    with rio.open(str(observation_path)) as observation_dataset, rio.open(
        str(reference_path)
    ) as reference_dataset:
        source_gray = read_raster_dataset_as_grayscale(observation_dataset).astype(
            np.float32
        )
        aligned = np.zeros(
            (reference_dataset.height, reference_dataset.width),
            dtype=np.float32,
        )
        reproject(
            source=source_gray,
            destination=aligned,
            src_transform=observation_dataset.transform,
            src_crs=observation_dataset.crs,
            dst_transform=reference_dataset.transform,
            dst_crs=reference_dataset.crs,
            resampling=Resampling.nearest,
            dst_nodata=0.0,
        )
    return np.clip(aligned, 0.0, 255.0).astype(np.uint8)


def is_supported_map_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MAP_FILE_SUFFIXES


def find_map_in_directory(map_dir: Path) -> Path:
    if not map_dir.exists():
        raise FileNotFoundError(
            "Harita klasoru bulunamadi: %s. Harita dosyasini bu klasore koyun "
            "veya --referans ile dosya/klasor verin." % map_dir
        )
    if not map_dir.is_dir():
        raise ValueError("Harita yolu klasor degil: %s" % map_dir)

    candidates = [
        path
        for path in map_dir.iterdir()
        if is_supported_map_file(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            "Harita klasorunde desteklenen harita dosyasi bulunamadi: %s "
            "(desteklenen uzantilar: %s)"
            % (map_dir, ", ".join(MAP_FILE_SUFFIXES))
        )

    candidates.sort(
        key=lambda path: (path.stat().st_mtime, path.name.lower()),
        reverse=True,
    )
    return candidates[0]


def resolve_map_path(map_path: Path) -> Path:
    if map_path.is_dir():
        return find_map_in_directory(map_path)
    if map_path.exists():
        if is_supported_map_file(map_path):
            return map_path
        raise ValueError(
            "Desteklenmeyen harita dosyasi: %s (desteklenen uzantilar: %s)"
            % (map_path, ", ".join(MAP_FILE_SUFFIXES))
        )
    raise FileNotFoundError(
        "Harita yolu bulunamadi: %s. Harita dosyasini '%s' klasorune koyun "
        "veya --referans ile dosya/klasor verin." % (map_path, MAP_DIRECTORY)
    )


def is_supported_model_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in MODEL_FILE_SUFFIXES


def find_model_in_directory(model_dir: Path) -> Path:
    if not model_dir.exists():
        raise FileNotFoundError(
            "Model klasoru bulunamadi: %s. Model dosyasini bu klasore koyun "
            "veya --model ile dosya/klasor verin." % model_dir
        )
    if not model_dir.is_dir():
        raise ValueError("Model yolu klasor degil: %s" % model_dir)

    candidates = [
        path
        for path in model_dir.iterdir()
        if is_supported_model_file(path)
    ]
    if not candidates:
        raise FileNotFoundError(
            "Model klasorunde desteklenen model dosyasi bulunamadi: %s "
            "(desteklenen uzantilar: %s)"
            % (model_dir, ", ".join(MODEL_FILE_SUFFIXES))
        )

    candidates.sort(
        key=lambda path: (path.stat().st_mtime, path.name.lower()),
        reverse=True,
    )
    return candidates[0]


def resolve_model_path(model_path: Path) -> Path:
    if model_path.is_dir():
        return find_model_in_directory(model_path)
    if model_path.exists():
        if is_supported_model_file(model_path):
            return model_path
        raise ValueError(
            "Desteklenmeyen model dosyasi: %s (desteklenen uzantilar: %s)"
            % (model_path, ", ".join(MODEL_FILE_SUFFIXES))
        )
    raise FileNotFoundError(
        "Model yolu bulunamadi: %s. Model dosyasini '%s' klasorune koyun "
        "veya --model ile dosya/klasor verin." % (model_path, MODEL_DIRECTORY)
    )


def resolve_config_model_path(config: SimulationConfig) -> SimulationConfig:
    resolved_model_path = resolve_model_path(config.model_path)
    if resolved_model_path != config.model_path:
        print("Model secildi: %s" % resolved_model_path)
        return dataclasses.replace(config, model_path=resolved_model_path)
    return config


def resolve_config_paths(config: SimulationConfig) -> SimulationConfig:
    resolved_reference_map_path = resolve_map_path(config.reference_map_path)
    if resolved_reference_map_path != config.reference_map_path:
        print("Harita secildi: %s" % resolved_reference_map_path)
        config = dataclasses.replace(
            config,
            reference_map_path=resolved_reference_map_path,
        )
    return resolve_config_model_path(config)


def load_model_compat(model_path: Path):
    # TensorFlow is intentionally imported only when the worker actually loads
    # the model.  CLI help and the native window can therefore appear without
    # paying TensorFlow's multi-second import cost.
    from tensorflow.keras.layers import Conv2DTranspose
    from tensorflow.keras.models import load_model

    class CompatConv2DTranspose(Conv2DTranspose):
        @classmethod
        def from_config(cls, layer_config):
            compat_config = dict(layer_config or {})
            compat_config.pop("groups", None)
            return super().from_config(compat_config)

    try:
        return load_model(str(model_path), compile=False)
    except TypeError as exc:
        if "Conv2DTranspose" in str(exc) and "groups" in str(exc):
            return load_model(
                str(model_path),
                compile=False,
                custom_objects={"Conv2DTranspose": CompatConv2DTranspose},
            )
        raise


def get_output_template_size(config: SimulationConfig) -> int:
    return config.model_input_size - (2 * config.crop_margin)


def get_observation_model_scale(config: SimulationConfig) -> float:
    return float(config.sample_window_size) / float(config.model_input_size)


def _scale_model_pixels(value: float, config: SimulationConfig, minimum: int = 1) -> int:
    return max(minimum, int(round(float(value) * get_observation_model_scale(config))))


def get_effective_template_size(config: SimulationConfig) -> int:
    return _scale_model_pixels(config.template_size, config)


def get_effective_template_offset(config: SimulationConfig) -> int:
    return _scale_model_pixels(config.template_offset, config)


def resize_templates_to_effective_size(
    templates: List[np.ndarray],
    config: SimulationConfig,
    scale_factors: Optional[Tuple[float, ...]] = None,
) -> List[np.ndarray]:
    base_size = get_effective_template_size(config)
    factors = scale_factors or tuple(1.0 for _ in templates)
    if len(factors) != len(templates):
        raise ValueError("Her şablon için bir ölçek katsayısı gerekli.")
    resized_templates = []
    for template, scale_factor in zip(templates, factors, strict=True):
        target_size = max(1, int(round(base_size * max(0.05, float(scale_factor)))))
        if template.shape[:2] == (target_size, target_size):
            resized_templates.append(template)
            continue
        interpolation = (
            cv2.INTER_AREA
            if target_size < max(template.shape[:2])
            else cv2.INTER_LINEAR
        )
        resized_templates.append(
            cv2.resize(
                template,
                (target_size, target_size),
                interpolation=interpolation,
            )
        )
    return resized_templates


def validate_config(config: SimulationConfig) -> None:
    normalize_scenario_mode(config.scenario_mode)
    output_template_size = get_output_template_size(config)
    if output_template_size <= 0:
        raise ValueError("crop_margin model_input_size icin fazla buyuk.")
    if output_template_size != config.template_size:
        raise ValueError(
            "template_size=%d ama model cikti boyutu=%d."
            % (config.template_size, output_template_size)
        )
    if config.sample_window_size <= 0 or config.model_input_size <= 0:
        raise ValueError("Model pencere boyutlari pozitif olmali.")


def normalize_heading_degrees(heading_degrees: float) -> float:
    return float(heading_degrees % 360.0)


def rotate_image_offset(
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


def get_heading_vector(heading_degrees: float) -> Tuple[float, float]:
    return rotate_image_offset(0.0, -1.0, heading_degrees)


def get_heading_label(heading_degrees: float) -> str:
    normalized_heading = normalize_heading_degrees(heading_degrees)
    direction_index = int(((normalized_heading + 22.5) % 360.0) // 45.0)
    return "%.1f° %s" % (normalized_heading, COMPASS_LABELS[direction_index])


def get_action_label(action: str) -> str:
    action_labels = {
        "forward": "ileri",
        "backward": "geri",
        "strafe_left": "yan-sol",
        "strafe_right": "yan-sag",
        "rotate_left": "don-sol",
        "rotate_right": "don-sag",
        "altitude_up": "irtifa+",
        "altitude_down": "irtifa-",
    }
    return action_labels.get(action, "bekle")


def is_translation_action(action: str) -> bool:
    return action in TRANSLATION_ACTIONS


def get_autonomous_step_size(
    waypoint_distance_px: Optional[float],
    config: SimulationConfig,
) -> float:
    max_step = max(1.0, float(config.step_size))
    if waypoint_distance_px is None:
        return max_step

    acceptance_radius = max(1.0, float(config.waypoint_acceptance_radius_px))
    if waypoint_distance_px <= acceptance_radius:
        return 0.0

    min_step = max(1.0, min(max_step, float(config.autonomous_min_step_size_px)))
    approach_step = float(waypoint_distance_px) - (acceptance_radius * 0.65)
    return max(min_step, min(max_step, approach_step))


def get_triplet_rotation_margin(config: SimulationConfig) -> int:
    return get_effective_template_offset(config)


def get_scaled_observation_window_size(
    scale_factor: float,
    config: SimulationConfig,
) -> int:
    window_size = int(
        round(config.sample_window_size * max(0.05, float(scale_factor)))
    )
    window_size = max(32, window_size)
    if (window_size % 2) != (config.sample_window_size % 2):
        window_size += 1
    return window_size


def get_rotated_capture_size(config: SimulationConfig) -> int:
    max_scale_factor = 1.0
    if is_altitude_scenario(config):
        max_scale_factor = max(
            1.0,
            compute_scale_factor_for_altitude(config.max_altitude_agl_m, config),
        )
    max_window_size = get_scaled_observation_window_size(
        max_scale_factor * 1.15,
        config,
    )
    capture_size = int(math.ceil(max_window_size * math.sqrt(2.0))) + 2
    if (capture_size % 2) != (config.sample_window_size % 2):
        capture_size += 1
    return capture_size


def get_observation_cursor_limits(
    image_shape: Tuple[int, int],
    config: SimulationConfig,
) -> Tuple[int, int, int]:
    height, width = image_shape
    triplet_margin = get_triplet_rotation_margin(config)
    capture_half = get_rotated_capture_size(config) // 2
    sample_half = config.sample_window_size // 2
    minimum = sample_half + capture_half + triplet_margin
    maximum_row = max(minimum, height + sample_half - capture_half - triplet_margin)
    maximum_col = max(minimum, width + sample_half - capture_half - triplet_margin)
    return minimum, maximum_row, maximum_col


def format_heading_label(heading_degrees: float) -> str:
    normalized_heading = normalize_heading_degrees(heading_degrees)
    direction_index = int(((normalized_heading + 22.5) % 360.0) // 45.0)
    return "%.1f deg %s" % (normalized_heading, COMPASS_LABELS[direction_index])


def normalize_scenario_mode(scenario_mode: str) -> str:
    normalized_mode = str(scenario_mode).strip().lower()
    if normalized_mode in ("normal", "standart"):
        return "normal"
    if normalized_mode in ("irtifa", "altitude", "elevation"):
        return "irtifa"
    raise ValueError(
        "scenario_mode gecersiz: %s. Desteklenen degerler: normal, irtifa."
        % scenario_mode
    )


def is_altitude_scenario(config: SimulationConfig) -> bool:
    return normalize_scenario_mode(config.scenario_mode) == "irtifa"


def get_scenario_label(config: SimulationConfig) -> str:
    return normalize_scenario_mode(config.scenario_mode)


def clamp_altitude_agl(altitude_agl_m: float, config: SimulationConfig) -> float:
    return float(
        min(
            max(altitude_agl_m, config.min_altitude_agl_m),
            config.max_altitude_agl_m,
        )
    )


def compute_virtual_camera_gsd_cm_per_px(
    altitude_agl_m: float,
    config: SimulationConfig,
) -> float:
    effective_altitude = max(float(altitude_agl_m), float(config.minimum_patch_agl_m))
    return (
        float(config.camera_sensor_width_mm)
        * effective_altitude
        * 100.0
        / (float(config.camera_focal_length_mm) * float(config.virtual_camera_width_px))
    )


def compute_scale_factor_for_altitude(
    altitude_agl_m: float,
    config: SimulationConfig,
) -> float:
    return compute_virtual_camera_gsd_cm_per_px(altitude_agl_m, config) / float(
        config.reference_map_gsd_cm_per_px
    )


def build_normal_altitude_state(
    patch_count: int,
    config: SimulationConfig,
    altitude_agl_m: float = 0.0,
) -> AltitudeSimulationState:
    # Normal modda observation_map referans harita ile aynı GSD'de (scale=1.0).
    # İrtifa sadece gösterim için saklanır; eşleşme ölçeğini etkilemez.
    zero_tuple = tuple(0.0 for _ in range(patch_count))
    scale_tuple = tuple(1.0 for _ in range(patch_count))
    return AltitudeSimulationState(
        altitude_agl_m=float(altitude_agl_m),
        altitude_msl_m=float(altitude_agl_m),
        center_ground_elevation_m=0.0,
        patch_ground_elevations_m=zero_tuple,
        patch_agl_m=tuple(altitude_agl_m for _ in range(patch_count)),
        patch_scale_factors=scale_tuple,
        center_gsd_cm_per_px=float(config.reference_map_gsd_cm_per_px),
    )


def load_terrain_context(
    resized_observation_shape: Tuple[int, int],
    config: SimulationConfig,
) -> TerrainContext:
    if not config.dem_path.exists():
        raise FileNotFoundError("DEM could not be found: %s" % config.dem_path)
    observation_grid_georef_path = (
        config.observation_grid_georef_path or config.observation_georef_path
    )
    if not observation_grid_georef_path.exists():
        raise FileNotFoundError(
            "Observation grid georeference raster could not be found: %s"
            % observation_grid_georef_path
        )

    observation_dataset = rio.open(str(observation_grid_georef_path))
    dem_dataset = rio.open(str(config.dem_path))
    observation_to_dem_transformer = Transformer.from_crs(
        observation_dataset.crs,
        dem_dataset.crs,
        always_xy=True,
    )
    return TerrainContext(
        dem_dataset=dem_dataset,
        observation_dataset=observation_dataset,
        observation_to_dem_transformer=observation_to_dem_transformer,
        resized_observation_shape=resized_observation_shape,
    )


def close_terrain_context(terrain_context: Optional[TerrainContext]) -> None:
    if terrain_context is None:
        return
    try:
        terrain_context.observation_dataset.close()
    except Exception:
        pass
    try:
        terrain_context.dem_dataset.close()
    except Exception:
        pass


def sample_ground_elevation_at_resized_pixel(
    pixel_x: float,
    pixel_y: float,
    terrain_context: TerrainContext,
) -> float:
    resized_height, resized_width = terrain_context.resized_observation_shape
    source_col = float(pixel_x) * (
        terrain_context.observation_dataset.width / float(resized_width)
    )
    source_row = float(pixel_y) * (
        terrain_context.observation_dataset.height / float(resized_height)
    )

    source_col = min(
        max(source_col, 0.0),
        max(0.0, terrain_context.observation_dataset.width - 1.0),
    )
    source_row = min(
        max(source_row, 0.0),
        max(0.0, terrain_context.observation_dataset.height - 1.0),
    )

    world_x, world_y = terrain_context.observation_dataset.transform * (
        source_col + 0.5,
        source_row + 0.5,
    )
    dem_x, dem_y = terrain_context.observation_to_dem_transformer.transform(
        world_x,
        world_y,
    )

    dem_bounds = terrain_context.dem_dataset.bounds
    if not (
        dem_bounds.left <= dem_x <= dem_bounds.right
        and dem_bounds.bottom <= dem_y <= dem_bounds.top
    ):
        raise ValueError(
            "Point is outside DEM bounds: x=%.3f y=%.3f" % (dem_x, dem_y)
        )

    sample = next(terrain_context.dem_dataset.sample([(dem_x, dem_y)]))[0]
    if not np.isfinite(sample):
        raise ValueError(
            "DEM sample is invalid at x=%.3f y=%.3f" % (dem_x, dem_y)
        )
    return float(sample)


def compute_altitude_simulation_state(
    observation_boxes: List[Tuple[int, int, int, int]],
    altitude_agl_m: float,
    terrain_context: TerrainContext,
    config: SimulationConfig,
) -> AltitudeSimulationState:
    patch_ground_elevations = []
    for box in observation_boxes:
        center_x = box[0] + (box[2] / 2.0)
        center_y = box[1] + (box[3] / 2.0)
        patch_ground_elevations.append(
            sample_ground_elevation_at_resized_pixel(
                center_x,
                center_y,
                terrain_context,
            )
        )

    center_ground_elevation_m = patch_ground_elevations[1]
    altitude_agl_m = clamp_altitude_agl(altitude_agl_m, config)
    altitude_msl_m = center_ground_elevation_m + altitude_agl_m
    patch_agl_values = tuple(
        max(config.minimum_patch_agl_m, altitude_msl_m - elevation_m)
        for elevation_m in patch_ground_elevations
    )
    patch_scale_factors = tuple(
        compute_scale_factor_for_altitude(patch_agl_m, config)
        for patch_agl_m in patch_agl_values
    )
    return AltitudeSimulationState(
        altitude_agl_m=altitude_agl_m,
        altitude_msl_m=float(altitude_msl_m),
        center_ground_elevation_m=float(center_ground_elevation_m),
        patch_ground_elevations_m=tuple(float(value) for value in patch_ground_elevations),
        patch_agl_m=tuple(float(value) for value in patch_agl_values),
        patch_scale_factors=tuple(float(value) for value in patch_scale_factors),
        center_gsd_cm_per_px=float(
            compute_virtual_camera_gsd_cm_per_px(altitude_agl_m, config)
        ),
    )


def _draw_alpha_panel(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    x0 = max(0, min(int(x0), image.shape[1] - 1))
    x1 = max(0, min(int(x1), image.shape[1] - 1))
    y0 = max(0, min(int(y0), image.shape[0] - 1))
    y1 = max(0, min(int(y1), image.shape[0] - 1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = image[y0:y1, x0:x1]
    overlay = roi.copy()
    cv2.rectangle(overlay, (0, 0), (x1 - x0, y1 - y0), color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _draw_rounded_rect(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    radius: int,
    color: Tuple[int, int, int],
    thickness: int = -1,
) -> None:
    radius = max(0, min(int(radius), (x1 - x0) // 2, (y1 - y0) // 2))
    if radius < 2:
        cv2.rectangle(image, (x0, y0), (x1, y1), color, thickness)
        return
    cv2.rectangle(image, (x0 + radius, y0), (x1 - radius, y1), color, thickness)
    cv2.rectangle(image, (x0, y0 + radius), (x1, y1 - radius), color, thickness)
    cv2.circle(image, (x0 + radius, y0 + radius), radius, color, thickness)
    cv2.circle(image, (x1 - radius, y0 + radius), radius, color, thickness)
    cv2.circle(image, (x0 + radius, y1 - radius), radius, color, thickness)
    cv2.circle(image, (x1 - radius, y1 - radius), radius, color, thickness)


def _draw_alpha_rounded_panel(
    image: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    radius: int,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    x0 = max(0, min(int(x0), image.shape[1] - 1))
    x1 = max(0, min(int(x1), image.shape[1] - 1))
    y0 = max(0, min(int(y0), image.shape[0] - 1))
    y1 = max(0, min(int(y1), image.shape[0] - 1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = image[y0:y1, x0:x1]
    overlay = roi.copy()
    _draw_rounded_rect(overlay, 0, 0, x1 - x0, y1 - y0, radius, color, -1)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def _draw_toggle_switch(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    is_on: bool,
    view_scale: float,
) -> None:
    radius = height // 2
    background = UI_COLORS["toggle_on"] if is_on else UI_COLORS["toggle_off"]
    cv2.rectangle(image, (x + radius, y), (x + width - radius, y + height), background, -1)
    cv2.circle(image, (x + radius, y + radius), radius, background, -1)
    cv2.circle(image, (x + width - radius, y + radius), radius, background, -1)

    knob_pad = max(3, int(round(4 * view_scale)))
    knob_x = x + width - radius - knob_pad if is_on else x + radius + knob_pad
    cv2.circle(image, (knob_x, y + radius), radius - knob_pad, UI_COLORS["toggle_knob"], -1)

    border = max(1, int(round(1.5 * view_scale)))
    cv2.rectangle(image, (x + radius, y), (x + width - radius, y + height), (200, 200, 210), border)
    cv2.circle(image, (x + radius, y + radius), radius, (200, 200, 210), border)
    cv2.circle(image, (x + width - radius, y + radius), radius, (200, 200, 210), border)


def _draw_text_with_shadow(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    cv2.putText(
        image,
        text,
        (position[0] + 2, position[1] + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        UI_COLORS["text_shadow"],
        thickness + 1,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_info_panel(
    image: np.ndarray,
    lines: List[str],
    top_left: Tuple[int, int],
    font_scale: float,
    thickness: int,
    alpha: float = 0.55,
    padding: int = 18,
    corner_radius: int = 18,
) -> None:
    if not lines:
        return

    sizes = [cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0] for line in lines]
    max_width = max(width for width, _ in sizes)
    max_height = max(height for _, height in sizes)
    line_gap = int(max_height * 1.6)
    x, y = top_left
    panel_width = max_width + (2 * padding)
    panel_height = (line_gap * len(lines)) + padding
    panel_x0 = max(0, x - padding)
    panel_y0 = max(0, y - max_height - padding)
    panel_x1 = min(image.shape[1] - 1, panel_x0 + panel_width)
    panel_y1 = min(image.shape[0] - 1, panel_y0 + panel_height)

    _draw_alpha_rounded_panel(
        image,
        panel_x0,
        panel_y0,
        panel_x1,
        panel_y1,
        corner_radius,
        UI_COLORS["panel_bg"],
        alpha,
    )
    _draw_rounded_rect(
        image,
        panel_x0,
        panel_y0,
        panel_x1,
        panel_y1,
        corner_radius,
        UI_COLORS["panel_border"],
        max(1, thickness // 3),
    )

    for index, line in enumerate(lines):
        _draw_text_with_shadow(
            image,
            line,
            (x, y + (index * line_gap)),
            font_scale,
            UI_COLORS["text_primary"],
            thickness,
        )


def _build_runtime_buttons() -> List[dict]:
    return [
        {"key": "_panel_collapsed", "label": "", "hotkey": "H", "rect": (0, 0, 0, 0), "is_collapse": True},
        {"key": "autonomous_mode", "label": "Otonom Mod", "hotkey": "P", "rect": (0, 0, 0, 0)},
        {"key": "kalman_on", "label": "Kalman", "hotkey": "K", "rect": (0, 0, 0, 0)},
        {"key": "info_panel", "label": "Bilgi", "hotkey": "B", "rect": (0, 0, 0, 0)},
        {"key": "trajectory", "label": "Trajektori", "hotkey": "T", "rect": (0, 0, 0, 0)},
        {"key": "roi_frame", "label": "ROI Cerceve", "hotkey": "O", "rect": (0, 0, 0, 0)},
        {"key": "tm_boxes", "label": "TM Kutular", "hotkey": "R", "rect": (0, 0, 0, 0)},
        {"key": "heading_arrow", "label": "Yon Oku", "hotkey": "Y", "rect": (0, 0, 0, 0)},
        {"key": "observation_boxes", "label": "Gozlem Kutulari", "hotkey": "G", "rect": (0, 0, 0, 0)},
        {"key": "ref_patch", "label": "Haritada Bulunan", "hotkey": "M", "rect": (0, 0, 0, 0)},
    ]


def _draw_runtime_buttons(
    image: np.ndarray,
    ui_state: dict,
    buttons: List[dict],
    config: SimulationConfig,
) -> None:
    if image is None or not buttons:
        return

    view_scale = max(0.25, float(config.ui_button_scale))
    margin = max(12, int(round(18 * view_scale)))
    gap = max(6, int(round(10 * view_scale)))
    button_width = max(180, int(round(280 * view_scale)))
    button_height = max(38, int(round(56 * view_scale)))
    panel_padding = max(8, int(round(14 * view_scale)))
    header_height = max(28, int(round(38 * view_scale)))
    corner_radius = max(6, int(round(12 * view_scale)))
    button_radius = max(4, int(round(8 * view_scale)))
    font_scale = max(0.52, float(config.ui_button_font_scale) * view_scale)
    font_thickness = max(1, int(round(config.ui_button_thickness * view_scale)))
    toggle_width = max(46, int(round(72 * view_scale)))
    toggle_height = max(20, int(round(30 * view_scale)))

    x0 = margin
    y0 = margin
    hover_key = ui_state.get("_hover_key")
    is_collapsed = bool(ui_state.get("_panel_collapsed", False))

    collapse_button = None
    content_buttons = []
    for button in buttons:
        if button.get("is_collapse"):
            collapse_button = button
        else:
            content_buttons.append(button)

    if is_collapsed:
        size = max(34, int(round(44 * view_scale)))
        x1 = x0 + size
        y1 = y0 + size
        fill = UI_COLORS["collapse_hover"] if hover_key == "_panel_collapsed" else UI_COLORS["collapse_btn"]
        _draw_alpha_rounded_panel(image, x0, y0, x1, y1, button_radius, UI_COLORS["panel_bg"], 0.65)
        _draw_rounded_rect(image, x0, y0, x1, y1, button_radius, fill, -1)
        _draw_rounded_rect(image, x0, y0, x1, y1, button_radius, UI_COLORS["panel_border"], 1)
        if collapse_button is not None:
            collapse_button["rect"] = (x0, y0, size, size)
        bar_width = int(size * 0.45)
        bar_height = max(2, int(round(3 * view_scale)))
        bar_x = x0 + ((size - bar_width) // 2)
        bar_gap = max(4, int(round(6 * view_scale)))
        bar_center = y0 + (size // 2)
        for delta in (-bar_gap, 0, bar_gap):
            y_bar = bar_center + delta - (bar_height // 2)
            cv2.rectangle(image, (bar_x, y_bar), (bar_x + bar_width, y_bar + bar_height), UI_COLORS["text_primary"], -1)
        return

    panel_width = button_width + (2 * panel_padding)
    panel_height = (
        header_height
        + (2 * panel_padding)
        + (len(content_buttons) * button_height)
        + (max(0, len(content_buttons) - 1) * gap)
    )
    px0 = x0 - panel_padding
    py0 = y0 - panel_padding
    px1 = px0 + panel_width
    py1 = py0 + panel_height

    _draw_alpha_rounded_panel(image, px0, py0, px1, py1, corner_radius, UI_COLORS["panel_bg"], 0.60)
    _draw_rounded_rect(image, px0, py0, px1, py1, corner_radius, UI_COLORS["panel_border"], 1)
    _draw_alpha_rounded_panel(image, px0, py0, px1, py0 + header_height + panel_padding, corner_radius, UI_COLORS["header_bg"], 0.35)
    _draw_text_with_shadow(
        image,
        "GORUNUM",
        (x0 + int(round(8 * view_scale)), y0 + int(round(header_height * 0.62))),
        max(0.52, font_scale * 0.92),
        UI_COLORS["accent"],
        font_thickness,
    )

    if collapse_button is not None:
        size = max(26, int(round(34 * view_scale)))
        cb_x0 = px1 - size - max(4, int(round(6 * view_scale)))
        cb_y0 = py0 + max(4, int(round(6 * view_scale)))
        cb_x1 = cb_x0 + size
        cb_y1 = cb_y0 + size
        fill = UI_COLORS["collapse_hover"] if hover_key == "_panel_collapsed" else UI_COLORS["collapse_btn"]
        _draw_rounded_rect(image, cb_x0, cb_y0, cb_x1, cb_y1, button_radius // 2, fill, -1)
        _draw_rounded_rect(image, cb_x0, cb_y0, cb_x1, cb_y1, button_radius // 2, UI_COLORS["panel_border"], 1)
        collapse_button["rect"] = (cb_x0, cb_y0, size, size)
        line_width = int(size * 0.5)
        line_height = max(2, int(round(3 * view_scale)))
        line_x = cb_x0 + ((size - line_width) // 2)
        line_y = cb_y0 + (size // 2) - (line_height // 2)
        cv2.rectangle(image, (line_x, line_y), (line_x + line_width, line_y + line_height), UI_COLORS["text_primary"], -1)

    current_y = y0 + header_height
    for button in content_buttons:
        key = button["key"]
        is_on = bool(ui_state.get(key, False))
        is_hovered = hover_key == key
        fill = UI_COLORS["btn_hover_on"] if is_on and is_hovered else UI_COLORS["btn_on"] if is_on else UI_COLORS["btn_hover_off"] if is_hovered else UI_COLORS["btn_off"]
        edge = UI_COLORS["accent"] if is_hovered else UI_COLORS["panel_border"]
        button["rect"] = (x0, current_y, button_width, button_height)
        _draw_rounded_rect(image, x0, current_y, x0 + button_width, current_y + button_height, button_radius, fill, -1)
        _draw_rounded_rect(image, x0, current_y, x0 + button_width, current_y + button_height, button_radius, edge, 1)
        label = "%s [%s]" % (button["label"], button["hotkey"])
        _draw_text_with_shadow(
            image,
            label,
            (x0 + max(10, int(round(14 * view_scale))), current_y + int(round(button_height * 0.63))),
            font_scale,
            UI_COLORS["text_primary"],
            font_thickness,
        )
        toggle_x = x0 + button_width - toggle_width - max(8, int(round(10 * view_scale)))
        toggle_y = current_y + ((button_height - toggle_height) // 2)
        _draw_toggle_switch(image, toggle_x, toggle_y, toggle_width, toggle_height, is_on, view_scale)
        current_y += button_height + gap


# ---------------------------------------------------------------------------
# Sütun genişliği — sürüklenebilir ayraçlar + kalıcı düzen (ui_layout.json)
# ---------------------------------------------------------------------------
_UI_LAYOUT_PATH = Path("ui_layout.json")
_MIN_MAP_WIDTH_PX = 220


def _compute_panel_widths(
    config: "SimulationConfig",
    ui_state: Optional[dict] = None,
) -> Tuple[int, int]:
    """Sol ve sağ sütun genişliklerini (piksel) sınırlandırarak hesaplar.

    ui_state verilirse kullanıcının sürüklediği değerler kullanılır; aksi halde
    config varsayılanları geçerlidir. Orta (harita) sütununun daima en az
    _MIN_MAP_WIDTH_PX kalması garanti edilir.
    """
    dashboard_width = int(config.display_size[0])
    fixed = (4 * config.panel_padding) + (2 * config.panel_gap)
    available = dashboard_width - fixed

    if ui_state is not None:
        left_ratio = float(ui_state.get("left_panel_ratio", config.left_panel_width_ratio))
        right_width = int(ui_state.get("right_panel_width", config.right_info_panel_width))
    else:
        left_ratio = float(config.left_panel_width_ratio)
        right_width = int(config.right_info_panel_width)

    left_width = int(round(dashboard_width * left_ratio))

    min_left = max(120, int(dashboard_width * 0.12))
    max_left = int(dashboard_width * 0.42)
    min_right = 130
    max_right = int(dashboard_width * 0.34)

    left_width = max(min_left, min(max_left, left_width))
    right_width = max(min_right, min(max_right, right_width))

    # Harita sütunu minimumun altına düşerse önce sağ, sonra sol sütunu kıs.
    overflow = _MIN_MAP_WIDTH_PX - (available - left_width - right_width)
    if overflow > 0:
        shrink_right = min(overflow, right_width - min_right)
        right_width -= shrink_right
        overflow -= shrink_right
        if overflow > 0:
            left_width = max(min_left, left_width - overflow)
    return left_width, right_width


def _clamp_dragged_panel_width(
    which: str,
    proposed: float,
    fixed_other_width: int,
    config: "SimulationConfig",
) -> int:
    """Sürüklenen sütun genişliğini, diğer sütun sabit kabul edilerek sınırlar."""
    dashboard_width = int(config.display_size[0])
    fixed = (4 * config.panel_padding) + (2 * config.panel_gap)
    available = dashboard_width - fixed
    if which == "left":
        lo = max(120, int(dashboard_width * 0.12))
        hi = int(dashboard_width * 0.42)
    else:
        lo = 130
        hi = int(dashboard_width * 0.34)
    hi = min(hi, available - fixed_other_width - _MIN_MAP_WIDTH_PX)
    lo = min(lo, hi)
    return int(max(lo, min(hi, proposed)))


def _get_splitter_zones(
    config: "SimulationConfig",
    ui_state: Optional[dict],
) -> dict:
    """Sol/sağ ayraç çubuklarının merkez-x ve dikey aralığını döndürür."""
    left_width, right_width = _compute_panel_widths(config, ui_state)
    pad = config.panel_padding
    gap = config.panel_gap
    dashboard_width, dashboard_height = config.display_size
    map_x = pad + left_width + gap
    map_width = dashboard_width - (4 * pad) - (2 * gap) - left_width - right_width
    return {
        "left": (pad + left_width + (gap // 2), pad, dashboard_height - pad),
        "right": (map_x + map_width + (gap // 2), pad, dashboard_height - pad),
    }


def _hit_test_splitter(
    x: int,
    y: int,
    config: "SimulationConfig",
    ui_state: Optional[dict],
) -> Optional[str]:
    """(x, y) bir ayraç çubuğu üzerindeyse adını ('left'/'right') döndürür."""
    grab = max(8, (config.panel_gap // 2) + 4)
    for name, (center_x, y0, y1) in _get_splitter_zones(config, ui_state).items():
        if abs(x - center_x) <= grab and y0 <= y <= y1:
            return name
    return None


def _apply_splitter_drag(
    ui_state: dict,
    config: "SimulationConfig",
    x: int,
) -> None:
    """Sürüklenen ayraca göre sol/sağ sütun genişliğini günceller."""
    which = ui_state.get("_dragging_splitter")
    if which is None:
        return
    pad = config.panel_padding
    gap = config.panel_gap
    dashboard_width = int(config.display_size[0])
    left_width, right_width = _compute_panel_widths(config, ui_state)
    if which == "left":
        # Ayraç merkezi = pad + left_width + gap/2
        proposed = x - pad - (gap / 2.0)
        new_left = _clamp_dragged_panel_width("left", proposed, right_width, config)
        ui_state["left_panel_ratio"] = new_left / float(dashboard_width)
    else:
        # Ayraç merkezi = dashboard_width - pad - right_width - gap/2
        proposed = dashboard_width - pad - (gap / 2.0) - x
        new_right = _clamp_dragged_panel_width("right", proposed, left_width, config)
        ui_state["right_panel_width"] = int(new_right)
    ui_state["_dirty"] = True
    ui_state["_layout_dirty"] = True


def load_ui_layout_overrides(config: "SimulationConfig") -> Tuple[float, int]:
    """ui_layout.json varsa kayıtlı sütun düzenini, yoksa config varsayılanını verir."""
    left_ratio = float(config.left_panel_width_ratio)
    right_width = int(config.right_info_panel_width)
    try:
        if _UI_LAYOUT_PATH.exists():
            data = json.loads(_UI_LAYOUT_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                left_ratio = float(data.get("left_panel_ratio", left_ratio))
                right_width = int(data.get("right_panel_width", right_width))
    except (OSError, ValueError, TypeError):
        pass
    left_width, right_width = _compute_panel_widths(
        config,
        {"left_panel_ratio": left_ratio, "right_panel_width": right_width},
    )
    return left_width / float(config.display_size[0]), right_width


def save_ui_layout(ui_state: dict) -> None:
    """Geçerli sütun düzenini ui_layout.json dosyasına yazar."""
    try:
        _UI_LAYOUT_PATH.write_text(
            json.dumps(
                {
                    "left_panel_ratio": round(float(ui_state.get("left_panel_ratio", 0.20)), 5),
                    "right_panel_width": int(ui_state.get("right_panel_width", 180)),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except (OSError, TypeError, ValueError):
        pass


def _draw_splitter_handles(
    canvas: np.ndarray,
    config: "SimulationConfig",
    ui_state: dict,
) -> None:
    """Sürüklenebilir sütun ayraçlarını tutamak göstergesiyle çizer."""
    dragging = ui_state.get("_dragging_splitter")
    hover = ui_state.get("_splitter_hover")
    for name, (center_x, y0, y1) in _get_splitter_zones(config, ui_state).items():
        is_active = name == dragging
        is_hot = is_active or name == hover
        half_width = 5 if is_hot else 3
        bar_color = UI_COLORS["accent"] if is_hot else UI_COLORS["panel_border"]
        alpha = 0.95 if is_active else (0.75 if is_hot else 0.40)
        _draw_alpha_rounded_panel(
            canvas,
            center_x - half_width, y0,
            center_x + half_width, y1,
            half_width, bar_color, alpha,
        )
        grip_color = UI_COLORS["text_primary"] if is_hot else (175, 180, 190)
        mid_y = (y0 + y1) // 2
        for offset in (-16, -8, 0, 8, 16):
            cv2.circle(canvas, (center_x, mid_y + offset), 2, grip_color, -1, cv2.LINE_AA)


def _runtime_buttons_mouse_cb(event: int, x: int, y: int, flags: int, userdata: dict) -> None:
    _ = flags
    if not isinstance(userdata, dict):
        return
    # Letterbox ters dönüşümü: ekran koordinatlarını dashboard koordinatlarına çevir
    _lb = userdata.get("_lb", {})
    _scale = float(_lb.get("scale", 1.0))
    if _scale > 0.0:
        x = int((x - int(_lb.get("x_off", 0))) / _scale)
        y = int((y - int(_lb.get("y_off", 0))) / _scale)
    ui_state = userdata.get("state")
    buttons = userdata.get("buttons")
    config = userdata.get("config")
    if not isinstance(ui_state, dict) or not isinstance(buttons, list):
        return

    # --- Sütun ayracı sürükleme: bırakma olayı ---
    if event == cv2.EVENT_LBUTTONUP:
        if ui_state.get("_dragging_splitter") is not None:
            ui_state["_dragging_splitter"] = None
            save_ui_layout(ui_state)
            ui_state["_dirty"] = True
        return

    # --- Sütun ayracı sürükleme: sürüş sürüyor ---
    if ui_state.get("_dragging_splitter") is not None and config is not None:
        if event == cv2.EVENT_MOUSEMOVE:
            _apply_splitter_drag(ui_state, config, x)
        return

    if event == cv2.EVENT_MOUSEMOVE:
        previous_hover = ui_state.get("_hover_key")
        previous_split = ui_state.get("_splitter_hover")
        ui_state["_hover_key"] = None
        for button in buttons:
            bx, by, bw, bh = button.get("rect", (0, 0, 0, 0))
            if bx <= x <= (bx + bw) and by <= y <= (by + bh):
                ui_state["_hover_key"] = button.get("key")
                break
        split_hover = None
        if config is not None and ui_state.get("_hover_key") is None:
            split_hover = _hit_test_splitter(x, y, config, ui_state)
        ui_state["_splitter_hover"] = split_hover
        if (
            previous_hover != ui_state.get("_hover_key")
            or previous_split != split_hover
        ):
            ui_state["_dirty"] = True
        return

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Önce buton tıklamalarını kontrol et
    for button in buttons:
        bx, by, bw, bh = button.get("rect", (0, 0, 0, 0))
        if bx <= x <= (bx + bw) and by <= y <= (by + bh):
            key = button.get("key")
            ui_state[key] = not bool(ui_state.get(key, False))
            ui_state["_dirty"] = True
            return

    # Sütun ayracına basıldıysa sürüklemeyi başlat
    if config is not None:
        hit_splitter = _hit_test_splitter(x, y, config, ui_state)
        if hit_splitter is not None:
            ui_state["_dragging_splitter"] = hit_splitter
            ui_state["_splitter_hover"] = hit_splitter
            _apply_splitter_drag(ui_state, config, x)
            return

    # Otonom modda harita paneline tıklama → waypoint ayarla
    if not ui_state.get("autonomous_mode", False):
        return
    preview_state = userdata.get("reference_preview_state")
    if preview_state is None:
        return
    px0 = preview_state.paste_x
    py0 = preview_state.paste_y
    px1 = px0 + preview_state.preview_width
    py1 = py0 + preview_state.preview_height
    if px0 <= x <= px1 and py0 <= y <= py1:
        ref_x = (x - px0) / preview_state.scale_x + preview_state.viewport_left
        ref_y = (y - py0) / preview_state.scale_y + preview_state.viewport_top
        userdata["waypoint_target"] = (int(round(ref_x)), int(round(ref_y)))
        ui_state["_dirty"] = True


def create_runtime_ui_state(config: SimulationConfig) -> dict:
    left_panel_ratio, right_panel_width = load_ui_layout_overrides(config)
    return {
        "info_panel": bool(config.show_info_panel),
        "trajectory": bool(config.show_trajectory),
        "roi_frame": bool(config.show_roi_frame),
        "tm_boxes": bool(config.show_tm_boxes),
        "heading_arrow": bool(config.show_heading_arrow),
        "observation_boxes": bool(config.show_observation_boxes),
        "autonomous_mode": bool(config.autonomous_mode_enabled),
        "kalman_on": bool(config.kalman_enabled),
        "norm_mode": "HISTEQ",
        "obs_window_size": config.sample_window_size,
        "obs_window_default": config.sample_window_size,
        "obs_272_mode": False,
        "ref_patch": False,
        "left_panel_ratio": left_panel_ratio,
        "right_panel_width": right_panel_width,
        "_panel_collapsed": True,
        "_hover_key": None,
        "_splitter_hover": None,
        "_dragging_splitter": None,
        "_layout_dirty": False,
        "_dirty": True,
        "_quality": None,
    }


def apply_runtime_ui_hotkey(key: int, ui_state: dict) -> bool:
    hotkey_map = {
        ord("b"): "info_panel",
        ord("B"): "info_panel",
        ord("t"): "trajectory",
        ord("T"): "trajectory",
        ord("o"): "roi_frame",
        ord("O"): "roi_frame",
        ord("r"): "tm_boxes",
        ord("R"): "tm_boxes",
        ord("y"): "heading_arrow",
        ord("Y"): "heading_arrow",
        ord("g"): "observation_boxes",
        ord("G"): "observation_boxes",
        ord("h"): "_panel_collapsed",
        ord("H"): "_panel_collapsed",
        ord("p"): "autonomous_mode",
        ord("P"): "autonomous_mode",
        ord("k"): "kalman_on",
        ord("K"): "kalman_on",
        ord("m"): "ref_patch",
        ord("M"): "ref_patch",
    }
    if key in NORM_CYCLE_KEYS:
        current = ui_state.get("norm_mode", "HAM")
        idx = _NORM_MODES.index(current) if current in _NORM_MODES else 0
        ui_state["norm_mode"] = _NORM_MODES[(idx + 1) % len(_NORM_MODES)]
        ui_state["_dirty"] = True
        return True
    if key in OBS_WINDOW_CYCLE_KEYS:
        ui_state["obs_272_mode"] = not bool(ui_state.get("obs_272_mode", False))
        default_window_size = int(ui_state.get("obs_window_default", 544))
        ui_state["obs_window_size"] = (
            _OBS_WINDOW_SMALL if ui_state["obs_272_mode"] else default_window_size
        )
        ui_state["_dirty"] = True
        return True
    target = hotkey_map.get(key)
    if target is None:
        return False
    ui_state[target] = not bool(ui_state.get(target, False))
    ui_state["_dirty"] = True
    return True


def load_assets(
    config: SimulationConfig,
    status_callback=None,
) -> Tuple[object, object, object]:
    validate_config(config)
    reference_map_path = resolve_map_path(config.reference_map_path)
    reference_map = None
    observation_map = None
    try:
        if status_callback is not None:
            status_callback("Referans harita hazırlanıyor", "loading")
        use_streaming = (
            config.stream_rasters
            and is_georaster_path(reference_map_path)
            and is_georaster_path(config.observation_map_path)
        )
        if use_streaming:
            reference_map = RasterioGraySource(reference_map_path)
            observation_map = RasterioGraySource(
                config.observation_map_path,
                reference_dataset=(
                    reference_map.dataset
                    if config.align_observation_to_reference_grid
                    else None
                ),
            )
            if observation_map.shape != reference_map.shape:
                # Non-aligned streaming sources cannot be resized lazily. Keep
                # the legacy behavior explicit for this uncommon configuration.
                close_raster_source(observation_map)
                observation_map = load_grayscale_raster(config.observation_map_path)
                observation_map = cv2.resize(
                    observation_map,
                    (reference_map.shape[1], reference_map.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
        else:
            if is_georaster_path(reference_map_path):
                reference_map = load_grayscale_raster(reference_map_path)
            else:
                reference_map = load_grayscale_image(reference_map_path)

            if (
                config.align_observation_to_reference_grid
                and is_georaster_path(reference_map_path)
                and is_georaster_path(config.observation_map_path)
            ):
                observation_map = load_observation_aligned_to_reference_grid(
                    config.observation_map_path,
                    reference_map_path,
                )
            elif is_georaster_path(config.observation_map_path):
                observation_map = load_grayscale_raster(config.observation_map_path)
            else:
                observation_map = load_grayscale_image(config.observation_map_path)

            if observation_map.shape != reference_map.shape:
                observation_map = cv2.resize(
                    observation_map,
                    (reference_map.shape[1], reference_map.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )

        if status_callback is not None:
            status_callback("Yapay zekâ modeli yükleniyor", "loading")
        model = load_model_compat(resolve_model_path(config.model_path))
        if status_callback is not None:
            status_callback("Simülasyon hazır", "ready")
        return reference_map, observation_map, model
    except Exception:
        close_raster_source(observation_map)
        close_raster_source(reference_map)
        raise


def clamp_observation_cursor(
    row: int,
    col: int,
    image_shape: Tuple[int, int],
    config: SimulationConfig,
) -> Tuple[int, int]:
    minimum, maximum_row, maximum_col = get_observation_cursor_limits(image_shape, config)
    clamped_row = min(max(row, minimum), maximum_row)
    clamped_col = min(max(col, minimum), maximum_col)
    return clamped_row, clamped_col


def sample_center_biased_coordinate(
    minimum: int,
    maximum: int,
    band_ratio: float,
) -> int:
    if maximum <= minimum:
        return int(minimum)

    center = (minimum + maximum) / 2.0
    half_span = (maximum - minimum) * max(0.05, min(1.0, float(band_ratio))) / 2.0
    band_min = max(minimum, int(round(center - half_span)))
    band_max = min(maximum, int(round(center + half_span)))
    if band_max <= band_min:
        return int(round(center))
    return random.randint(band_min, band_max)


def get_observation_boxes(
    row: int,
    col: int,
    heading_degrees: float,
    config: SimulationConfig,
) -> List[Tuple[int, int, int, int]]:
    size = config.sample_window_size
    offset = get_effective_template_offset(config)
    # Offset vektörünü UAV başlık açısıyla döndür:
    # baseline her zaman gövde ekseniyle (diagonal forward-right) hizalı kalır
    raw_dx, raw_dy = rotate_image_offset(float(offset), float(offset), heading_degrees)
    idx = int(round(raw_dx))
    idy = int(round(raw_dy))
    return [
        (col - size - idx, row - size - idy, size, size),
        (col - size,       row - size,       size, size),
        (col - size + idx, row - size + idy, size, size),
    ]


def get_template_boxes_from_observation_boxes(
    observation_boxes: List[Tuple[int, int, int, int]],
    config: SimulationConfig,
    scale_factors: Optional[Tuple[float, ...]] = None,
) -> List[Tuple[int, int, int, int]]:
    # Pencere boyutu model girdisinden küçükse (272 modu), model çıktısındaki
    # 512 px şablon gerçek haritada 256 px'e karşılık gelir.
    base_size = get_effective_template_size(config)
    factors = scale_factors or tuple(1.0 for _ in observation_boxes)
    if len(factors) != len(observation_boxes):
        raise ValueError("Her gözlem kutusu için bir ölçek katsayısı gerekli.")
    template_boxes = []
    for (x, y, width, height), scale_factor in zip(
        observation_boxes,
        factors,
        strict=True,
    ):
        target_size = max(1, int(round(base_size * max(0.05, float(scale_factor)))))
        center_x = x + (width / 2.0)
        center_y = y + (height / 2.0)
        template_boxes.append(
            (
                int(round(center_x - (target_size / 2.0))),
                int(round(center_y - (target_size / 2.0))),
                target_size,
                target_size,
            )
        )
    return template_boxes


def rotate_square_capture(
    image: np.ndarray,
    angle_degrees: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0),
        angle_degrees,
        1.0,
    )
    return cv2.warpAffine(
        image,
        rotation_matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def extract_rotated_observation_window(
    observation_map: np.ndarray,
    observation_box: Tuple[int, int, int, int],
    heading_degrees: float,
    scale_factor: float,
    config: SimulationConfig,
) -> np.ndarray:
    capture_size = get_rotated_capture_size(config)
    center_x = observation_box[0] + (observation_box[2] // 2)
    center_y = observation_box[1] + (observation_box[3] // 2)
    capture_left = int(round(center_x - (capture_size / 2.0)))
    capture_top = int(round(center_y - (capture_size / 2.0)))
    capture_right = capture_left + capture_size
    capture_bottom = capture_top + capture_size

    raw_capture = observation_map[capture_top:capture_bottom, capture_left:capture_right]
    if raw_capture.shape[:2] != (capture_size, capture_size):
        pad_bottom = max(0, capture_size - raw_capture.shape[0])
        pad_right = max(0, capture_size - raw_capture.shape[1])
        raw_capture = cv2.copyMakeBorder(
            raw_capture,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_REPLICATE,
        )

    simulated_flight_capture = rotate_square_capture(raw_capture, heading_degrees)
    north_aligned_capture = rotate_square_capture(
        simulated_flight_capture,
        -heading_degrees,
    )

    scaled_window_size = get_scaled_observation_window_size(scale_factor, config)
    crop_start = (capture_size - scaled_window_size) // 2
    crop_end = crop_start + scaled_window_size
    scaled_window = north_aligned_capture[crop_start:crop_end, crop_start:crop_end]

    if scaled_window.shape[:2] != (scaled_window_size, scaled_window_size):
        pad_bottom = max(0, scaled_window_size - scaled_window.shape[0])
        pad_right = max(0, scaled_window_size - scaled_window.shape[1])
        scaled_window = cv2.copyMakeBorder(
            scaled_window,
            0,
            pad_bottom,
            0,
            pad_right,
            cv2.BORDER_REPLICATE,
        )

    if scaled_window_size != config.sample_window_size:
        interp = (
            cv2.INTER_AREA
            if scaled_window_size > config.sample_window_size
            else cv2.INTER_LINEAR
        )
        scaled_window = cv2.resize(
            scaled_window,
            (config.sample_window_size, config.sample_window_size),
            interpolation=interp,
        )

    return scaled_window


def prepare_triplet_for_model(
    observation_windows: List[np.ndarray],
    config: SimulationConfig,
    norm_mode: str = "HISTEQ",
) -> np.ndarray:
    prepared_windows = []
    for observation_window in observation_windows:
        # Normalizasyonu resize öncesi uygula: doğal piksel histogramı korunur.
        normed_src = apply_observation_norm(observation_window, norm_mode)
        # Küçültmede INTER_AREA (antialias), büyütmede INTER_LINEAR (yumuşak).
        src_size = max(normed_src.shape[0], normed_src.shape[1])
        interp = cv2.INTER_AREA if src_size > config.model_input_size else cv2.INTER_LINEAR
        resized_window = cv2.resize(
            normed_src,
            (config.model_input_size, config.model_input_size),
            interpolation=interp,
        )
        normalized_window = (resized_window.astype(np.float32) - 127.5) / 127.5
        prepared_windows.append(normalized_window)

    return np.stack(prepared_windows, axis=0).reshape(
        -1,
        config.model_input_size,
        config.model_input_size,
        1,
    )


def predict_template_triplet(
    model: object,
    model_input_triplet: np.ndarray,
    config: SimulationConfig,
) -> List[np.ndarray]:
    predictions = model.predict(model_input_triplet, verbose=0)
    templates = []
    for prediction in predictions:
        prediction_2d = prediction.reshape(config.model_input_size, config.model_input_size)
        cropped_prediction = prediction_2d[
            config.crop_margin : config.model_input_size - config.crop_margin,
            config.crop_margin : config.model_input_size - config.crop_margin,
        ]
        scaled_prediction = np.clip(cropped_prediction, 0.0, 1.0)
        template_image = np.asarray(
            np.round(scaled_prediction * 255.0),
            dtype=np.uint8,
        )
        templates.append(template_image)
    return templates


def extract_template_triplet(
    observation_map: np.ndarray,
    row: int,
    col: int,
    heading_degrees: float,
    altitude_agl_m: float,
    terrain_context: Optional[TerrainContext],
    model: object,
    config: SimulationConfig,
    norm_mode: str = "HISTEQ",
    obs_window_size: Optional[int] = None,
) -> Tuple[
    List[np.ndarray],
    List[np.ndarray],
    List[Tuple[int, int, int, int]],
    List[Tuple[int, int, int, int]],
    int,
    int,
    AltitudeSimulationState,
]:
    if obs_window_size is not None and obs_window_size != config.sample_window_size:
        config = dataclasses.replace(config, sample_window_size=obs_window_size)
    row, col = clamp_observation_cursor(row, col, observation_map.shape, config)
    observation_boxes = get_observation_boxes(row, col, heading_degrees, config)
    if is_altitude_scenario(config):
        if terrain_context is None:
            raise ValueError("Irtifa senaryosu icin terrain_context gerekli.")
        altitude_state = compute_altitude_simulation_state(
            observation_boxes,
            altitude_agl_m,
            terrain_context,
            config,
        )
    else:
        altitude_state = build_normal_altitude_state(len(observation_boxes), config, altitude_agl_m)

    observation_windows = []
    for box, scale_factor in zip(
        observation_boxes,
        altitude_state.patch_scale_factors,
        strict=True,
    ):
        observation_windows.append(
            extract_rotated_observation_window(
                observation_map,
                box,
                heading_degrees,
                scale_factor,
                config,
            )
        )

    model_input_triplet = prepare_triplet_for_model(observation_windows, config, norm_mode)
    templates = predict_template_triplet(model, model_input_triplet, config)
    template_boxes = get_template_boxes_from_observation_boxes(
        observation_boxes,
        config,
        altitude_state.patch_scale_factors,
    )

    return (
        templates,
        observation_windows,
        observation_boxes,
        template_boxes,
        row,
        col,
        altitude_state,
    )


def is_sqdiff_method(match_method: int) -> bool:
    return match_method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED)


def extract_match_score_and_location(
    response_map: np.ndarray,
    match_method: int,
) -> Tuple[float, Tuple[int, int]]:
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(response_map)
    if is_sqdiff_method(match_method):
        return float(min_val), min_loc
    return float(max_val), max_loc


def extract_match_evidence(
    response_map: np.ndarray,
    template: np.ndarray,
    match_method: int,
) -> TemplateMatchEvidence:
    score, top_left = extract_match_score_and_location(response_map, match_method)
    response_height, response_width = response_map.shape[:2]
    radius_x = max(2, int(template.shape[1] // 3))
    radius_y = max(2, int(template.shape[0] // 3))
    x0 = max(0, top_left[0] - radius_x)
    x1 = min(response_width, top_left[0] + radius_x + 1)
    y0 = max(0, top_left[1] - radius_y)
    y1 = min(response_height, top_left[1] + radius_y + 1)
    excluded = response_map[y0:y1, x0:x1].copy()
    try:
        response_map[y0:y1, x0:x1] = (
            float("inf") if is_sqdiff_method(match_method) else float("-inf")
        )
        second_score, _ = extract_match_score_and_location(
            response_map,
            match_method,
        )
    finally:
        response_map[y0:y1, x0:x1] = excluded
    if not math.isfinite(second_score):
        peak_margin = 0.0
    elif is_sqdiff_method(match_method):
        peak_margin = max(0.0, float(second_score - score))
    else:
        peak_margin = max(0.0, float(score - second_score))
    return TemplateMatchEvidence(
        score=float(score),
        top_left=(int(top_left[0]), int(top_left[1])),
        peak_margin=peak_margin,
        template_stddev=float(np.std(template)),
    )


def run_template_match(
    reference_map: np.ndarray,
    template: np.ndarray,
    match_method: int,
) -> TemplateMatchEvidence:
    template_stddev = float(np.std(template))
    if template_stddev <= 1e-6:
        return TemplateMatchEvidence(
            score=(float("inf") if is_sqdiff_method(match_method) else 0.0),
            top_left=(0, 0),
            peak_margin=0.0,
            template_stddev=template_stddev,
        )
    response_map = cv2.matchTemplate(reference_map, template, match_method, None)
    return extract_match_evidence(response_map, template, match_method)


def run_template_match_pyramid(
    search_region: np.ndarray,
    template: np.ndarray,
    config: SimulationConfig,
) -> TemplateMatchEvidence:
    match_method = config.match_method
    template_stddev = float(np.std(template))
    if template_stddev <= 1e-6:
        return TemplateMatchEvidence(
            score=(float("inf") if is_sqdiff_method(match_method) else 0.0),
            top_left=(0, 0),
            peak_margin=0.0,
            template_stddev=template_stddev,
        )
    region_height, region_width = search_region.shape[:2]
    template_height, template_width = template.shape[:2]
    result_height = region_height - template_height + 1
    result_width = region_width - template_width + 1

    if result_height <= 0 or result_width <= 0:
        fallback_score = float("inf") if is_sqdiff_method(match_method) else float("-inf")
        return TemplateMatchEvidence(
            score=fallback_score,
            top_left=(0, 0),
            peak_margin=0.0,
            template_stddev=float(np.std(template)),
        )

    scale = float(config.coarse_scale)
    small_width = max(1, int(region_width * scale))
    small_height = max(1, int(region_height * scale))
    small_template_width = max(1, int(template_width * scale))
    small_template_height = max(1, int(template_height * scale))

    if (
        small_template_width > small_width
        or small_template_height > small_height
        or scale <= 0.0
    ):
        coarse_x = result_width // 2
        coarse_y = result_height // 2
        coarse_peak_margin = 0.0
    else:
        region_small = cv2.resize(
            search_region,
            (small_width, small_height),
            interpolation=cv2.INTER_AREA,
        )
        template_small = cv2.resize(
            template,
            (small_template_width, small_template_height),
            interpolation=cv2.INTER_AREA,
        )
        coarse_response = cv2.matchTemplate(region_small, template_small, match_method, None)
        if coarse_response.size == 0:
            coarse_x = result_width // 2
            coarse_y = result_height // 2
            coarse_peak_margin = 0.0
        else:
            coarse_evidence = extract_match_evidence(
                coarse_response,
                template_small,
                match_method,
            )
            coarse_loc = coarse_evidence.top_left
            coarse_x = int(coarse_loc[0] / scale)
            coarse_y = int(coarse_loc[1] / scale)
            coarse_peak_margin = coarse_evidence.peak_margin

    pad = max(8, int(max(template_width, template_height) * config.roi_pad_factor))
    x1 = max(0, coarse_x - pad)
    y1 = max(0, coarse_y - pad)
    x2 = min(result_width - 1, coarse_x + pad)
    y2 = min(result_height - 1, coarse_y + pad)

    if x2 < x1:
        x1 = x2 = max(0, min(coarse_x, result_width - 1))
    if y2 < y1:
        y1 = y2 = max(0, min(coarse_y, result_height - 1))

    region_roi = search_region[y1 : y2 + template_height, x1 : x2 + template_width]
    roi_response = cv2.matchTemplate(region_roi, template, match_method, None)
    fine_evidence = extract_match_evidence(roi_response, template, match_method)
    return TemplateMatchEvidence(
        score=fine_evidence.score,
        top_left=(fine_evidence.top_left[0] + x1, fine_evidence.top_left[1] + y1),
        peak_margin=min(coarse_peak_margin, fine_evidence.peak_margin),
        template_stddev=fine_evidence.template_stddev,
    )


def match_three(
    search_region: np.ndarray,
    templates: List[np.ndarray],
    config: SimulationConfig,
) -> Tuple[List[TemplateMatchEvidence], str]:
    if config.use_pyramid_matching:
        worker = lambda template: run_template_match_pyramid(search_region, template, config)
        backend_label = "parallel-pyramid" if config.use_parallel_matching else "serial-pyramid"
    else:
        worker = lambda template: run_template_match(search_region, template, config.match_method)
        backend_label = "parallel-direct" if config.use_parallel_matching else "serial-direct"

    if config.use_parallel_matching:
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(worker, template) for template in templates]
            results = [future.result() for future in futures]
    else:
        results = [worker(template) for template in templates]

    return results, backend_label


def intersect_boxes(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    x = max(box_a[0], box_b[0])
    y = max(box_a[1], box_b[1])
    width = min(box_a[0] + box_a[2], box_b[0] + box_b[2]) - x
    height = min(box_a[1] + box_a[3], box_b[1] + box_b[3]) - y

    if width <= 0 or height <= 0:
        return None
    return (x, y, width, height)


def compute_intersection_box(
    boxes: List[Tuple[int, int, int, int]],
) -> Tuple[Tuple[int, int, int, int], str]:
    intersection_ab = intersect_boxes(boxes[0], boxes[1])
    intersection_bc = intersect_boxes(boxes[1], boxes[2])
    intersection_ac = intersect_boxes(boxes[0], boxes[2])

    if intersection_ab and intersection_bc:
        intersection_abc = intersect_boxes(intersection_ab, intersection_bc)
        if intersection_abc:
            return intersection_abc, "abc"

    if intersection_ab:
        return intersection_ab, "ab"
    if intersection_bc:
        return intersection_bc, "bc"
    if intersection_ac:
        return intersection_ac, "ac"

    return boxes[1], "center_fallback"


def get_search_window_box(
    reference_map_shape: Tuple[int, int],
    center: Tuple[int, int],
    window_size: int,
) -> Tuple[int, int, int, int]:
    height, width = reference_map_shape
    half_window = max(1, int(window_size // 2))
    center_x, center_y = center

    left = center_x - half_window
    right = center_x + half_window
    top = center_y - half_window
    bottom = center_y + half_window

    # Kenar taşması olduğunda karşı tarafı genişlet — merkez hizası korunur
    if left < 0:
        right -= left
        left = 0
    if right > width:
        left -= right - width
        right = width
        left = max(0, left)
    if top < 0:
        bottom -= top
        top = 0
    if bottom > height:
        top -= bottom - height
        bottom = height
        top = max(0, top)

    return left, top, right, bottom


def extract_search_region(
    reference_map: np.ndarray,
    previous_predicted_center: Optional[Tuple[int, int]],
    search_window_size: int,
    step_count: int,
    config: SimulationConfig,
    force_global: bool = False,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int], str]:
    should_search_global = (
        bool(force_global)
        or previous_predicted_center is None
        or config.global_refresh_interval > 0
        and step_count > 0
        and (step_count % config.global_refresh_interval) == 0
    )
    if should_search_global:
        full_box = (0, 0, reference_map.shape[1], reference_map.shape[0])
        materialize = getattr(reference_map, "read_full", None)
        full_map = materialize() if callable(materialize) else reference_map
        return full_map, (0, 0), full_box, "global"

    search_box = get_search_window_box(
        reference_map.shape,
        previous_predicted_center,
        search_window_size,
    )
    left, top, right, bottom = search_box
    return reference_map[top:bottom, left:right], (left, top), search_box, "adaptive-roi"


def localize_template_triplet(
    search_region: np.ndarray,
    search_origin: Tuple[int, int],
    templates: List[np.ndarray],
    config: SimulationConfig,
    match_downsample_size: int = 0,
) -> Tuple[
    List[float],
    List[Tuple[int, int, int, int]],
    Tuple[int, int, int, int],
    str,
    str,
    List[TemplateMatchEvidence],
]:
    scores = []
    matched_boxes = []

    # Eşleştirme öncesi downsample (hız/hassasiyet dengesi)
    ds_scale = 1.0
    orig_template_sizes = [(t.shape[1], t.shape[0]) for t in templates]
    if match_downsample_size > 0 and templates:
        tpl_h, tpl_w = templates[0].shape[:2]
        if max(tpl_h, tpl_w) > match_downsample_size:
            ds_scale = match_downsample_size / float(max(tpl_h, tpl_w))
            ds_tw = max(1, int(round(tpl_w * ds_scale)))
            ds_th = max(1, int(round(tpl_h * ds_scale)))
            ds_sw = max(ds_tw + 1, int(round(search_region.shape[1] * ds_scale)))
            ds_sh = max(ds_th + 1, int(round(search_region.shape[0] * ds_scale)))
            templates = [
                cv2.resize(t, (ds_tw, ds_th), interpolation=cv2.INTER_AREA)
                for t in templates
            ]
            search_region = cv2.resize(
                search_region, (ds_sw, ds_sh), interpolation=cv2.INTER_AREA
            )

    match_results, match_backend = match_three(search_region, templates, config)

    for (orig_w, orig_h), match_result in zip(
        orig_template_sizes,
        match_results,
        strict=True,
    ):
        local_top_left = match_result.top_left
        scores.append(match_result.score)
        matched_boxes.append(
            (
                int(round(local_top_left[0] / ds_scale)) + search_origin[0],
                int(round(local_top_left[1] / ds_scale)) + search_origin[1],
                orig_w,
                orig_h,
            )
        )

    intersection_box, intersection_mode = compute_intersection_box(matched_boxes)
    return (
        scores,
        matched_boxes,
        intersection_box,
        intersection_mode,
        match_backend,
        match_results,
    )


def get_box_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    return (box[0] + (box[2] // 2), box[1] + (box[3] // 2))


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


def is_strict_triplet_alignment(
    matched_boxes: List[Tuple[int, int, int, int]],
    intersection_mode: str,
    config: SimulationConfig,
    heading_degrees: float = 0.0,
) -> bool:
    if intersection_mode != "abc" or len(matched_boxes) != 3:
        return False

    center_a = get_box_center(matched_boxes[0])
    center_b = get_box_center(matched_boxes[1])
    center_c = get_box_center(matched_boxes[2])

    delta_ab_x = float(center_b[0] - center_a[0])
    delta_ab_y = float(center_b[1] - center_a[1])
    delta_bc_x = float(center_c[0] - center_b[0])
    delta_bc_y = float(center_c[1] - center_b[1])
    midpoint_x = (center_a[0] + center_c[0]) / 2.0
    midpoint_y = (center_a[1] + center_c[1]) / 2.0
    midpoint_error = math.hypot(center_b[0] - midpoint_x, center_b[1] - midpoint_y)

    observation_scale = get_observation_model_scale(config)
    expected_offset = float(config.template_offset) * observation_scale
    alignment_tolerance = max(
        8.0,
        float(config.triplet_alignment_tolerance_px) * observation_scale,
    )

    # Başlık açısına göre döndürülmüş beklenen delta
    exp_dx, exp_dy = rotate_image_offset(
        expected_offset, expected_offset, heading_degrees
    )
    step_error = max(
        abs(delta_ab_x - exp_dx),
        abs(delta_ab_y - exp_dy),
        abs(delta_bc_x - exp_dx),
        abs(delta_bc_y - exp_dy),
    )
    symmetry_error = max(
        abs(delta_ab_x - delta_bc_x),
        abs(delta_ab_y - delta_bc_y),
    )
    # Yön kontrolü: gerçek delta beklenen yönle aynı tarafta mı?
    # (monotonic_diagonal yerine; beklenen delta negatif bileşen içerebilir)
    dot_ab = delta_ab_x * exp_dx + delta_ab_y * exp_dy
    correct_direction = dot_ab > 0.0

    alignment_error = max(midpoint_error, step_error, symmetry_error)
    return correct_direction and (
        alignment_error <= alignment_tolerance
    )


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def calculate_fit_size(
    image_width: int,
    image_height: int,
    target_width: int,
    target_height: int,
) -> Tuple[int, int]:
    scale = min(float(target_width) / image_width, float(target_height) / image_height)
    return (
        max(1, int(round(image_width * scale))),
        max(1, int(round(image_height * scale))),
    )


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


def get_dashboard_layout(
    config: SimulationConfig,
    show_ref_patch: bool = False,
    ui_state: Optional[dict] = None,
) -> Tuple[
    Tuple[int, int, int, int],
    Tuple[int, int, int, int],
    Tuple[int, int, int, int],
    Optional[Tuple[int, int, int, int]],
    Tuple[int, int, int, int],
]:
    dashboard_width, dashboard_height = config.display_size
    left_panel_width, right_panel_width = _compute_panel_widths(config, ui_state)
    # Harita: sol kolon + sağ telemetri kolonunu çıkar
    # Kenar boşluğu yapısı: pad | sol | gap | harita | gap | sağ | pad
    map_width = (
        dashboard_width
        - (4 * config.panel_padding)
        - (2 * config.panel_gap)
        - left_panel_width
        - right_panel_width
    )
    map_x = config.panel_padding + left_panel_width + config.panel_gap
    map_rect = (
        map_x,
        config.panel_padding,
        map_width,
        dashboard_height - (2 * config.panel_padding),
    )
    right_panel_rect = (
        map_x + map_width + config.panel_gap,
        config.panel_padding,
        right_panel_width,
        dashboard_height - (2 * config.panel_padding),
    )

    _ = show_ref_patch
    panel_height = (
        dashboard_height - 2 * config.panel_padding - 2 * config.panel_gap
    ) // 3
    observation_rect = (
        config.panel_padding, config.panel_padding,
        left_panel_width, panel_height,
    )
    template_rect = (
        config.panel_padding,
        config.panel_padding + panel_height + config.panel_gap,
        left_panel_width,
        panel_height,
    )
    ref_patch_rect: Optional[Tuple[int, int, int, int]] = (
        config.panel_padding,
        config.panel_padding + 2 * (panel_height + config.panel_gap),
        left_panel_width,
        panel_height,
    )

    return observation_rect, template_rect, map_rect, ref_patch_rect, right_panel_rect


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


def draw_panel_frame(
    canvas: np.ndarray,
    panel_rect: Tuple[int, int, int, int],
    title: str,
    config: SimulationConfig,
    accent_color: Optional[Tuple[int, int, int]] = None,
) -> None:
    x, y, width, height = panel_rect
    hdr_h = config.panel_title_height
    # Panel background
    cv2.rectangle(canvas, (x, y), (x + width, y + height), config.panel_background_color, -1)
    # Header background — slightly lighter strip
    hdr_bg = tuple(min(255, c + 14) for c in config.panel_background_color)
    cv2.rectangle(canvas, (x, y), (x + width, y + hdr_h), hdr_bg, -1)
    # Outer border
    cv2.rectangle(canvas, (x, y), (x + width, y + height), config.panel_border_color, 1)
    # Accent left bar (full height)
    if accent_color is not None:
        cv2.rectangle(canvas, (x, y), (x + 3, y + height), accent_color, -1)
    title_x = x + (10 if accent_color is not None else 12)
    _put_text_tr(canvas, title, (title_x, y + hdr_h - 8), 17, config.panel_title_color, 2)


def draw_panel(
    canvas: np.ndarray,
    image: np.ndarray,
    panel_rect: Tuple[int, int, int, int],
    title: str,
    config: SimulationConfig,
    accent_color: Optional[Tuple[int, int, int]] = None,
) -> None:
    draw_panel_frame(canvas, panel_rect, title, config, accent_color=accent_color)
    content_x, content_y, content_width, content_height = get_panel_content_rect(
        panel_rect,
        config,
    )
    fitted = resize_to_fit(ensure_bgr(image), content_width, content_height)
    fitted_height, fitted_width = fitted.shape[:2]
    paste_x = content_x + (content_width - fitted_width) // 2
    paste_y = content_y + (content_height - fitted_height) // 2
    canvas[paste_y : paste_y + fitted_height, paste_x : paste_x + fitted_width] = fitted


def get_reference_viewport_box(
    reference_map_shape: Tuple[int, int],
    predicted_intersection_box: Tuple[int, int, int, int],
    actual_intersection_box: Tuple[int, int, int, int],
    search_window_box: Tuple[int, int, int, int],
    search_mode: str,
    config: SimulationConfig,
) -> Tuple[int, int, int, int]:
    map_height, map_width = reference_map_shape
    if search_mode != "global":
        padding = int(config.reference_viewport_search_padding)
        search_width = int(search_window_box[2] - search_window_box[0])
        search_height = int(search_window_box[3] - search_window_box[1])

        # Gerçek konum her zaman görünür olsun: arama penceresi + actual box'u kapsayan bb
        all_x = [
            search_window_box[0], search_window_box[2],
            actual_intersection_box[0],
            actual_intersection_box[0] + actual_intersection_box[2],
        ]
        all_y = [
            search_window_box[1], search_window_box[3],
            actual_intersection_box[1],
            actual_intersection_box[1] + actual_intersection_box[3],
        ]
        span_x = int(max(all_x) - min(all_x))
        span_y = int(max(all_y) - min(all_y))

        viewport_width = min(
            map_width,
            max(
                int(config.reference_viewport_search_min_size),
                search_width + (2 * padding),
                span_x + (2 * padding),
            ),
        )
        viewport_height = min(
            map_height,
            max(
                int(config.reference_viewport_search_min_size),
                search_height + (2 * padding),
                span_y + (2 * padding),
            ),
        )
        center_x = (min(all_x) + max(all_x)) / 2.0
        center_y = (min(all_y) + max(all_y)) / 2.0
        viewport_left = int(round(center_x - (viewport_width / 2.0)))
        viewport_top = int(round(center_y - (viewport_height / 2.0)))
        viewport_left = min(max(viewport_left, 0), max(0, map_width - viewport_width))
        viewport_top = min(max(viewport_top, 0), max(0, map_height - viewport_height))
        viewport_right = viewport_left + viewport_width
        viewport_bottom = viewport_top + viewport_height
        return viewport_left, viewport_top, viewport_right, viewport_bottom

    relevant_boxes = [
        predicted_intersection_box,
        actual_intersection_box,
    ]
    left = min(box[0] for box in relevant_boxes)
    top = min(box[1] for box in relevant_boxes)
    right = max(box[0] + box[2] for box in relevant_boxes)
    bottom = max(box[1] + box[3] for box in relevant_boxes)
    padding = int(config.reference_viewport_padding)

    viewport_width = min(
        map_width,
        max(
            int(config.reference_viewport_base_size),
            int((right - left) + (2 * padding)),
        ),
    )
    viewport_height = min(
        map_height,
        max(
            int(config.reference_viewport_base_size),
            int((bottom - top) + (2 * padding)),
        ),
    )

    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    viewport_left = int(round(center_x - (viewport_width / 2.0)))
    viewport_top = int(round(center_y - (viewport_height / 2.0)))
    viewport_left = min(max(viewport_left, 0), max(0, map_width - viewport_width))
    viewport_top = min(max(viewport_top, 0), max(0, map_height - viewport_height))
    viewport_right = viewport_left + viewport_width
    viewport_bottom = viewport_top + viewport_height
    return viewport_left, viewport_top, viewport_right, viewport_bottom


def create_reference_preview_state(
    reference_map: np.ndarray,
    panel_rect: Tuple[int, int, int, int],
    viewport_box: Tuple[int, int, int, int],
    config: SimulationConfig,
) -> ReferencePreviewState:
    content_x, content_y, content_width, content_height = get_panel_content_rect(
        panel_rect,
        config,
    )
    viewport_left, viewport_top, viewport_right, viewport_bottom = viewport_box
    reference_crop = reference_map[viewport_top:viewport_bottom, viewport_left:viewport_right]
    if reference_crop.size == 0:
        reference_crop = reference_map
        viewport_left = 0
        viewport_top = 0
        viewport_right = reference_map.shape[1]
        viewport_bottom = reference_map.shape[0]

    preview_width, preview_height = calculate_fit_size(
        reference_crop.shape[1],
        reference_crop.shape[0],
        content_width,
        content_height,
    )
    preview_image = cv2.resize(
        reference_crop,
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
        scale_x=preview_width / float(viewport_right - viewport_left),
        scale_y=preview_height / float(viewport_bottom - viewport_top),
        viewport_left=viewport_left,
        viewport_top=viewport_top,
        viewport_width=viewport_right - viewport_left,
        viewport_height=viewport_bottom - viewport_top,
        base_preview=preview_image,
    )


def scale_point_to_preview(
    point: Tuple[int, int],
    preview_state: ReferencePreviewState,
) -> Tuple[int, int]:
    return (
        int(round((point[0] - preview_state.viewport_left) * preview_state.scale_x)),
        int(round((point[1] - preview_state.viewport_top) * preview_state.scale_y)),
    )


def scale_box_to_preview(
    box: Tuple[int, int, int, int],
    preview_state: ReferencePreviewState,
) -> Tuple[int, int, int, int]:
    left = int(round((box[0] - preview_state.viewport_left) * preview_state.scale_x))
    top = int(round((box[1] - preview_state.viewport_top) * preview_state.scale_y))
    right = int(
        round(
            ((box[0] + box[2]) - preview_state.viewport_left) * preview_state.scale_x
        )
    )
    bottom = int(
        round(
            ((box[1] + box[3]) - preview_state.viewport_top) * preview_state.scale_y
        )
    )
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

    margin = max(50, int(round(120 / max(preview_state.scale_x, preview_state.scale_y, 1e-6))))
    filtered_points = [
        point
        for point in points
        if (
            (preview_state.viewport_left - margin)
            <= point[0]
            <= (preview_state.viewport_left + preview_state.viewport_width + margin)
            and (preview_state.viewport_top - margin)
            <= point[1]
            <= (preview_state.viewport_top + preview_state.viewport_height + margin)
        )
    ]
    if len(filtered_points) < 2:
        return

    scaled_points = [
        scale_point_to_preview(point, preview_state) for point in filtered_points
    ]
    cv2.polylines(
        preview_image,
        [np.array(scaled_points, dtype=np.int32)],
        False,
        color,
        thickness,
    )


def draw_heading_arrow(
    image: np.ndarray,
    origin: Tuple[int, int],
    heading_degrees: float,
    length: int,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    if length <= 0:
        return

    direction_x, direction_y = get_heading_vector(heading_degrees)
    arrow_tip = (
        int(round(origin[0] + (direction_x * length))),
        int(round(origin[1] + (direction_y * length))),
    )
    cv2.arrowedLine(
        image,
        origin,
        arrow_tip,
        color,
        thickness,
        cv2.LINE_AA,
        0,
        0.24,
    )


# ---------------------------------------------------------------------------
# Uçak simgesi (yöne göre dönen araç göstergesi)
# ---------------------------------------------------------------------------
# Tüm noktalar normalize edilmiş gövde çerçevesinde tanımlıdır:
#   - dönüş merkezi (0, 0)
#   - burun (0, -1), kuyruk (0, +1)
#   - +x sağ kanat yönü
# Çizimden önce her nokta başlık açısı kadar döndürülür.
_AIRCRAFT_OUTLINE = (
    (0.000, -1.000),   # burun
    (0.071, -0.786),
    (0.083, -0.048),   # ön gövde / kanat kökü öncesi
    (0.119,  0.048),   # sağ kanat hücum kenarı kökü
    (0.929,  0.452),   # sağ kanat ucu (hücum)
    (0.952,  0.548),   # sağ kanat ucu (firar)
    (0.143,  0.381),   # sağ kanat firar kenarı kökü
    (0.095,  0.667),   # arka gövde
    (0.405,  0.857),   # sağ yatay dengeleyici ucu (hücum)
    (0.405,  0.929),   # sağ yatay dengeleyici ucu (firar)
    (0.060,  0.881),   # dengeleyici kökü
    (0.060,  1.000),   # kuyruk (sağ)
    (-0.060, 1.000),   # kuyruk (sol)
    (-0.060, 0.881),
    (-0.405, 0.929),
    (-0.405, 0.857),
    (-0.095, 0.667),
    (-0.143, 0.381),
    (-0.952, 0.548),
    (-0.929, 0.452),
    (-0.119, 0.048),
    (-0.083, -0.048),
    (-0.071, -0.786),
)

_AIRCRAFT_FUSELAGE = (
    (0.000, -1.000),
    (0.072, -0.700),
    (0.090,  0.100),
    (0.072,  0.620),
    (0.045,  0.930),
    (-0.045, 0.930),
    (-0.072, 0.620),
    (-0.090, 0.100),
    (-0.072, -0.700),
)

_AIRCRAFT_TAILFIN = (
    (0.000, 0.500),
    (0.050, 0.930),
    (-0.050, 0.930),
)

_AIRCRAFT_COCKPIT = (
    (0.000, -0.860),
    (0.050, -0.700),
    (0.000, -0.520),
    (-0.050, -0.700),
)


def _scale_color(
    color: Tuple[int, int, int],
    factor: float,
) -> Tuple[int, int, int]:
    """factor < 1 rengi karartır, factor > 1 beyaza doğru açar."""
    if factor <= 1.0:
        return tuple(int(max(0, min(255, round(channel * factor)))) for channel in color)
    blend = min(1.0, factor - 1.0)
    return tuple(
        int(max(0, min(255, round(channel + (255 - channel) * blend))))
        for channel in color
    )


def _blend_filled_poly(
    image: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float,
) -> None:
    """Dolu poligonu sınırlayıcı kutu içinde saydam olarak harmanlar."""
    x, y, width, height = cv2.boundingRect(points)
    pad = 2
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(image.shape[1], x + width + pad)
    y1 = min(image.shape[0], y + height + pad)
    if x1 <= x0 or y1 <= y0:
        return
    roi = image[y0:y1, x0:x1]
    overlay = roi.copy()
    local_points = points - np.array((x0, y0), dtype=np.int32)
    cv2.fillPoly(overlay, [local_points], color, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)


def draw_aircraft_marker(
    image: np.ndarray,
    origin: Tuple[int, int],
    heading_degrees: float,
    length: int,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    """Aracı, başlık açısına göre dönen bir uçak silueti olarak çizer.

    Burun her zaman uçuş yönünü gösterir; araç döndükçe simge de döner.
    Uzaklaştırılmış (küçük) görünümlerde ayrıntılar kademeli olarak gizlenir.
    """
    if length <= 0:
        return
    radius = max(6.0, float(length) * 0.55)

    def _project(points: tuple) -> np.ndarray:
        projected = []
        for local_x, local_y in points:
            offset_x, offset_y = rotate_image_offset(
                local_x * radius, local_y * radius, heading_degrees
            )
            projected.append(
                (
                    int(round(origin[0] + offset_x)),
                    int(round(origin[1] + offset_y)),
                )
            )
        return np.array(projected, dtype=np.int32)

    outline_points = _project(_AIRCRAFT_OUTLINE)

    # Derinlik hissi için sağ-alta kaydırılmış yumuşak gölge.
    shadow_offset = max(2, int(round(radius * 0.10)))
    _blend_filled_poly(
        image,
        outline_points + np.array((shadow_offset, shadow_offset), dtype=np.int32),
        (8, 8, 12),
        0.35,
    )

    wing_color = _scale_color(color, 0.78)
    outline_color = _scale_color(color, 0.30)
    line_width = max(1, min(3, int(round(radius / 46.0))))

    # Kanatlar + kuyruk silueti (gövdeden biraz koyu).
    cv2.fillPoly(image, [outline_points], wing_color, cv2.LINE_AA)
    cv2.polylines(image, [outline_points], True, outline_color, line_width, cv2.LINE_AA)

    # Gövde — kanatların üzerinde, ana renkte: katmanlı/3B görünüm verir.
    if radius >= 18.0:
        fuselage_points = _project(_AIRCRAFT_FUSELAGE)
        cv2.fillPoly(image, [fuselage_points], color, cv2.LINE_AA)
        cv2.polylines(
            image, [fuselage_points], True, outline_color,
            max(1, line_width - 1), cv2.LINE_AA,
        )

    # Dikey dengeleyici (kuyrukta koyu üçgen) + kokpit camı.
    if radius >= 30.0:
        cv2.fillPoly(image, [_project(_AIRCRAFT_TAILFIN)], _scale_color(color, 0.5), cv2.LINE_AA)
        cv2.fillPoly(image, [_project(_AIRCRAFT_COCKPIT)], (28, 30, 38), cv2.LINE_AA)


def create_observation_context_view(
    observation_map: np.ndarray,
    observation_boxes: List[Tuple[int, int, int, int]],
    actual_boxes: List[Tuple[int, int, int, int]],
    actual_intersection_box: Tuple[int, int, int, int],
    heading_degrees: float,
    ui_state: dict,
    config: SimulationConfig,
) -> np.ndarray:
    context_boxes = observation_boxes + actual_boxes + [actual_intersection_box]
    left = max(0, min(box[0] for box in context_boxes) - config.observation_context_margin)
    top = max(0, min(box[1] for box in context_boxes) - config.observation_context_margin)
    right = min(
        observation_map.shape[1],
        max(box[0] + box[2] for box in context_boxes) + config.observation_context_margin,
    )
    bottom = min(
        observation_map.shape[0],
        max(box[1] + box[3] for box in context_boxes) + config.observation_context_margin,
    )

    view = ensure_bgr(observation_map[top:bottom, left:right])
    if view.size == 0:
        return np.zeros((config.sample_window_size, config.sample_window_size, 3), dtype=np.uint8)

    if ui_state.get("observation_boxes", True):
        for index, box in enumerate(observation_boxes):
            box_left = box[0] - left
            box_top = box[1] - top
            box_right = box_left + box[2]
            box_bottom = box_top + box[3]
            cv2.rectangle(
                view,
                (box_left, box_top),
                (box_right, box_bottom),
                TEMPLATE_COLORS[index],
                2,
            )
            cv2.putText(
                view,
                "O%d" % (index + 1),
                (box_left + 8, box_top + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                TEMPLATE_COLORS[index],
                2,
            )

        for index, box in enumerate(actual_boxes):
            box_left = box[0] - left
            box_top = box[1] - top
            box_right = box_left + box[2]
            box_bottom = box_top + box[3]
            cv2.rectangle(
                view,
                (box_left, box_top),
                (box_right, box_bottom),
                TEMPLATE_COLORS[index],
                1,
            )

    intersection_left = actual_intersection_box[0] - left
    intersection_top = actual_intersection_box[1] - top
    intersection_right = intersection_left + actual_intersection_box[2]
    intersection_bottom = intersection_top + actual_intersection_box[3]
    cv2.rectangle(
        view,
        (intersection_left, intersection_top),
        (intersection_right, intersection_bottom),
        config.actual_intersection_color,
        2,
    )

    actual_center = get_box_center(actual_intersection_box)
    local_center = (actual_center[0] - left, actual_center[1] - top)
    cv2.circle(view, local_center, 6, config.actual_intersection_color, -1)
    if ui_state.get("heading_arrow", True):
        draw_aircraft_marker(
            view,
            local_center,
            heading_degrees,
            max(36, min(view.shape[0], view.shape[1]) // 5),
            config.heading_indicator_color,
            2,
        )
    cv2.putText(
        view,
        "Baslik: %s" % format_heading_label(heading_degrees),
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        config.panel_title_color,
        2,
    )
    return view


def _draw_observation_tile(
    canvas: np.ndarray,
    image: np.ndarray,
    top_left: Tuple[int, int],
    size: int,
    label: str,
    color: Tuple[int, int, int],
    subtitle: str,
) -> None:
    x, y = top_left
    tile = cv2.resize(
        ensure_bgr(image),
        (size, size),
        interpolation=cv2.INTER_AREA if image.shape[0] >= size else cv2.INTER_NEAREST,
    )
    canvas[y : y + size, x : x + size] = tile
    cv2.rectangle(canvas, (x, y), (x + size, y + size), color, 2)
    cv2.putText(
        canvas,
        label,
        (x + 10, y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        color,
        2,
    )
    cv2.putText(
        canvas,
        subtitle,
        (x + 10, y + size - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
    )


def format_patch_subtitle(
    patch_index: int,
    altitude_state: AltitudeSimulationState,
    config: SimulationConfig,
) -> str:
    if is_altitude_scenario(config):
        return "%.1fm | x%.2f" % (
            altitude_state.patch_agl_m[patch_index],
            altitude_state.patch_scale_factors[patch_index],
        )
    return "normal | x%.2f" % altitude_state.patch_scale_factors[patch_index]


def create_observation_view(
    observation_map: np.ndarray,
    observation_boxes: List[Tuple[int, int, int, int]],
    actual_boxes: List[Tuple[int, int, int, int]],
    actual_intersection_box: Tuple[int, int, int, int],
    observation_windows: List[np.ndarray],
    altitude_state: AltitudeSimulationState,
    heading_degrees: float,
    ui_state: dict,
    config: SimulationConfig,
    actual_crop: Optional[np.ndarray] = None,
) -> np.ndarray:
    _ = (actual_boxes, actual_intersection_box, actual_crop, heading_degrees)
    if len(observation_boxes) < 3:
        return np.zeros(
            (config.sample_window_size, config.sample_window_size, 3),
            dtype=np.uint8,
        )

    hero_size = config.sample_window_size
    padding = 8
    canvas_width = hero_size
    canvas_height = padding * 2 + hero_size
    canvas = np.full(
        (canvas_height, canvas_width, 3),
        config.panel_background_color,
        dtype=np.uint8,
    )

    # Merkez gözlem penceresi (O2) yeşil kenarlıkla gösterilir
    if len(observation_windows) >= 2 and observation_windows[1] is not None:
        display_img = observation_windows[1]
    else:
        display_img = extract_padded_patch(observation_map, observation_boxes[1])

    norm_mode = ui_state.get("norm_mode", "HAM")
    display_img = apply_observation_norm(display_img, norm_mode)

    _draw_observation_tile(
        canvas,
        display_img,
        (padding, padding),
        hero_size - padding * 2,
        "O2",
        (0, 204, 0),
        format_patch_subtitle(1, altitude_state, config),
    )

    return canvas


def create_template_strip(
    templates: List[np.ndarray],
    config: SimulationConfig,
) -> np.ndarray:
    if len(templates) < 3:
        return np.zeros(
            (config.sample_window_size, config.sample_window_size, 3),
            dtype=np.uint8,
        )

    hero_size = config.sample_window_size
    padding = 8
    strip = np.full(
        (padding * 2 + hero_size, hero_size, 3),
        config.panel_background_color,
        dtype=np.uint8,
    )

    tile_sz = hero_size - padding * 2
    model_out = cv2.resize(
        ensure_bgr(templates[1]),
        (tile_sz, tile_sz),
        interpolation=cv2.INTER_AREA,
    )
    strip[padding : padding + tile_sz, padding : padding + tile_sz] = model_out
    cv2.rectangle(strip, (padding, padding), (padding + tile_sz, padding + tile_sz), (0, 204, 0), 2)
    cv2.putText(strip, "O2", (padding + 8, padding + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 204, 0), 2)

    return strip


def extract_padded_patch(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
) -> np.ndarray:
    x, y, width, height = box
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image.shape[1], x + width)
    y2 = min(image.shape[0], y + height)
    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((height, width), dtype=image.dtype)

    pad_top = max(0, -y)
    pad_left = max(0, -x)
    pad_bottom = max(0, (y + height) - image.shape[0])
    pad_right = max(0, (x + width) - image.shape[1])
    if pad_top or pad_bottom or pad_left or pad_right:
        patch = cv2.copyMakeBorder(
            patch,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REPLICATE,
        )
    return patch


def compose_triplet_diagnostic_image(
    reference_map: np.ndarray,
    actual_boxes: List[Tuple[int, int, int, int]],
    matched_boxes: List[Tuple[int, int, int, int]],
    templates: List[np.ndarray],
    observation_windows: List[np.ndarray],
    score_values: List[float],
    error_pixels: float,
    point_label: str,
    altitude_state: AltitudeSimulationState,
    config: SimulationConfig,
) -> np.ndarray:
    tile_size = int(config.diagnostic_tile_size)
    padding = 18
    gap = 14
    header_height = 116
    row_gap = 18
    cols = 4
    rows = 3
    canvas_width = (padding * 2) + (cols * tile_size) + ((cols - 1) * gap)
    canvas_height = (
        header_height
        + (padding * 2)
        + (rows * tile_size)
        + ((rows - 1) * row_gap)
    )
    canvas = np.full(
        (canvas_height, canvas_width, 3),
        config.panel_background_color,
        dtype=np.uint8,
    )

    header_lines = [
        "%s | scenario=%s | err=%.1f px"
        % (
            point_label,
            get_scenario_label(config),
            error_pixels,
        ),
        "Kolonlar: Gozlem | Gercek Patch | Model Template | Eslesen Patch",
    ]
    if is_altitude_scenario(config):
        header_lines[0] += " | alt=%.1f m agl | gsd=%.2f cm/px" % (
            altitude_state.altitude_agl_m,
            altitude_state.center_gsd_cm_per_px,
        )
    for line_index, line in enumerate(header_lines):
        cv2.putText(
            canvas,
            line,
            (padding, 32 + (line_index * 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.74,
            config.panel_title_color,
            2,
            cv2.LINE_AA,
        )

    column_titles = (
        "Gozlem",
        "Gercek Patch",
        "Model Template",
        "Eslesen Patch",
    )
    for column_index, title in enumerate(column_titles):
        title_x = padding + (column_index * (tile_size + gap))
        cv2.putText(
            canvas,
            title,
            (title_x, header_height - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            config.panel_title_color,
            2,
            cv2.LINE_AA,
        )

    for row_index in range(rows):
        base_y = header_height + padding + (row_index * (tile_size + row_gap))
        row_label = "O%d %s | score=%.4f" % (
            row_index + 1,
            format_patch_subtitle(row_index, altitude_state, config),
            score_values[row_index],
        )
        cv2.putText(
            canvas,
            row_label,
            (padding, base_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            TEMPLATE_COLORS[row_index],
            2,
            cv2.LINE_AA,
        )
        row_images = [
            observation_windows[row_index],
            extract_padded_patch(reference_map, actual_boxes[row_index]),
            templates[row_index],
            extract_padded_patch(reference_map, matched_boxes[row_index]),
        ]
        for column_index, row_image in enumerate(row_images):
            tile_x = padding + (column_index * (tile_size + gap))
            tile = cv2.resize(
                ensure_bgr(row_image),
                (tile_size, tile_size),
                interpolation=cv2.INTER_NEAREST,
            )
            canvas[base_y : base_y + tile_size, tile_x : tile_x + tile_size] = tile
            border_color = (
                TEMPLATE_COLORS[row_index]
                if column_index in (0, 1, 2, 3)
                else config.panel_border_color
            )
            cv2.rectangle(
                canvas,
                (tile_x, base_y),
                (tile_x + tile_size, base_y + tile_size),
                border_color,
                2,
            )

    return canvas


def run_template_diagnostics(
    reference_map: np.ndarray,
    observation_map: np.ndarray,
    model: object,
    terrain_context: Optional[TerrainContext],
    config: SimulationConfig,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.diagnostic_output_dir / ("template_diag_" + timestamp)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    heading_degrees = normalize_heading_degrees(config.initial_heading_degrees)

    for case_index, (row_seed, col_seed) in enumerate(
        config.diagnostic_benchmark_points,
        start=1,
    ):
        row, col = clamp_observation_cursor(
            row_seed,
            col_seed,
            observation_map.shape,
            config,
        )
        (
            templates,
            observation_windows,
            observation_boxes,
            actual_boxes,
            row,
            col,
            altitude_state,
        ) = extract_template_triplet(
            observation_map,
            row,
            col,
            heading_degrees,
            config.initial_altitude_agl_m,
            terrain_context,
            model,
            config,
        )
        actual_intersection_box, _ = compute_intersection_box(actual_boxes)
        actual_center = get_box_center(actual_intersection_box)
        search_region, search_origin, search_window_box, search_mode = extract_search_region(
            reference_map,
            actual_center,
            config.base_search_window_size,
            0,
            config,
        )
        (
            score_values,
            matched_boxes,
            predicted_intersection_box,
            intersection_mode,
            match_backend,
            match_evidence,
        ) = localize_template_triplet(
            search_region,
            search_origin,
            resize_templates_to_effective_size(
                templates,
                config,
                altitude_state.patch_scale_factors,
            ),
            config,
        )
        predicted_center = get_box_center(predicted_intersection_box)
        error_pixels = compute_error_pixels(predicted_center, actual_center)

        case_name = "case_%02d_r%d_c%d" % (case_index, row, col)
        diagnostic_image = compose_triplet_diagnostic_image(
            reference_map,
            actual_boxes,
            matched_boxes,
            templates,
            observation_windows,
            score_values,
            error_pixels,
            case_name,
            altitude_state,
            config,
        )
        cv2.imwrite(str(output_dir / (case_name + "_triptych.png")), diagnostic_image)

        metadata = {
            "case_index": case_index,
            "row": row,
            "col": col,
            "actual_center": list(actual_center),
            "predicted_center": list(predicted_center),
            "error_pixels": float(error_pixels),
            "score_values": [float(value) for value in score_values],
            "intersection_mode": intersection_mode,
            "search_mode": search_mode,
            "match_backend": match_backend,
            "peak_margins": [
                float(evidence.peak_margin) for evidence in match_evidence
            ],
            "template_stddevs": [
                float(evidence.template_stddev) for evidence in match_evidence
            ],
            "search_window_box": list(search_window_box),
            "actual_boxes": [list(box) for box in actual_boxes],
            "matched_boxes": [list(box) for box in matched_boxes],
            "patch_scale_factors": [
                float(value) for value in altitude_state.patch_scale_factors
            ],
            "patch_agl_m": [float(value) for value in altitude_state.patch_agl_m],
        }
        (output_dir / (case_name + "_meta.json")).write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )
        summary_rows.append(metadata)

    summary = {
        "scenario_mode": get_scenario_label(config),
        "reference_map_path": str(config.reference_map_path),
        "observation_map_path": str(config.observation_map_path),
        "model_path": str(config.model_path),
        "case_count": len(summary_rows),
        "cases": summary_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print("Template diagnostics exported to %s" % output_dir)
    return output_dir


def _draw_score_bars(
    canvas: np.ndarray,
    score_values: List[float],
    x: int,
    y: int,
    bar_w: int = 80,
    bar_h: int = 10,
    gap: int = 4,
) -> None:
    """Her template için normalize edilmiş skor çubuğu çizer."""
    for idx, raw_score in enumerate(score_values[:3]):
        # TM_CCOEFF_NORMED: [-1, 1] → [0, 1]
        norm = max(0.0, min(1.0, (float(raw_score) + 1.0) / 2.0))
        color_r = int(255 * (1.0 - norm))
        color_g = int(255 * norm)
        bar_y = y + idx * (bar_h + gap)
        cv2.rectangle(canvas, (x, bar_y), (x + bar_w, bar_y + bar_h), (50, 50, 60), -1)
        fill_w = max(1, int(round(bar_w * norm)))
        cv2.rectangle(canvas, (x, bar_y), (x + fill_w, bar_y + bar_h), (0, color_g, color_r), -1)
        cv2.rectangle(canvas, (x, bar_y), (x + bar_w, bar_y + bar_h), (80, 80, 90), 1)
        cv2.putText(
            canvas,
            "T%d %.3f" % (idx + 1, raw_score),
            (x + bar_w + 6, bar_y + bar_h - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            TEMPLATE_COLORS[idx],
            1,
            cv2.LINE_AA,
        )


def _draw_confidence_bar(
    canvas: np.ndarray,
    confidence: float,
    is_reliable: bool,
    x: int,
    y: int,
    bar_w: int = 120,
    bar_h: int = 14,
) -> None:
    conf = max(0.0, min(1.0, confidence))
    r = int(255 * (1.0 - conf))
    g = int(255 * conf)
    border_color = (0, 200, 80) if is_reliable else (0, 80, 220)
    cv2.rectangle(canvas, (x, y), (x + bar_w, y + bar_h), (40, 40, 50), -1)
    fill_w = max(1, int(round(bar_w * conf)))
    cv2.rectangle(canvas, (x, y), (x + fill_w, y + bar_h), (0, g, r), -1)
    cv2.rectangle(canvas, (x, y), (x + bar_w, y + bar_h), border_color, 1)
    label = "TAMAM" if is_reliable else "DUSUK"
    cv2.putText(
        canvas,
        "%.2f %s" % (conf, label),
        (x + bar_w + 6, y + bar_h - 1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        border_color,
        1,
        cv2.LINE_AA,
    )


_TP_BG      = (42, 38, 37)     # C_CARD_BG  #252740 → BGR
_TP_BORDER  = (85, 61, 58)     # C_BORDER   #3a3d55
_TP_LABEL   = (158, 127, 123)  # C_MUTED    #7b7f9e
_TP_VALUE   = (240, 234, 232)  # C_TEXT     #e8eaf0
_TP_ACCENT  = (244, 133, 66)   # C_ACCENT   #4285f4
_TP_SUCCESS = (80, 175, 76)    # C_SUCCESS  #4caf50
_TP_WARN    = (0, 152, 255)    # C_WARN     #ff9800
_TP_DANGER  = (68, 68, 244)    # red        #f44336
_TP_PANEL_W = 180              # sağ panel genişliği (piksel) — config.right_info_panel_width ile eşleşmeli


def _draw_metric_card(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    label: str,
    value: str,
    value_color: Tuple[int, int, int] = _TP_VALUE,
    label_color: Tuple[int, int, int] = _TP_LABEL,
) -> None:
    _draw_alpha_rounded_panel(canvas, x, y, x + w, y + h, 7, _TP_BG, 0.88)
    _draw_rounded_rect(canvas, x, y, x + w, y + h, 7, _TP_BORDER, 1)
    cv2.putText(
        canvas, label, (x + 9, y + 18),
        cv2.FONT_HERSHEY_SIMPLEX, 0.44, label_color, 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas, value, (x + 9, y + h - 9),
        cv2.FONT_HERSHEY_SIMPLEX, 0.78, value_color, 2, cv2.LINE_AA,
    )


def draw_right_telemetry_panel(
    canvas: np.ndarray,
    panel_rect: Tuple[int, int, int, int],
    heading_degrees: float,
    altitude_state: "AltitudeSimulationState",
    error_pixels: float,
    step_count: int,
    last_action: str,
    score_values: List[float],
    intersection_mode: str,
    search_window_size: int,
    ui_state: dict,
    config: "SimulationConfig",
    quality: Optional["LocalizationQuality"] = None,
    kalman_error_pixels: Optional[float] = None,
    autonomous_mode: bool = False,
    waypoint_distance_px: Optional[float] = None,
) -> None:
    px, py, pw, ph = panel_rect

    # Panel arka planı
    _draw_alpha_rounded_panel(canvas, px, py, px + pw, py + ph, 10, (30, 28, 26), 0.82)
    _draw_rounded_rect(canvas, px, py, px + pw, py + ph, 10, _TP_BORDER, 1)

    # Başlık
    title = "OTONOM" if autonomous_mode else "NAVİGASYON"
    title_color = _TP_SUCCESS if autonomous_mode else _TP_ACCENT
    _put_text_tr(canvas, title, (px + 12, py + 22), 17, title_color, 2)
    sep_y = py + 30
    cv2.line(canvas, (px + 8, sep_y), (px + pw - 8, sep_y), _TP_BORDER, 1, cv2.LINE_AA)

    gsd_cm = (
        altitude_state.center_gsd_cm_per_px
        if altitude_state.center_gsd_cm_per_px > 0.0
        else float(config.reference_map_gsd_cm_per_px)
    )
    error_m = error_pixels * gsd_cm / 100.0

    # Hata rengini belirle
    if error_m < 30:
        err_color = _TP_SUCCESS
    elif error_m < 80:
        err_color = _TP_WARN
    else:
        err_color = _TP_DANGER

    # Güven rengi
    if quality is not None:
        conf_pct = quality.confidence * 100.0
        conf_color = _TP_SUCCESS if quality.is_reliable else (_TP_WARN if conf_pct >= 40 else _TP_DANGER)
        conf_str = "%.0f%%" % conf_pct
    else:
        conf_color = _TP_LABEL
        conf_str = "--"

    # Kartlar — 2 sütunlu ızgara
    card_h   = 52
    card_gap = 6
    cw2      = (pw - 12 - card_gap) // 2   # tek kart genişliği (2 sütun)
    cw1      = pw - 12                      # tam genişlik kart

    cards_2col = [
        ("HDG", format_heading_label(heading_degrees), _TP_VALUE),
        ("ALT", "%.0fm" % altitude_state.altitude_agl_m, _TP_VALUE),
        ("GSD", "%.1fcm" % gsd_cm, _TP_VALUE),
        ("ERR", "%.1fm" % error_m, err_color),
        ("ADIM", str(step_count), _TP_VALUE),
        ("GUVEN", conf_str, conf_color),
    ]

    cx0 = py + 38
    for i, (lbl, val, col) in enumerate(cards_2col):
        row = i // 2
        col_idx = i % 2
        cx = px + 6 + col_idx * (cw2 + card_gap)
        cy = cx0 + row * (card_h + card_gap)
        _draw_metric_card(canvas, cx, cy, cw2, card_h, lbl, val, col)

    # Tam genişlik kartlar (KLM, ISC, ROI, CMD)
    full_y = cx0 + 3 * (card_h + card_gap)
    _kalman_is_on = bool(ui_state.get("kalman_on", False))
    if _kalman_is_on and kalman_error_pixels is not None:
        _klm_val = "%.1fm" % (kalman_error_pixels * gsd_cm / 100.0)
    else:
        _klm_val = "AKTIF" if _kalman_is_on else "KAPALI"
    _klm_color = _TP_SUCCESS if _kalman_is_on else _TP_LABEL
    _norm_mode = ui_state.get("norm_mode", "HAM")
    _nrm_color = _TP_SUCCESS if _norm_mode != "HAM" else _TP_LABEL
    _obs_272 = bool(ui_state.get("obs_272_mode", False))
    _obw_color = _TP_SUCCESS if _obs_272 else _TP_LABEL
    full_cards = [
        ("KLM", _klm_val, _klm_color),
        ("NRM", _norm_mode, _nrm_color),
        ("OBW", "272>256" if _obs_272 else "544 std", _obw_color),
        ("ISC", intersection_mode, _TP_VALUE),
        ("ROI", "%d px" % search_window_size, _TP_VALUE),
        ("CMD", get_action_label(last_action), _TP_VALUE),
    ]
    if waypoint_distance_px is not None:
        wpt_m = waypoint_distance_px * gsd_cm / 100.0
        full_cards.insert(0, ("WPT", "%.0fm" % wpt_m, _TP_SUCCESS))

    for lbl, val, col in full_cards:
        _draw_metric_card(canvas, px + 6, full_y, cw1, card_h - 8, lbl, val, col)
        full_y += (card_h - 8) + card_gap

    # Skor çubukları
    full_y += 4
    if full_y + 60 < py + ph:
        cv2.putText(
            canvas, "SKORLAR", (px + 12, full_y + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.40, _TP_LABEL, 1, cv2.LINE_AA,
        )
        _draw_score_bars(canvas, score_values, px + 12, full_y + 18)

    # Güven çubuğu
    if quality is not None:
        bar_y = full_y + 18 + 3 * 14 + 14
        if bar_y + 20 < py + ph:
            _draw_confidence_bar(
                canvas, quality.confidence, quality.is_reliable, px + 12, bar_y,
            )


def draw_hud(
    canvas: np.ndarray,
    map_rect: Tuple[int, int, int, int],
    score_values: List[float],
    observation_cursor: Tuple[int, int],
    predicted_center: Tuple[int, int],
    actual_center: Tuple[int, int],
    error_pixels: float,
    kalman_error_pixels: Optional[float],
    step_count: int,
    last_action: str,
    heading_degrees: float,
    altitude_state: AltitudeSimulationState,
    intersection_mode: str,
    search_mode: str,
    match_backend: str,
    search_window_size: int,
    ui_state: dict,
    config: SimulationConfig,
    quality: Optional[LocalizationQuality] = None,
    autonomous_mode: bool = False,
    waypoint_distance_px: Optional[float] = None,
) -> None:
    x, y, width, height = map_rect

    help_line_1 = "WASD hareket  |  Q/E dönüş  |  P otonom"
    if is_altitude_scenario(config):
        help_line_1 += "  |  +/- irtifa"
    help_line_2 = "H panel  |  T iz  |  O ROI  |  K Kalman  |  N norm  |  V 272-mod  |  ESC/X çıkış"
    _hl1_w, _hl1_h = cv2.getTextSize(
        help_line_1, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2
    )[0]
    _hl2_w, _hl2_h = cv2.getTextSize(
        help_line_2, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2
    )[0]
    _help_bg_x0 = x + 8
    _help_bg_y0 = y + height - _hl1_h - _hl2_h - 28
    _help_bg_x1 = min(x + width - 4, x + 8 + max(_hl1_w, _hl2_w) + 8)
    _help_bg_y1 = y + height - 4
    _draw_alpha_rounded_panel(
        canvas, _help_bg_x0, _help_bg_y0, _help_bg_x1, _help_bg_y1,
        6, (20, 22, 30), 0.60,
    )
    _put_text_tr(canvas, help_line_1, (x + 12, y + height - 36), 15, config.panel_title_color, 2)
    _put_text_tr(canvas, help_line_2, (x + 12, y + height - 14), 14, config.panel_title_color, 1)


def draw_localization_dashboard(
    observation_rect: Tuple[int, int, int, int],
    template_rect: Tuple[int, int, int, int],
    reference_preview_state: ReferencePreviewState,
    observation_view: np.ndarray,
    template_strip: np.ndarray,
    matched_boxes: List[Tuple[int, int, int, int]],
    predicted_intersection_box: Tuple[int, int, int, int],
    actual_intersection_box: Tuple[int, int, int, int],
    search_window_box: Tuple[int, int, int, int],
    predicted_history: List[Tuple[int, int]],
    actual_history: List[Tuple[int, int]],
    score_values: List[float],
    observation_cursor: Tuple[int, int],
    step_count: int,
    last_action: str,
    heading_degrees: float,
    altitude_state: AltitudeSimulationState,
    intersection_mode: str,
    search_mode: str,
    match_backend: str,
    search_window_size: int,
    ui_state: dict,
    runtime_ui_buttons: List[dict],
    config: SimulationConfig,
    kalman_center: Optional[Tuple[int, int]] = None,
    waypoint_target: Optional[Tuple[int, int]] = None,
    autonomous_mode: bool = False,
    quality: Optional[LocalizationQuality] = None,
    kalman_error_pixels: Optional[float] = None,
    waypoint_distance_px: Optional[float] = None,
    ref_patch_image: Optional[np.ndarray] = None,
    ref_patch_rect: Optional[Tuple[int, int, int, int]] = None,
    right_panel_rect: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    dashboard_width, dashboard_height = config.display_size
    canvas = np.full(
        (dashboard_height, dashboard_width, 3),
        config.dashboard_background_color,
        dtype=np.uint8,
    )

    map_rect = reference_preview_state.panel_rect
    preview = reference_preview_state.base_preview.copy()

    predicted_center = get_box_center(predicted_intersection_box)
    actual_center = get_box_center(actual_intersection_box)

    overlay_thickness = max(2, int(round(6 * reference_preview_state.scale_x)))
    marker_radius = max(4, int(round(14 * reference_preview_state.scale_x)))

    if ui_state.get("trajectory", True):
        draw_scaled_path(
            preview,
            actual_history,
            reference_preview_state,
            config.actual_path_color,
            overlay_thickness,
        )
        draw_scaled_path(
            preview,
            predicted_history,
            reference_preview_state,
            config.predicted_path_color,
            overlay_thickness,
        )

    if ui_state.get("tm_boxes", True):
        for index, box in enumerate(matched_boxes):
            scaled_box = scale_box_to_preview(box, reference_preview_state)
            cv2.rectangle(
                preview,
                (scaled_box[0], scaled_box[1]),
                (scaled_box[2], scaled_box[3]),
                TEMPLATE_COLORS[index],
                overlay_thickness,
            )
            cv2.putText(
                preview,
                "T%d" % (index + 1),
                (scaled_box[0] + 6, scaled_box[1] + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                TEMPLATE_COLORS[index],
                2,
            )

    scaled_actual_box = scale_box_to_preview(actual_intersection_box, reference_preview_state)
    scaled_predicted_box = scale_box_to_preview(predicted_intersection_box, reference_preview_state)
    scaled_search_box = scale_box_to_preview(
        (
            search_window_box[0],
            search_window_box[1],
            search_window_box[2] - search_window_box[0],
            search_window_box[3] - search_window_box[1],
        ),
        reference_preview_state,
    )
    scaled_actual_center = scale_point_to_preview(actual_center, reference_preview_state)
    scaled_predicted_center = scale_point_to_preview(predicted_center, reference_preview_state)

    if ui_state.get("roi_frame", True):
        cv2.rectangle(
            preview,
            (scaled_search_box[0], scaled_search_box[1]),
            (scaled_search_box[2], scaled_search_box[3]),
            config.search_window_color,
            max(1, overlay_thickness - 1),
        )

    cv2.rectangle(
        preview,
        (scaled_actual_box[0], scaled_actual_box[1]),
        (scaled_actual_box[2], scaled_actual_box[3]),
        config.actual_intersection_color,
        overlay_thickness,
    )
    cv2.rectangle(
        preview,
        (scaled_predicted_box[0], scaled_predicted_box[1]),
        (scaled_predicted_box[2], scaled_predicted_box[3]),
        config.predicted_intersection_color,
        overlay_thickness,
    )
    cv2.circle(preview, scaled_actual_center, marker_radius, config.actual_intersection_color, -1)
    cv2.circle(
        preview,
        scaled_predicted_center,
        marker_radius,
        config.predicted_intersection_color,
        -1,
    )
    cv2.line(
        preview,
        scaled_actual_center,
        scaled_predicted_center,
        config.error_line_color,
        overlay_thickness,
    )
    if ui_state.get("heading_arrow", True):
        draw_aircraft_marker(
            preview,
            scaled_actual_center,
            heading_degrees,
            max(marker_radius * 4, 28),
            config.heading_indicator_color,
            max(2, overlay_thickness - 1),
        )

    # --- Kalman tahmin noktası (sarı) ---
    if kalman_center is not None and config.kalman_enabled:
        scaled_kalman = scale_point_to_preview(kalman_center, reference_preview_state)
        kalman_r = max(3, marker_radius - 2)
        cv2.circle(preview, scaled_kalman, kalman_r + 2, (0, 0, 0), -1)
        cv2.circle(preview, scaled_kalman, kalman_r, (0, 220, 255), -1)
        cv2.putText(
            preview, "K",
            (scaled_kalman[0] + kalman_r + 3, scaled_kalman[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 1, cv2.LINE_AA,
        )

    # --- Waypoint işaretçisi (magenta) ---
    if waypoint_target is not None:
        scaled_wp = scale_point_to_preview(waypoint_target, reference_preview_state)
        wp_color = (255, 40, 200)
        cv2.drawMarker(preview, scaled_wp, wp_color, cv2.MARKER_CROSS, 28, 2, cv2.LINE_AA)
        accept_r = max(4, int(round(config.waypoint_acceptance_radius_px * reference_preview_state.scale_x)))
        cv2.circle(preview, scaled_wp, accept_r, wp_color, 1, cv2.LINE_AA)
        cv2.putText(
            preview, "WP",
            (scaled_wp[0] + accept_r + 3, scaled_wp[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, wp_color, 1, cv2.LINE_AA,
        )
        # Kalman → waypoint yönlendirme çizgisi (otonom modda)
        if autonomous_mode and kalman_center is not None:
            scaled_kc = scale_point_to_preview(kalman_center, reference_preview_state)
            cv2.line(preview, scaled_kc, scaled_wp, wp_color, 1, cv2.LINE_AA)

    _GREEN = (0, 204, 0)
    _BLUE = TEMPLATE_COLORS[2]
    draw_panel(canvas, observation_view, observation_rect, config.observation_panel_title, config, accent_color=_GREEN)
    draw_panel(canvas, template_strip, template_rect, config.template_panel_title, config, accent_color=_BLUE)
    if ref_patch_rect is not None:
        if ref_patch_image is not None:
            _rp_img = (
                ref_patch_image if ref_patch_image.ndim == 3
                else cv2.cvtColor(ref_patch_image, cv2.COLOR_GRAY2BGR)
            )
        else:
            _rp_img = np.full(
                (config.sample_window_size, config.sample_window_size, 3),
                config.panel_background_color,
                dtype=np.uint8,
            )
        draw_panel(canvas, _rp_img, ref_patch_rect, "Eşleşen Bölge", config, accent_color=_GREEN)
    draw_panel_frame(canvas, map_rect, config.reference_panel_title, config)
    canvas[
        reference_preview_state.paste_y : reference_preview_state.paste_y
        + reference_preview_state.preview_height,
        reference_preview_state.paste_x : reference_preview_state.paste_x
        + reference_preview_state.preview_width,
    ] = preview

    _err_px = compute_error_pixels(predicted_center, actual_center)
    draw_hud(
        canvas,
        map_rect,
        score_values,
        observation_cursor,
        predicted_center,
        actual_center,
        _err_px,
        kalman_error_pixels=kalman_error_pixels,
        step_count=step_count,
        last_action=last_action,
        heading_degrees=heading_degrees,
        altitude_state=altitude_state,
        intersection_mode=intersection_mode,
        search_mode=search_mode,
        match_backend=match_backend,
        search_window_size=search_window_size,
        ui_state=ui_state,
        config=config,
        quality=quality,
        autonomous_mode=autonomous_mode,
        waypoint_distance_px=waypoint_distance_px,
    )
    _rp_rect = right_panel_rect if right_panel_rect is not None else (
        map_rect[0] + map_rect[2] + 4, map_rect[1], _TP_PANEL_W, map_rect[3]
    )
    draw_right_telemetry_panel(
        canvas,
        _rp_rect,
        heading_degrees=heading_degrees,
        altitude_state=altitude_state,
        error_pixels=_err_px,
        step_count=step_count,
        last_action=last_action,
        score_values=score_values,
        intersection_mode=intersection_mode,
        search_window_size=search_window_size,
        ui_state=ui_state,
        config=config,
        quality=quality,
        kalman_error_pixels=kalman_error_pixels,
        autonomous_mode=autonomous_mode,
        waypoint_distance_px=waypoint_distance_px,
    )
    _draw_splitter_handles(canvas, config, ui_state)

    if config.ui_buttons_enabled:
        _draw_runtime_buttons(canvas, ui_state, runtime_ui_buttons, config)

    return canvas


def move_observation_cursor(
    row: int,
    col: int,
    action: str,
    image_shape: Tuple[int, int],
    heading_degrees: float,
    config: SimulationConfig,
    step_size_px: Optional[float] = None,
) -> Tuple[int, int]:
    move_x = 0.0
    move_y = 0.0
    action_step_size = float(config.step_size if step_size_px is None else step_size_px)

    if action == "forward":
        move_x, move_y = rotate_image_offset(0.0, -action_step_size, heading_degrees)
    elif action == "backward":
        move_x, move_y = rotate_image_offset(0.0, action_step_size, heading_degrees)
    elif action == "strafe_right":
        move_x, move_y = rotate_image_offset(action_step_size, 0.0, heading_degrees)
    elif action == "strafe_left":
        move_x, move_y = rotate_image_offset(-action_step_size, 0.0, heading_degrees)

    row += int(round(move_y))
    col += int(round(move_x))

    return clamp_observation_cursor(row, col, image_shape, config)


def apply_control_action(
    row: int,
    col: int,
    heading_degrees: float,
    altitude_agl_m: float,
    action: str,
    image_shape: Tuple[int, int],
    config: SimulationConfig,
    step_size_px: Optional[float] = None,
) -> Tuple[int, int, float, float]:
    if action == "rotate_left":
        return (
            row,
            col,
            normalize_heading_degrees(heading_degrees - config.rotation_step_degrees),
            altitude_agl_m,
        )
    if action == "rotate_right":
        return (
            row,
            col,
            normalize_heading_degrees(heading_degrees + config.rotation_step_degrees),
            altitude_agl_m,
        )
    if action == "altitude_up":
        if not is_altitude_scenario(config):
            return row, col, heading_degrees, altitude_agl_m
        return (
            row,
            col,
            heading_degrees,
            clamp_altitude_agl(altitude_agl_m + config.altitude_step_m, config),
        )
    if action == "altitude_down":
        if not is_altitude_scenario(config):
            return row, col, heading_degrees, altitude_agl_m
        return (
            row,
            col,
            heading_degrees,
            clamp_altitude_agl(altitude_agl_m - config.altitude_step_m, config),
        )

    row, col = move_observation_cursor(
        row,
        col,
        action,
        image_shape,
        heading_degrees,
        config,
        step_size_px,
    )
    return row, col, heading_degrees, altitude_agl_m


def get_action_from_key(key: int) -> str:
    if key in UP_KEYS:
        return "forward"
    if key in DOWN_KEYS:
        return "backward"
    if key in LEFT_KEYS:
        return "strafe_left"
    if key in RIGHT_KEYS:
        return "strafe_right"
    if key in ROTATE_LEFT_KEYS:
        return "rotate_left"
    if key in ROTATE_RIGHT_KEYS:
        return "rotate_right"
    if key in ALTITUDE_UP_KEYS:
        return "altitude_up"
    if key in ALTITUDE_DOWN_KEYS:
        return "altitude_down"
    return ""


def print_localization_status(
    score_values: List[float],
    matched_boxes: List[Tuple[int, int, int, int]],
    predicted_intersection_box: Tuple[int, int, int, int],
    actual_intersection_box: Tuple[int, int, int, int],
    row: int,
    col: int,
    error_pixels: float,
    step_count: int,
    last_action: str,
    heading_degrees: float,
    altitude_state: AltitudeSimulationState,
    intersection_mode: str,
    search_mode: str,
    match_backend: str,
    search_window_size: int,
    config: SimulationConfig,
) -> None:
    strict_triplet_lock = is_strict_triplet_alignment(
        matched_boxes,
        intersection_mode,
        config,
        heading_degrees,
    )
    print("cursor=(row=%d, col=%d)" % (row, col))
    print(
        "scores=(%.4f, %.4f, %.4f) scenario=%s intersection_mode=%s search=%s backend=%s"
        % (
            score_values[0],
            score_values[1],
            score_values[2],
            get_scenario_label(config),
            intersection_mode,
            search_mode,
            match_backend,
        )
    )
    print("matched_boxes=%s" % (matched_boxes,))
    status_line = (
        "predicted_intersection=%s actual_intersection=%s error=%.1fpx step=%d action=%s heading=%s"
        % (
            predicted_intersection_box,
            actual_intersection_box,
            error_pixels,
            step_count,
            get_action_label(last_action),
            format_heading_label(heading_degrees),
        )
    )
    if is_altitude_scenario(config):
        status_line += " alt=%.1fm agl gsd=%.2fcm/px" % (
            altitude_state.altitude_agl_m,
            altitude_state.center_gsd_cm_per_px,
        )
    status_line += " window=%d lock=%s" % (
        search_window_size,
        "strict-triplet" if strict_triplet_lock else "partial",
    )
    print(status_line)


def update_search_window_size(
    current_search_window_size: int,
    matched_boxes: List[Tuple[int, int, int, int]],
    intersection_mode: str,
    config: SimulationConfig,
    heading_degrees: float = 0.0,
    use_kalman: bool = False,
    quality_reliable: bool = True,
    reference_map_shape: Optional[Tuple[int, int]] = None,
) -> int:
    factor = float(config.kalman_window_growth_factor) if use_kalman else 1.0
    maximum_window_size = config.max_search_window_size
    if config.progressive_global_recovery and reference_map_shape is not None:
        maximum_window_size = max(
            maximum_window_size,
            int(max(reference_map_shape[:2])),
        )
    if not quality_reliable:
        return min(
            maximum_window_size,
            current_search_window_size
            + max(1, int(config.search_window_failure_growth * factor)),
        )
    if is_strict_triplet_alignment(matched_boxes, intersection_mode, config, heading_degrees):
        return config.base_search_window_size
    if intersection_mode in ("abc", "ab", "bc", "ac"):
        return min(
            maximum_window_size,
            current_search_window_size + max(1, int(config.search_window_growth_step * factor)),
        )
    return min(
        maximum_window_size,
        current_search_window_size + max(1, int(config.search_window_failure_growth * factor)),
    )


def should_force_global_recovery(
    search_anchor_center: Optional[Tuple[int, int]],
    low_confidence_steps: int,
    current_search_window_size: int,
    config: SimulationConfig,
    reference_map_shape: Optional[Tuple[int, int]] = None,
) -> bool:
    minimum_recovery_window = min(
        config.max_search_window_size,
        max(config.base_search_window_size, config.global_recovery_min_window_size),
    )
    if config.progressive_global_recovery and reference_map_shape is not None:
        # Tam taramaya ancak turuncu ROI zaten haritanın tüm uzun kenarını
        # kaplayacak kadar büyüdüğünde geç. Böylece görünüm bir anda sıçramaz.
        minimum_recovery_window = max(
            minimum_recovery_window,
            int(max(reference_map_shape[:2])),
        )
    return bool(
        search_anchor_center is not None
        and low_confidence_steps
        >= max(1, config.global_recovery_after_low_confidence_steps)
        and current_search_window_size >= minimum_recovery_window
    )


def choose_initial_cursor(
    observation_map_shape: Tuple[int, int],
    config: SimulationConfig,
) -> Tuple[int, int]:
    if not config.random_start:
        return clamp_observation_cursor(
            config.initial_row,
            config.initial_col,
            observation_map_shape,
            config,
        )

    minimum, maximum_row, maximum_col = get_observation_cursor_limits(
        observation_map_shape,
        config,
    )
    return (
        sample_center_biased_coordinate(
            minimum,
            maximum_row,
            config.random_start_middle_band_ratio,
        ),
        sample_center_biased_coordinate(
            minimum,
            maximum_col,
            config.random_start_middle_band_ratio,
        ),
    )


def seed_known_initial_position(
    previous_predicted_center: Optional[Tuple[int, int]],
    actual_center: Tuple[int, int],
    step_count: int,
    config: SimulationConfig,
) -> Optional[Tuple[int, int]]:
    """Use the declared start fix as a prior, without exposing later truth."""
    if (
        config.initial_position_known
        and step_count == 0
        and previous_predicted_center is None
    ):
        return actual_center
    return previous_predicted_center


def main(
    config=None,
    _display_fn=None,
    _getkey_fn=None,
    _use_qt: bool = False,
    _ctx_holder=None,
    _telemetry_fn=None,
    _status_fn=None,
) -> None:
    def emit_status(message: str, level: str = "info") -> None:
        if _status_fn is not None:
            _status_fn(message, level)

    if config is None:
        config = SimulationConfig()
        config = _apply_args_to_config(config, _parse_args())
    emit_status("Veri yolları doğrulanıyor", "loading")
    config = resolve_config_paths(config)
    reference_map, observation_map, model = load_assets(config, emit_status)
    terrain_context: Optional[TerrainContext] = None

    try:
        if is_altitude_scenario(config):
            terrain_context = load_terrain_context(observation_map.shape, config)

        if config.diagnostic_benchmark_enabled:
            run_template_diagnostics(
                reference_map, observation_map, model, terrain_context, config,
            )
            if config.diagnostic_benchmark_only:
                return

        observation_rect, template_rect, map_rect, ref_patch_rect, right_panel_rect = get_dashboard_layout(config)

        row, col = choose_initial_cursor(observation_map.shape, config)
        predicted_history: List[Tuple[int, int]] = []
        actual_history: List[Tuple[int, int]] = []
        step_count = 0
        last_action = ""
        last_action_step_size = 0.0
        heading_degrees = normalize_heading_degrees(config.initial_heading_degrees)
        altitude_agl_m = clamp_altitude_agl(config.initial_altitude_agl_m, config)
        previous_predicted_center: Optional[Tuple[int, int]] = None
        tentative_search_center: Optional[Tuple[int, int]] = None
        search_window_size = config.base_search_window_size
        kalman: Optional[PositionKalmanFilter] = None
        low_confidence_steps = 0
        waypoint_hits = 0
        active_waypoint_target: Optional[Tuple[int, int]] = None
        previous_waypoint_distance_px: Optional[float] = None
        auto_no_progress_steps = 0
        auto_last_action = ""
        auto_same_action_steps = 0

        runtime_ui_state = create_runtime_ui_state(config)
        runtime_ui_buttons = _build_runtime_buttons() if config.ui_buttons_enabled else []
        _lb_state: dict = {"scale": 1.0, "x_off": 0, "y_off": 0}
        runtime_ui_context: dict = {
            "state": runtime_ui_state,
            "buttons": runtime_ui_buttons,
            "reference_preview_state": None,
            "waypoint_target": None,
            "config": config,
            "_lb": _lb_state,
        }
        if _ctx_holder is not None:
            _ctx_holder[0] = runtime_ui_context

        csv_writer, csv_file = _open_csv_log(config)

        try:
            if not _use_qt:
                cv2.namedWindow(config.dashboard_window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(
                    config.dashboard_window_name,
                    config.display_size[0],
                    config.display_size[1],
                )
                cv2.setMouseCallback(
                    config.dashboard_window_name,
                    _runtime_buttons_mouse_cb,
                    runtime_ui_context,
                )

            while True:
                step_started_at = time.perf_counter()
                current_search_window_size = search_window_size
                runtime_obs_window_size = int(
                    runtime_ui_state.get("obs_window_size", config.sample_window_size)
                )
                runtime_match_config = dataclasses.replace(
                    config,
                    sample_window_size=runtime_obs_window_size,
                )
                (
                    templates,
                    observation_windows,
                    observation_boxes,
                    actual_boxes,
                    row,
                    col,
                    altitude_state,
                ) = extract_template_triplet(
                    observation_map, row, col, heading_degrees,
                    altitude_agl_m, terrain_context, model, config,
                    norm_mode=runtime_ui_state.get("norm_mode", "HISTEQ"),
                    obs_window_size=runtime_obs_window_size,
                )
                altitude_agl_m = altitude_state.altitude_agl_m
                actual_intersection_box, _ = compute_intersection_box(actual_boxes)
                actual_center = get_box_center(actual_intersection_box)

                # Senaryoda yalnız ilk konum bilinir. Bu açık başlangıç sabiti,
                # ilk eşlemeyi tam harita yerine turuncu ROI içinde başlatır;
                # sonraki gerçek konumlar lokalizasyon durumuna enjekte edilmez.
                previous_predicted_center = seed_known_initial_position(
                    previous_predicted_center,
                    actual_center,
                    step_count,
                    config,
                )

                motion_prior_center = propagate_center_with_action(
                    previous_predicted_center,
                    last_action,
                    heading_degrees,
                    last_action_step_size,
                )
                motion_tentative_center = propagate_center_with_action(
                    tentative_search_center,
                    last_action,
                    heading_degrees,
                    last_action_step_size,
                )
                search_anchor_center = (
                    motion_prior_center
                    if motion_prior_center is not None
                    else motion_tentative_center
                )
                force_global_recovery = should_force_global_recovery(
                    search_anchor_center,
                    low_confidence_steps,
                    current_search_window_size,
                    config,
                    reference_map.shape,
                )
                search_region, search_origin, _search_window_box, search_mode = extract_search_region(
                    reference_map, search_anchor_center, current_search_window_size,
                    step_count, config, force_global=force_global_recovery,
                )
                # Arama bölgesi 1:1 ölçekte kalır; modelin 512 px çıktısı,
                # aktif gözlem penceresinin gerçek harita ölçeğine indirilir.
                _match_templates = resize_templates_to_effective_size(
                    templates,
                    runtime_match_config,
                    altitude_state.patch_scale_factors,
                )
                (
                    score_values,
                    matched_boxes,
                    predicted_intersection_box,
                    intersection_mode,
                    match_backend,
                    match_evidence,
                ) = (
                    localize_template_triplet(
                        search_region,
                        search_origin,
                        _match_templates,
                        config,
                        0,
                    )
                )

                raw_predicted_center = get_box_center(predicted_intersection_box)
                # --- Lokalizasyon kalitesi (gps_denied_autonomy) ---
                quality = compute_localization_quality(
                    score_values,
                    matched_boxes,
                    predicted_intersection_box,
                    intersection_mode,
                    is_sqdiff_method(config.match_method),
                    config.localization_score_threshold,
                    config.localization_confidence_threshold,
                    config.localization_spread_threshold_px,
                    peak_margins=tuple(
                        evidence.peak_margin for evidence in match_evidence
                    ),
                    peak_margin_threshold=config.localization_peak_margin_threshold,
                    template_stddevs=tuple(
                        evidence.template_stddev for evidence in match_evidence
                    ),
                    template_std_threshold=config.localization_template_std_threshold,
                    strict_alignment=(
                        is_strict_triplet_alignment(
                            matched_boxes,
                            intersection_mode,
                            runtime_match_config,
                            heading_degrees,
                        )
                        if config.localization_require_strict_triplet
                        else None
                    ),
                )

                # --- Kalman filtresi (K tuşuyla çalışma anında aç/kapat) ---
                use_kalman = bool(runtime_ui_state.get("kalman_on", config.kalman_enabled))
                if use_kalman:
                    kalman_initialized_from_current_measurement = False
                    if kalman is None:
                        init_pos = previous_predicted_center
                        if init_pos is None and quality.is_reliable:
                            init_pos = raw_predicted_center
                            kalman_initialized_from_current_measurement = True
                        if init_pos is not None:
                            kalman = PositionKalmanFilter(
                                init_pos,
                                config.kalman_process_noise,
                                config.kalman_measurement_noise,
                            )
                    if kalman is not None:
                        # Filtre bu adımın görsel ölçümünden kurulmuşsa hareket zaten
                        # ölçümün içindedir; aynı komutu yeniden uygulamak konumu iki
                        # kez ilerletir ve bir sonraki ROI merkezini kaydırır.
                        if not kalman_initialized_from_current_measurement:
                            moved = propagate_center_with_action(
                                (0, 0), last_action, heading_degrees, last_action_step_size,
                            )
                            kalman.predict(
                                float(moved[0]) if moved is not None else 0.0,
                                float(moved[1]) if moved is not None else 0.0,
                            )
                        if quality.is_reliable:
                            kalman.update(
                                float(raw_predicted_center[0]),
                                float(raw_predicted_center[1]),
                                quality.confidence,
                            )
                else:
                    # Kalman kapalıyken filtreyi sıfırla; sonraki açılışta yeniden başlasın
                    kalman = None

                kalman_center: Optional[Tuple[int, int]] = (
                    kalman.position if (kalman is not None and use_kalman) else None
                )

                # --- Sensör füzyonu ---
                _prior_was_none = motion_prior_center is None
                fused_center, _fusion_ok, _jump_px = fuse_measurement_with_prior(
                    motion_prior_center,
                    raw_predicted_center,
                    quality,
                    config.max_visual_jump_px,
                    config.sensor_fusion_blend_gain,
                )
                # Step-0'da prior yokken kalite düşükse güvenli başlangıç konumuna dön
                if _prior_was_none and not _fusion_ok:
                    previous_predicted_center = None
                elif kalman_center is not None and use_kalman:
                    # Kalman açıkken arama çerçevesi Kalman pozisyonuna odaklanır;
                    # tek-adım yanlış eşleşmelerine karşı daha dayanıklı
                    previous_predicted_center = kalman_center
                else:
                    previous_predicted_center = fused_center

                if previous_predicted_center is not None and quality.is_reliable:
                    tentative_search_center = None
                elif previous_predicted_center is None:
                    candidate_is_usable_for_roi = (
                        quality.reason != "template_variance"
                        and all(math.isfinite(value) for value in raw_predicted_center)
                    )
                    tentative_search_center = (
                        raw_predicted_center
                        if candidate_is_usable_for_roi
                        else motion_tentative_center
                    )

                search_window_size = update_search_window_size(
                    current_search_window_size,
                    matched_boxes,
                    intersection_mode,
                    runtime_match_config,
                    heading_degrees,
                    use_kalman=use_kalman,
                    quality_reliable=quality.is_reliable,
                    reference_map_shape=reference_map.shape,
                )

                low_confidence_steps = (
                    0
                    if quality.is_reliable
                    else (
                        2
                        if search_mode == "global"
                        else low_confidence_steps + 1
                    )
                )

                display_predicted_center = (
                    kalman_center if kalman_center is not None else fused_center
                )
                error_pixels = compute_error_pixels(display_predicted_center, actual_center)
                kalman_error_pixels: Optional[float] = (
                    compute_error_pixels(kalman_center, actual_center)
                    if kalman_center is not None else None
                )

                autonomous_mode = bool(runtime_ui_state.get("autonomous_mode", False))
                waypoint_target: Optional[Tuple[int, int]] = runtime_ui_context.get("waypoint_target")
                if waypoint_target != active_waypoint_target:
                    active_waypoint_target = waypoint_target
                    waypoint_hits = 0
                    previous_waypoint_distance_px = None
                    auto_no_progress_steps = 0
                    auto_last_action = ""
                    auto_same_action_steps = 0
                if autonomous_mode and waypoint_target is not None:
                    waypoint_index, waypoint_hits = update_waypoint_progress(
                        0,
                        waypoint_hits,
                        display_predicted_center,
                        (waypoint_target,),
                        config.waypoint_acceptance_radius_px,
                        quality.confidence if quality.is_reliable else 0.0,
                        config.waypoint_acceptance_confidence_threshold,
                        config.waypoint_required_consecutive_hits,
                    )
                    if waypoint_index >= 1:
                        runtime_ui_context["waypoint_target"] = None
                        runtime_ui_state["autonomous_mode"] = False
                        runtime_ui_state["_dirty"] = True
                        active_waypoint_target = None
                        waypoint_target = None
                        waypoint_hits = 0
                        previous_waypoint_distance_px = None
                        auto_no_progress_steps = 0
                        auto_last_action = ""
                        auto_same_action_steps = 0
                        last_action = "hold"
                        last_action_step_size = 0.0
                        autonomous_mode = False
                waypoint_distance_px: Optional[float] = (
                    math.hypot(
                        display_predicted_center[0] - waypoint_target[0],
                        display_predicted_center[1] - waypoint_target[1],
                    )
                    if waypoint_target is not None else None
                )

                predicted_history.append(display_predicted_center)
                actual_history.append(actual_center)
                predicted_history = predicted_history[-config.path_history_limit :]
                actual_history = actual_history[-config.path_history_limit :]

                # Gösterim için arama çerçevesini güncel tahmin merkezine ortala;
                # üçlü kesişim sonrası küçülen pencere de UAV'ı merkeze alır
                display_search_window_box = get_search_window_box(
                    reference_map.shape, display_predicted_center, search_window_size,
                )

                # Ref-patch toggle'a ve sütun genişliklerine göre layout yeniden hesapla
                _show_ref_patch = bool(runtime_ui_state.get("ref_patch", False))
                observation_rect, template_rect, map_rect, ref_patch_rect, right_panel_rect = get_dashboard_layout(
                    config, _show_ref_patch, runtime_ui_state,
                )
                # Haritada yeşil (O2) eşleşmesinin bulduğu bölgeyi her adımda çıkar
                ref_patch_image: Optional[np.ndarray] = None
                if len(matched_boxes) >= 2:
                    _bx, _by, _bw, _bh = matched_boxes[1]
                    _ref_crop = reference_map[
                        max(0, _by) : min(reference_map.shape[0], _by + _bh),
                        max(0, _bx) : min(reference_map.shape[1], _bx + _bw),
                    ]
                    if _ref_crop.size > 0:
                        ref_patch_image = _ref_crop.copy()

                reference_viewport_box = get_reference_viewport_box(
                    reference_map.shape, predicted_intersection_box,
                    actual_intersection_box, display_search_window_box, search_mode, config,
                )
                reference_preview_state = create_reference_preview_state(
                    reference_map, map_rect, reference_viewport_box, config,
                )
                runtime_ui_context["reference_preview_state"] = reference_preview_state

                observation_view = create_observation_view(
                    observation_map, observation_boxes, actual_boxes, actual_intersection_box,
                    observation_windows, altitude_state, heading_degrees, runtime_ui_state, config,
                )
                template_strip = create_template_strip(templates, config)

                print_localization_status(
                    score_values, matched_boxes, predicted_intersection_box,
                    actual_intersection_box, row, col, error_pixels, step_count,
                    last_action, heading_degrees, altitude_state, intersection_mode,
                    search_mode,
                    match_backend,
                    current_search_window_size,
                    runtime_match_config,
                )

                processing_ms = (time.perf_counter() - step_started_at) * 1000.0
                if csv_writer is not None:
                    _write_csv_row(
                        csv_writer, step_count, row, col, heading_degrees, altitude_state,
                        last_action, score_values, intersection_mode, search_mode, match_backend,
                        actual_center, raw_predicted_center, kalman_center, actual_center,
                        quality, current_search_window_size, processing_ms,
                    )

                # Tekrar kullanılacak dashboard kwargs
                _dash_kw: dict = dict(
                    observation_rect=observation_rect,
                    template_rect=template_rect,
                    reference_preview_state=reference_preview_state,
                    observation_view=observation_view,
                    template_strip=template_strip,
                    matched_boxes=matched_boxes,
                    predicted_intersection_box=predicted_intersection_box,
                    actual_intersection_box=actual_intersection_box,
                    search_window_box=display_search_window_box,
                    predicted_history=predicted_history,
                    actual_history=actual_history,
                    score_values=score_values,
                    observation_cursor=(row, col),
                    step_count=step_count,
                    last_action=last_action,
                    heading_degrees=heading_degrees,
                    altitude_state=altitude_state,
                    intersection_mode=intersection_mode,
                    search_mode=search_mode,
                    match_backend=match_backend,
                    search_window_size=current_search_window_size,
                    ui_state=runtime_ui_state,
                    runtime_ui_buttons=runtime_ui_buttons,
                    config=runtime_match_config,
                    kalman_center=kalman_center,
                    waypoint_target=waypoint_target,
                    autonomous_mode=autonomous_mode,
                    quality=quality,
                    kalman_error_pixels=kalman_error_pixels,
                    waypoint_distance_px=waypoint_distance_px,
                    ref_patch_image=ref_patch_image,
                    ref_patch_rect=ref_patch_rect,
                    right_panel_rect=right_panel_rect,
                )
                dashboard = draw_localization_dashboard(**_dash_kw)
                runtime_ui_context.update(
                    map_rect=map_rect,
                    observation_view=observation_view,
                    template_strip=template_strip,
                    ref_patch_image=ref_patch_image,
                )
                if _telemetry_fn is not None:
                    _telemetry_fn(
                        {
                            "step": step_count,
                            "heading": get_heading_label(heading_degrees),
                            "altitude_m": altitude_state.altitude_agl_m,
                            "gsd_cm": altitude_state.center_gsd_cm_per_px,
                            "error_px": error_pixels,
                            "error_m": error_pixels * config.reference_map_gsd_cm_per_px / 100.0,
                            "confidence": quality.confidence,
                            "reliable": quality.is_reliable,
                            "reason": quality.reason,
                            "scores": tuple(float(score) for score in score_values),
                            "peak_margins": tuple(
                                float(evidence.peak_margin)
                                for evidence in match_evidence
                            ),
                            "template_stddevs": tuple(
                                float(evidence.template_stddev)
                                for evidence in match_evidence
                            ),
                            "intersection_mode": intersection_mode,
                            "search_mode": search_mode,
                            "search_window_px": current_search_window_size,
                            "backend": match_backend,
                            "action": get_action_label(last_action),
                            "kalman_on": use_kalman,
                            "autonomous": autonomous_mode,
                            "obs_window_size": runtime_obs_window_size,
                            "norm_mode": runtime_ui_state.get("norm_mode", "HISTEQ"),
                            "processing_ms": processing_ms,
                        }
                    )

                # --- İç döngü: tuş bekleme / otonom adım ---
                should_exit = False
                while True:
                    if _display_fn is not None:
                        _display_fn(dashboard, _lb_state)
                    else:
                        _imshow_keepratio(config.dashboard_window_name, dashboard, _lb_state)
                    runtime_ui_state["_dirty"] = False
                    wait_ms = (
                        config.autonomous_step_interval_ms
                        if bool(runtime_ui_state.get("autonomous_mode", False))
                        else 30
                    )
                    key = _getkey_fn(wait_ms) if _getkey_fn is not None else cv2.waitKeyEx(wait_ms)

                    if runtime_ui_state.get("_dirty"):
                        # Sütun ayracı sürüklendiyse layout + harita önizlemesini yenile
                        if runtime_ui_state.get("_layout_dirty"):
                            runtime_ui_state["_layout_dirty"] = False
                            (
                                observation_rect,
                                template_rect,
                                map_rect,
                                ref_patch_rect,
                                right_panel_rect,
                            ) = get_dashboard_layout(
                                config,
                                bool(runtime_ui_state.get("ref_patch", False)),
                                runtime_ui_state,
                            )
                            reference_preview_state = create_reference_preview_state(
                                reference_map, map_rect, reference_viewport_box, config,
                            )
                            runtime_ui_context["map_rect"] = map_rect
                            runtime_ui_context["reference_preview_state"] = reference_preview_state
                            _dash_kw["observation_rect"] = observation_rect
                            _dash_kw["template_rect"] = template_rect
                            _dash_kw["ref_patch_rect"] = ref_patch_rect
                            _dash_kw["right_panel_rect"] = right_panel_rect
                            _dash_kw["reference_preview_state"] = reference_preview_state
                        # waypoint veya toggle değişmiş olabilir
                        dirty_waypoint_target = runtime_ui_context.get("waypoint_target")
                        if dirty_waypoint_target != active_waypoint_target:
                            active_waypoint_target = dirty_waypoint_target
                            waypoint_hits = 0
                            previous_waypoint_distance_px = None
                            auto_no_progress_steps = 0
                            auto_last_action = ""
                            auto_same_action_steps = 0
                        _dash_kw["waypoint_target"] = dirty_waypoint_target
                        _dash_kw["autonomous_mode"] = bool(runtime_ui_state.get("autonomous_mode", False))
                        _dash_kw["waypoint_distance_px"] = (
                            math.hypot(
                                display_predicted_center[0] - dirty_waypoint_target[0],
                                display_predicted_center[1] - dirty_waypoint_target[1],
                            )
                            if dirty_waypoint_target is not None else None
                        )
                        dashboard = draw_localization_dashboard(**_dash_kw)
                        continue

                    if key in EXIT_KEYS:
                        should_exit = True
                        break

                    if key != -1 and config.ui_buttons_enabled and apply_runtime_ui_hotkey(
                        key, runtime_ui_state,
                    ):
                        _dash_kw["autonomous_mode"] = bool(runtime_ui_state.get("autonomous_mode", False))
                        # Kalman toggle: bir sonraki adımda kalman bloğu sıfırlanacak;
                        # şimdilik HUD'ı güncelle
                        dashboard = draw_localization_dashboard(**_dash_kw)
                        # İşlem yöntemleri yeni model/eşleme sonucu üretmelidir;
                        # yalnız HUD etiketini değiştirmek kullanıcıyı yanıltır.
                        if (
                            key in KALMAN_TOGGLE_KEYS
                            or key in NORM_CYCLE_KEYS
                            or key in OBS_WINDOW_CYCLE_KEYS
                            or key in REF_PATCH_TOGGLE_KEYS
                        ):
                            last_action = "hold"
                            last_action_step_size = 0.0
                            step_count += 1
                            break
                        continue

                    if bool(runtime_ui_state.get("autonomous_mode", False)):
                        auto_waypoint_target = runtime_ui_context.get("waypoint_target")
                        if auto_waypoint_target is None:
                            last_action = "hold"
                            last_action_step_size = 0.0
                            continue
                        current_waypoint_distance_px = math.hypot(
                            display_predicted_center[0] - auto_waypoint_target[0],
                            display_predicted_center[1] - auto_waypoint_target[1],
                        )
                        action_low_confidence_steps = (
                            low_confidence_steps
                            if low_confidence_steps <= config.autonomous_low_confidence_recovery_steps
                            else 1
                        )
                        auto_action = choose_autonomous_action(
                            display_predicted_center,
                            auto_waypoint_target,
                            heading_degrees,
                            action_low_confidence_steps,
                            config.waypoint_acceptance_radius_px,
                            config.waypoint_rotation_tolerance_deg,
                            config.waypoint_body_axis_deadband_px,
                        )
                        if previous_waypoint_distance_px is not None:
                            if current_waypoint_distance_px >= (
                                previous_waypoint_distance_px
                                - config.autonomous_stuck_distance_epsilon_px
                            ):
                                auto_no_progress_steps += 1
                            else:
                                auto_no_progress_steps = 0
                        previous_waypoint_distance_px = current_waypoint_distance_px

                        if auto_action == auto_last_action:
                            auto_same_action_steps += 1
                        else:
                            auto_last_action = auto_action
                            auto_same_action_steps = 1

                        if (
                            auto_no_progress_steps >= config.autonomous_stuck_max_steps
                            and is_translation_action(auto_action)
                            and auto_same_action_steps >= config.autonomous_stuck_max_steps
                        ):
                            auto_action = (
                                "rotate_right"
                                if (step_count % 2 == 0)
                                else "rotate_left"
                            )
                            auto_no_progress_steps = 0
                            auto_last_action = auto_action
                            auto_same_action_steps = 1

                        last_action = auto_action or "hold"
                        if auto_action not in ("hold", ""):
                            auto_step_size = (
                                get_autonomous_step_size(current_waypoint_distance_px, config)
                                if is_translation_action(auto_action)
                                else 0.0
                            )
                            row, col, heading_degrees, altitude_agl_m = apply_control_action(
                                row, col, heading_degrees, altitude_agl_m,
                                auto_action, observation_map.shape, config,
                                step_size_px=auto_step_size,
                            )
                            last_action_step_size = auto_step_size
                        else:
                            last_action_step_size = 0.0
                        step_count += 1
                        break

                    # Manuel mod
                    if key == -1:
                        continue

                    action = get_action_from_key(key)
                    if action:
                        row, col, heading_degrees, altitude_agl_m = apply_control_action(
                            row, col, heading_degrees, altitude_agl_m,
                            action, observation_map.shape, config,
                        )
                        last_action = action
                        last_action_step_size = float(config.step_size) if is_translation_action(action) else 0.0
                        step_count += 1
                        break

                    print("Unrecognized key code: %s" % key)

                if should_exit:
                    break

        finally:
            if csv_file is not None:
                csv_file.close()
                print("CSV log kaydedildi: %s" % (config.log_csv_path or "log_simulasyon_*.csv"))

    finally:
        close_terrain_context(terrain_context)
        close_raster_source(observation_map)
        close_raster_source(reference_map)
        if not _use_qt:
            cv2.destroyAllWindows()


def main_qt(config=None) -> None:
    if not _HAS_QT:
        print("PySide6/PyQt5 bulunamadı, OpenCV moduna geçiliyor.")
        main(config=config)
        return
    if config is None:
        config = SimulationConfig()
        config = _apply_args_to_config(config, _parse_args())
    config = resolve_config_paths(config)
    from mission_control_ui import run_mission_control

    # Qt kabuğu eski OpenCV yan panellerini ayrı bileşenler olarak gösterir.
    # İç kompozisyonu geniş formata almak, ortadaki harita karesinin dar/küçük
    # üretilmesini önler; Qt tarafı görüntü oranını koruyarak ölçekler.
    mission_control_config = dataclasses.replace(
        config,
        display_size=config.mission_control_canvas_size,
        left_panel_width_ratio=0.12,
        right_info_panel_width=130,
    )

    raise SystemExit(
        run_mission_control(
            mission_control_config,
            main,
            _runtime_buttons_mouse_cb,
            _QT_KEY_MAP,
        )
    )


if __name__ == "__main__":
    main_qt()

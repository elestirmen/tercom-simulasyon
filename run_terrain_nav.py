"""Command-line and desktop entry point for TERCOM terrain navigation."""

import argparse
from dataclasses import replace
from pathlib import Path

from terrain_nav.config import (
    MOTION_MODE_KNOWN_DISTANCE,
    MOTION_MODE_MEASURED_SPEED,
    MOTION_MODE_UNKNOWN_CONSTANT_SPEED,
    MOTION_MODES,
    LocalizationConfig,
    apply_realistic_noise_mode,
)
from terrain_nav.logging_io import save_config, save_results
from terrain_nav.simulation import SimulationEngine

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DEM_PATH = Path(
    r"C:\d_surucusu\visual_navigation\template-matching\karlik_30_cm_bingmap_utm_elevation.tif"
)


def _resolve_dem_path(explicit_path: str | None, fast_mode: bool) -> str:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"DEM dosyası bulunamadı: {path}")
        return str(path)
    if not fast_mode and DEFAULT_DEM_PATH.is_file():
        return str(DEFAULT_DEM_PATH)
    return ""


def build_config(
    *,
    fast_mode: bool = False,
    dem_path: str | None = None,
    dem_target_size: int | None = None,
    search_roi_size_px: int | None = None,
    start_row: int | None = None,
    start_col: int | None = None,
    realistic_noise: bool = False,
    motion_mode: str | None = None,
    speed_search_min_m_s: float | None = None,
    speed_search_max_m_s: float | None = None,
) -> LocalizationConfig:
    config = LocalizationConfig()
    resolved_dem_path = _resolve_dem_path(dem_path, fast_mode)
    selected_motion_mode = motion_mode or (
        MOTION_MODE_MEASURED_SPEED if realistic_noise else MOTION_MODE_KNOWN_DISTANCE
    )
    if selected_motion_mode not in MOTION_MODES:
        expected = ", ".join(sorted(MOTION_MODES))
        raise ValueError(f"motion_mode must be one of: {expected}")

    terrain_updates = {}
    if resolved_dem_path:
        terrain_updates.update(
            dem_path=resolved_dem_path,
            external_auto_center_start=start_row is None and start_col is None,
        )
    if dem_target_size is not None:
        terrain_updates["dem_target_size"] = max(2, int(dem_target_size))
    terrain = replace(config.terrain, **terrain_updates)

    algorithm = config.algorithm
    if search_roi_size_px is not None:
        algorithm = replace(algorithm, search_roi_size_px=max(0, int(search_roi_size_px)))
    speed_updates = {}
    if speed_search_min_m_s is not None:
        speed_updates["speed_search_min_m_s"] = float(speed_search_min_m_s)
    if speed_search_max_m_s is not None:
        speed_updates["speed_search_max_m_s"] = float(speed_search_max_m_s)
    if speed_updates:
        algorithm = replace(algorithm, **speed_updates)

    route_updates = {}
    if start_row is not None:
        route_updates["start_row"] = int(start_row)
    if start_col is not None:
        route_updates["start_col"] = int(start_col)
    if fast_mode and not resolved_dem_path:
        if selected_motion_mode == MOTION_MODE_UNKNOWN_CONSTANT_SPEED:
            terrain = replace(terrain, rows=120, cols=160)
        else:
            terrain = replace(terrain, rows=100, cols=100)
        if start_row is None:
            route_updates["start_row"] = terrain.rows // 2
        if start_col is None:
            route_updates["start_col"] = (
                30
                if selected_motion_mode == MOTION_MODE_UNKNOWN_CONSTANT_SPEED
                else terrain.cols // 2
            )
    route = replace(config.route, **route_updates)
    if fast_mode:
        if selected_motion_mode == MOTION_MODE_UNKNOWN_CONSTANT_SPEED:
            route = replace(
                route,
                heading_deg=90.0,
                sample_spacing_m=5.0,
                route_length_m=80.0,
            )
        else:
            route = replace(route, route_length_m=50.0)

    sensor = config.sensor
    built_config = replace(
        config,
        terrain=terrain,
        route=route,
        sensor=sensor,
        algorithm=algorithm,
        motion_mode=selected_motion_mode,
    )
    if realistic_noise:
        built_config = apply_realistic_noise_mode(
            built_config,
            True,
            fast_synthetic=fast_mode and not resolved_dem_path,
        )
        built_config = replace(built_config, motion_mode=selected_motion_mode)

    if selected_motion_mode == MOTION_MODE_UNKNOWN_CONSTANT_SPEED:
        unknown_algorithm = built_config.algorithm
        if fast_mode and not resolved_dem_path:
            unknown_algorithm = replace(
                unknown_algorithm,
                min_profile_length=10,
                min_profile_duration_s=4.0,
                max_profile_duration_s=20.0,
                coarse_stride=5,
                medium_stride=2,
                refinement_radius_px=8,
                max_match_jump_m=0.0,
            )
        built_config = replace(
            built_config,
            sensor=replace(built_config.sensor, altitude_mode="barometric_altitude"),
            algorithm=unknown_algorithm,
        )
    return built_config


def run_headless(**config_kwargs) -> None:
    print("Headless TERCOM simülasyonu çalışıyor...")
    simulation = SimulationEngine(build_config(**config_kwargs))
    try:
        results = []
        for step in range(simulation.get_total_steps()):
            true_state, estimated_state, _measurement = simulation.step()
            results.append((true_state, estimated_state))
            if step % 10 == 0:
                print(f"Adım {step}/{simulation.get_total_steps()}")

        output_dir = PROJECT_ROOT / "results"
        output_dir.mkdir(exist_ok=True)
        save_config(simulation.config, output_dir / "config.json")
        save_results(
            results,
            output_dir / "results.csv",
            true_speed_m_s=simulation.config.route.speed_m_s,
        )
        print(f"Sonuçlar yazıldı: {output_dir}")
    finally:
        simulation.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TERCOM terrain-profile navigation")
    parser.add_argument("--headless", action="store_true", help="Run without the desktop UI")
    parser.add_argument("--fast", action="store_true", help="Use a small synthetic terrain")
    parser.add_argument("--dem", metavar="PATH", help="External GeoTIFF DEM")
    parser.add_argument("--dem-target-size", type=int, metavar="PX")
    parser.add_argument("--search-roi-size", type=int, metavar="PX")
    parser.add_argument("--start-row", type=int, metavar="ROW")
    parser.add_argument("--start-col", type=int, metavar="COL")
    parser.add_argument(
        "--realistic-noise",
        action="store_true",
        help="Use barometric relative altitude and noisy speed measurements",
    )
    parser.add_argument(
        "--motion-mode",
        choices=sorted(MOTION_MODES),
        help="Motion input available to localization",
    )
    parser.add_argument(
        "--unknown-speed",
        action="store_true",
        help="Estimate constant speed from timestamps and DEM profile matching",
    )
    parser.add_argument("--speed-search-min", type=float, metavar="M_S")
    parser.add_argument("--speed-search-max", type=float, metavar="M_S")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    motion_mode = (
        MOTION_MODE_UNKNOWN_CONSTANT_SPEED if args.unknown_speed else args.motion_mode
    )
    config_kwargs = {
        "fast_mode": args.fast,
        "dem_path": args.dem,
        "dem_target_size": args.dem_target_size,
        "search_roi_size_px": args.search_roi_size,
        "start_row": args.start_row,
        "start_col": args.start_col,
        "realistic_noise": args.realistic_noise,
        "motion_mode": motion_mode,
        "speed_search_min_m_s": args.speed_search_min,
        "speed_search_max_m_s": args.speed_search_max,
    }
    if args.headless:
        run_headless(**config_kwargs)
        return

    from terrain_nav.ui import run_ui

    run_ui(build_config(**config_kwargs))


if __name__ == "__main__":
    main()

"""Command-line and desktop entry point for TERCOM terrain navigation."""

import argparse
from dataclasses import replace
from pathlib import Path

from terrain_nav.config import LocalizationConfig
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
    dem_window_size: int | None = None,
    dem_target_size: int | None = None,
    dem_window_row: int | None = None,
    dem_window_col: int | None = None,
    search_roi_size_px: int | None = None,
    start_row: int | None = None,
    start_col: int | None = None,
) -> LocalizationConfig:
    config = LocalizationConfig()
    resolved_dem_path = _resolve_dem_path(dem_path, fast_mode)

    terrain_updates = {}
    if resolved_dem_path:
        terrain_updates.update(
            dem_path=resolved_dem_path,
            external_auto_center_start=start_row is None and start_col is None,
        )
    if dem_window_size is not None:
        terrain_updates["dem_window_size"] = max(2, int(dem_window_size))
    if dem_target_size is not None:
        terrain_updates["dem_target_size"] = max(2, int(dem_target_size))
    if dem_window_row is not None:
        terrain_updates["dem_window_row"] = int(dem_window_row)
    if dem_window_col is not None:
        terrain_updates["dem_window_col"] = int(dem_window_col)
    terrain = replace(config.terrain, **terrain_updates)

    algorithm = config.algorithm
    if search_roi_size_px is not None:
        algorithm = replace(algorithm, search_roi_size_px=max(0, int(search_roi_size_px)))

    route_updates = {}
    if start_row is not None:
        route_updates["start_row"] = int(start_row)
    if start_col is not None:
        route_updates["start_col"] = int(start_col)
    if fast_mode and not resolved_dem_path:
        terrain = replace(terrain, rows=100, cols=100)
        if start_row is None:
            route_updates["start_row"] = terrain.rows // 2
        if start_col is None:
            route_updates["start_col"] = terrain.cols // 2
    route = replace(config.route, **route_updates)
    if fast_mode:
        route = replace(route, route_length_m=50.0)

    return replace(config, terrain=terrain, route=route, algorithm=algorithm)


def run_headless(**config_kwargs) -> None:
    print("Headless TERCOM simülasyonu çalışıyor...")
    simulation = SimulationEngine(build_config(**config_kwargs))
    results = []
    for step in range(simulation.get_total_steps()):
        true_state, estimated_state, _measurement = simulation.step()
        results.append((true_state, estimated_state))
        if step % 10 == 0:
            print(f"Adım {step}/{simulation.get_total_steps()}")

    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    save_config(simulation.config, output_dir / "config.json")
    save_results(results, output_dir / "results.csv")
    print(f"Sonuçlar yazıldı: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TERCOM terrain-profile navigation")
    parser.add_argument("--headless", action="store_true", help="Run without the desktop UI")
    parser.add_argument("--fast", action="store_true", help="Use a small synthetic terrain")
    parser.add_argument("--dem", metavar="PATH", help="External GeoTIFF DEM")
    parser.add_argument("--dem-window-size", type=int, metavar="PX")
    parser.add_argument("--dem-target-size", type=int, metavar="PX")
    parser.add_argument("--dem-row", type=int, metavar="ROW")
    parser.add_argument("--dem-col", type=int, metavar="COL")
    parser.add_argument("--search-roi-size", type=int, metavar="PX")
    parser.add_argument("--start-row", type=int, metavar="ROW")
    parser.add_argument("--start-col", type=int, metavar="COL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_kwargs = {
        "fast_mode": args.fast,
        "dem_path": args.dem,
        "dem_window_size": args.dem_window_size,
        "dem_target_size": args.dem_target_size,
        "dem_window_row": args.dem_row,
        "dem_window_col": args.dem_col,
        "search_roi_size_px": args.search_roi_size,
        "start_row": args.start_row,
        "start_col": args.start_col,
    }
    if args.headless:
        run_headless(**config_kwargs)
        return

    from terrain_nav.ui import run_ui

    run_ui(build_config(**config_kwargs))


if __name__ == "__main__":
    main()

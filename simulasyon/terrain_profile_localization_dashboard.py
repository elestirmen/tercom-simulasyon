"""Entry point for Dashboard and Headless CLI."""

import argparse
from dataclasses import replace
from pathlib import Path

from terrain_nav.config import LocalizationConfig
from terrain_nav.logging_io import save_config, save_results
from terrain_nav.simulation import SimulationEngine

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
    route = replace(config.route, **route_updates)
    if fast_mode:
        route = replace(route, route_length_m=50.0)
        if not resolved_dem_path:
            terrain = replace(terrain, rows=100, cols=100)
    return replace(config, terrain=terrain, route=route, algorithm=algorithm)


def run_headless(
    fast_mode=False,
    *,
    dem_path: str | None = None,
    dem_window_size: int | None = None,
    dem_target_size: int | None = None,
    dem_window_row: int | None = None,
    dem_window_col: int | None = None,
    search_roi_size_px: int | None = None,
    start_row: int | None = None,
    start_col: int | None = None,
):
    print("Running in headless mode...")
    config = build_config(
        fast_mode=fast_mode,
        dem_path=dem_path,
        dem_window_size=dem_window_size,
        dem_target_size=dem_target_size,
        dem_window_row=dem_window_row,
        dem_window_col=dem_window_col,
        search_roi_size_px=search_roi_size_px,
        start_row=start_row,
        start_col=start_col,
    )
    
    sim = SimulationEngine(config)
    results = []
    
    total = sim.get_total_steps()
    for i in range(total):
        true_s, est_s, _measurement = sim.step()
        results.append((true_s, est_s))
        if i % 10 == 0:
            print(f"Step {i}/{total}")
            
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    
    save_config(sim.config, str(out_dir / "config.json"))
    save_results(results, str(out_dir / "results.csv"))
    print("Headless simulation complete. Results saved to 'results/' directory.")

def main():
    parser = argparse.ArgumentParser(description="Terrain Navigation Dashboard")
    parser.add_argument("--headless", action="store_true", help="Run without UI and output CSV/JSON")
    parser.add_argument("--fast", action="store_true", help="Run in fast mode (small terrain)")
    parser.add_argument("--dem", default=None, metavar="YOL", help="External GeoTIFF DEM path")
    parser.add_argument(
        "--dem-window-size",
        type=int,
        default=None,
        metavar="PX",
        help="External DEM window edge length in pixels (default: 4096)",
    )
    parser.add_argument(
        "--dem-target-size",
        type=int,
        default=None,
        metavar="PX",
        help="In-memory/search raster max edge (default: 2048; physical extent is preserved)",
    )
    parser.add_argument("--dem-row", type=int, default=None, metavar="ROW", help="External DEM window top row")
    parser.add_argument("--dem-col", type=int, default=None, metavar="COL", help="External DEM window left column")
    parser.add_argument(
        "--search-roi-size",
        type=int,
        default=None,
        metavar="PX",
        help="Local search window after first match (default: 512; 0 disables)",
    )
    parser.add_argument("--start-row", type=int, default=None, metavar="ROW", help="Local simulation start row")
    parser.add_argument("--start-col", type=int, default=None, metavar="COL", help="Local simulation start column")
    args = parser.parse_args()

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

    try:
        from terrain_nav.ui import run_ui

        run_ui(build_config(**config_kwargs))
    except ImportError as e:
        print(f"Error starting UI: {e}")
        print("To run in headless mode, use --headless flag.")

if __name__ == "__main__":
    main()

"""Terrain generation adapter for GNSS-denied simulation."""

import pathlib
import sys
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from terrain_nav.config import TerrainConfig

# Setup import path for yuzey_alani_hesaplama
_repo_root = pathlib.Path(__file__).parent.parent.parent / "yuzey_alani_hesaplama"
if str(_repo_root.absolute()) not in sys.path:
    sys.path.insert(0, str(_repo_root.absolute()))

try:
    from surface_area.synthetic import generate_synthetic_dsm
except ImportError:
    raise ImportError(
        f"Could not import generate_synthetic_dsm from {_repo_root.absolute()}"
    ) from None

class TerrainManager:
    """Manages ground-truth DEM and navigation DEM.
    Ensures that algorithmic localization uses only nav_dem.
    """
    def __init__(self, config: TerrainConfig):
        self.config = config
        self.truth_dem = None
        self.nav_dem = None
        self.dx = float(config.dx)
        self.dy = float(config.dy)
        self.window_offset = (0, 0)
        self.source_shape: Optional[Tuple[int, int]] = None
        self._generate()
        
    def _generate(self) -> None:
        """Load a bounded external DEM window or generate a synthetic one."""
        dem_path = pathlib.Path(self.config.dem_path) if self.config.dem_path else None
        if dem_path is not None:
            self.truth_dem = self._load_external_dem(dem_path)
        else:
            base_dem = generate_synthetic_dsm(
                rows=self.config.rows,
                cols=self.config.cols,
                dx=self.config.dx,
                dy=self.config.dy,
                preset=self.config.preset,
                relief=self.config.relief,
                roughness_m=self.config.roughness,
                seed=self.config.seed
            )
            # Synthetic truth DEM is shifted to the configured base elevation.
            self.truth_dem = base_dem + self.config.base_elevation
            self.source_shape = self.truth_dem.shape
        
        # Keep both rasters in float32. A float64 navigation DEM plus a second
        # localizer copy roughly triples persistent memory on large windows.
        rng = np.random.default_rng(self.config.seed + 123)
        self.truth_dem = np.asarray(self.truth_dem, dtype=np.float32)
        self.nav_dem = self.truth_dem.copy()
        self.nav_dem += np.float32(self.config.dem_bias_m)
        if self.config.dem_noise_std_m > 0:
            # Generate the same float64 normal sequence as earlier versions,
            # but add it in row chunks to avoid a full-size temporary array.
            chunk_rows = min(256, self.truth_dem.shape[0])
            for row_start in range(0, self.truth_dem.shape[0], chunk_rows):
                row_end = min(self.truth_dem.shape[0], row_start + chunk_rows)
                noise = rng.normal(
                    loc=0.0,
                    scale=self.config.dem_noise_std_m,
                    size=(row_end - row_start, self.truth_dem.shape[1]),
                )
                np.add(
                    self.nav_dem[row_start:row_end],
                    noise,
                    out=self.nav_dem[row_start:row_end],
                    casting="unsafe",
                )

    def _load_external_dem(self, dem_path: pathlib.Path) -> np.ndarray:
        if not dem_path.exists():
            raise FileNotFoundError(f"External DEM not found: {dem_path}")
        if not dem_path.is_file():
            raise ValueError(f"External DEM path is not a file: {dem_path}")

        with rasterio.open(dem_path) as dataset:
            if dataset.count < 1:
                raise ValueError(f"External DEM has no raster bands: {dem_path}")

            self.source_shape = (dataset.height, dataset.width)
            window_size = max(2, int(self.config.dem_window_size))
            window_height = min(window_size, dataset.height)
            window_width = min(window_size, dataset.width)

            row_off = int(self.config.dem_window_row)
            col_off = int(self.config.dem_window_col)
            if row_off < 0:
                row_off = (dataset.height - window_height) // 2
            if col_off < 0:
                col_off = (dataset.width - window_width) // 2
            row_off = min(max(0, row_off), dataset.height - window_height)
            col_off = min(max(0, col_off), dataset.width - window_width)
            self.window_offset = (row_off, col_off)

            window = Window(col_off, row_off, window_width, window_height)
            target_size = max(2, int(self.config.dem_target_size))
            scale = min(1.0, target_size / max(window_width, window_height))
            target_height = max(2, int(round(window_height * scale)))
            target_width = max(2, int(round(window_width * scale)))

            read_kwargs = {
                "window": window,
                "masked": True,
                "out_dtype": "float32",
            }
            if (target_height, target_width) != (window_height, window_width):
                read_kwargs["out_shape"] = (1, target_height, target_width)
                read_kwargs["resampling"] = Resampling.bilinear
            dem = dataset.read(1, **read_kwargs)
            if np.ma.isMaskedArray(dem):
                dem = dem.filled(np.nan)
            dem = np.asarray(dem, dtype=np.float32)

            # Keep the source window's physical extent after resampling. This
            # is critical: route distances and coordinate conversion must not
            # accidentally become four times smaller with a 2x downsample.
            source_dx, source_dy = map(float, dataset.res)
            self.dx = source_dx * window_width / dem.shape[1]
            self.dy = source_dy * window_height / dem.shape[0]
            finite_ratio = float(np.isfinite(dem).mean())
            if finite_ratio < 0.99:
                raise ValueError(
                    f"External DEM window contains too much nodata ({finite_ratio:.1%} valid): "
                    f"{dem_path}"
                )
            if not np.isfinite(dem).all():
                fill_value = float(np.nanmedian(dem))
                dem = np.nan_to_num(dem, nan=fill_value)
            return dem

    def get_truth_dem(self) -> np.ndarray:
        return self.truth_dem.copy() # Return copy to prevent accidental mutation
        
    def get_navigation_dem(self, *, copy: bool = True) -> np.ndarray:
        if copy:
            return self.nav_dem.copy()
        view = self.nav_dem.view()
        view.flags.writeable = False
        return view

    def get_center_pixel(self) -> Tuple[int, int]:
        return self.truth_dem.shape[0] // 2, self.truth_dem.shape[1] // 2

    def sample_truth_elevation(self, row: int, col: int) -> float:
        return float(self.truth_dem[row, col])
        
    def get_extent(self) -> Tuple[float, float]:
        """Returns max_x, max_y in meters."""
        rows, cols = self.truth_dem.shape
        return cols * self.dx, rows * self.dy

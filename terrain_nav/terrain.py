"""Terrain generation adapter for GNSS-denied simulation."""

import pathlib
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window

from terrain_nav.config import TerrainConfig
from terrain_nav.synthetic import generate_synthetic_dem


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
        self.source_dx = float(config.dx)
        self.source_dy = float(config.dy)
        self.source_bounds: Optional[Tuple[float, float, float, float]] = None
        self.window_offset = (0, 0)
        self.source_shape: Optional[Tuple[int, int]] = None
        self._display_dem: Optional[np.ndarray] = None
        self._display_dem_max_edge = 0
        self._source_max_elevation: Optional[float] = None
        self._generate()

    def _generate(self) -> None:
        """Load a bounded external DEM window or generate a synthetic one."""
        dem_path = pathlib.Path(self.config.dem_path) if self.config.dem_path else None
        if dem_path is not None:
            self.truth_dem = self._load_external_dem(dem_path)
        else:
            base_dem = generate_synthetic_dem(
                rows=self.config.rows,
                cols=self.config.cols,
                dx=self.config.dx,
                dy=self.config.dy,
                preset=self.config.preset,
                relief=self.config.relief,
                roughness=self.config.roughness,
                seed=self.config.seed,
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
            self.source_bounds = (
                float(dataset.bounds.left),
                float(dataset.bounds.right),
                float(dataset.bounds.bottom),
                float(dataset.bounds.top),
            )
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
            self.source_dx = abs(source_dx)
            self.source_dy = abs(source_dy)
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
        return self.truth_dem.copy()  # Return copy to prevent accidental mutation

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

    def navigation_world_to_display(
        self,
        x: float,
        y: float,
    ) -> Tuple[float, float]:
        """Convert loaded-window coordinates to complete source-map coordinates."""
        offset_x, offset_y = self.get_display_offset()
        return x + offset_x, y + offset_y

    def is_inside_navigation_window(self, x: float, y: float) -> bool:
        """Return whether a navigation-world point is inside the loaded DEM window."""
        col = x / self.dx
        row = -y / self.dy
        row_index = int(round(row))
        col_index = int(round(col))
        rows, cols = self.truth_dem.shape
        return 0 <= row_index < rows and 0 <= col_index < cols

    def is_inside_source_map(self, x: float, y: float) -> bool:
        """Return whether a navigation-world point is inside the full source map."""
        display_x, display_y = self.navigation_world_to_display(x, y)
        left, right, bottom, top = self.get_display_bounds()
        return left <= display_x < right and bottom < display_y <= top

    def sample_elevation_at_world(self, x: float, y: float) -> float:
        """Sample truth elevation anywhere inside the complete source map."""
        if self.is_inside_navigation_window(x, y):
            row = int(round(-y / self.dy))
            col = int(round(x / self.dx))
            return self.sample_truth_elevation(row, col)

        dem_path = pathlib.Path(self.config.dem_path) if self.config.dem_path else None
        if dem_path is None or not self.is_inside_source_map(x, y):
            raise ValueError("Konum kaynak DEM kapsamının dışında.")

        display_x, display_y = self.navigation_world_to_display(x, y)
        left, _right, _bottom, top = self.get_display_bounds()
        source_col = int((display_x - left) / self.source_dx)
        source_row = int((top - display_y) / self.source_dy)
        with rasterio.open(dem_path) as dataset:
            sample = dataset.read(
                1,
                window=Window(source_col, source_row, 1, 1),
                masked=True,
                out_dtype="float32",
            )
        if np.ma.isMaskedArray(sample) and np.ma.is_masked(sample[0, 0]):
            raise ValueError("Konumda geçerli kaynak DEM yüksekliği bulunamadı.")
        value = float(np.asarray(sample)[0, 0])
        if not np.isfinite(value):
            raise ValueError("Konumda geçerli kaynak DEM yüksekliği bulunamadı.")
        return value

    def get_source_max_elevation(self, max_edge: int = 1024) -> float:
        """Estimate a conservative full-source maximum with max resampling."""
        if self._source_max_elevation is not None:
            return self._source_max_elevation
        dem_path = pathlib.Path(self.config.dem_path) if self.config.dem_path else None
        if dem_path is None:
            self._source_max_elevation = float(np.nanmax(self.truth_dem))
            return self._source_max_elevation

        with rasterio.open(dem_path) as dataset:
            scale = min(1.0, max(2, int(max_edge)) / max(dataset.height, dataset.width))
            out_height = max(2, int(round(dataset.height * scale)))
            out_width = max(2, int(round(dataset.width * scale)))
            overview_transform = from_bounds(
                *dataset.bounds,
                width=out_width,
                height=out_height,
            )
            with WarpedVRT(
                dataset,
                width=out_width,
                height=out_height,
                transform=overview_transform,
                resampling=Resampling.max,
            ) as overview:
                maxima = overview.read(1, masked=True, out_dtype="float32")
        self._source_max_elevation = float(np.ma.max(maxima))
        return self._source_max_elevation

    def get_extent(self) -> Tuple[float, float]:
        """Returns max_x, max_y in meters."""
        rows, cols = self.truth_dem.shape
        return cols * self.dx, rows * self.dy

    def get_display_dem(self, max_edge: int = 1600) -> np.ndarray:
        """Return a bounded overview raster for UI display only.

        Localization keeps using the configured high-detail window. For an
        external DEM this method separately downsamples the complete source so
        the operator can see where that window and the aircraft sit on the map.
        """
        dem_path = pathlib.Path(self.config.dem_path) if self.config.dem_path else None
        if dem_path is None:
            return self.get_navigation_dem(copy=False)

        max_edge = max(2, int(max_edge))
        if self._display_dem is not None and self._display_dem_max_edge == max_edge:
            return self._display_dem

        with rasterio.open(dem_path) as dataset:
            scale = min(1.0, max_edge / max(dataset.height, dataset.width))
            out_height = max(2, int(round(dataset.height * scale)))
            out_width = max(2, int(round(dataset.width * scale)))
            overview = dataset.read(
                1,
                masked=True,
                out_shape=(1, out_height, out_width),
                out_dtype="float32",
                resampling=Resampling.bilinear,
            )

        if np.ma.isMaskedArray(overview):
            overview = overview.filled(np.nan)
        overview = np.asarray(overview, dtype=np.float32)
        if not np.isfinite(overview).all():
            overview = np.nan_to_num(overview, nan=float(np.nanmedian(overview)))
        overview.flags.writeable = False
        self._display_dem = overview
        self._display_dem_max_edge = max_edge
        return overview

    def get_display_extent(self) -> Tuple[float, float]:
        """Return the complete source-map extent used by the UI overview."""
        if self.source_shape is None:
            return self.get_extent()
        rows, cols = self.source_shape
        return cols * self.source_dx, rows * self.source_dy

    def get_display_bounds(self) -> Tuple[float, float, float, float]:
        """Return overview bounds as left, right, bottom, top coordinates."""
        if self.source_bounds is not None:
            return self.source_bounds
        width, height = self.get_display_extent()
        return 0.0, width, -height, 0.0

    def get_display_offset(self) -> Tuple[float, float]:
        """Translate local navigation-world coordinates into overview coordinates."""
        row_off, col_off = self.window_offset
        left, _right, _bottom, top = self.get_display_bounds()
        return (
            left + col_off * self.source_dx,
            top - row_off * self.source_dy,
        )

    def get_navigation_bounds_on_display(self) -> Tuple[float, float, float, float]:
        """Return local navigation-window bounds in complete-map coordinates."""
        offset_x, offset_y = self.get_display_offset()
        width, height = self.get_extent()
        return offset_x, offset_x + width, offset_y - height, offset_y

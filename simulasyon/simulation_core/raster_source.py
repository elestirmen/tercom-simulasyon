"""Low-memory, windowed grayscale access for geospatial rasters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window


def _read_window_as_gray(dataset: Any, window: Window) -> np.ndarray:
    if dataset.count <= 0:
        raise ValueError(f"Raster has no bands: {getattr(dataset, 'name', '<unknown>')}")

    band_count = min(3, int(dataset.count))
    data = dataset.read(list(range(1, band_count + 1)), window=window)
    if data.shape[0] == 1:
        gray = data[0]
    elif data.shape[0] >= 3:
        rgb = np.moveaxis(data[:3], 0, -1)
        if rgb.dtype != np.uint8:
            rgb = cv2.normalize(rgb, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = data.astype(np.float32).mean(axis=0)

    if gray.dtype == np.uint8:
        return np.ascontiguousarray(gray)
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)


class RasterioGraySource:
    """Array-like 2D grayscale raster backed by Rasterio window reads."""

    ndim = 2
    dtype = np.dtype(np.uint8)

    def __init__(
        self,
        path: Path,
        *,
        reference_dataset: Optional[Any] = None,
        resampling: Resampling = Resampling.nearest,
    ) -> None:
        self.path = Path(path)
        self._source = rio.open(str(self.path))
        self._dataset: Any = self._source
        self._vrt: Optional[WarpedVRT] = None
        if reference_dataset is not None:
            self._vrt = WarpedVRT(
                self._source,
                crs=reference_dataset.crs,
                transform=reference_dataset.transform,
                width=reference_dataset.width,
                height=reference_dataset.height,
                resampling=resampling,
            )
            self._dataset = self._vrt
        self.shape: Tuple[int, int] = (
            int(self._dataset.height),
            int(self._dataset.width),
        )

    @property
    def dataset(self) -> Any:
        return self._dataset

    def __getitem__(self, key: Any) -> np.ndarray:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("RasterioGraySource expects [rows, columns] slicing")
        row_key, col_key = key
        if not isinstance(row_key, slice) or not isinstance(col_key, slice):
            raise TypeError("RasterioGraySource supports slice-based access only")
        if row_key.step not in (None, 1) or col_key.step not in (None, 1):
            raise ValueError("RasterioGraySource does not support stepped slices")

        row_start = 0 if row_key.start is None else max(0, int(row_key.start))
        row_stop = self.shape[0] if row_key.stop is None else min(self.shape[0], int(row_key.stop))
        col_start = 0 if col_key.start is None else max(0, int(col_key.start))
        col_stop = self.shape[1] if col_key.stop is None else min(self.shape[1], int(col_key.stop))
        height = max(0, row_stop - row_start)
        width = max(0, col_stop - col_start)
        if height == 0 or width == 0:
            return np.empty((height, width), dtype=np.uint8)
        window = Window(col_start, row_start, width, height)
        return _read_window_as_gray(self._dataset, window)

    def read_full(self) -> np.ndarray:
        return self[:, :]

    def close(self) -> None:
        if self._vrt is not None:
            self._vrt.close()
            self._vrt = None
        if self._source is not None:
            self._source.close()
            self._source = None


def close_raster_source(source: Any) -> None:
    close = getattr(source, "close", None)
    if callable(close):
        close()

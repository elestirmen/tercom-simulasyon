from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.transform import from_origin

from simulation_core.raster_source import RasterioGraySource


def _write_rgb(path: Path, data: np.ndarray, transform=None) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=3,
        dtype="uint8",
        crs="EPSG:32636",
        transform=transform or from_origin(0, data.shape[1], 1, 1),
        tiled=False,
    ) as dataset:
        dataset.write(data)


def test_window_read_matches_rgb_grayscale(tmp_path: Path) -> None:
    rgb_bands = np.zeros((3, 8, 10), dtype=np.uint8)
    rgb_bands[0] = 200
    rgb_bands[1] = np.arange(80, dtype=np.uint8).reshape(8, 10)
    rgb_bands[2] = 20
    path = tmp_path / "rgb.tif"
    _write_rgb(path, rgb_bands)

    source = RasterioGraySource(path)
    try:
        actual = source[2:6, 3:8]
    finally:
        source.close()

    rgb = np.moveaxis(rgb_bands[:, 2:6, 3:8], 0, -1)
    expected = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    assert actual.shape == (4, 5)
    np.testing.assert_array_equal(actual, expected)


def test_vrt_alignment_adopts_reference_shape(tmp_path: Path) -> None:
    reference_data = np.full((3, 6, 7), 100, dtype=np.uint8)
    observation_data = np.full((3, 8, 9), 150, dtype=np.uint8)
    reference_path = tmp_path / "reference.tif"
    observation_path = tmp_path / "observation.tif"
    _write_rgb(reference_path, reference_data)
    _write_rgb(observation_path, observation_data)

    reference = RasterioGraySource(reference_path)
    observation = RasterioGraySource(
        observation_path,
        reference_dataset=reference.dataset,
    )
    try:
        assert observation.shape == reference.shape == (6, 7)
        assert observation[0:3, 0:4].shape == (3, 4)
    finally:
        observation.close()
        reference.close()

"""Tests for coordinate conventions."""

import math

from terrain_nav.coordinates import CoordinateTransform, circular_heading_error, normalize_heading


def test_heading_normalization():
    assert math.isclose(normalize_heading(360.0), 0.0)
    assert math.isclose(normalize_heading(0.0), 0.0)
    assert math.isclose(normalize_heading(-90.0), 270.0)
    assert math.isclose(normalize_heading(450.0), 90.0)
    assert math.isclose(normalize_heading(-360.0), 0.0)


def test_circular_heading_error():
    assert math.isclose(circular_heading_error(1.0, 359.0), 2.0)
    assert math.isclose(circular_heading_error(359.0, 1.0), 2.0)
    assert math.isclose(circular_heading_error(90.0, 180.0), 90.0)
    assert math.isclose(circular_heading_error(180.0, 90.0), 90.0)
    assert math.isclose(circular_heading_error(270.0, 10.0), 100.0)


def test_offset_pixels_directions():
    ct = CoordinateTransform(dx=1.0, dy=1.0)

    # 1. 0° yönünde satır azalır, sütun değişmez.
    d_row, d_col = ct.offset_pixels(10.0, 0.0)
    assert d_row < 0
    assert math.isclose(d_col, 0.0, abs_tol=1e-9)

    # 2. 90° yönünde sütun artar, satır değişmez.
    d_row, d_col = ct.offset_pixels(10.0, 90.0)
    assert math.isclose(d_row, 0.0, abs_tol=1e-9)
    assert d_col > 0

    # 3. 180° yönünde satır artar, sütun değişmez.
    d_row, d_col = ct.offset_pixels(10.0, 180.0)
    assert d_row > 0
    assert math.isclose(d_col, 0.0, abs_tol=1e-9)

    # 4. 270° yönünde sütun azalır, satır değişmez.
    d_row, d_col = ct.offset_pixels(10.0, 270.0)
    assert math.isclose(d_row, 0.0, abs_tol=1e-9)
    assert d_col < 0

    # 5. 45° yönünde doğu ve kuzey bileşenleri eşittir.
    dx_meters, dy_meters = ct.offset_meters(10.0, 45.0)
    assert math.isclose(dx_meters, dy_meters)

    d_row, d_col = ct.offset_pixels(10.0, 45.0)
    assert d_row < 0
    assert d_col > 0
    assert math.isclose(-d_row, d_col)


def test_world_pixel_conversions():
    ct = CoordinateTransform(dx=2.0, dy=3.0, origin_x=10.0, origin_y=20.0)

    # Start at origin
    row, col = ct.world_to_pixel(10.0, 20.0)
    assert math.isclose(row, 0.0)
    assert math.isclose(col, 0.0)

    # Move East by dx (2.0)
    row, col = ct.world_to_pixel(12.0, 20.0)
    assert math.isclose(row, 0.0)
    assert math.isclose(col, 1.0)

    # Move North by dy (3.0) -> row should decrease by 1
    row, col = ct.world_to_pixel(10.0, 23.0)
    assert math.isclose(row, -1.0)
    assert math.isclose(col, 0.0)

    # Back to world
    x, y = ct.pixel_to_world(-1.0, 0.0)
    assert math.isclose(x, 10.0)
    assert math.isclose(y, 23.0)

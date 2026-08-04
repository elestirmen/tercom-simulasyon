"""Tests for trajectory generation."""

import math
from terrain_nav.config import RouteConfig
from terrain_nav.trajectory import RouteGenerator
from terrain_nav.coordinates import CoordinateTransform

def test_straight_heading():
    c = RouteConfig(mode="straight_heading", heading_deg=90.0, route_length_m=100.0, sample_spacing_m=10.0)
    ct = CoordinateTransform(dx=1.0, dy=1.0)
    rg = RouteGenerator(c, ct)
    
    offsets = rg.generate_offsets()
    assert len(offsets) == 10
    
    # 0, 10, 20... East (x increases), North (y) stays 0
    # heading 90 degrees means East
    for i, (dx, dy, h) in enumerate(offsets):
        assert math.isclose(dx, i * 10.0)
        assert math.isclose(dy, 0.0, abs_tol=1e-9)
        assert math.isclose(h, 90.0)

def test_heading_sequence():
    c = RouteConfig(mode="heading_sequence", heading_sequence="0:50,90:50", sample_spacing_m=10.0)
    ct = CoordinateTransform(dx=1.0, dy=1.0)
    rg = RouteGenerator(c, ct)
    
    offsets = rg.generate_offsets()
    assert len(offsets) == 10
    
    # First 5 points moving North (heading 0) -> y increases
    # Last 5 points moving East (heading 90) -> x increases
    assert math.isclose(offsets[4][1], 40.0)
    assert math.isclose(offsets[4][0], 0.0)
    
    # Start of 90 degrees segment
    assert math.isclose(offsets[5][1], 50.0) # y reached 50
    assert math.isclose(offsets[5][0], 0.0)  # x is still 0
    
    # End of 90 degrees segment
    assert math.isclose(offsets[9][1], 50.0)
    assert math.isclose(offsets[9][0], 40.0)

def test_waypoint_route():
    # 40. Waypoint dönüşlerinde profil geometrisi bozulmaz.
    # relative to origin
    wps = [(0.0, 50.0), (50.0, 50.0)]
    c = RouteConfig(mode="waypoint_route", waypoints=wps, sample_spacing_m=10.0)
    ct = CoordinateTransform(dx=1.0, dy=1.0)
    rg = RouteGenerator(c, ct)
    
    offsets = rg.generate_offsets()
    assert len(offsets) == 10 # 50/10 + 50/10 = 10
    
    # Same as heading sequence above basically
    assert math.isclose(offsets[9][0], 40.0)
    assert math.isclose(offsets[9][1], 50.0)
    assert math.isclose(offsets[9][2], 90.0) # pointing East

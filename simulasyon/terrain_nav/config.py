"""Configuration classes for terrain navigation simulation."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class TerrainConfig:
    preset: str = "valley"
    seed: int = 42
    rows: int = 1000
    cols: int = 1000
    dx: float = 1.0
    dy: float = 1.0
    relief: float = 1.0
    roughness: float = 1.0
    base_elevation: float = 1000.0
    dem_noise_std_m: float = 0.5
    dem_bias_m: float = 0.0
    # Optional external DEM. When set, only a bounded raster window is read.
    dem_path: str = ""
    dem_window_size: int = 4096
    # Maximum edge length of the in-memory/search raster. The source window's
    # physical extent is preserved while reducing the number of search cells.
    dem_target_size: int = 2048
    dem_window_row: int = -1
    dem_window_col: int = -1
    external_auto_center_start: bool = True

@dataclass(frozen=True)
class RouteConfig:
    mode: str = "straight_heading"  # straight_heading, heading_sequence, waypoint_route, free_manual
    start_row: int = 500
    start_col: int = 500
    heading_deg: float = 0.0
    speed_m_s: float = 10.0
    sample_spacing_m: float = 10.0
    route_length_m: float = 1000.0
    heading_sequence: str = "" # e.g. "0:500,45:300"
    waypoints: List[Tuple[float, float]] = field(default_factory=list) # x, y

@dataclass(frozen=True)
class SensorConfig:
    # Altitude modes: known_msl_altitude, unknown_constant_msl_altitude, barometric_altitude
    altitude_mode: str = "known_msl_altitude"
    constant_msl_m: float = 1500.0
    min_safe_agl_m: float = 50.0
    
    # Barometer
    baro_bias_m: float = 0.0
    baro_noise_std_m: float = 1.0
    baro_drift_rate_m_s: float = 0.01
    baro_random_walk_std_m: float = 0.1
    
    # Laser Altimeter
    laser_noise_std_m: float = 0.5
    laser_bias_m: float = 0.0
    laser_outlier_prob: float = 0.01
    laser_drop_prob: float = 0.02
    laser_outlier_magnitude_m: float = 50.0
    laser_min_range_m: float = 0.5
    laser_max_range_m: float = 3000.0
    
    # Heading modes: known_heading, noisy_heading, unknown_heading
    heading_mode: str = "known_heading"
    sensor_heading_bias_deg: float = 0.0
    sensor_heading_noise_std_deg: float = 1.0
    heading_uncertainty_deg: float = 5.0

@dataclass(frozen=True)
class AlgorithmConfig:
    method: str = "coarse_to_fine" # exhaustive, coarse_to_fine, particle_filter
    top_k: int = 5
    min_profile_length: int = 10
    
    # Coarse to fine
    coarse_stride: int = 10
    medium_stride: int = 3
    fine_stride: int = 1
    coarse_heading_step_deg: float = 15.0
    medium_heading_step_deg: float = 3.0
    fine_heading_step_deg: float = 0.5
    refinement_radius_px: int = 20
    
    # Particle Filter
    num_particles: int = 5000
    
    # Loss method: rmse, mae, huber
    loss_method: str = "huber"
    huber_delta: float = 10.0
    
    # Online localization window
    profile_window_size: int = 100 # How many past measurements to use
    # After the first reliable match, restrict the next search around it.
    # Zero disables the optimization and keeps a global search every step.
    search_roi_size_px: int = 512
    # Maximum physically plausible correction of the profile-start anchor.
    # Set to zero to disable motion-continuity gating.
    max_match_jump_m: float = 10.0

@dataclass(frozen=True)
class LocalizationConfig:
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    route: RouteConfig = field(default_factory=RouteConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    
    headless: bool = False

"""Configuration classes for terrain navigation simulation."""

from dataclasses import dataclass, field


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
    # Optional external DEM. The complete source is resampled to a bounded
    # in-memory raster so localization can search everywhere the aircraft flies.
    dem_path: str = ""
    # Maximum edge length of the complete in-memory/search raster. The source
    # map's physical extent is preserved while reducing the number of cells.
    dem_target_size: int = 2048
    external_auto_center_start: bool = True


@dataclass(frozen=True)
class RouteConfig:
    start_row: int = 500
    start_col: int = 500
    heading_deg: float = 0.0
    speed_m_s: float = 10.0
    sample_spacing_m: float = 10.0
    route_length_m: float = 1000.0
    # UI manual-control defaults. Headless simulations keep the automatic route.
    manual_step_distance_m: float = 100.0
    manual_turn_step_deg: float = 15.0


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
    top_k: int = 5
    min_profile_length: int = 10

    # Coarse to fine
    coarse_stride: int = 10
    medium_stride: int = 3
    fine_stride: int = 1
    fine_heading_step_deg: float = 0.5
    refinement_radius_px: int = 20

    # Loss method: rmse, mae, huber
    loss_method: str = "huber"
    huber_delta: float = 10.0
    # A candidate must pass absolute quality checks before it can become a fix.
    # Quality is measured on the inlier profile so a single laser outlier does
    # not poison the whole sliding window.
    quality_trim_fraction: float = 0.05
    max_match_inlier_rmse_m: float = 3.0
    min_match_inlier_correlation: float = 0.80
    min_match_valid_ratio: float = 0.80

    # Online localization window
    profile_window_size: int = 100  # How many past measurements to use
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

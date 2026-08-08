"""Configuration classes for terrain navigation simulation."""

from dataclasses import dataclass, field, replace

MOTION_MODE_KNOWN_DISTANCE = "known_distance"
MOTION_MODE_MEASURED_SPEED = "measured_speed"
MOTION_MODE_UNKNOWN_CONSTANT_SPEED = "unknown_constant_speed"
MOTION_MODES = frozenset(
    {
        MOTION_MODE_KNOWN_DISTANCE,
        MOTION_MODE_MEASURED_SPEED,
        MOTION_MODE_UNKNOWN_CONSTANT_SPEED,
    }
)


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
    manual_sample_spacing_m: float = 20.0
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

    # Speed sensor / odometry. The simulator moves with the commanded true
    # distance, while localization receives this measured distance.
    speed_bias_m_s: float = 0.0
    speed_noise_std_m_s: float = 0.0
    speed_random_walk_std_m_s: float = 0.0


@dataclass(frozen=True)
class AlgorithmConfig:
    top_k: int = 5
    min_profile_length: int = 10
    min_profile_distance_m: float = 0.0
    # Optional measured profile span cap. Zero leaves the count-based window as
    # the only limit.
    max_profile_distance_m: float = 0.0
    # Unknown-speed profiles are gated and trimmed by elapsed time because
    # distance is deliberately unavailable to the localizer in that mode.
    min_profile_duration_s: float = 30.0
    max_profile_duration_s: float = 120.0

    # Constant-speed hypothesis search.
    speed_search_min_m_s: float = 5.0
    speed_search_max_m_s: float = 30.0
    speed_search_coarse_step_m_s: float = 5.0
    speed_search_medium_step_m_s: float = 1.0
    speed_search_fine_step_m_s: float = 0.2
    speed_search_keep_hypotheses: int = 3
    speed_tracking_half_range_m_s: float = 1.0
    speed_tracking_step_m_s: float = 1.0
    speed_ambiguity_top_k: int = 5
    speed_ambiguity_score_margin: float = 0.05
    speed_ambiguity_std_threshold_m_s: float = 2.0

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
    # Optional localization profile reduction. ``raw`` preserves the current
    # online behavior; optimizer runs can select uniform or interpolated profile
    # points without shrinking the retained measurement history.
    profile_resampling_mode: str = "raw"
    profile_points: int = 0
    # After the first reliable match, restrict the next search around it.
    # Zero disables the optimization and keeps a global search every step.
    search_roi_size_px: int = 0
    # Maximum physically plausible correction of the profile-start anchor.
    # Set to zero to disable motion-continuity gating.
    max_match_jump_m: float = 10.0

    def __post_init__(self) -> None:
        if self.speed_search_min_m_s <= 0.0:
            raise ValueError("speed_search_min_m_s must be positive")
        if self.speed_search_max_m_s <= self.speed_search_min_m_s:
            raise ValueError("speed_search_max_m_s must be greater than speed_search_min_m_s")
        for name in (
            "speed_search_coarse_step_m_s",
            "speed_search_medium_step_m_s",
            "speed_search_fine_step_m_s",
            "speed_tracking_half_range_m_s",
            "speed_tracking_step_m_s",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.speed_search_keep_hypotheses < 1:
            raise ValueError("speed_search_keep_hypotheses must be at least 1")
        if self.speed_ambiguity_top_k < 2:
            raise ValueError("speed_ambiguity_top_k must be at least 2")
        if self.speed_ambiguity_score_margin < 0.0:
            raise ValueError("speed_ambiguity_score_margin cannot be negative")
        if self.speed_ambiguity_std_threshold_m_s < 0.0:
            raise ValueError("speed_ambiguity_std_threshold_m_s cannot be negative")
        if self.min_profile_duration_s < 0.0:
            raise ValueError("min_profile_duration_s cannot be negative")
        if self.max_profile_duration_s <= self.min_profile_duration_s:
            raise ValueError(
                "max_profile_duration_s must be greater than min_profile_duration_s"
            )
        if self.profile_resampling_mode not in {"raw", "uniform", "interpolated"}:
            raise ValueError(
                "profile_resampling_mode must be one of: raw, uniform, interpolated"
            )
        if self.profile_points < 0:
            raise ValueError("profile_points cannot be negative")


@dataclass(frozen=True)
class LocalizationRuntimeConfig:
    """Configuration visible inside localization, excluding all truth route data."""

    sensor: SensorConfig
    algorithm: AlgorithmConfig
    motion_mode: str


@dataclass(frozen=True)
class LocalizationConfig:
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    route: RouteConfig = field(default_factory=RouteConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    motion_mode: str = MOTION_MODE_KNOWN_DISTANCE

    def __post_init__(self) -> None:
        if self.motion_mode not in MOTION_MODES:
            expected = ", ".join(sorted(MOTION_MODES))
            raise ValueError(f"motion_mode must be one of: {expected}")


def localization_runtime_config(config: LocalizationConfig) -> LocalizationRuntimeConfig:
    """Copy only configuration that the localization algorithm is allowed to observe."""
    return LocalizationRuntimeConfig(
        sensor=config.sensor,
        algorithm=config.algorithm,
        motion_mode=config.motion_mode,
    )


def uses_realistic_noise_mode(config: LocalizationConfig) -> bool:
    """Return whether the config has the realistic sensor-noise preset enabled."""
    return (
        config.sensor.altitude_mode == "barometric_altitude"
        and config.sensor.speed_noise_std_m_s > 0.0
    )


def apply_realistic_noise_mode(
    config: LocalizationConfig,
    enabled: bool,
    *,
    fast_synthetic: bool | None = None,
) -> LocalizationConfig:
    """Apply or remove the realistic sensor-noise preset while preserving map/route settings."""
    if fast_synthetic is None:
        fast_synthetic = (
            not config.terrain.dem_path
            and config.terrain.rows <= 100
            and config.terrain.cols <= 100
        )

    if enabled:
        min_profile_distance_m = 40.0 if fast_synthetic else 800.0
        max_profile_distance_m = 0.0 if fast_synthetic else 2000.0
        return replace(
            config,
            sensor=replace(
                config.sensor,
                altitude_mode="barometric_altitude",
                baro_bias_m=75.0,
                baro_noise_std_m=2.0,
                baro_drift_rate_m_s=0.0,
                baro_random_walk_std_m=0.03,
                speed_bias_m_s=0.0,
                speed_noise_std_m_s=0.25,
                speed_random_walk_std_m_s=0.03,
            ),
            algorithm=replace(
                config.algorithm,
                min_profile_length=5,
                min_profile_distance_m=min_profile_distance_m,
                max_profile_distance_m=max_profile_distance_m,
                max_match_inlier_rmse_m=5.0,
                max_match_jump_m=50.0,
            ),
            motion_mode=(
                config.motion_mode
                if config.motion_mode == MOTION_MODE_UNKNOWN_CONSTANT_SPEED
                else MOTION_MODE_MEASURED_SPEED
            ),
        )

    default_sensor = SensorConfig()
    default_algorithm = AlgorithmConfig()
    return replace(
        config,
        sensor=replace(
            config.sensor,
            altitude_mode=default_sensor.altitude_mode,
            baro_bias_m=default_sensor.baro_bias_m,
            baro_noise_std_m=default_sensor.baro_noise_std_m,
            baro_drift_rate_m_s=default_sensor.baro_drift_rate_m_s,
            baro_random_walk_std_m=default_sensor.baro_random_walk_std_m,
            speed_bias_m_s=default_sensor.speed_bias_m_s,
            speed_noise_std_m_s=default_sensor.speed_noise_std_m_s,
            speed_random_walk_std_m_s=default_sensor.speed_random_walk_std_m_s,
        ),
        algorithm=replace(
            config.algorithm,
            min_profile_length=default_algorithm.min_profile_length,
            min_profile_distance_m=default_algorithm.min_profile_distance_m,
            max_profile_distance_m=default_algorithm.max_profile_distance_m,
            max_match_inlier_rmse_m=default_algorithm.max_match_inlier_rmse_m,
            max_match_jump_m=default_algorithm.max_match_jump_m,
        ),
        motion_mode=(
            config.motion_mode
            if config.motion_mode == MOTION_MODE_UNKNOWN_CONSTANT_SPEED
            else MOTION_MODE_KNOWN_DISTANCE
        ),
    )

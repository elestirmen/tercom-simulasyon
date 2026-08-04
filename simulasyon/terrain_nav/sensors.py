"""Sensor models for generating noisy measurements from ground truth."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple
from terrain_nav.config import SensorConfig
from terrain_nav.coordinates import normalize_heading

@dataclass
class Measurement:
    laser_agl_m: float
    laser_valid: bool
    baro_msl_m: float
    sensor_heading_deg: float
    traveled_distance_m: float

class SensorSimulator:
    """Generates sensor measurements using models for Barometer, Laser, and Heading."""
    
    def __init__(self, config: SensorConfig, seed: int = 123):
        self.config = config
        self.rng = np.random.default_rng(seed)
        
        # Internal state
        self.baro_drift_m = 0.0
        
    def step_barometer(self, true_msl_m: float, dt_s: float = 1.0) -> float:
        """Calculate next barometer reading."""
        # Random walk for drift
        walk = self.rng.normal(loc=self.config.baro_drift_rate_m_s * dt_s, 
                               scale=self.config.baro_random_walk_std_m * np.sqrt(dt_s))
        self.baro_drift_m += walk
        
        noise = self.rng.normal(loc=0.0, scale=self.config.baro_noise_std_m)
        reading = true_msl_m + self.config.baro_bias_m + self.baro_drift_m + noise
        return reading

    def measure_laser(self, true_msl_m: float, terrain_elevation_m: float) -> Tuple[float, bool]:
        """Calculate next laser reading."""
        true_agl = true_msl_m - terrain_elevation_m
        
        # Check range constraints (without noise first, or with? Usually true range matters for drops)
        if true_agl < self.config.laser_min_range_m or true_agl > self.config.laser_max_range_m:
            return 0.0, False
            
        # Check drop probability
        if self.rng.random() < self.config.laser_drop_prob:
            return 0.0, False
            
        # Check outlier
        if self.rng.random() < self.config.laser_outlier_prob:
            sign = 1 if self.rng.random() < 0.5 else -1
            measured_agl = true_agl + sign * self.config.laser_outlier_magnitude_m
            return measured_agl, True
            
        # Normal measurement
        noise = self.rng.normal(loc=0.0, scale=self.config.laser_noise_std_m)
        measured_agl = true_agl + self.config.laser_bias_m + noise
        
        # Re-check range with noise? Usually sensors return max or invalid if it goes out.
        if measured_agl < self.config.laser_min_range_m or measured_agl > self.config.laser_max_range_m:
            return 0.0, False
            
        return measured_agl, True

    def measure_heading(self, true_heading_deg: float) -> float:
        """Calculate noisy heading."""
        if self.config.heading_mode == "known_heading":
            return true_heading_deg
            
        noise = self.rng.normal(loc=0.0, scale=self.config.sensor_heading_noise_std_deg)
        measured = true_heading_deg + self.config.sensor_heading_bias_deg + noise
        return normalize_heading(measured)

    def generate_measurement(
        self, 
        true_msl_m: float, 
        terrain_elevation_m: float, 
        true_heading_deg: float, 
        traveled_distance_m: float,
        dt_s: float = 1.0
    ) -> Measurement:
        
        baro = self.step_barometer(true_msl_m, dt_s)
        laser, valid = self.measure_laser(true_msl_m, terrain_elevation_m)
        heading = self.measure_heading(true_heading_deg)
        
        return Measurement(
            laser_agl_m=laser,
            laser_valid=valid,
            baro_msl_m=baro,
            sensor_heading_deg=heading,
            traveled_distance_m=traveled_distance_m
        )

"""Particle filter for localization."""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from terrain_nav.config import LocalizationConfig
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.matcher import evaluate_candidate, Candidate

class ParticleFilter:
    def __init__(self, config: LocalizationConfig, dem: np.ndarray, ct: CoordinateTransform):
        self.config = config
        self.dem = dem
        self.ct = ct
        self.N = config.algorithm.num_particles
        
        # State: [row, col, heading, msl_est]
        self.particles = np.zeros((self.N, 4))
        self.weights = np.ones(self.N) / self.N
        self.initialized = False
        
    def initialize(self, bounds: Tuple[float, float, float, float]):
        """Initialize particles uniformly within bounds (min_r, max_r, min_c, max_c)."""
        min_r, max_r, min_c, max_c = bounds
        self.particles[:, 0] = np.random.uniform(min_r, max_r, self.N)
        self.particles[:, 1] = np.random.uniform(min_c, max_c, self.N)
        self.particles[:, 2] = np.random.uniform(0.0, 360.0, self.N)
        self.particles[:, 3] = np.random.uniform(
            self.config.sensor.constant_msl_m - 100, 
            self.config.sensor.constant_msl_m + 100, 
            self.N
        )
        self.weights.fill(1.0 / self.N)
        self.initialized = True
        
    def predict(self, d_row: float, d_col: float, d_heading: float, std_r: float, std_c: float, std_h: float):
        """Move particles and add noise."""
        if not self.initialized: return
        
        self.particles[:, 0] += d_row + np.random.normal(0, std_r, self.N)
        self.particles[:, 1] += d_col + np.random.normal(0, std_c, self.N)
        self.particles[:, 2] += d_heading + np.random.normal(0, std_h, self.N)
        self.particles[:, 2] %= 360.0
        
    def update(
        self,
        laser_agl: np.ndarray,
        laser_valid: np.ndarray,
        baro_msl: np.ndarray,
        base_offsets: List[Tuple[float, float, float]]
    ):
        """Update weights based on measurement likelihood."""
        if not self.initialized: return
        
        # We can optimize this by vectorizing evaluate_candidate over particles,
        # but for simplicity and since PF is a stretch goal, we loop.
        # Alternatively, we just use the evaluate_candidate which returns a score (lower is better, RMSE)
        # Likelihood = exp(-score^2 / (2 * sigma^2))
        sigma = self.config.sensor.laser_noise_std_m * 2.0
        if sigma <= 0: sigma = 1.0
        
        for i in range(self.N):
            r = self.particles[i, 0]
            c = self.particles[i, 1]
            h = self.particles[i, 2]
            
            cand = evaluate_candidate(
                r, c, h, self.dem, laser_agl, laser_valid, baro_msl,
                base_offsets, self.ct, self.config
            )
            
            if cand is None:
                self.weights[i] = 1e-300 # almost zero
            else:
                self.weights[i] *= np.exp(- (cand.score**2) / (2 * sigma**2))
                self.particles[i, 3] = cand.estimated_msl_m
                
        # Normalize weights
        w_sum = np.sum(self.weights)
        if w_sum > 0:
            self.weights /= w_sum
        else:
            self.weights.fill(1.0 / self.N)
            
        self.resample()
        
    def resample(self):
        """Systematic resampling if ESS < N/2."""
        ess = 1.0 / np.sum(self.weights**2)
        if ess < self.N / 2.0:
            positions = (np.arange(self.N) + np.random.random()) / self.N
            indexes = np.zeros(self.N, dtype=int)
            cumulative_sum = np.cumsum(self.weights)
            i, j = 0, 0
            while i < self.N and j < self.N:
                if positions[i] < cumulative_sum[j]:
                    indexes[i] = j
                    i += 1
                else:
                    j += 1
            
            self.particles = self.particles[indexes]
            self.weights.fill(1.0 / self.N)
            
    def get_estimate(self) -> Candidate:
        """Return mean of particles as a Candidate."""
        if not self.initialized:
            return Candidate(0, 0, 0, 0, 9999.0, 0.0, {})
            
        # Weighted mean
        mean_r = np.average(self.particles[:, 0], weights=self.weights)
        mean_c = np.average(self.particles[:, 1], weights=self.weights)
        
        # Heading mean needs circular mean
        h_rad = np.radians(self.particles[:, 2])
        mean_sin = np.average(np.sin(h_rad), weights=self.weights)
        mean_cos = np.average(np.cos(h_rad), weights=self.weights)
        mean_h = np.degrees(np.arctan2(mean_sin, mean_cos)) % 360.0
        
        mean_msl = np.average(self.particles[:, 3], weights=self.weights)
        
        # We don't have exact metrics for the mean without re-evaluating,
        # but we can return the mean state.
        return Candidate(
            row=float(mean_r),
            col=float(mean_c),
            heading_deg=float(mean_h),
            estimated_msl_m=float(mean_msl),
            score=0.0,
            valid_ratio=1.0,
            metrics={}
        )

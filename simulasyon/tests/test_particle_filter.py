"""Tests for particle filter."""

import numpy as np
import math
from terrain_nav.config import LocalizationConfig
from terrain_nav.coordinates import CoordinateTransform
from terrain_nav.particle_filter import ParticleFilter

def test_pf_resampling():
    # 29. Resampling algoritması efektif parçacık sayısı %50'nin altına düştüğünde tetiklenir.
    config = LocalizationConfig()
    object.__setattr__(config.algorithm, 'num_particles', 10)
    dem = np.zeros((10, 10))
    ct = CoordinateTransform(1.0, 1.0)
    pf = ParticleFilter(config, dem, ct)
    
    pf.initialize((0, 10, 0, 10))
    
    # Make weights very skewed to trigger ESS < N/2
    pf.weights = np.zeros(10)
    pf.weights[0] = 0.99
    pf.weights[1] = 0.01
    
    ess_before = 1.0 / np.sum(pf.weights**2)
    assert ess_before < 5.0
    
    pf.resample()
    
    ess_after = 1.0 / np.sum(pf.weights**2)
    assert math.isclose(ess_after, 10.0) # Reset to uniform weights

def test_pf_predict():
    config = LocalizationConfig()
    object.__setattr__(config.algorithm, 'num_particles', 100)
    dem = np.zeros((10, 10))
    ct = CoordinateTransform(1.0, 1.0)
    pf = ParticleFilter(config, dem, ct)
    
    pf.initialize((0, 10, 0, 10))
    
    # Predict step
    pf.predict(d_row=1.0, d_col=-1.0, d_heading=10.0, std_r=0.0, std_c=0.0, std_h=0.0)
    
    # It should have shifted deterministically since std=0
    assert (pf.particles[:, 0] >= 1.0).all() and (pf.particles[:, 0] <= 11.0).all()
    assert (pf.particles[:, 1] >= -1.0).all() and (pf.particles[:, 1] <= 9.0).all()

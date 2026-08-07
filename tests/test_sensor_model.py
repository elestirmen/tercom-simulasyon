"""Tests for sensor models."""

import math

from terrain_nav.config import SensorConfig
from terrain_nav.sensors import SensorSimulator


def test_laser_noise_and_bias():
    # 16. Gürültüsüz lazer doğru AGL üretir.
    # 17. Lazer bias doğru eklenir.
    c = SensorConfig(
        laser_noise_std_m=0.0, laser_bias_m=5.0, laser_drop_prob=0.0, laser_outlier_prob=0.0
    )
    sim = SensorSimulator(c)

    m, valid = sim.measure_laser(true_msl_m=1000.0, terrain_elevation_m=900.0)
    assert valid
    assert math.isclose(m, 105.0)  # 100 true AGL + 5 bias


def test_laser_drop_and_out_of_range():
    # 20. Ölçüm kaybında sistem çökmez.
    # 21. Menzil dışı lazer ölçümü geçersiz sayılır.
    c = SensorConfig(
        laser_min_range_m=10.0, laser_max_range_m=100.0, laser_drop_prob=1.0
    )  # Always drop
    sim = SensorSimulator(c)
    m, valid = sim.measure_laser(true_msl_m=1000.0, terrain_elevation_m=950.0)
    assert not valid

    # Out of range max
    c2 = SensorConfig(laser_min_range_m=10.0, laser_max_range_m=100.0, laser_drop_prob=0.0)
    sim2 = SensorSimulator(c2)
    m, valid = sim2.measure_laser(true_msl_m=1000.0, terrain_elevation_m=800.0)  # AGL 200 > 100
    assert not valid

    # Out of range min
    m, valid = sim2.measure_laser(true_msl_m=1000.0, terrain_elevation_m=995.0)  # AGL 5 < 10
    assert not valid


def test_barometer_bias_and_drift():
    # 18. Barometre bias doğru eklenir.
    # 19. Barometre drift zamanla doğru değişir.
    c = SensorConfig(
        baro_bias_m=10.0, baro_noise_std_m=0.0, baro_random_walk_std_m=0.0, baro_drift_rate_m_s=2.0
    )
    sim = SensorSimulator(c)

    # Step 1: dt=1s => drift adds 2.0
    m1 = sim.step_barometer(1000.0, 1.0)
    assert math.isclose(m1, 1012.0)  # 1000 + 10 bias + 2 drift

    # Step 2: dt=1s => drift adds another 2.0 (total 4.0)
    m2 = sim.step_barometer(1000.0, 1.0)
    assert math.isclose(m2, 1014.0)


def test_heading_model():
    # known_heading
    c1 = SensorConfig(heading_mode="known_heading", sensor_heading_bias_deg=10.0)
    sim1 = SensorSimulator(c1)
    h1 = sim1.measure_heading(90.0)
    assert math.isclose(h1, 90.0)  # bias ignored in known_heading

    # noisy_heading
    c2 = SensorConfig(
        heading_mode="noisy_heading", sensor_heading_bias_deg=10.0, sensor_heading_noise_std_deg=0.0
    )
    sim2 = SensorSimulator(c2)
    h2 = sim2.measure_heading(90.0)
    assert math.isclose(h2, 100.0)


def test_speed_measurement_bias_changes_localizer_distance():
    c = SensorConfig(speed_bias_m_s=1.0, speed_noise_std_m_s=0.0)
    sim = SensorSimulator(c)

    measured_distance, measured_speed = sim.measure_traveled_distance(
        true_distance_m=10.0,
        dt_s=2.0,
    )

    assert math.isclose(measured_speed, 6.0)
    assert math.isclose(measured_distance, 12.0)

from simulation_core.filters import ConstantVelocityKalmanFilter


def test_predict_applies_commanded_displacement() -> None:
    kalman = ConstantVelocityKalmanFilter((100, 200), 2.0, 5.0)
    kalman.predict(10.0, -5.0)
    assert kalman.position == (110, 195)


def test_measurements_update_velocity_state() -> None:
    kalman = ConstantVelocityKalmanFilter((0, 0), 1.0, 2.0)
    kalman.predict()
    kalman.update(10.0, 0.0, confidence=1.0)
    assert kalman.position[0] > 0
    assert kalman.velocity[0] > 0.0


def test_reliable_update_reduces_position_uncertainty() -> None:
    kalman = ConstantVelocityKalmanFilter((0, 0), 2.0, 8.0)
    kalman.predict(5.0, 5.0)
    before = kalman.uncertainty_px
    kalman.update(5.0, 5.0, confidence=0.9)
    assert kalman.uncertainty_px < before

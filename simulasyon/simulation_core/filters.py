"""State-estimation filters used by the localization pipeline."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class ConstantVelocityKalmanFilter:
    """Four-state ``x, y, vx, vy`` Kalman filter with optional motion control.

    ``motion_x`` and ``motion_y`` are known commanded displacements. Velocity
    states model residual motion between measurements instead of replacing the
    command model.
    """

    def __init__(
        self,
        initial_position: Tuple[int, int],
        process_noise: float = 50.0,
        measurement_noise: float = 80.0,
    ) -> None:
        self._state = np.array(
            [float(initial_position[0]), float(initial_position[1]), 0.0, 0.0],
            dtype=np.float64,
        )
        measurement_variance = max(1e-6, float(measurement_noise) ** 2)
        velocity_variance = max(1e-6, float(process_noise) ** 2)
        self._covariance = np.diag(
            [measurement_variance, measurement_variance, velocity_variance, velocity_variance]
        )
        self._process_variance = max(1e-6, float(process_noise) ** 2)
        self._measurement_variance = measurement_variance

    def predict(
        self,
        motion_x: float = 0.0,
        motion_y: float = 0.0,
        dt: float = 1.0,
    ) -> None:
        dt = max(1e-3, float(dt))
        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        self._state = transition @ self._state
        self._state[0] += float(motion_x)
        self._state[1] += float(motion_y)

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = self._process_variance
        process_covariance = q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )
        self._covariance = (
            transition @ self._covariance @ transition.T + process_covariance
        )

    def update(
        self,
        measured_x: float,
        measured_y: float,
        confidence: float = 1.0,
    ) -> None:
        observation = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        confidence = max(0.01, min(1.0, float(confidence)))
        measurement_covariance = np.eye(2, dtype=np.float64) * (
            self._measurement_variance / confidence
        )
        measurement = np.array([float(measured_x), float(measured_y)], dtype=np.float64)
        innovation = measurement - observation @ self._state
        innovation_covariance = (
            observation @ self._covariance @ observation.T + measurement_covariance
        )
        kalman_gain = (
            self._covariance
            @ observation.T
            @ np.linalg.inv(innovation_covariance)
        )
        self._state = self._state + kalman_gain @ innovation
        identity = np.eye(4, dtype=np.float64)
        # Joseph form is numerically stable and keeps covariance symmetric.
        residual = identity - kalman_gain @ observation
        self._covariance = (
            residual @ self._covariance @ residual.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )

    @property
    def position(self) -> Tuple[int, int]:
        return (int(round(self._state[0])), int(round(self._state[1])))

    @property
    def velocity(self) -> Tuple[float, float]:
        return (float(self._state[2]), float(self._state[3]))

    @property
    def uncertainty_px(self) -> float:
        position_variance = max(
            0.0,
            float((self._covariance[0, 0] + self._covariance[1, 1]) / 2.0),
        )
        return float(np.sqrt(position_variance))

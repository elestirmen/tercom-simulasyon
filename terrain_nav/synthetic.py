"""Small deterministic synthetic terrains for tests and fast demos."""

import numpy as np


def generate_synthetic_dem(
    *,
    rows: int,
    cols: int,
    dx: float,
    dy: float,
    preset: str,
    relief: float,
    roughness: float,
    seed: int,
) -> np.ndarray:
    """Generate a deterministic plane or valley without external repositories."""
    if rows < 2 or cols < 2:
        raise ValueError("Synthetic DEM dimensions must be at least 2 x 2")
    if dx <= 0 or dy <= 0:
        raise ValueError("Synthetic DEM cell sizes must be positive")

    y = np.linspace(-1.0, 1.0, rows, dtype=np.float64)[:, None]
    x = np.linspace(-1.0, 1.0, cols, dtype=np.float64)[None, :]
    rng = np.random.default_rng(seed)

    if preset == "plane":
        terrain = np.zeros((rows, cols), dtype=np.float64)
    elif preset == "valley":
        terrain = 90.0 * x**2 + 18.0 * y + 8.0 * np.sin(2.5 * np.pi * y)
    else:
        raise ValueError(f"Unsupported synthetic terrain preset: {preset}")

    texture = np.zeros((rows, cols), dtype=np.float64)
    for frequency in (1.0, 1.7, 2.8, 4.3, 6.1):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amplitude = rng.uniform(0.8, 2.2) / frequency
        projected = np.cos(angle) * x + np.sin(angle) * y
        texture += amplitude * np.sin(np.pi * frequency * projected + phase)

    terrain = relief * terrain + roughness * texture
    return np.asarray(terrain, dtype=np.float64)

"""Tests for headless execution."""

import csv
import os
import subprocess
from pathlib import Path


def test_headless_mode_creates_files(tmp_path):
    # 43. Başarılı bir --headless koşusu JSON ve CSV üretir.
    env = os.environ.copy()

    # Run the entry script
    script = Path(__file__).parent.parent / "run_terrain_nav.py"

    # Call script in subprocess to avoid polluting test environment
    result = subprocess.run(
        ["python", str(script), "--headless", "--fast"], env=env, cwd=str(script.parent)
    )
    assert result.returncode == 0

    # Check if results/config.json and results/results.csv exist
    out_dir = script.parent / "results"
    assert (out_dir / "config.json").exists()
    assert (out_dir / "results.csv").exists()
    with (out_dir / "results.csv").open(encoding="utf-8", newline="") as result_file:
        headers = next(csv.reader(result_file))
    assert "estimated_speed_m_s" in headers
    assert "speed_confidence" in headers

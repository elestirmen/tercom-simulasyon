import subprocess
import sys
from pathlib import Path


def test_help_does_not_initialize_tensorflow() -> None:
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "simulasyon_yonlendirme_uclu_dashboard.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0
    assert "Üçlü şablon eşleme" in result.stdout
    assert "oneDNN" not in result.stderr
    assert "tensorflow" not in result.stderr.lower()

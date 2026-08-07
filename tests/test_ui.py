"""Focused checks for the active terrain-navigation desktop UI."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from run_terrain_nav import build_config
from terrain_nav.ui import MissionControlWindow


def test_ui_starts_with_active_map_scope_and_heading_arrow() -> None:
    app = QApplication.instance() or QApplication([])
    window = MissionControlWindow(build_config(fast_mode=True))
    try:
        window.start_sim()
        app.processEvents()

        assert window.worker is not None
        assert window.worker.isRunning()
        assert "uçuş, sensör ve eşleştirme kapsamı" in window.lbl_map_scope.text()
        assert "Tam harita lokalizasyon DEM'i" in window.lbl_loaded_scope.text()
        assert "tam kaynak harita" in window.lbl_search_scope.text()
        assert window.map_canvas.true_heading_arrow is not None
        assert window.lbl_true_pos.text() != "-"
    finally:
        window.stop_sim()
        window.worker.wait(2000)
        window.close()


def test_ui_realistic_noise_toggle_changes_simulation_config() -> None:
    app = QApplication.instance() or QApplication([])
    window = MissionControlWindow(build_config(fast_mode=True))
    try:
        assert not window.chk_realistic_noise.isChecked()
        window.chk_realistic_noise.setChecked(True)
        window.start_sim()
        app.processEvents()

        assert window.worker is not None
        assert window.worker.sim is not None
        assert window.worker.sim.config.sensor.altitude_mode == "barometric_altitude"
        assert window.worker.sim.config.sensor.speed_noise_std_m_s > 0.0
        assert window.worker.sim.config.algorithm.min_profile_distance_m == 40.0
        assert not window.chk_realistic_noise.isEnabled()
    finally:
        window.stop_sim()
        if window.worker is not None:
            window.worker.wait(2000)
        window.close()


def test_ui_realistic_noise_toggle_can_start_ideal_mode_from_realistic_cli_config() -> None:
    app = QApplication.instance() or QApplication([])
    window = MissionControlWindow(build_config(fast_mode=True, realistic_noise=True))
    try:
        assert window.chk_realistic_noise.isChecked()
        window.chk_realistic_noise.setChecked(False)
        window.start_sim()
        app.processEvents()

        assert window.worker is not None
        assert window.worker.sim is not None
        assert window.worker.sim.config.sensor.altitude_mode == "known_msl_altitude"
        assert window.worker.sim.config.sensor.speed_noise_std_m_s == 0.0
        assert window.worker.sim.config.algorithm.min_profile_distance_m == 0.0
    finally:
        window.stop_sim()
        if window.worker is not None:
            window.worker.wait(2000)
        window.close()

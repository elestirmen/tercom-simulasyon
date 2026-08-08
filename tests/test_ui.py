"""Focused checks for the active terrain-navigation desktop UI."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
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
        assert window.profile_canvas is not None
        assert window.main_splitter.count() == 3
        assert window.main_splitter.orientation() == Qt.Horizontal
        assert window.main_splitter.handleWidth() == 8
        assert window.profile_group.minimumWidth() == 280
        assert window.btn_benchmark.text() == "Benchmark Modu"
        assert not window.btn_benchmark.isEnabled()
        assert "Benchmark" in window.lbl_benchmark.text()
        assert window.benchmark_tabs.count() == 3
        assert window.benchmark_variant_table.columnCount() == 8
        assert window.cmb_motion_mode.count() == 3
        assert window.cmb_motion_mode.currentData() == "known_distance"
        assert window.lbl_est_speed.text() == "-"
        assert window.lbl_speed_confidence.text() == "-"
        assert window.lbl_cpu_workers.text() == "0 / 1 işçi (eşleştirme bekleniyor)"
        assert window.spin_parallel_workers.minimum() == 1
        assert window.spin_parallel_workers.maximum() == max(1, os.cpu_count() or 1)
        assert window.spin_parallel_workers.value() == 1
        assert "Yükseklik profili" in window.profile_canvas.axes.get_title()
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
        assert not window.spin_parallel_workers.isEnabled()
    finally:
        window.stop_sim()
        if window.worker is not None:
            window.worker.wait(2000)
        window.close()


def test_ui_parallel_worker_selection_updates_prestart_config() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None
    window = MissionControlWindow(build_config(fast_mode=True))
    try:
        selected_workers = min(4, window.spin_parallel_workers.maximum())
        window.spin_parallel_workers.setValue(selected_workers)

        config = window._simulation_config()

        assert config.algorithm.parallel_workers == selected_workers
    finally:
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


def test_ui_can_select_unknown_speed_mode() -> None:
    app = QApplication.instance() or QApplication([])
    assert app is not None
    window = MissionControlWindow(build_config(fast_mode=True))
    try:
        index = window.cmb_motion_mode.findData("unknown_constant_speed")
        window.cmb_motion_mode.setCurrentIndex(index)

        config = window._simulation_config()

        assert config.motion_mode == "unknown_constant_speed"
        assert config.sensor.altitude_mode == "barometric_altitude"
        assert config.algorithm.min_profile_duration_s == 5.0
    finally:
        window.close()

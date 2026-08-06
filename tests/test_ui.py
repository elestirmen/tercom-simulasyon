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
        assert "uçuş ve sensör kapsamı" in window.lbl_map_scope.text()
        assert "eşleştirme kapsaması" in window.lbl_loaded_scope.text()
        assert window.map_canvas.true_heading_arrow is not None
        assert window.lbl_true_pos.text() != "-"
    finally:
        window.stop_sim()
        window.worker.wait(2000)
        window.close()

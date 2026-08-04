import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from mission_control_ui import MissionControlWindow
from simulasyon_yonlendirme_uclu_dashboard import (
    SimulationConfig,
    apply_runtime_ui_hotkey,
    create_runtime_ui_state,
)


def test_mission_control_exposes_semantic_controls() -> None:
    app = QApplication.instance() or QApplication([])
    window = MissionControlWindow(
        SimulationConfig(),
        lambda **kwargs: None,
        lambda *args: None,
        {},
    )
    try:
        assert window.auto_button.accessibleName().startswith("Otonom kontrolü")
        assert window.kalman_button.accessibleName().startswith("Kalman kontrolü")
        assert window.input_size_button.accessibleName().startswith("Girdi 544")
        assert window.norm_button.accessibleName().startswith("Norm HISTEQ")
        assert window.input_size_button.property("controlRole") == "method"
        assert window.norm_button.property("controlRole") == "method"
        assert window.trajectory_button.property("controlRole") == "visual"
        assert "hesaplamayı etkilemez" in window.roi_button.toolTip()
        assert window.stop_button.accessibleName().startswith("Simülasyonu durdur")
        assert not window.stop_button.isEnabled()
        assert window.error_card.value.accessibleName() == "Simülasyon doğrulama hatası"
        assert window.evidence_splitter.count() == 3
        assert window.map_canvas.accessibleName().startswith("Operasyon haritası")
        assert window.confidence_bar.accessibleName() == "Lokalizasyon güven yüzdesi"
        window._on_telemetry({"obs_window_size": 272, "norm_mode": "CLAHE"})
        assert window.input_size_button.text().startswith("Girdi 272 px")
        assert window.input_size_button.isChecked()
        assert window.norm_button.text().startswith("Norm CLAHE")
        assert window.norm_button.isChecked()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_input_size_hotkey_changes_processing_window() -> None:
    state = create_runtime_ui_state(SimulationConfig())
    assert state["obs_window_size"] == 544
    assert apply_runtime_ui_hotkey(ord("V"), state)
    assert state["obs_window_size"] == 272
    assert apply_runtime_ui_hotkey(ord("V"), state)
    assert state["obs_window_size"] == 544


def test_normalization_hotkey_cycles_model_preprocessing() -> None:
    state = create_runtime_ui_state(SimulationConfig())
    assert state["norm_mode"] == "HISTEQ"
    assert apply_runtime_ui_hotkey(ord("N"), state)
    assert state["norm_mode"] == "EDGE"
    assert apply_runtime_ui_hotkey(ord("N"), state)
    assert state["norm_mode"] == "HAM"

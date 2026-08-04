"""Semantic, high-DPI Mission Control shell for the simulation engine."""

from __future__ import annotations

import queue
import sys
import traceback
from typing import Any, Callable, Dict, Optional

import cv2
import numpy as np

try:
    from PySide6.QtCore import QSettings, Qt, QThread, Signal
    from PySide6.QtGui import QImage, QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PySide6"
except ImportError:  # pragma: no cover - compatibility fallback
    from PyQt5.QtCore import QSettings, Qt, QThread
    from PyQt5.QtCore import pyqtSignal as Signal
    from PyQt5.QtGui import QImage, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QFrame,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMessageBox,
        QProgressBar,
        QPushButton,
        QSizePolicy,
        QSplitter,
        QVBoxLayout,
        QWidget,
    )

    QT_BINDING = "PyQt5"


APP_STYLE = """
QMainWindow, QWidget#Root {
    background: #DCE4EB;
    color: #183247;
    font-family: "Segoe UI Variable", "Segoe UI";
    font-size: 13px;
}
QFrame#Header, QFrame#Footer, QFrame#SidePanel, QFrame#MapPanel {
    background: #F4F7FA;
    border: 1px solid #AEBECC;
    border-radius: 12px;
}
QFrame#Header { border-bottom: 3px solid #2A789E; }
QLabel#AppTitle { font-size: 18px; font-weight: 700; color: #102E45; }
QLabel#Muted { color: #536B7D; }
QLabel#SectionTitle { font-size: 12px; font-weight: 700; color: #36566F; }
QLabel#ControlGroupLabel {
    color: #547087; font-size: 9px; font-weight: 800; letter-spacing: 1px;
}
QLabel#StatusPill {
    background: #D9EFE6; color: #146047; border: 1px solid #83BCA6;
    border-radius: 10px; padding: 5px 10px; font-weight: 700;
}
QPushButton {
    background: #E8EEF3; color: #234258; border: 1px solid #AFC0CE;
    border-radius: 8px; padding: 8px 12px; font-weight: 600;
}
QPushButton:hover { background: #DCE8F0; border-color: #6F96B2; }
QPushButton:pressed { background: #C8DCE9; }
QPushButton:focus { border: 2px solid #145D7E; }
QPushButton#StopButton {
    background: #F3DDE1; color: #8A2E3C; border: 1px solid #D59BA5;
    border-radius: 8px; padding: 8px 12px; font-weight: 700;
}
QPushButton#StopButton:hover { background: #EFCBD2; border-color: #B96C7A; }
QPushButton#StopButton:disabled { background: #E8EEF3; color: #718291; border-color: #C5D0D8; }
QPushButton[controlRole="method"] { border-color: #7BA9C3; }
QPushButton[controlRole="method"]:checked {
    background: #176B91; border-color: #145D7E; color: #FFFFFF;
}
QPushButton[controlRole="visual"] { color: #2D4B61; }
QPushButton[controlRole="visual"]:checked {
    background: #D7E4ED; border-color: #7C9CAF; color: #25475E;
}
QFrame#ControlGroup {
    background: #EAF0F4; border: 1px solid #C1CDD6; border-radius: 9px;
}
QFrame#EvidenceCard {
    background: #EDF2F5; border: 1px solid #C3CFD8; border-radius: 9px;
}
QProgressBar {
    background: #D4DFE7; border: 1px solid #B5C4CF; border-radius: 6px;
    color: #234258; text-align: center; min-height: 12px;
}
QProgressBar::chunk { background: #1886A5; border-radius: 5px; }
QSplitter::handle { background: #A8BAC7; }
QSplitter::handle:horizontal { width: 8px; margin: 6px 2px; }
QSplitter::handle:vertical { height: 8px; margin: 2px 6px; }
QSplitter::handle:hover { background: #2787A9; }
QToolTip {
    background: #F7FAFC; color: #183247; border: 1px solid #8EA5B6;
    padding: 5px;
}
"""


def _as_bgr(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image is None or image.size == 0:
        return None
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


class ImagePane(QLabel):
    def __init__(self, empty_text: str, parent=None) -> None:
        super().__init__(empty_text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(160, 100)
        self.setStyleSheet(
            "background:#C9D4DC; border:1px solid #98AAB8; "
            "border-radius:9px; color:#3E596B;"
        )
        self._pixmap_source: Optional[QPixmap] = None

    def set_frame(self, frame: Optional[np.ndarray]) -> None:
        bgr = _as_bgr(frame)
        if bgr is None:
            return
        height, width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        qimage = QImage(rgb.data, width, height, width * 3, QImage.Format_RGB888).copy()
        self._pixmap_source = QPixmap.fromImage(qimage)
        self._refresh()

    def _refresh(self) -> None:
        if self._pixmap_source is not None:
            self.setPixmap(
                self._pixmap_source.scaled(
                    self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()


class MapCanvas(ImagePane):
    source_pressed = Signal(int, int)
    source_moved = Signal(int, int)

    def __init__(self, parent=None) -> None:
        super().__init__("Haritalar ve model hazırlanıyor…", parent)
        self.setAccessibleName("Operasyon haritası; tıklayarak waypoint seçin")
        self.setMinimumSize(520, 520)
        self.setMouseTracking(True)
        self._source_origin = (0, 0)
        self._image_size = (1, 1)

    def set_map_frame(self, frame: np.ndarray, source_origin=(0, 0)) -> None:
        self._source_origin = (int(source_origin[0]), int(source_origin[1]))
        self._image_size = (int(frame.shape[1]), int(frame.shape[0]))
        self.set_frame(frame)

    def _to_source(self, window_x: int, window_y: int) -> tuple[int, int]:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return window_x, window_y
        pix_w, pix_h = pixmap.width(), pixmap.height()
        x_offset = (self.width() - pix_w) // 2
        y_offset = (self.height() - pix_h) // 2
        image_x = int((window_x - x_offset) * self._image_size[0] / max(1, pix_w))
        image_y = int((window_y - y_offset) * self._image_size[1] / max(1, pix_h))
        return image_x + self._source_origin[0], image_y + self._source_origin[1]

    @staticmethod
    def _event_xy(event) -> tuple[int, int]:
        if hasattr(event, "position"):
            point = event.position()
            return int(point.x()), int(point.y())
        return int(event.x()), int(event.y())

    def mousePressEvent(self, event) -> None:
        x, y = self._event_xy(event)
        source_x, source_y = self._to_source(x, y)
        self.source_pressed.emit(source_x, source_y)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        x, y = self._event_xy(event)
        source_x, source_y = self._to_source(x, y)
        self.source_moved.emit(source_x, source_y)
        super().mouseMoveEvent(event)


class MetricCard(QFrame):
    def __init__(self, title: str, initial: str = "—", parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(
            "QFrame{background:#E8EEF2;border:1px solid #C5D0D8;border-radius:9px;}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(11, 9, 11, 9)
        layout.setSpacing(2)
        label = QLabel(title.upper())
        label.setStyleSheet("color:#597084;font-size:10px;font-weight:700;border:0;")
        self.value = QLabel(initial)
        self.value.setStyleSheet("color:#102E45;font-size:18px;font-weight:700;border:0;")
        self.value.setAccessibleName(title)
        layout.addWidget(label)
        layout.addWidget(self.value)


class SimulationWorker(QThread):
    frame_ready = Signal(object)
    telemetry_ready = Signal(object)
    status_ready = Signal(str, str)
    failed = Signal(str)

    def __init__(self, config, simulation_main: Callable[..., None], parent=None) -> None:
        super().__init__(parent)
        self._config = config
        self._simulation_main = simulation_main
        self._key_queue: queue.Queue[int] = queue.Queue()
        self.context_holder = [None]

    def post_key(self, key: int) -> None:
        self._key_queue.put(int(key))

    def _get_key(self, wait_ms: int) -> int:
        try:
            return self._key_queue.get(timeout=max(0.001, wait_ms / 1000.0))
        except queue.Empty:
            return -1

    def _display(self, dashboard: np.ndarray, label_state: dict) -> None:
        label_state.update(scale=1.0, x_off=0, y_off=0)
        context = self.context_holder[0] or {}
        map_rect = context.get("map_rect", (0, 0, dashboard.shape[1], dashboard.shape[0]))
        x, y, width, height = [int(value) for value in map_rect]
        x = max(0, min(x, dashboard.shape[1] - 1))
        y = max(0, min(y, dashboard.shape[0] - 1))
        width = max(1, min(width, dashboard.shape[1] - x))
        height = max(1, min(height, dashboard.shape[0] - y))
        map_frame = dashboard[y : y + height, x : x + width].copy()
        self.frame_ready.emit(
            {
                "map": map_frame,
                "map_origin": (x, y),
                "observation": context.get("observation_view"),
                "template": context.get("template_strip"),
                "reference_patch": context.get("ref_patch_image"),
            }
        )

    def run(self) -> None:
        try:
            self._simulation_main(
                config=self._config,
                _display_fn=self._display,
                _getkey_fn=self._get_key,
                _use_qt=True,
                _ctx_holder=self.context_holder,
                _telemetry_fn=self.telemetry_ready.emit,
                _status_fn=self.status_ready.emit,
            )
        except Exception:
            self.failed.emit(traceback.format_exc())


class MissionControlWindow(QMainWindow):
    def __init__(
        self,
        config,
        simulation_main: Callable[..., None],
        runtime_mouse_callback: Callable[..., None],
        qt_key_map: Dict[int, int],
    ) -> None:
        super().__init__()
        self._config = config
        self._runtime_mouse_callback = runtime_mouse_callback
        self._qt_key_map = qt_key_map
        self._worker = SimulationWorker(config, simulation_main, self)
        self._closing = False
        self._was_shown = False
        self._stop_requested = False
        self._failure_seen = False
        self._settings = QSettings(
            "KapadokyaUniversity", "GPSDeniedMissionControl"
        )

        self.setWindowTitle("GPS-Denied Mission Control")
        self.setMinimumSize(1180, 760)
        self.resize(1480, 920)
        self.setStyleSheet(APP_STYLE)
        self._build_ui()

        self._worker.frame_ready.connect(self._on_frame)
        self._worker.telemetry_ready.connect(self._on_telemetry)
        self._worker.status_ready.connect(self._on_status)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self.map_canvas.source_pressed.connect(self._on_map_press)
        self.map_canvas.source_moved.connect(self._on_map_move)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("Root")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(14, 14, 14, 12)
        root_layout.setSpacing(10)
        self.setCentralWidget(root)

        header = QFrame()
        header.setObjectName("Header")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 11, 14, 11)
        header_layout.setSpacing(7)
        identity_row = QHBoxLayout()
        identity_row.setSpacing(10)
        brand = QLabel("◈")
        brand.setStyleSheet("font-size:26px;color:#1682A3;")
        title_box = QVBoxLayout()
        title_box.setSpacing(0)
        title = QLabel("GPS-Denied Mission Control")
        title.setObjectName("AppTitle")
        subtitle = QLabel("Görsel lokalizasyon ve otonom görev konsolu")
        subtitle.setObjectName("Muted")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        identity_row.addWidget(brand)
        identity_row.addLayout(title_box)
        identity_row.addStretch(1)

        scenario = QLabel(str(self._config.scenario_mode).upper())
        scenario.setStyleSheet(
            "background:#DFEAF2;color:#234F6B;border:1px solid #ABC1D0;"
            "border-radius:10px;padding:5px 10px;font-weight:700;"
        )
        scenario.setAccessibleName("Aktif senaryo")
        self.status_pill = QLabel("BAŞLATILIYOR")
        self.status_pill.setObjectName("StatusPill")
        self.status_pill.setAccessibleName("Lokalizasyon durumu")
        identity_row.addWidget(scenario)
        identity_row.addWidget(self.status_pill)
        self.stop_button = QPushButton("Durdur  Esc")
        self.stop_button.setObjectName("StopButton")
        self.stop_button.setAccessibleName("Simülasyonu durdur, kısayol Escape")
        self.stop_button.setAccessibleDescription(
            "Çalışan simülasyonu güvenli şekilde durdurur; pencereyi kapatmaz."
        )
        self.stop_button.setToolTip(self.stop_button.accessibleDescription())
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_simulation)
        identity_row.addWidget(self.stop_button)
        header_layout.addLayout(identity_row)

        method_group = QFrame()
        method_group.setObjectName("ControlGroup")
        method_layout = QVBoxLayout(method_group)
        method_layout.setContentsMargins(8, 4, 8, 7)
        method_layout.setSpacing(4)
        method_title = QLabel("İŞLEM / YÖNTEM")
        method_title.setObjectName("ControlGroupLabel")
        method_row = QHBoxLayout()
        method_row.setSpacing(6)

        self.auto_button = self._control_button(
            "Otonom",
            "P",
            True,
            "method",
            "Uçuş komutlarını hedefe göre otomatik üretir; simülasyon davranışını değiştirir.",
        )
        self.auto_button.setChecked(bool(self._config.autonomous_mode_enabled))
        self.kalman_button = self._control_button(
            "Kalman",
            "K",
            True,
            "method",
            "Konum kestirim filtresini açar veya kapatır; lokalizasyon dinamiğini değiştirir.",
        )
        self.kalman_button.setChecked(bool(self._config.kalman_enabled))
        self.input_size_button = self._control_button(
            f"Girdi {int(self._config.sample_window_size)} px",
            "V",
            True,
            "method",
            "Model için alınan gözlem penceresini 544 ve 272 piksel arasında değiştirir.",
        )
        self.input_size_button.setChecked(int(self._config.sample_window_size) == 272)
        self.norm_button = self._control_button(
            "Norm HISTEQ",
            "N",
            True,
            "method",
            "Model girdisi normalizasyonunu HAM, CLAHE, HISTEQ ve EDGE arasında değiştirir.",
        )
        self.norm_button.setChecked(True)
        for button in (
            self.auto_button,
            self.kalman_button,
            self.input_size_button,
            self.norm_button,
        ):
            method_row.addWidget(button)
        method_layout.addWidget(method_title)
        method_layout.addLayout(method_row)

        visual_group = QFrame()
        visual_group.setObjectName("ControlGroup")
        visual_layout = QVBoxLayout(visual_group)
        visual_layout.setContentsMargins(8, 4, 8, 7)
        visual_layout.setSpacing(4)
        visual_title = QLabel("YALNIZ GÖRÜNÜM")
        visual_title.setObjectName("ControlGroupLabel")
        visual_row = QHBoxLayout()
        visual_row.setSpacing(6)
        self.trajectory_button = self._control_button(
            "Rota",
            "T",
            True,
            "visual",
            "Yalnızca rota çizgisinin görünürlüğünü değiştirir; hesaplama yöntemini etkilemez.",
        )
        self.trajectory_button.setChecked(bool(self._config.show_trajectory))
        self.roi_button = self._control_button(
            "Arama alanı",
            "O",
            True,
            "visual",
            "Yalnızca arama alanı çerçevesini gösterir veya gizler; hesaplamayı etkilemez.",
        )
        self.roi_button.setChecked(bool(self._config.show_roi_frame))
        visual_row.addWidget(self.trajectory_button)
        visual_row.addWidget(self.roi_button)
        visual_layout.addWidget(visual_title)
        visual_layout.addLayout(visual_row)
        controls_row = QHBoxLayout()
        controls_row.setSpacing(8)
        controls_row.addWidget(method_group)
        controls_row.addWidget(visual_group)
        controls_row.addStretch(1)
        header_layout.addLayout(controls_row)
        root_layout.addWidget(header)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setAccessibleName("Ana panel genişlik ayırıcısı")
        self.main_splitter.setChildrenCollapsible(False)

        left_panel = QFrame()
        left_panel.setObjectName("SidePanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(11, 11, 11, 11)
        left_title = QLabel("GÖRSEL KANIT")
        left_title.setObjectName("SectionTitle")
        left_layout.addWidget(left_title)
        self.observation_view = ImagePane("Canlı görüntü bekleniyor")
        self.observation_view.setAccessibleName("Canlı gözlem görüntüsü")
        self.template_view = ImagePane("Model çıktısı bekleniyor")
        self.template_view.setAccessibleName("Model şablon çıktısı")
        self.reference_patch_view = ImagePane("Eşleşme bekleniyor")
        self.reference_patch_view.setAccessibleName("Referansta eşleşen bölge")
        self.evidence_splitter = QSplitter(Qt.Vertical)
        self.evidence_splitter.setAccessibleName("Görsel kanıt yükseklik ayırıcısı")
        self.evidence_splitter.setChildrenCollapsible(False)
        self.evidence_splitter.addWidget(
            self._evidence_section("GÖZLEM", self.observation_view)
        )
        self.evidence_splitter.addWidget(
            self._evidence_section("MODEL", self.template_view)
        )
        self.evidence_splitter.addWidget(
            self._evidence_section("EŞLEŞME", self.reference_patch_view)
        )
        self.evidence_splitter.setSizes([240, 240, 240])
        left_layout.addWidget(self.evidence_splitter, 1)

        map_panel = QFrame()
        map_panel.setObjectName("MapPanel")
        map_layout = QVBoxLayout(map_panel)
        map_layout.setContentsMargins(10, 10, 10, 10)
        map_header = QHBoxLayout()
        map_title = QLabel("OPERASYON HARİTASI")
        map_title.setObjectName("SectionTitle")
        self.map_hint = QLabel("Haritada hedef seçmek için tıklayın")
        self.map_hint.setObjectName("Muted")
        map_header.addWidget(map_title)
        map_header.addStretch(1)
        map_header.addWidget(self.map_hint)
        map_layout.addLayout(map_header)
        self.map_canvas = MapCanvas()
        map_layout.addWidget(self.map_canvas, 1)
        self.loading_bar = QProgressBar()
        self.loading_bar.setRange(0, 0)
        self.loading_bar.setAccessibleName("Başlatma ilerlemesi")
        map_layout.addWidget(self.loading_bar)

        right_panel = QFrame()
        right_panel.setObjectName("SidePanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(11, 11, 11, 11)
        right_layout.setSpacing(8)
        nav_title = QLabel("NAVİGASYON")
        nav_title.setObjectName("SectionTitle")
        right_layout.addWidget(nav_title)
        self.heading_card = MetricCard("Başlık")
        self.altitude_card = MetricCard("AGL")
        self.gsd_card = MetricCard("GSD")
        self.error_card = MetricCard("Simülasyon doğrulama hatası")
        cards_row_1 = QHBoxLayout()
        cards_row_1.addWidget(self.heading_card)
        cards_row_1.addWidget(self.altitude_card)
        cards_row_2 = QHBoxLayout()
        cards_row_2.addWidget(self.gsd_card)
        cards_row_2.addWidget(self.error_card)
        right_layout.addLayout(cards_row_1)
        right_layout.addLayout(cards_row_2)

        confidence_title = QLabel("LOKALİZASYON GÜVENİ")
        confidence_title.setObjectName("SectionTitle")
        right_layout.addWidget(confidence_title)
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.confidence_bar.setAccessibleName("Lokalizasyon güven yüzdesi")
        right_layout.addWidget(self.confidence_bar)
        self.detail_label = QLabel("Model ve raster kaynakları hazırlanıyor")
        self.detail_label.setWordWrap(True)
        self.detail_label.setObjectName("Muted")
        self.detail_label.setAccessibleName("Lokalizasyon ayrıntıları")
        right_layout.addWidget(self.detail_label)
        right_layout.addStretch(1)

        self.main_splitter.addWidget(left_panel)
        self.main_splitter.addWidget(map_panel)
        self.main_splitter.addWidget(right_panel)
        self.main_splitter.setSizes([285, 1050, 300])
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setStretchFactor(2, 0)
        saved_main_splitter = self._settings.value("main_splitter_state")
        if saved_main_splitter is not None:
            self.main_splitter.restoreState(saved_main_splitter)
        saved_evidence_splitter = self._settings.value("evidence_splitter_state")
        if saved_evidence_splitter is not None:
            self.evidence_splitter.restoreState(saved_evidence_splitter)
        root_layout.addWidget(self.main_splitter, 1)

        footer = QFrame()
        footer.setObjectName("Footer")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(13, 7, 13, 7)
        shortcuts = QLabel(
            "Ayırıcıları sürükleyerek panel boyutlarını ayarlayın  •  "
            "WASD hareket  •  Q/E dönüş  •  N normalizasyon  •  V 544/272  •  ESC çıkış"
        )
        shortcuts.setObjectName("Muted")
        self.performance_label = QLabel(f"{QT_BINDING} • bekleniyor")
        self.performance_label.setObjectName("Muted")
        self.performance_label.setAccessibleName("İşleme performansı")
        footer_layout.addWidget(shortcuts)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.performance_label)
        root_layout.addWidget(footer)

        self.setFocusPolicy(Qt.StrongFocus)
        self.map_canvas.setFocusPolicy(Qt.StrongFocus)
        self.setTabOrder(self.stop_button, self.auto_button)
        self.setTabOrder(self.auto_button, self.kalman_button)
        self.setTabOrder(self.kalman_button, self.input_size_button)
        self.setTabOrder(self.input_size_button, self.norm_button)
        self.setTabOrder(self.norm_button, self.trajectory_button)
        self.setTabOrder(self.trajectory_button, self.roi_button)
        self.setTabOrder(self.roi_button, self.map_canvas)

    @staticmethod
    def _evidence_section(title: str, pane: ImagePane) -> QFrame:
        section = QFrame()
        section.setObjectName("EvidenceCard")
        layout = QVBoxLayout(section)
        layout.setContentsMargins(7, 6, 7, 7)
        layout.setSpacing(5)
        label = QLabel(title)
        label.setObjectName("SectionTitle")
        layout.addWidget(label)
        layout.addWidget(pane, 1)
        return section

    def _control_button(
        self,
        label: str,
        hotkey: str,
        checkable: bool,
        role: str,
        description: str,
    ) -> QPushButton:
        button = QPushButton(f"{label}  {hotkey}")
        button.setCheckable(checkable)
        button.setProperty("controlRole", role)
        button.setAccessibleName(f"{label} kontrolü, kısayol {hotkey}")
        button.setAccessibleDescription(description)
        button.setToolTip(description)
        button.clicked.connect(lambda _checked=False, key=hotkey: self._worker.post_key(ord(key)))
        return button

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._was_shown = True
        if not self._worker.isRunning():
            self._worker.start()
        self.stop_button.setEnabled(True)

    def _set_status(self, text: str, state: str) -> None:
        styles = {
            "ready": (
                "background:#D9EFE6;color:#146047;border:1px solid #83BCA6;"
                "border-radius:10px;padding:5px 10px;font-weight:700;"
            ),
            "warning": (
                "background:#F7E8C8;color:#755019;border:1px solid #D6B66F;"
                "border-radius:10px;padding:5px 10px;font-weight:700;"
            ),
            "loading": (
                "background:#DFEAF2;color:#234F6B;border:1px solid #ABC1D0;"
                "border-radius:10px;padding:5px 10px;font-weight:700;"
            ),
            "error": (
                "background:#F3DDE1;color:#8A2E3C;border:1px solid #D59BA5;"
                "border-radius:10px;padding:5px 10px;font-weight:700;"
            ),
            "stopped": (
                "background:#E8EEF3;color:#52687A;border:1px solid #B5C4CF;"
                "border-radius:10px;padding:5px 10px;font-weight:700;"
            ),
        }
        self.status_pill.setText(text)
        self.status_pill.setStyleSheet(styles.get(state, styles["ready"]))

    def _stop_simulation(self) -> None:
        if not self._worker.isRunning():
            return
        self._stop_requested = True
        self.stop_button.setEnabled(False)
        self._set_status("DURDURULUYOR", "warning")
        self._worker.post_key(27)

    def _on_frame(self, bundle: Dict[str, Any]) -> None:
        self.map_canvas.set_map_frame(bundle["map"], bundle.get("map_origin", (0, 0)))
        self.observation_view.set_frame(bundle.get("observation"))
        self.template_view.set_frame(bundle.get("template"))
        self.reference_patch_view.set_frame(bundle.get("reference_patch"))
        self.loading_bar.hide()

    def _on_telemetry(self, data: Dict[str, Any]) -> None:
        self.heading_card.value.setText(str(data.get("heading", "—")))
        self.altitude_card.value.setText(f"{float(data.get('altitude_m', 0.0)):.0f} m")
        self.gsd_card.value.setText(f"{float(data.get('gsd_cm', 0.0)):.1f} cm")
        self.error_card.value.setText(f"{float(data.get('error_m', 0.0)):.1f} m")
        confidence = max(0, min(100, int(round(float(data.get("confidence", 0.0)) * 100))))
        self.confidence_bar.setValue(confidence)
        reliable = bool(data.get("reliable"))
        if reliable:
            self._set_status("KİLİT SAĞLAM", "ready")
        else:
            self._set_status("YENİDEN KAZANIM", "warning")
        scores = data.get("scores", ())
        score_text = " / ".join(f"{float(score):.3f}" for score in scores)
        peak_margins = data.get("peak_margins", ())
        peak_floor = min((float(value) for value in peak_margins), default=0.0)
        template_stddevs = data.get("template_stddevs", ())
        template_std_floor = min(
            (float(value) for value in template_stddevs),
            default=0.0,
        )
        self.detail_label.setText(
            f"Adım: {data.get('step', 0)}\n"
            f"Arama: {data.get('search_mode', '—')} • "
            f"Kesişim: {data.get('intersection_mode', '—')}\n"
            f"Skorlar: {score_text or '—'} • Tepe marjı: {peak_floor:.3f}\n"
            f"Şablon σ: {template_std_floor:.1f}\n"
            f"Karar: {data.get('reason', '—')} • Eylem: {data.get('action', '—')}"
        )
        self.performance_label.setText(
            f"{QT_BINDING} • {float(data.get('processing_ms', 0.0)):.0f} ms • "
            f"{data.get('backend', '—')}"
        )
        for button, state in (
            (self.auto_button, bool(data.get("autonomous"))),
            (self.kalman_button, bool(data.get("kalman_on"))),
        ):
            button.blockSignals(True)
            button.setChecked(state)
            button.blockSignals(False)
        input_size = int(data.get("obs_window_size", self._config.sample_window_size))
        self.input_size_button.blockSignals(True)
        self.input_size_button.setText(f"Girdi {input_size} px  V")
        self.input_size_button.setChecked(input_size == 272)
        self.input_size_button.setAccessibleName(
            f"Girdi çözünürlüğü {input_size} piksel, kısayol V"
        )
        self.input_size_button.blockSignals(False)
        norm_mode = str(data.get("norm_mode", "HISTEQ")).upper()
        self.norm_button.blockSignals(True)
        self.norm_button.setText(f"Norm {norm_mode}  N")
        self.norm_button.setChecked(norm_mode != "HAM")
        self.norm_button.setAccessibleName(
            f"Normalizasyon yöntemi {norm_mode}, kısayol N"
        )
        self.norm_button.blockSignals(False)

    def _on_status(self, message: str, level: str) -> None:
        self.map_canvas.setText(message)
        self.map_canvas.setAccessibleDescription(message)
        if level == "loading":
            self.loading_bar.show()
            self._set_status("YÜKLENİYOR", "loading")
        elif level == "ready":
            self.loading_bar.hide()
            self._set_status("HAZIR", "ready")

    def _on_failed(self, traceback_text: str) -> None:
        self._failure_seen = True
        self.stop_button.setEnabled(False)
        self.loading_bar.hide()
        self._set_status("HATA", "error")
        summary = traceback_text.strip().splitlines()[-1] if traceback_text.strip() else "Bilinmeyen hata"
        self.map_canvas.setText(f"Simülasyon başlatılamadı\n{summary}")
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Simülasyon başlatılamadı")
        dialog.setIcon(QMessageBox.Critical)
        dialog.setText(summary)
        dialog.setDetailedText(traceback_text)
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.open()

    def _context(self) -> Optional[dict]:
        return self._worker.context_holder[0]

    def _on_map_press(self, x: int, y: int) -> None:
        context = self._context()
        if context is not None:
            self._runtime_mouse_callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, context)

    def _on_map_move(self, x: int, y: int) -> None:
        context = self._context()
        if context is not None:
            self._runtime_mouse_callback(cv2.EVENT_MOUSEMOVE, x, y, 0, context)

    def keyPressEvent(self, event) -> None:
        qt_key = int(event.key())
        cv2_key = self._qt_key_map.get(qt_key)
        if cv2_key is None:
            text = event.text()
            cv2_key = ord(text) if text and len(text) == 1 else qt_key
        self._worker.post_key(cv2_key)
        event.accept()

    def closeEvent(self, event) -> None:
        self._closing = True
        if self._was_shown:
            self._settings.setValue(
                "main_splitter_state", self.main_splitter.saveState()
            )
            self._settings.setValue(
                "evidence_splitter_state", self.evidence_splitter.saveState()
            )
        if self._worker.isRunning():
            self._worker.post_key(27)
            self._worker.wait(5000)
        event.accept()

    def _on_worker_finished(self) -> None:
        if self._closing:
            self.close()
            return
        self.stop_button.setEnabled(False)
        if self._failure_seen:
            return
        if self._stop_requested:
            self._set_status("DURDURULDU", "stopped")
        else:
            self._set_status("TAMAMLANDI", "ready")


def run_mission_control(
    config,
    simulation_main: Callable[..., None],
    runtime_mouse_callback: Callable[..., None],
    qt_key_map: Dict[int, int],
) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("GPS-Denied Mission Control")
    window = MissionControlWindow(
        config,
        simulation_main,
        runtime_mouse_callback,
        qt_key_map,
    )
    window.show()
    return int(app.exec())

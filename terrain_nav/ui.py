"""PySide6 UI for Terrain Navigation."""

import sys
from queue import Empty, Queue

from PySide6.QtCore import QEvent, Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from terrain_nav.config import (
    LocalizationConfig,
    apply_realistic_noise_mode,
    uses_realistic_noise_mode,
)
from terrain_nav.rendering import MapCanvas
from terrain_nav.simulation import MotionOutOfBoundsError, SimulationEngine

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', Arial, sans-serif;
}
QGroupBox {
    border: 1px solid #45475a;
    border-radius: 6px;
    margin-top: 12px;
    font-weight: bold;
    color: #89b4fa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
}
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 4px;
    padding: 8px;
    font-weight: bold;
}
QPushButton:hover {
    background-color: #45475a;
}
QPushButton#btnStart {
    background-color: #a6e3a1;
    color: #11111b;
}
QPushButton#btnStart:hover {
    background-color: #94e2d5;
}
QPushButton#btnStop {
    background-color: #f38ba8;
    color: #11111b;
}
QPushButton#btnStop:hover {
    background-color: #eba0ac;
}
QTextEdit {
    background-color: #11111b;
    color: #a6e3a1;
    font-family: 'Consolas', monospace;
    border: 1px solid #45475a;
    border-radius: 4px;
}
QLabel#metricLabel {
    font-size: 18px;
    font-weight: bold;
    color: #f9e2af;
}
QLabel#titleLabel {
    font-size: 14px;
    color: #a6adc8;
}
"""


class SimulationWorker(QThread):
    step_done = Signal(int, object, object, object)  # step_idx, true_state, est_state, measurement
    state_changed = Signal(object)
    command_rejected = Signal(str)
    finished_sim = Signal()

    def __init__(
        self,
        config: LocalizationConfig,
        simulation: SimulationEngine | None = None,
        manual_control: bool = True,
    ):
        super().__init__()
        self.config = config
        self.sim = simulation
        self.manual_control = manual_control
        self._is_running = False
        self._commands = Queue()
        self.stopped_by_user = False

    def run(self):
        if self.sim is None:
            self.sim = SimulationEngine(self.config, manual_control=self.manual_control)
        self._is_running = True
        self.stopped_by_user = False
        total = self.sim.get_total_steps()
        try:
            if self.manual_control:
                # Manual mode is event-driven. The route length is a benchmark
                # hint, not a hard command limit; keep waiting for W/S/A/D/Q/E.
                while self._is_running:
                    try:
                        command, value = self._commands.get(timeout=0.1)
                    except Empty:
                        continue

                    if command == "rotate":
                        self.sim.turn_vehicle(value)
                        self.state_changed.emit(self.sim.get_current_state())
                        continue

                    if command == "move":
                        try:
                            true_s, est_s, measurement = self.sim.execute_motion(
                                self.config.route.manual_step_distance_m,
                                value,
                            )
                        except MotionOutOfBoundsError as exc:
                            self.command_rejected.emit(str(exc))
                            continue
                        self.step_done.emit(self.sim.step_idx - 1, true_s, est_s, measurement)

                    # Keep a small pause so a burst of queued commands still lets
                    # the Qt event loop repaint telemetry and the map.
                    self.msleep(50)
            else:
                for i in range(total):
                    if not self._is_running:
                        break
                    true_s, est_s, m = self.sim.step()
                    self.step_done.emit(i, true_s, est_s, m)
                    self.msleep(100)  # Sleep to allow UI to update and user to see
        finally:
            self.sim.close()
            self.finished_sim.emit()

    def stop(self):
        self.stopped_by_user = True
        self._is_running = False

    def turn_vehicle(self, angle_deg: float):
        if self.manual_control:
            self._commands.put(("rotate", angle_deg))
        elif self.sim:
            self.sim.turn_vehicle(angle_deg)

    def post_motion(self, relative_heading_deg: float):
        """Queue one movement command for the worker thread."""
        self._commands.put(("move", relative_heading_deg))


class MissionControlWindow(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
        incoming_config = config or LocalizationConfig()
        self.base_config = apply_realistic_noise_mode(incoming_config, False)
        self.config = self.base_config
        self.setWindowTitle("TERCOM Mission Control")
        self.resize(1280, 800)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.setFocusPolicy(Qt.StrongFocus)

        # Main vertical layout: Top (Map + Sidebar) and Bottom (Console)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)

        top_layout = QHBoxLayout()

        # --- Left Panel: Map ---
        self.map_canvas = MapCanvas(self)
        top_layout.addWidget(self.map_canvas, 4)

        # --- Right Panel: Sidebar ---
        sidebar_layout = QVBoxLayout()

        # Controls Group
        ctrl_group = QGroupBox("Simülasyon Kontrolü")
        ctrl_layout = QVBoxLayout()
        self.btn_start = QPushButton("Manuel Simülasyonu Başlat")
        self.btn_start.setObjectName("btnStart")
        self.btn_stop = QPushButton("Acil Durdurma")
        self.btn_stop.setObjectName("btnStop")
        self.chk_realistic_noise = QCheckBox("Gerçekçi sensör gürültüsü")
        self.chk_realistic_noise.setChecked(uses_realistic_noise_mode(incoming_config))
        self.chk_realistic_noise.setToolTip(
            "Barometre bias/gürültüsü ve hız ölçüm sapmasını etkinleştirir."
        )

        self.btn_start.clicked.connect(self.start_sim)
        self.btn_stop.clicked.connect(self.stop_sim)

        ctrl_layout.addWidget(self.chk_realistic_noise)
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        self.lbl_controls = QLabel(
            f"W ileri / S geri / A sol / D sağ "
            f"({self.config.route.manual_step_distance_m:.0f} m)\n"
            f"Q sola / E sağa dön ({self.config.route.manual_turn_step_deg:.0f}°). "
            "Her komut sonrası sistem bekler."
        )
        self.lbl_controls.setObjectName("titleLabel")
        self.lbl_controls.setWordWrap(True)
        ctrl_layout.addWidget(self.lbl_controls)
        ctrl_group.setLayout(ctrl_layout)
        sidebar_layout.addWidget(ctrl_group)

        scope_group = QGroupBox("Harita ve Arama Kapsamı")
        scope_layout = QVBoxLayout()
        self.lbl_map_scope = QLabel("Tam harita: simülasyon başlayınca hazırlanacak")
        self.lbl_loaded_scope = QLabel("Lokalizasyon DEM'i: simülasyon başlayınca hazırlanacak")
        self.lbl_search_scope = QLabel("Aktif arama: bekleniyor")
        for label in (
            self.lbl_map_scope,
            self.lbl_loaded_scope,
            self.lbl_search_scope,
        ):
            label.setObjectName("titleLabel")
            label.setWordWrap(True)
            scope_layout.addWidget(label)
        scope_group.setLayout(scope_layout)
        sidebar_layout.addWidget(scope_group)

        # Telemetry Group
        telemetry_group = QGroupBox("Canlı Telemetri")
        tel_layout = QFormLayout()

        self.lbl_step = QLabel("-")
        self.lbl_step.setObjectName("metricLabel")

        self.lbl_true_pos = QLabel("-")
        self.lbl_true_pos.setObjectName("metricLabel")

        self.lbl_est_pos = QLabel("-")
        self.lbl_est_pos.setObjectName("metricLabel")

        self.lbl_true_heading = QLabel("-")
        self.lbl_true_heading.setObjectName("metricLabel")

        self.lbl_est_heading = QLabel("-")
        self.lbl_est_heading.setObjectName("metricLabel")

        self.lbl_msl = QLabel("-")
        self.lbl_msl.setObjectName("metricLabel")

        self.lbl_agl = QLabel("-")
        self.lbl_agl.setObjectName("metricLabel")

        self.lbl_error_pos = QLabel("0.0 m")
        self.lbl_error_pos.setObjectName("metricLabel")

        self.lbl_score = QLabel("-")
        self.lbl_score.setObjectName("metricLabel")

        self.lbl_spread = QLabel("-")
        self.lbl_spread.setObjectName("metricLabel")

        self.lbl_ambig = QLabel("GÜVENLİ")
        self.lbl_ambig.setObjectName("metricLabel")
        self.lbl_ambig.setStyleSheet("color: #a6e3a1;")

        tel_layout.addRow(self._create_title("Adım (Step):"), self.lbl_step)
        tel_layout.addRow(self._create_title("Gerçek Konum (harita X,Y):"), self.lbl_true_pos)
        tel_layout.addRow(self._create_title("Tahmin Konumu (harita X,Y):"), self.lbl_est_pos)
        tel_layout.addRow(self._create_title("Gerçek Yön (Pusula):"), self.lbl_true_heading)
        tel_layout.addRow(self._create_title("Tahmin Yönü:"), self.lbl_est_heading)
        tel_layout.addRow(self._create_title("Sensör MSL (Deniz Seviyesi):"), self.lbl_msl)
        tel_layout.addRow(self._create_title("Lazer AGL (Zeminden Yükseklik):"), self.lbl_agl)
        tel_layout.addRow(self._create_title("Konum Hatası:"), self.lbl_error_pos)
        tel_layout.addRow(self._create_title("Arama Dağılımı:"), self.lbl_spread)
        tel_layout.addRow(self._create_title("Eşleşme Skoru:"), self.lbl_score)
        tel_layout.addRow(self._create_title("Güven Durumu:"), self.lbl_ambig)

        telemetry_group.setLayout(tel_layout)
        sidebar_layout.addWidget(telemetry_group)
        sidebar_layout.addStretch()

        top_layout.addLayout(sidebar_layout, 1)

        # --- Bottom Panel: Console ---
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(160)

        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.log_text)

        self.worker = None
        self.true_path = []
        self.est_path = []
        self._app = QApplication.instance()
        if self._app is not None:
            self._app.installEventFilter(self)

    def _create_title(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("titleLabel")
        return lbl

    @staticmethod
    def _format_extent(width_m: float, height_m: float) -> str:
        if max(width_m, height_m) >= 1000.0:
            return f"{width_m / 1000.0:.2f} × {height_m / 1000.0:.2f} km"
        return f"{width_m:.0f} × {height_m:.0f} m"

    def _display_position(self, state) -> tuple[float, float]:
        return self.map_canvas.to_display_point((state[0], state[1]))

    def _update_scope_status(self, simulation: SimulationEngine) -> None:
        map_width, map_height = simulation.terrain.get_display_extent()
        nav_rows, nav_cols = simulation.terrain.nav_dem.shape
        self.lbl_map_scope.setText(
            "Tam kaynak harita: "
            f"{self._format_extent(map_width, map_height)} — uçuş, sensör ve eşleştirme kapsamı"
        )
        self.lbl_loaded_scope.setText(
            f"Tam harita lokalizasyon DEM'i: {nav_cols} × {nav_rows} px — "
            f"{simulation.terrain.dx:.2f} × {simulation.terrain.dy:.2f} m/px"
        )
        self._update_search_status(simulation.get_localization_status())

    def _update_search_status(self, status: dict) -> None:
        self.map_canvas.update_search_roi(status.get("draw_bounds"))
        if status["mode"] == "global_search":
            if status.get("phase") == "recovery":
                message = "Yeniden yakalama: tam kaynak haritada küresel arama"
            elif status.get("phase") == "tracking":
                message = "Aktif arama: tam kaynak harita (yerel ROI harita boyutunda)"
            else:
                message = "Aktif arama: tam kaynak harita (ilk güvenilir eşleşme bekleniyor)"
            self.lbl_search_scope.setText(message)
            return

        row_start, row_end, col_start, col_end = status["bounds"]
        width_m = (col_end - col_start) * self.map_canvas.nav_dx
        height_m = (row_end - row_start) * self.map_canvas.nav_dy
        prefix = (
            "Yeniden yakalama ROI'si: "
            if status.get("phase") == "recovery"
            else "Aktif yerel eşleştirme ROI'si: "
        )
        self.lbl_search_scope.setText(
            prefix + f"{self._format_extent(width_m, height_m)} "
            f"({col_end - col_start} × {row_end - row_start} px)"
        )

    def _handle_manual_key(self, key: int, auto_repeat: bool = False) -> bool:
        if auto_repeat or self.worker is None or not self.worker.isRunning():
            return False

        distance = self.config.route.manual_step_distance_m
        turn = self.config.route.manual_turn_step_deg
        commands = {
            Qt.Key_W: (0.0, f"[MANUAL] W: {distance:.0f} m ileri"),
            Qt.Key_S: (180.0, f"[MANUAL] S: {distance:.0f} m geri"),
            Qt.Key_A: (-90.0, f"[MANUAL] A: {distance:.0f} m sola"),
            Qt.Key_D: (90.0, f"[MANUAL] D: {distance:.0f} m saga"),
        }
        if key in commands:
            relative_heading, message = commands[key]
            self.worker.post_motion(relative_heading)
            self.log_text.append(f"{message}; komut kuyruğa alındı.")
            return True

        if key == Qt.Key_Q:
            self.worker.turn_vehicle(-turn)
            self.log_text.append(f"[MANUAL] Q: sola donus ({turn:.0f} derece); bekliyor.")
            return True
        if key == Qt.Key_E:
            self.worker.turn_vehicle(turn)
            self.log_text.append(f"[MANUAL] E: saga donus ({turn:.0f} derece); bekliyor.")
            return True
        return False

    def _simulation_config(self) -> LocalizationConfig:
        return apply_realistic_noise_mode(
            self.base_config,
            self.chk_realistic_noise.isChecked(),
        )

    def eventFilter(self, watched, event):
        if event.type() == QEvent.KeyPress and self.isActiveWindow():
            if self._handle_manual_key(event.key(), event.isAutoRepeat()):
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event):
        if not self._handle_manual_key(event.key(), event.isAutoRepeat()):
            super().keyPressEvent(event)

    def start_sim(self):
        if self.worker is not None and self.worker.isRunning():
            return

        self.log_text.append("[SYSTEM] Simülasyon başlatılıyor...")
        self.chk_realistic_noise.setEnabled(False)

        # Reset paths
        self.true_path = []
        self.est_path = []

        # Build one shared simulation/terrain instance for both map rendering
        # and localization instead of loading the DEM twice.
        simulation = SimulationEngine(self._simulation_config(), manual_control=True)
        mode_text = (
            "gerçekçi sensör gürültüsü" if self.chk_realistic_noise.isChecked() else "ideal sensör"
        )
        self.log_text.append(f"[SYSTEM] Sensör modu: {mode_text}.")
        self.map_canvas.plot_terrain(simulation.terrain)
        initial_state = simulation.get_current_state()
        self.true_path = [(initial_state[0], initial_state[1])]
        self._update_scope_status(simulation)
        initial_x, initial_y = self._display_position(initial_state)
        self.lbl_true_pos.setText(f"{initial_x:.0f}, {initial_y:.0f}")
        self.lbl_true_heading.setText(f"{initial_state[2]:.1f}°")
        self.map_canvas.update_trajectory(
            self.true_path,
            self.est_path,
            true_heading_deg=initial_state[2],
        )

        self.worker = SimulationWorker(simulation.config, simulation, manual_control=True)
        self.worker.step_done.connect(self.on_step)
        self.worker.state_changed.connect(self.on_state_changed)
        self.worker.command_rejected.connect(self.on_command_rejected)
        self.worker.finished_sim.connect(self.on_finished)
        self.worker.start()
        self.setFocus()
        self.log_text.append(
            "[SYSTEM] Manuel mod aktif: her hareket komutu sonrası simülasyon bekler."
        )

    def stop_sim(self):
        if self.worker:
            self.worker.stop()
            self.log_text.append("[SYSTEM] Simülasyon iptal edildi.")

    def on_state_changed(self, true_state):
        """Refresh truth telemetry immediately after Q/E without a sample."""
        display_x, display_y = self._display_position(true_state)
        self.lbl_true_pos.setText(f"{display_x:.0f}, {display_y:.0f}")
        self.lbl_true_heading.setText(f"{true_state[2]:.1f} derece")
        if not self.true_path:
            self.true_path.append((true_state[0], true_state[1]))
        self.map_canvas.update_trajectory(
            self.true_path,
            self.est_path,
            true_heading_deg=true_state[2],
        )

    def on_command_rejected(self, message: str) -> None:
        self.log_text.append(f"[SINIR] {message}")

    def closeEvent(self, event):
        if self._app is not None:
            self._app.removeEventFilter(self)
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(1500)
        super().closeEvent(event)

    def on_step(self, step_idx, true_state, est_state, measurement):
        self.lbl_step.setText(str(step_idx))
        display_true_x, display_true_y = self._display_position(true_state)
        self.lbl_true_pos.setText(f"{display_true_x:.0f}, {display_true_y:.0f}")
        self.lbl_true_heading.setText(f"{true_state[2]:.1f}°")

        if measurement:
            self.lbl_msl.setText(f"{measurement.baro_msl_m:.1f} m")
            self.lbl_agl.setText(f"{measurement.laser_agl_m:.1f} m")

        self.true_path.append((true_state[0], true_state[1]))
        localization_status = (
            self.worker.sim.get_localization_status()
            if self.worker is not None and self.worker.sim is not None
            else {"mode": "global_search", "phase": "initial", "draw_bounds": None}
        )

        if est_state is None:
            self.lbl_est_pos.setText("-")
            self.lbl_est_heading.setText("-")
            self.lbl_error_pos.setText("-")
            self.lbl_score.setText("-")
            self.lbl_spread.setText("-")

            if localization_status.get("rejection_reason") == "quality":
                rejected_score = localization_status.get("rejected_score")
                score_text = f"{rejected_score:.2f}" if rejected_score is not None else "-"
                self.lbl_score.setText(score_text)
                self.log_text.append(
                    f"[{step_idx:03d}] Aday eşleşme kalite kapısından reddedildi "
                    f"(skor: {score_text})."
                )
                self.lbl_ambig.setText("KALİTE YETERSİZ")
                self.lbl_ambig.setStyleSheet("color: #f38ba8;")
            elif localization_status.get("phase") == "recovery":
                self.log_text.append(
                    f"[{step_idx:03d}] Eşleşme kayboldu; arama alanı genişletiliyor."
                )
                self.lbl_ambig.setText("YENİDEN ARANIYOR")
                self.lbl_ambig.setStyleSheet("color: #fab387;")
            else:
                self.log_text.append(f"[{step_idx:03d}] Harita eşleştirme verisi bekleniyor...")
                self.lbl_ambig.setText("EŞLEŞME YOK")
                self.lbl_ambig.setStyleSheet("color: #f38ba8;")
        else:
            display_est_x, display_est_y = self._display_position(
                (est_state.estimated_x, est_state.estimated_y)
            )
            self.lbl_est_pos.setText(f"{display_est_x:.0f}, {display_est_y:.0f}")
            self.lbl_est_heading.setText(f"{est_state.estimated_heading_deg:.1f}°")
            self.lbl_spread.setText(f"{est_state.spatial_spread:.1f} m")

            self.est_path.append((est_state.estimated_x, est_state.estimated_y))
            err_x = est_state.estimated_x - true_state[0]
            err_y = est_state.estimated_y - true_state[1]
            err_pos = (err_x**2 + err_y**2) ** 0.5

            self.lbl_error_pos.setText(f"{err_pos:.1f} m")
            quality_score = est_state.quality_score
            if quality_score is None:
                quality_score = est_state.score
            self.lbl_score.setText(f"{quality_score:.2f}")

            if est_state.is_ambiguous:
                self.lbl_ambig.setText("BELİRSİZ (AMBIG)")
                self.lbl_ambig.setStyleSheet("color: #fab387;")
            else:
                self.lbl_ambig.setText("GÜVENLİ (FIX)")
                self.lbl_ambig.setStyleSheet("color: #a6e3a1;")

            self.log_text.append(
                f"[{step_idx:03d}] Fix: ({est_state.estimated_x:.0f}, {est_state.estimated_y:.0f}), Hata: {err_pos:.1f}m, Dağılım: {est_state.spatial_spread:.0f}m"
            )

        self._update_search_status(localization_status)
        self.map_canvas.update_trajectory(
            self.true_path,
            self.est_path,
            true_heading_deg=true_state[2],
        )
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_finished(self):
        self.chk_realistic_noise.setEnabled(True)
        if self.worker is not None and self.worker.stopped_by_user:
            return
        self.log_text.append("[SYSTEM] Simülasyon tamamlandı.")


def run_ui(config=None):
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MissionControlWindow(config)
    window.show()
    sys.exit(app.exec())

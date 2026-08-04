"""PySide6 UI for Terrain Navigation."""

import sys

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
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

from terrain_nav.config import LocalizationConfig
from terrain_nav.rendering import MapCanvas
from terrain_nav.simulation import SimulationEngine

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
    font-size: 24px;
    font-weight: bold;
    color: #f9e2af;
}
QLabel#titleLabel {
    font-size: 14px;
    color: #a6adc8;
}
"""

class SimulationWorker(QThread):
    step_done = Signal(int, object, object, object) # step_idx, true_state, est_state, measurement
    finished_sim = Signal()
    
    def __init__(
        self,
        config: LocalizationConfig,
        simulation: SimulationEngine | None = None,
    ):
        super().__init__()
        self.config = config
        self.sim = simulation
        self._is_running = False
        
    def run(self):
        if self.sim is None:
            self.sim = SimulationEngine(self.config)
        self._is_running = True
        total = self.sim.get_total_steps()
        
        for i in range(total):
            if not self._is_running:
                break
            true_s, est_s, m = self.sim.step()
            self.step_done.emit(i, true_s, est_s, m)
            self.msleep(100) # Sleep to allow UI to update and user to see
            
        self.finished_sim.emit()
        
    def stop(self):
        self._is_running = False
        
    def turn_vehicle(self, angle_deg: float):
        if self.sim:
            self.sim.turn_vehicle(angle_deg)

class MissionControlWindow(QMainWindow):
    def __init__(self, config=None):
        super().__init__()
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
        self.btn_start = QPushButton("Bölge Taramasını Başlat")
        self.btn_start.setObjectName("btnStart")
        self.btn_stop = QPushButton("Acil Durdurma")
        self.btn_stop.setObjectName("btnStop")
        
        self.btn_start.clicked.connect(self.start_sim)
        self.btn_stop.clicked.connect(self.stop_sim)
        
        ctrl_layout.addWidget(self.btn_start)
        ctrl_layout.addWidget(self.btn_stop)
        ctrl_group.setLayout(ctrl_layout)
        sidebar_layout.addWidget(ctrl_group)
        
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
        tel_layout.addRow(self._create_title("Gerçek Konum (X,Y):"), self.lbl_true_pos)
        tel_layout.addRow(self._create_title("Tahmin Konumu (X,Y):"), self.lbl_est_pos)
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
        self.log_text.setMaximumHeight(200)
        
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.log_text)
        
        self.worker = None
        self.config = config or LocalizationConfig()
        self.true_path = []
        self.est_path = []
        
    def _create_title(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("titleLabel")
        return lbl
        
    def keyPressEvent(self, event):
        if self.worker and self.worker.isRunning():
            if event.key() == Qt.Key_A:
                self.worker.turn_vehicle(-15.0)
                self.log_text.append("[MANUAL] İHA Sola Döndü (-15°)")
            elif event.key() == Qt.Key_D:
                self.worker.turn_vehicle(15.0)
                self.log_text.append("[MANUAL] İHA Sağa Döndü (+15°)")
        
    def start_sim(self):
        if self.worker is not None and self.worker.isRunning():
            return
            
        self.log_text.append("[SYSTEM] Simülasyon başlatılıyor...")
        
        # Reset paths
        self.true_path = []
        self.est_path = []
        
        # Build one shared simulation/terrain instance for both map rendering
        # and localization instead of loading the DEM twice.
        simulation = SimulationEngine(self.config)
        self.map_canvas.plot_terrain(simulation.terrain)

        self.worker = SimulationWorker(simulation.config, simulation)
        self.worker.step_done.connect(self.on_step)
        self.worker.finished_sim.connect(self.on_finished)
        self.worker.start()
        
    def stop_sim(self):
        if self.worker:
            self.worker.stop()
            self.log_text.append("[SYSTEM] Simülasyon iptal edildi.")
            
    def on_step(self, step_idx, true_state, est_state, measurement):
        self.lbl_step.setText(str(step_idx))
        self.lbl_true_pos.setText(f"{true_state[0]:.0f}, {true_state[1]:.0f}")
        self.lbl_true_heading.setText(f"{true_state[2]:.1f}°")
        
        if measurement:
            self.lbl_msl.setText(f"{measurement.baro_msl_m:.1f} m")
            self.lbl_agl.setText(f"{measurement.laser_agl_m:.1f} m")
        
        self.true_path.append((true_state[0], true_state[1]))
        
        if est_state is None:
            self.lbl_est_pos.setText("-")
            self.lbl_est_heading.setText("-")
            self.lbl_error_pos.setText("-")
            self.lbl_score.setText("-")
            self.lbl_spread.setText("-")
            
            self.log_text.append(f"[{step_idx:03d}] Harita eşleştirme verisi bekleniyor...")
            self.lbl_ambig.setText("EŞLEŞME YOK")
            self.lbl_ambig.setStyleSheet("color: #f38ba8;")
        else:
            self.lbl_est_pos.setText(f"{est_state.estimated_x:.0f}, {est_state.estimated_y:.0f}")
            self.lbl_est_heading.setText(f"{est_state.estimated_heading_deg:.1f}°")
            self.lbl_spread.setText(f"{est_state.spatial_spread:.1f} m")
            
            self.est_path.append((est_state.estimated_x, est_state.estimated_y))
            err_x = est_state.estimated_x - true_state[0]
            err_y = est_state.estimated_y - true_state[1]
            err_pos = (err_x**2 + err_y**2)**0.5
            
            self.lbl_error_pos.setText(f"{err_pos:.1f} m")
            self.lbl_score.setText(f"{est_state.score:.2f}")
            
            if est_state.is_ambiguous:
                self.lbl_ambig.setText("BELİRSİZ (AMBIG)")
                self.lbl_ambig.setStyleSheet("color: #fab387;")
            else:
                self.lbl_ambig.setText("GÜVENLİ (FIX)")
                self.lbl_ambig.setStyleSheet("color: #a6e3a1;")
                
            self.log_text.append(f"[{step_idx:03d}] Fix: ({est_state.estimated_x:.0f}, {est_state.estimated_y:.0f}), Hata: {err_pos:.1f}m, Dağılım: {est_state.spatial_spread:.0f}m")
            
        self.map_canvas.update_trajectory(self.true_path, self.est_path)
        # Scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
            
    def on_finished(self):
        self.log_text.append("[SYSTEM] Simülasyon tamamlandı.")

def run_ui(config=None):
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLESHEET)
    window = MissionControlWindow(config)
    window.show()
    sys.exit(app.exec())

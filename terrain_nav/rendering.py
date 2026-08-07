"""Rendering utilities for localization results."""

import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter

from terrain_nav.metrics import ProfileComparison
from terrain_nav.terrain import TerrainManager


class ProfileCanvas(FigureCanvas):
    STATUS_LABELS = {
        "profile_incomplete": "profil birikiyor",
        "no_candidates": "aday yok",
        "continuity_rejected": "süreklilik reddi",
        "quality_rejected": "kalite reddi",
        "ambiguous": "belirsiz aday",
        "speed_ambiguous": "hız belirsiz",
        "fix": "en iyi fix",
    }
    STATUS_COLORS = {
        "profile_incomplete": "#a6adc8",
        "no_candidates": "#f38ba8",
        "continuity_rejected": "#fab387",
        "quality_rejected": "#f38ba8",
        "ambiguous": "#fab387",
        "speed_ambiguous": "#fab387",
        "fix": "#a6e3a1",
    }

    def __init__(self, parent=None, width=3.2, height=4, dpi=100):
        plt.style.use("dark_background")
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor("#1e1e2e")
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor("#11111b")
        super().__init__(self.fig)
        self.setParent(parent)

        (self.measured_line,) = self.axes.plot(
            [],
            [],
            color="#00f0ff",
            linewidth=2.2,
            label="Aranan profil",
        )
        (self.matched_line,) = self.axes.plot(
            [],
            [],
            color="#f9e2af",
            linestyle="--",
            linewidth=2.2,
            label="En iyi DEM",
        )
        self.axes.grid(color="#333333", linestyle="--", linewidth=0.5, alpha=0.7)
        self.axes.legend(
            loc="upper right",
            facecolor="#1e1e2e",
            edgecolor="#444444",
            fontsize=8,
        )
        self.axes.tick_params(axis="x", labelrotation=25, labelsize=8)
        self.axes.tick_params(axis="y", labelsize=8)
        self.axes.set_xlabel("Profil mesafesi (m)", color="gray")
        self.axes.set_ylabel("Arazi kotu (m)", color="gray")
        self.clear_profile()

    def clear_profile(self) -> None:
        self.measured_line.set_data([], [])
        self.matched_line.set_data([], [])
        self.axes.set_title("Yükseklik profili bekleniyor", color="#a6adc8", pad=12)
        self.axes.set_xlim(0.0, 1.0)
        self.axes.set_ylim(0.0, 1.0)
        self.fig.tight_layout()
        self.draw()

    def update_profile(self, comparison: ProfileComparison | None) -> None:
        if comparison is None or not comparison.distances_m:
            self.clear_profile()
            return

        x = np.asarray(comparison.distances_m, dtype=np.float64)
        measured = np.asarray(comparison.measured_elevation_m, dtype=np.float64)
        matched = np.asarray(comparison.matched_elevation_m, dtype=np.float64)
        measured_valid = np.isfinite(x) & np.isfinite(measured)
        matched_valid = np.isfinite(x) & np.isfinite(matched)

        self.measured_line.set_data(x[measured_valid], measured[measured_valid])
        self.matched_line.set_data(x[matched_valid], matched[matched_valid])

        visible_values = []
        if np.any(measured_valid):
            visible_values.append(measured[measured_valid])
        if np.any(matched_valid):
            visible_values.append(matched[matched_valid])
        if not visible_values:
            self.clear_profile()
            return

        y_values = np.concatenate(visible_values)
        y_min = float(np.nanmin(y_values))
        y_max = float(np.nanmax(y_values))
        y_pad = max(1.0, (y_max - y_min) * 0.12)
        x_min = float(np.nanmin(x))
        x_max = float(np.nanmax(x))
        x_pad = max(1.0, (x_max - x_min) * 0.04)
        self.axes.set_xlim(x_min - x_pad, x_max + x_pad)
        self.axes.set_ylim(y_min - y_pad, y_max + y_pad)

        status = comparison.status
        status_text = self.STATUS_LABELS.get(status, status)
        details = []
        if comparison.quality_score is not None and np.isfinite(comparison.quality_score):
            details.append(f"hata {comparison.quality_score:.2f} m")
        if (
            comparison.quality_correlation is not None
            and np.isfinite(comparison.quality_correlation)
        ):
            details.append(f"korr {comparison.quality_correlation:.2f}")
        if comparison.estimated_speed_m_s is not None:
            details.append(f"hız {comparison.estimated_speed_m_s:.1f} m/s")
        suffix = f" | {', '.join(details)}" if details else ""
        self.axes.set_title(
            f"Yükseklik profili - {status_text}{suffix}",
            color=self.STATUS_COLORS.get(status, "#cdd6f4"),
            pad=12,
        )
        self.fig.tight_layout()
        self.draw()


class MapCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.style.use("dark_background")
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor("#1e1e2e")
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor("#1e1e2e")

        super().__init__(self.fig)
        self.setParent(parent)

        self.true_line = None
        self.est_line = None
        self.est_point = None
        self.true_heading_arrow = None
        self.search_roi_patch = None
        self.display_offset = (0.0, 0.0)
        self.nav_dx = 1.0
        self.nav_dy = 1.0
        self.heading_arrow_length = 1.0
        self.cbar = None

    def plot_terrain(self, tm: TerrainManager):
        self.axes.clear()
        if self.cbar:
            self.cbar.remove()

        dem = tm.get_display_dem()
        max_x, max_y = tm.get_display_extent()
        left, right, bottom, top = tm.get_display_bounds()
        self.display_offset = tm.get_display_offset()
        self.nav_dx = float(tm.dx)
        self.nav_dy = float(tm.dy)
        self.heading_arrow_length = max(1.0, min(max_x, max_y) * 0.065)

        # Plot dem with nice colormap matching our coordinate system
        # coordinates.py uses y = -row*dy, so row 0 is y=0, row N is y=-max_y
        im = self.axes.imshow(
            dem, cmap="magma", extent=[left, right, bottom, top], origin="upper", alpha=0.85
        )

        self.cbar = self.fig.colorbar(im, ax=self.axes, fraction=0.046, pad=0.04)
        self.cbar.set_label("Elevation (m)", color="white")
        self.cbar.ax.yaxis.set_tick_params(color="white")
        self.cbar.outline.set_edgecolor("none")

        # Neon lines
        (self.true_line,) = self.axes.plot(
            [], [], color="#00f0ff", linestyle="-", linewidth=2.5, label="Gerçek rota"
        )
        (self.est_line,) = self.axes.plot(
            [], [], color="#ff00ff", linestyle="--", linewidth=2.5, label="Tahmin rotası"
        )

        (self.est_point,) = self.axes.plot(
            [], [], marker="X", color="#ff00ff", markersize=8, markeredgecolor="white"
        )
        self.true_heading_arrow = FancyArrowPatch(
            (0.0, 0.0),
            (0.0, 0.0),
            arrowstyle="-|>",
            mutation_scale=24,
            linewidth=3.0,
            edgecolor="white",
            facecolor="#00f0ff",
            zorder=8,
            label="İHA ve gerçek yön",
        )
        self.axes.add_patch(self.true_heading_arrow)

        self.search_roi_patch = Rectangle(
            (0.0, 0.0),
            0.0,
            0.0,
            fill=False,
            edgecolor="#fab387",
            linewidth=2.2,
            linestyle="-",
            zorder=6,
            label="Aktif eşleştirme ROI'si",
            visible=False,
        )
        self.axes.add_patch(self.search_roi_patch)

        # Clean axes
        self.axes.spines["top"].set_visible(False)
        self.axes.spines["right"].set_visible(False)
        self.axes.spines["bottom"].set_color("#555555")
        self.axes.spines["left"].set_color("#555555")

        self.axes.grid(color="#333333", linestyle="--", linewidth=0.5, alpha=0.7)
        self.axes.legend(
            loc="upper right",
            facecolor="#1e1e2e",
            edgecolor="#444444",
            fontsize=8,
        )
        self.axes.set_title("GNSS Olmadan Seyrüsefer — Tam Harita", color="white", pad=15)
        coordinate_prefix = "UTM " if tm.source_bounds is not None else "Yerel "
        self.axes.set_xlabel(f"{coordinate_prefix}Doğu (m)", color="gray")
        self.axes.set_ylabel(f"{coordinate_prefix}Kuzey (m)", color="gray")
        meter_formatter = FuncFormatter(lambda value, _position: f"{value:,.0f}".replace(",", " "))
        self.axes.xaxis.set_major_formatter(meter_formatter)
        self.axes.yaxis.set_major_formatter(meter_formatter)
        self.axes.tick_params(axis="x", labelrotation=30, labelsize=8)
        self.axes.tick_params(axis="y", labelsize=8)
        # Dynamic artists must never autoscale the view down to the short
        # trajectory; keep the complete source map locked on screen.
        self.axes.set_xlim(left, right)
        self.axes.set_ylim(bottom, top)

        self.fig.tight_layout()
        self.draw()

    def to_display_point(self, point: Tuple[float, float]) -> Tuple[float, float]:
        """Convert navigation-window coordinates to coordinates shown on the map."""
        return (
            point[0] + self.display_offset[0],
            point[1] + self.display_offset[1],
        )

    def update_search_roi(
        self,
        bounds: Tuple[int, int, int, int] | None,
    ) -> None:
        """Show the algorithm's current local search ROI in overview coordinates."""
        if self.search_roi_patch is None:
            return
        if bounds is None:
            self.search_roi_patch.set_visible(False)
            return

        row_start, row_end, col_start, col_end = bounds
        x = self.display_offset[0] + col_start * self.nav_dx
        y_bottom = self.display_offset[1] - row_end * self.nav_dy
        self.search_roi_patch.set_xy((x, y_bottom))
        self.search_roi_patch.set_width((col_end - col_start) * self.nav_dx)
        self.search_roi_patch.set_height((row_end - row_start) * self.nav_dy)
        self.search_roi_patch.set_visible(True)

    def update_trajectory(
        self,
        true_path: List[Tuple[float, float]],
        est_path: List[Tuple[float, float]],
        true_heading_deg: float | None = None,
    ):
        if self.true_line is None:
            return

        if true_path:
            display_true = [self.to_display_point(point) for point in true_path]
            tx = [point[0] for point in display_true]
            ty = [point[1] for point in display_true]
            self.true_line.set_data(tx, ty)
            if self.true_heading_arrow is not None and true_heading_deg is not None:
                angle = math.radians(float(true_heading_deg) % 360.0)
                tip = (
                    tx[-1] + self.heading_arrow_length * math.sin(angle),
                    ty[-1] + self.heading_arrow_length * math.cos(angle),
                )
                self.true_heading_arrow.set_positions((tx[-1], ty[-1]), tip)

        if est_path:
            display_est = [self.to_display_point(point) for point in est_path]
            ex = [point[0] for point in display_est]
            ey = [point[1] for point in display_est]
            self.est_line.set_data(ex, ey)
            self.est_point.set_data([ex[-1]], [ey[-1]])

        self.draw()

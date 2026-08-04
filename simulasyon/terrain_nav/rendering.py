"""Rendering utilities for localization results."""

from typing import List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from terrain_nav.terrain import TerrainManager


class MapCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        plt.style.use('dark_background')
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.fig.patch.set_facecolor('#1e1e2e')
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#1e1e2e')
        
        super().__init__(self.fig)
        self.setParent(parent)
        
        self.true_line = None
        self.est_line = None
        self.true_point = None
        self.est_point = None
        self.cbar = None
        
    def plot_terrain(self, tm: TerrainManager):
        self.axes.clear()
        if self.cbar:
            self.cbar.remove()
            
        dem = tm.get_navigation_dem(copy=False)
        max_x, max_y = tm.get_extent()
        
        # Plot dem with nice colormap matching our coordinate system
        # coordinates.py uses y = -row*dy, so row 0 is y=0, row N is y=-max_y
        im = self.axes.imshow(
            dem, cmap='magma', 
            extent=[0, max_x, -max_y, 0],
            origin='upper',
            alpha=0.85
        )
        
        self.cbar = self.fig.colorbar(im, ax=self.axes, fraction=0.046, pad=0.04)
        self.cbar.set_label('Elevation (m)', color='white')
        self.cbar.ax.yaxis.set_tick_params(color='white')
        self.cbar.outline.set_edgecolor('none')
        
        # Neon lines
        self.true_line, = self.axes.plot([], [], color='#00f0ff', linestyle='-', linewidth=2.5, label='True Path')
        self.est_line, = self.axes.plot([], [], color='#ff00ff', linestyle='--', linewidth=2.5, label='Est Path')
        
        self.true_point, = self.axes.plot([], [], marker='o', color='#00f0ff', markersize=8, markeredgecolor='white')
        self.est_point, = self.axes.plot([], [], marker='X', color='#ff00ff', markersize=8, markeredgecolor='white')
        
        # Clean axes
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_color('#555555')
        self.axes.spines['left'].set_color('#555555')
        
        self.axes.grid(color='#333333', linestyle='--', linewidth=0.5, alpha=0.7)
        self.axes.legend(loc='upper right', facecolor='#1e1e2e', edgecolor='#444444')
        self.axes.set_title("GNSS-Denied Navigation Map", color='white', pad=15)
        self.axes.set_xlabel("East (m)", color='gray')
        self.axes.set_ylabel("North (m)", color='gray')
        
        self.fig.tight_layout()
        self.draw()

    def update_trajectory(self, true_path: List[Tuple[float, float]], est_path: List[Tuple[float, float]]):
        if self.true_line is None:
            return
        
        if true_path:
            tx = [p[0] for p in true_path]
            ty = [p[1] for p in true_path]
            self.true_line.set_data(tx, ty)
            self.true_point.set_data([tx[-1]], [ty[-1]])
            
        if est_path:
            ex = [p[0] for p in est_path]
            ey = [p[1] for p in est_path]
            self.est_line.set_data(ex, ey)
            self.est_point.set_data([ex[-1]], [ey[-1]])
            
        self.draw()

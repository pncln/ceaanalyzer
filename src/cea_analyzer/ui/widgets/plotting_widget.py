"""
Plotting Widget
-------------

Widget for visualizing CEA data with interactive plots.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
    QComboBox, QLabel, QPushButton, QGroupBox, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from ...utils.plots import create_graphs, create_optimization_plot


class PlottingWidget(QWidget):
    """
    Widget for visualizing CEA analysis data with interactive plots.
    """
    
    def __init__(self, parent=None):
        """Initialize the plotting widget."""
        super().__init__(parent)
        
        # Data
        self.df = None
        
        # Figures and canvases
        self.figures = {}
        self.canvases = {}
        self.toolbars = {}
        
        # Setup UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Plot controls
        control_group = QGroupBox("Plot Controls")
        control_layout = QHBoxLayout()
        
        # Plot type selection
        self.plot_type_label = QLabel("Plot Type:")
        self.plot_type_combo = QComboBox()
        plot_types = ["Isp vs O/F", "Temperature vs O/F", "Pressure Ratio vs O/F", 
                     "Enthalpy vs O/F", "Area Ratio vs Isp", "3D Optimization"]
        self.plot_type_combo.addItems(plot_types)
        self.plot_type_combo.currentIndexChanged.connect(self._update_plot)
        
        # Other controls
        self.log_scale_check = QCheckBox("Logarithmic Scale")
        self.log_scale_check.setChecked(False)
        self.log_scale_check.stateChanged.connect(self._update_plot)
        
        self.grid_check = QCheckBox("Show Grid")
        self.grid_check.setChecked(True)
        self.grid_check.stateChanged.connect(self._update_plot)
        
        self.export_button = QPushButton("Export Plot")
        self.export_button.clicked.connect(self._export_plot)
        
        # Add controls to layout
        control_layout.addWidget(self.plot_type_label)
        control_layout.addWidget(self.plot_type_combo)
        control_layout.addWidget(self.log_scale_check)
        control_layout.addWidget(self.grid_check)
        control_layout.addStretch()
        control_layout.addWidget(self.export_button)
        
        control_group.setLayout(control_layout)
        
        # Tab widget for different plots
        self.plot_tabs = QTabWidget()
        self.plot_tabs.setTabPosition(QTabWidget.South)
        
        # Create standard plot tabs
        standard_plot_names = ["Isp", "Temperature", "Pressure", "Enthalpy", "Area Ratio", "Optimization"]
        
        for name in standard_plot_names:
            # Create figure and canvas
            fig = Figure(figsize=(8, 6), dpi=100)
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, self)
            
            # Store references
            self.figures[name] = fig
            self.canvases[name] = canvas
            self.toolbars[name] = toolbar
            
            # Create tab widget
            tab_widget = QWidget()
            tab_layout = QVBoxLayout(tab_widget)
            tab_layout.addWidget(toolbar)
            tab_layout.addWidget(canvas)
            
            # Add to tab widget
            self.plot_tabs.addTab(tab_widget, name)
        
        # Add to main layout
        main_layout.addWidget(control_group)
        main_layout.addWidget(self.plot_tabs)
        
        # Initialize with empty plots
        self._create_empty_plots()
        
        # Disable controls until data is loaded
        self._set_controls_enabled(False)
        
    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable controls."""
        self.plot_type_combo.setEnabled(enabled)
        self.log_scale_check.setEnabled(enabled)
        self.grid_check.setEnabled(enabled)
        self.export_button.setEnabled(enabled)
        
    def _create_empty_plots(self):
        """Create empty plots with placeholder text."""
        for name, fig in self.figures.items():
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_title(f"{name} Plot")
            ax.text(0.5, 0.5, "No data available", 
                   ha='center', va='center', fontsize=14, 
                   transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            self.canvases[name].draw()
            
    def set_data(self, df: pd.DataFrame):
        """Set the data for plotting."""
        if df is None or df.empty:
            self._set_controls_enabled(False)
            self._create_empty_plots()
            return
            
        # Store data
        self.df = df
        
        # Enable controls
        self._set_controls_enabled(True)
        
        # Create plots
        self._update_all_plots()
        
    def _update_all_plots(self):
        """Update all plots with current data."""
        if self.df is None:
            return
            
        # Create standard graphs
        plot_dict = create_graphs(self.df)
        
        # Update each plot
        for name, fig in plot_dict.items():
            if name in self.figures:
                # Clear existing figure
                self.figures[name].clear()
                
                # Copy axes from new figure to existing one
                new_axes = fig.get_axes()
                for i, ax in enumerate(new_axes):
                    new_ax = self.figures[name].add_subplot(1, 1, i+1)
                    new_ax.set_title(ax.get_title())
                    new_ax.set_xlabel(ax.get_xlabel())
                    new_ax.set_ylabel(ax.get_ylabel())
                    
                    # Copy lines and collections
                    for line in ax.get_lines():
                        x_data = line.get_xdata()
                        y_data = line.get_ydata()
                        new_ax.plot(x_data, y_data, 
                                   color=line.get_color(),
                                   linestyle=line.get_linestyle(),
                                   marker=line.get_marker(),
                                   label=line.get_label())
                    
                    # Apply grid and legend
                    new_ax.grid(self.grid_check.isChecked())
                    if ax.get_legend():
                        new_ax.legend()
                    
                    # Apply log scale if checked
                    if self.log_scale_check.isChecked():
                        new_ax.set_xscale('log')
                
                # Redraw canvas
                self.figures[name].tight_layout()
                self.canvases[name].draw()
        
        # Create optimization plot
        opt_fig = create_optimization_plot(self.df)
        
        # Update optimization plot
        self.figures["Optimization"].clear()
        
        # Copy 3D axes
        new_axes = opt_fig.get_axes()
        if new_axes:
            new_ax = self.figures["Optimization"].add_subplot(111, projection='3d')
            
            # Copy surface plot if it exists
            for collection in new_axes[0].collections:
                # This is a simplified approach - 3D plots are complex
                # and may require more detailed copying logic
                new_ax.add_collection3d(collection)
            
            # Set axis labels and title
            new_ax.set_title(new_axes[0].get_title())
            new_ax.set_xlabel(new_axes[0].get_xlabel())
            new_ax.set_ylabel(new_axes[0].get_ylabel())
            new_ax.set_zlabel(new_axes[0].get_zlabel())
            
            # Set axis limits
            new_ax.set_xlim(new_axes[0].get_xlim())
            new_ax.set_ylim(new_axes[0].get_ylim())
            new_ax.set_zlim(new_axes[0].get_zlim())
        
        # Redraw canvas
        self.figures["Optimization"].tight_layout()
        self.canvases["Optimization"].draw()
        
    def _update_plot(self):
        """Update the current plot based on selected options."""
        plot_type = self.plot_type_combo.currentText()
        
        # Map plot type to tab index
        plot_map = {
            "Isp vs O/F": 0,
            "Temperature vs O/F": 1,
            "Pressure Ratio vs O/F": 2,
            "Enthalpy vs O/F": 3,
            "Area Ratio vs Isp": 4,
            "3D Optimization": 5
        }
        
        # Set the current tab
        if plot_type in plot_map:
            self.plot_tabs.setCurrentIndex(plot_map[plot_type])
        
        # Update all plots with current settings
        self._update_all_plots()
        
    def _export_plot(self):
        """Export the current plot to a file."""
        from PyQt5.QtWidgets import QFileDialog
        
        # Get current figure
        current_tab = self.plot_tabs.currentIndex()
        tab_name = self.plot_tabs.tabText(current_tab)
        fig = self.figures.get(tab_name)
        
        if fig is None:
            return
            
        # Ask for file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Plot", "", "PNG Image (*.png);;PDF File (*.pdf);;SVG Image (*.svg)"
        )
        
        if file_path:
            # Save figure
            fig.savefig(file_path, bbox_inches='tight', dpi=300)

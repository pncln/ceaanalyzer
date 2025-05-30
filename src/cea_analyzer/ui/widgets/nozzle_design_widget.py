"""
Nozzle Design Widget
-----------------

Widget for designing and visualizing rocket nozzle contours.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QComboBox, QLineEdit, QDoubleSpinBox, QCheckBox, QPushButton,
    QFileDialog, QTabWidget, QSplitter, QMessageBox, QRadioButton,
    QButtonGroup, QGridLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from ...propulsion.nozzle import (
    conical_nozzle, bell_nozzle, rao_optimum_nozzle, 
    moc_nozzle, truncated_ideal_contour, 
    add_inlet_section, calculate_performance
)


class NozzleDesignWidget(QWidget):
    """
    Widget for designing and visualizing rocket nozzle contours using various methods.
    """
    
    # Signal emitted when a nozzle is designed
    nozzle_designed = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """Initialize the nozzle design widget."""
        super().__init__(parent)
        
        # Data
        self.cea_data = None
        self.current_nozzle_coords = None
        self.current_performance = None
        
        # Setup UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout as a splitter for resizable sections
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Create a splitter for the main panels
        self.splitter = QSplitter(Qt.Vertical)
        
        # Top panel: Design controls and options
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget for different design approaches
        self.design_tabs = QTabWidget()
        
        # 1. Basic Design Tab
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        
        # Nozzle type selection
        type_group = QGroupBox("Nozzle Type")
        type_layout = QGridLayout()
        
        self.nozzle_type_label = QLabel("Select Nozzle Type:")
        self.nozzle_type_combo = QComboBox()
        self.nozzle_type_combo.addItems([
            "Conical", 
            "Bell (80%)", 
            "Rao Optimum", 
            "Method of Characteristics (MOC)",
            "Truncated Ideal Contour (TIC)"
        ])
        self.nozzle_type_combo.currentIndexChanged.connect(self._update_parameter_visibility)
        
        type_layout.addWidget(self.nozzle_type_label, 0, 0)
        type_layout.addWidget(self.nozzle_type_combo, 0, 1)
        type_group.setLayout(type_layout)
        
        # Basic parameters
        params_group = QGroupBox("Basic Parameters")
        params_layout = QGridLayout()
        
        # Throat radius
        self.throat_radius_label = QLabel("Throat Radius (m):")
        self.throat_radius_spin = QDoubleSpinBox()
        self.throat_radius_spin.setDecimals(4)
        self.throat_radius_spin.setRange(0.001, 1.0)
        self.throat_radius_spin.setValue(0.05)
        self.throat_radius_spin.setSingleStep(0.001)
        
        # Expansion ratio
        self.expansion_ratio_label = QLabel("Expansion Ratio:")
        self.expansion_ratio_spin = QDoubleSpinBox()
        self.expansion_ratio_spin.setDecimals(1)
        self.expansion_ratio_spin.setRange(1.1, 100.0)
        self.expansion_ratio_spin.setValue(8.0)
        self.expansion_ratio_spin.setSingleStep(0.5)
        
        # Half angle (for conical)
        self.half_angle_label = QLabel("Half Angle (°):")
        self.half_angle_spin = QDoubleSpinBox()
        self.half_angle_spin.setDecimals(1)
        self.half_angle_spin.setRange(5.0, 30.0)
        self.half_angle_spin.setValue(15.0)
        self.half_angle_spin.setSingleStep(0.5)
        
        # Bell percentage (for bell)
        self.bell_percent_label = QLabel("Bell Percentage (%):")
        self.bell_percent_spin = QDoubleSpinBox()
        self.bell_percent_spin.setDecimals(1)
        self.bell_percent_spin.setRange(60.0, 100.0)
        self.bell_percent_spin.setValue(80.0)
        self.bell_percent_spin.setSingleStep(5.0)
        
        # Add parameters to grid
        params_layout.addWidget(self.throat_radius_label, 0, 0)
        params_layout.addWidget(self.throat_radius_spin, 0, 1)
        params_layout.addWidget(self.expansion_ratio_label, 1, 0)
        params_layout.addWidget(self.expansion_ratio_spin, 1, 1)
        params_layout.addWidget(self.half_angle_label, 2, 0)
        params_layout.addWidget(self.half_angle_spin, 2, 1)
        params_layout.addWidget(self.bell_percent_label, 3, 0)
        params_layout.addWidget(self.bell_percent_spin, 3, 1)
        
        params_group.setLayout(params_layout)
        
        # Options
        options_group = QGroupBox("Design Options")
        options_layout = QVBoxLayout()
        
        self.include_inlet_check = QCheckBox("Include Inlet Section")
        self.include_inlet_check.setChecked(True)
        
        options_layout.addWidget(self.include_inlet_check)
        options_group.setLayout(options_layout)
        
        # Design button
        self.design_button = QPushButton("Generate Nozzle Design")
        self.design_button.clicked.connect(self._generate_nozzle)
        
        # Export button
        self.export_button = QPushButton("Export Nozzle Coordinates")
        self.export_button.clicked.connect(self._export_coordinates)
        self.export_button.setEnabled(False)
        
        # Add to basic layout
        basic_layout.addWidget(type_group)
        basic_layout.addWidget(params_group)
        basic_layout.addWidget(options_group)
        basic_layout.addWidget(self.design_button)
        basic_layout.addWidget(self.export_button)
        basic_layout.addStretch()
        
        # 2. Advanced Design Tab (MOC Parameters)
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)
        
        moc_group = QGroupBox("Method of Characteristics Parameters")
        moc_layout = QGridLayout()
        
        # Number of characteristic lines
        self.num_char_label = QLabel("Number of Characteristic Lines:")
        self.num_char_spin = QDoubleSpinBox()
        self.num_char_spin.setDecimals(0)
        self.num_char_spin.setRange(10.0, 100.0)
        self.num_char_spin.setValue(30.0)
        self.num_char_spin.setSingleStep(5.0)
        
        # Maximum Prandtl-Meyer angle
        self.nu_max_label = QLabel("Maximum Prandtl-Meyer Angle (°):")
        self.nu_max_spin = QDoubleSpinBox()
        self.nu_max_spin.setDecimals(1)
        self.nu_max_spin.setRange(10.0, 90.0)
        self.nu_max_spin.setValue(30.0)
        self.nu_max_spin.setSingleStep(1.0)
        
        # Add to layout
        moc_layout.addWidget(self.num_char_label, 0, 0)
        moc_layout.addWidget(self.num_char_spin, 0, 1)
        moc_layout.addWidget(self.nu_max_label, 1, 0)
        moc_layout.addWidget(self.nu_max_spin, 1, 1)
        
        moc_group.setLayout(moc_layout)
        advanced_layout.addWidget(moc_group)
        advanced_layout.addStretch()
        
        # Add tabs to tab widget
        self.design_tabs.addTab(basic_tab, "Basic Design")
        self.design_tabs.addTab(advanced_tab, "Advanced Parameters")
        
        # Add tab widget to top layout
        top_layout.addWidget(self.design_tabs)
        
        # Bottom panel: Visualization and Results
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        # Visualization tabs
        viz_tabs = QTabWidget()
        
        # 1. Nozzle Contour Tab
        contour_tab = QWidget()
        contour_layout = QVBoxLayout(contour_tab)
        
        # Figure for nozzle contour
        self.contour_fig = Figure(figsize=(10, 6), dpi=100)
        self.contour_canvas = FigureCanvas(self.contour_fig)
        self.contour_toolbar = NavigationToolbar(self.contour_canvas, self)
        
        contour_layout.addWidget(self.contour_toolbar)
        contour_layout.addWidget(self.contour_canvas)
        
        # 2. Performance Tab
        performance_tab = QWidget()
        performance_layout = QVBoxLayout(performance_tab)
        
        # Text display for performance
        self.performance_text = QLabel("No nozzle design available.")
        self.performance_text.setAlignment(Qt.AlignCenter)
        self.performance_text.setWordWrap(True)
        self.performance_text.setStyleSheet("font-family: monospace;")
        
        performance_layout.addWidget(self.performance_text)
        
        # Add tabs to viz_tabs
        viz_tabs.addTab(contour_tab, "Nozzle Contour")
        viz_tabs.addTab(performance_tab, "Performance")
        
        # Add viz_tabs to bottom layout
        bottom_layout.addWidget(viz_tabs)
        
        # Add panels to splitter
        self.splitter.addWidget(top_panel)
        self.splitter.addWidget(bottom_panel)
        self.splitter.setSizes([200, 400])  # Initial sizes
        
        # Add splitter to main layout
        main_layout.addWidget(self.splitter)
        
        # Initialize with empty contour
        self._create_empty_contour()
        
        # Update parameter visibility based on nozzle type
        self._update_parameter_visibility()
        
        # Disable controls until CEA data is loaded
        self._set_controls_enabled(False)
        
    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable controls."""
        self.design_tabs.setEnabled(enabled)
        self.design_button.setEnabled(enabled)
        
    def _create_empty_contour(self):
        """Create an empty contour plot with placeholder text."""
        self.contour_fig.clear()
        ax = self.contour_fig.add_subplot(111)
        ax.set_title("Nozzle Contour")
        ax.text(0.5, 0.5, "No nozzle design available", 
               ha='center', va='center', fontsize=14, 
               transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        self.contour_canvas.draw()
        
    def _update_parameter_visibility(self):
        """Update visibility of parameters based on selected nozzle type."""
        nozzle_type = self.nozzle_type_combo.currentText()
        
        # Hide all specific parameters first
        self.half_angle_label.setVisible(False)
        self.half_angle_spin.setVisible(False)
        self.bell_percent_label.setVisible(False)
        self.bell_percent_spin.setVisible(False)
        
        # Show relevant parameters based on nozzle type
        if "Conical" in nozzle_type:
            self.half_angle_label.setVisible(True)
            self.half_angle_spin.setVisible(True)
        elif "Bell" in nozzle_type:
            self.bell_percent_label.setVisible(True)
            self.bell_percent_spin.setVisible(True)
            
    def set_cea_data(self, cea_data: Dict[str, Any]):
        """Set the CEA data for nozzle design."""
        if cea_data is None:
            self._set_controls_enabled(False)
            return
            
        # Store CEA data
        self.cea_data = cea_data
        
        # Set expansion ratio from CEA data if available
        if 'area_ratio' in cea_data:
            self.expansion_ratio_spin.setValue(cea_data['area_ratio'])
            
        # Enable controls
        self._set_controls_enabled(True)
        
    def _generate_nozzle(self):
        """Generate nozzle contour based on current parameters."""
        if self.cea_data is None:
            QMessageBox.warning(self, "Warning", "No CEA data available. Load data first.")
            return
            
        # Get parameters
        nozzle_type = self.nozzle_type_combo.currentText()
        throat_radius = self.throat_radius_spin.value()
        expansion_ratio = self.expansion_ratio_spin.value()
        include_inlet = self.include_inlet_check.isChecked()
        
        # Update CEA data with current expansion ratio
        cea_data = self.cea_data.copy()
        cea_data['area_ratio'] = expansion_ratio
        
        # Generate nozzle contour based on type
        try:
            if "Conical" in nozzle_type:
                half_angle = self.half_angle_spin.value()
                x, r = conical_nozzle(cea_data, half_angle=half_angle, R_throat=throat_radius)
                cea_data['nozzle_type'] = "conical"
                
            elif "Bell" in nozzle_type:
                percent_bell = self.bell_percent_spin.value()
                x, r = bell_nozzle(cea_data, R_throat=throat_radius, percent_bell=percent_bell)
                cea_data['nozzle_type'] = "bell"
                
            elif "Rao Optimum" in nozzle_type:
                x, r = rao_optimum_nozzle(cea_data, R_throat=throat_radius)
                cea_data['nozzle_type'] = "rao"
                
            elif "Method of Characteristics" in nozzle_type:
                num_char = int(self.num_char_spin.value())
                nu_max = self.nu_max_spin.value() if self.nu_max_spin.value() > 0 else None
                x, r = moc_nozzle(cea_data, R_throat=throat_radius, N=num_char, nu_max=nu_max)
                cea_data['nozzle_type'] = "moc"
                
            elif "Truncated Ideal Contour" in nozzle_type:
                x, r = truncated_ideal_contour(cea_data, R_throat=throat_radius)
                cea_data['nozzle_type'] = "tic"
                
            else:
                # Default to conical
                x, r = conical_nozzle(cea_data, R_throat=throat_radius)
                cea_data['nozzle_type'] = "conical"
                
            # Add inlet section if requested
            if include_inlet:
                x, r = add_inlet_section(x, r, throat_radius)
                
            # Store current coordinates
            self.current_nozzle_coords = (x, r)
            
            # Calculate performance
            self.current_performance = calculate_performance(cea_data, (x, r))
            
            # Plot nozzle contour
            self._plot_nozzle_contour(x, r)
            
            # Update performance display
            self._update_performance_display()
            
            # Enable export button
            self.export_button.setEnabled(True)
            
            # Emit signal
            self.nozzle_designed.emit(self.current_performance)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating nozzle contour: {str(e)}")
            
    def _plot_nozzle_contour(self, x: np.ndarray, r: np.ndarray):
        """Plot the nozzle contour."""
        # Clear figure
        self.contour_fig.clear()
        ax = self.contour_fig.add_subplot(111)
        
        # Find throat position
        throat_idx = np.argmin(r)
        throat_x = x[throat_idx]
        throat_r = r[throat_idx]
        
        # Plot with engineering-standard styling
        # Outer contour (thick blue line)
        ax.plot(x, r, 'b-', lw=2.5)
        ax.plot(x, -r, 'b-', lw=2.5)
        
        # Fill the nozzle with a subtle gradient
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list('nozzle_gradient', ['#d0d0d0', '#f8f8f8'])
        for i in range(len(x)-1):
            ax.fill_between(x[i:i+2], r[i:i+2], -r[i:i+2], 
                           color=cmap(i/len(x)), alpha=0.7, linewidth=0)
            
        # Add centerline
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.5, lw=0.5)
        
        # Add throat marker
        ax.plot([throat_x], [0], 'ro', markersize=4)
        
        # Add title and labels
        nozzle_type = self.nozzle_type_combo.currentText()
        ax.set_title(f"{nozzle_type} Nozzle Design")
        ax.set_xlabel("Axial Distance (m)")
        ax.set_ylabel("Radial Distance (m)")
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Equal aspect ratio
        ax.set_aspect('equal')
        
        # Add annotations
        area_ratio = self.expansion_ratio_spin.value()
        ax.text(0.98, 0.02, f"Area Ratio: {area_ratio:.2f}", 
               transform=ax.transAxes, ha='right', va='bottom',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Redraw canvas
        self.contour_fig.tight_layout()
        self.contour_canvas.draw()
        
    def _update_performance_display(self):
        """Update the performance display with current performance data."""
        if self.current_performance is None:
            self.performance_text.setText("No performance data available.")
            return
            
        # Format performance text
        perf = self.current_performance
        text = f"""<html>
        <style>
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
        </style>
        <h2>Nozzle Performance Analysis</h2>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Area Ratio (Ae/At)</td><td>{perf['area_ratio']:.2f}</td></tr>
            <tr><td>Thrust Coefficient (Cf)</td><td>{perf['thrust_coefficient']:.3f}</td></tr>
            <tr><td>Ideal Thrust Coefficient</td><td>{perf['ideal_thrust_coefficient']:.3f}</td></tr>
            <tr><td>Divergence Loss Factor</td><td>{perf['divergence_loss_factor']:.3f}</td></tr>
            <tr><td>Divergence Angle</td><td>{perf['divergence_angle_deg']:.2f}°</td></tr>
            <tr><td>Nozzle Efficiency</td><td>{perf['nozzle_efficiency']:.2%}</td></tr>
            <tr><td>Length to Throat Ratio</td><td>{perf['length_to_throat_ratio']:.2f}</td></tr>
            <tr><td>Surface Area</td><td>{perf['surface_area']:.4f} m²</td></tr>
            <tr><td>Exit Mach Number</td><td>{perf['exit_mach_number']:.2f}</td></tr>
        </table>
        </html>"""
        
        self.performance_text.setText(text)
        
    def _export_coordinates(self):
        """Export nozzle coordinates to a file."""
        if self.current_nozzle_coords is None:
            return
            
        # Get file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Nozzle Coordinates", "", 
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        
        if not file_path:
            return
            
        # Get coordinates
        x, r = self.current_nozzle_coords
        
        try:
            # Write to file
            with open(file_path, 'w') as f:
                f.write("x,r\n")  # Header
                for i in range(len(x)):
                    f.write(f"{x[i]:.6f},{r[i]:.6f}\n")
                    
            QMessageBox.information(self, "Success", f"Nozzle coordinates exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting coordinates: {str(e)}")

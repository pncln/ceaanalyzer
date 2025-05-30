"""
Motor Design Widget
----------------

Widget for designing and analyzing rocket motors with various grain configurations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QComboBox, QLineEdit, QDoubleSpinBox, QCheckBox, QPushButton,
    QFileDialog, QTabWidget, QSplitter, QMessageBox, QRadioButton,
    QButtonGroup, QGridLayout, QSpacerItem, QSizePolicy, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from ...propulsion.motor.design import MotorDesign
from ...propulsion.motor.types import MotorType
from ...propulsion.motor.components import MotorCase, Nozzle
from ...propulsion.grain import MotorGrain, GrainType, PropellantProperties
from ...propulsion.grain import BatesGrain, StarGrain, EndBurnerGrain


class MotorDesignWidget(QWidget):
    """
    Widget for designing and analyzing rocket motors with various grain configurations.
    """
    
    # Signal emitted when a motor is designed
    motor_designed = pyqtSignal(object)
    
    def __init__(self, parent=None):
        """Initialize the motor design widget."""
        super().__init__(parent)
        
        # Data
        self.cea_data = None
        self.motor_design = None
        
        # Setup UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout with a splitter for resizable sections
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Create a splitter for the main panels
        self.splitter = QSplitter(Qt.Vertical)
        
        # Top panel: Design controls
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget for different design components
        self.design_tabs = QTabWidget()
        
        # 1. Basic Motor Tab
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)
        
        # Motor type selection
        type_group = QGroupBox("Motor Type")
        type_layout = QHBoxLayout()
        
        self.motor_type_combo = QComboBox()
        self.motor_type_combo.addItems(["Solid Rocket Motor", "Hybrid Rocket Motor", "Liquid Rocket Engine"])
        self.motor_type_combo.currentIndexChanged.connect(self._update_controls_visibility)
        
        type_layout.addWidget(QLabel("Select Motor Type:"))
        type_layout.addWidget(self.motor_type_combo)
        type_group.setLayout(type_layout)
        
        # Motor name
        name_group = QGroupBox("Motor Designation")
        name_layout = QHBoxLayout()
        
        self.motor_name_edit = QLineEdit()
        self.motor_name_edit.setPlaceholderText("Enter motor designation (e.g., J450)")
        
        name_layout.addWidget(QLabel("Motor Name:"))
        name_layout.addWidget(self.motor_name_edit)
        name_group.setLayout(name_layout)
        
        # Design button
        self.design_button = QPushButton("Generate Motor Design")
        self.design_button.clicked.connect(self._generate_motor)
        
        # Save/Load buttons
        buttons_layout = QHBoxLayout()
        
        self.save_button = QPushButton("Save Motor Design")
        self.save_button.clicked.connect(self._save_motor)
        self.save_button.setEnabled(False)
        
        self.load_button = QPushButton("Load Motor Design")
        self.load_button.clicked.connect(self._load_motor)
        
        buttons_layout.addWidget(self.save_button)
        buttons_layout.addWidget(self.load_button)
        
        # Add to basic layout
        basic_layout.addWidget(type_group)
        basic_layout.addWidget(name_group)
        basic_layout.addWidget(self.design_button)
        basic_layout.addLayout(buttons_layout)
        basic_layout.addStretch()
        
        # 2. Grain Tab
        grain_tab = QWidget()
        grain_layout = QVBoxLayout(grain_tab)
        
        # Grain type
        grain_type_group = QGroupBox("Grain Configuration")
        grain_type_layout = QFormLayout()
        
        self.grain_type_combo = QComboBox()
        self.grain_type_combo.addItems(["BATES", "Star", "End Burner"])
        self.grain_type_combo.currentIndexChanged.connect(self._update_grain_controls)
        
        grain_type_layout.addRow("Grain Type:", self.grain_type_combo)
        
        # Basic grain parameters
        self.grain_length_spin = QDoubleSpinBox()
        self.grain_length_spin.setDecimals(3)
        self.grain_length_spin.setRange(0.01, 2.0)
        self.grain_length_spin.setValue(0.2)
        self.grain_length_spin.setSingleStep(0.01)
        grain_type_layout.addRow("Grain Length (m):", self.grain_length_spin)
        
        self.grain_od_spin = QDoubleSpinBox()
        self.grain_od_spin.setDecimals(3)
        self.grain_od_spin.setRange(0.01, 0.5)
        self.grain_od_spin.setValue(0.075)
        self.grain_od_spin.setSingleStep(0.005)
        grain_type_layout.addRow("Outer Diameter (m):", self.grain_od_spin)
        
        self.grain_id_spin = QDoubleSpinBox()
        self.grain_id_spin.setDecimals(3)
        self.grain_id_spin.setRange(0.001, 0.4)
        self.grain_id_spin.setValue(0.025)
        self.grain_id_spin.setSingleStep(0.005)
        grain_type_layout.addRow("Core Diameter (m):", self.grain_id_spin)
        
        grain_type_group.setLayout(grain_type_layout)
        
        # Propellant properties
        prop_group = QGroupBox("Propellant Properties")
        prop_layout = QFormLayout()
        
        self.prop_name_edit = QLineEdit("APCP")
        prop_layout.addRow("Propellant Name:", self.prop_name_edit)
        
        self.prop_density_spin = QDoubleSpinBox()
        self.prop_density_spin.setDecimals(1)
        self.prop_density_spin.setRange(500, 2500)
        self.prop_density_spin.setValue(1750)
        self.prop_density_spin.setSingleStep(10)
        prop_layout.addRow("Density (kg/m³):", self.prop_density_spin)
        
        self.prop_a_coef_spin = QDoubleSpinBox()
        self.prop_a_coef_spin.setDecimals(4)
        self.prop_a_coef_spin.setRange(0.0001, 1.0)
        self.prop_a_coef_spin.setValue(0.0076)
        self.prop_a_coef_spin.setSingleStep(0.0001)
        prop_layout.addRow("Burn Rate Coefficient (a):", self.prop_a_coef_spin)
        
        self.prop_n_exp_spin = QDoubleSpinBox()
        self.prop_n_exp_spin.setDecimals(2)
        self.prop_n_exp_spin.setRange(0.1, 2.0)
        self.prop_n_exp_spin.setValue(0.36)
        self.prop_n_exp_spin.setSingleStep(0.01)
        prop_layout.addRow("Pressure Exponent (n):", self.prop_n_exp_spin)
        
        prop_group.setLayout(prop_layout)
        
        # Add to grain layout
        grain_layout.addWidget(grain_type_group)
        grain_layout.addWidget(prop_group)
        grain_layout.addStretch()
        
        # 3. Case Tab
        case_tab = QWidget()
        case_layout = QVBoxLayout(case_tab)
        
        case_group = QGroupBox("Motor Case")
        case_form = QFormLayout()
        
        self.case_material_combo = QComboBox()
        self.case_material_combo.addItems(["Aluminum", "Steel", "Carbon Fiber", "Fiberglass"])
        case_form.addRow("Case Material:", self.case_material_combo)
        
        self.case_wall_spin = QDoubleSpinBox()
        self.case_wall_spin.setDecimals(3)
        self.case_wall_spin.setRange(0.001, 0.05)
        self.case_wall_spin.setValue(0.003)
        self.case_wall_spin.setSingleStep(0.001)
        case_form.addRow("Wall Thickness (m):", self.case_wall_spin)
        
        self.case_sf_spin = QDoubleSpinBox()
        self.case_sf_spin.setDecimals(1)
        self.case_sf_spin.setRange(1.1, 5.0)
        self.case_sf_spin.setValue(2.0)
        self.case_sf_spin.setSingleStep(0.1)
        case_form.addRow("Safety Factor:", self.case_sf_spin)
        
        case_group.setLayout(case_form)
        case_layout.addWidget(case_group)
        case_layout.addStretch()
        
        # 4. Nozzle Tab
        nozzle_tab = QWidget()
        nozzle_layout = QVBoxLayout(nozzle_tab)
        
        nozzle_group = QGroupBox("Nozzle Parameters")
        nozzle_form = QFormLayout()
        
        self.nozzle_throat_spin = QDoubleSpinBox()
        self.nozzle_throat_spin.setDecimals(3)
        self.nozzle_throat_spin.setRange(0.002, 0.1)
        self.nozzle_throat_spin.setValue(0.015)
        self.nozzle_throat_spin.setSingleStep(0.001)
        nozzle_form.addRow("Throat Diameter (m):", self.nozzle_throat_spin)
        
        self.nozzle_er_spin = QDoubleSpinBox()
        self.nozzle_er_spin.setDecimals(1)
        self.nozzle_er_spin.setRange(2.0, 20.0)
        self.nozzle_er_spin.setValue(8.0)
        self.nozzle_er_spin.setSingleStep(0.5)
        nozzle_form.addRow("Expansion Ratio:", self.nozzle_er_spin)
        
        self.nozzle_type_combo = QComboBox()
        self.nozzle_type_combo.addItems(["Conical", "Bell", "Rao Optimum"])
        nozzle_form.addRow("Nozzle Type:", self.nozzle_type_combo)
        
        nozzle_group.setLayout(nozzle_form)
        nozzle_layout.addWidget(nozzle_group)
        nozzle_layout.addStretch()
        
        # Add tabs to tab widget
        self.design_tabs.addTab(basic_tab, "Basic")
        self.design_tabs.addTab(grain_tab, "Grain")
        self.design_tabs.addTab(case_tab, "Case")
        self.design_tabs.addTab(nozzle_tab, "Nozzle")
        
        # Add tab widget to top layout
        top_layout.addWidget(self.design_tabs)
        
        # Bottom panel: Visualization and Results
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        # Results tabs
        results_tabs = QTabWidget()
        
        # 1. Performance Tab
        perf_tab = QWidget()
        perf_layout = QVBoxLayout(perf_tab)
        
        self.thrust_curve_fig = Figure(figsize=(8, 4))
        self.thrust_curve_canvas = FigureCanvas(self.thrust_curve_fig)
        self.thrust_curve_toolbar = NavigationToolbar(self.thrust_curve_canvas, self)
        
        perf_layout.addWidget(self.thrust_curve_toolbar)
        perf_layout.addWidget(self.thrust_curve_canvas)
        
        # 2. Summary Tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        self.summary_text = QLabel("No motor design available.")
        self.summary_text.setAlignment(Qt.AlignCenter)
        self.summary_text.setWordWrap(True)
        self.summary_text.setStyleSheet("font-family: monospace;")
        
        summary_layout.addWidget(self.summary_text)
        
        # Add tabs to results_tabs
        results_tabs.addTab(perf_tab, "Performance")
        results_tabs.addTab(summary_tab, "Summary")
        
        # Add results_tabs to bottom layout
        bottom_layout.addWidget(results_tabs)
        
        # Add panels to splitter
        self.splitter.addWidget(top_panel)
        self.splitter.addWidget(bottom_panel)
        self.splitter.setSizes([300, 300])  # Initial sizes
        
        # Add splitter to main layout
        main_layout.addWidget(self.splitter)
        
        # Initialize with empty plots
        self._create_empty_plots()
        
        # Update controls visibility
        self._update_controls_visibility()
        
        # Disable controls until CEA data is loaded
        self._set_controls_enabled(False)
        
    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable controls."""
        self.design_tabs.setEnabled(enabled)
        self.design_button.setEnabled(enabled)
        
    def _create_empty_plots(self):
        """Create empty plots with placeholder text."""
        # Thrust curve
        self.thrust_curve_fig.clear()
        ax = self.thrust_curve_fig.add_subplot(111)
        ax.set_title("Thrust Curve")
        ax.text(0.5, 0.5, "No motor design available", 
               ha='center', va='center', fontsize=14, 
               transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        self.thrust_curve_canvas.draw()
        
    def _update_controls_visibility(self):
        """Update visibility of controls based on selected motor type."""
        motor_type = self.motor_type_combo.currentText()
        
        # Enable/disable tabs based on motor type
        if "Solid" in motor_type:
            self.design_tabs.setTabEnabled(1, True)  # Grain tab
        else:
            self.design_tabs.setTabEnabled(1, False)  # Grain tab
            
    def _update_grain_controls(self):
        """Update grain controls based on selected grain type."""
        grain_type = self.grain_type_combo.currentText()
        
        # Show/hide core diameter for end burner
        if grain_type == "End Burner":
            self.grain_id_spin.setEnabled(False)
        else:
            self.grain_id_spin.setEnabled(True)
            
    def set_cea_data(self, cea_data: Dict[str, Any]):
        """Set the CEA data for motor design."""
        if cea_data is None:
            self._set_controls_enabled(False)
            return
            
        # Store CEA data
        self.cea_data = cea_data
        
        # Set expansion ratio from CEA data if available
        if 'area_ratio' in cea_data:
            self.nozzle_er_spin.setValue(cea_data['area_ratio'])
            
        # Enable controls
        self._set_controls_enabled(True)
        
    def update_nozzle(self, nozzle_data: Dict[str, Any]):
        """Update the nozzle design with data from nozzle designer."""
        if nozzle_data is None:
            return
            
        # Update nozzle controls
        if 'area_ratio' in nozzle_data:
            self.nozzle_er_spin.setValue(nozzle_data['area_ratio'])
            
        # Update motor design if it exists
        if self.motor_design:
            # Update nozzle
            nozzle = Nozzle(
                throat_diameter=self.nozzle_throat_spin.value(),
                expansion_ratio=nozzle_data['area_ratio'],
                contour_type=self.nozzle_type_combo.currentText()
            )
            self.motor_design.set_nozzle(nozzle)
            
            # Update display
            self._update_motor_display()
            
    def _generate_motor(self):
        """Generate motor design based on current parameters."""
        if self.cea_data is None:
            QMessageBox.warning(self, "Warning", "No CEA data available. Load data first.")
            return
            
        try:
            # Get parameters
            motor_name = self.motor_name_edit.text() or "Unnamed Motor"
            motor_type_str = self.motor_type_combo.currentText()
            motor_type = MotorType.SOLID if "Solid" in motor_type_str else (
                MotorType.HYBRID if "Hybrid" in motor_type_str else MotorType.LIQUID
            )
            
            # Create motor design
            self.motor_design = MotorDesign(name=motor_name, motor_type=motor_type, cea_data=self.cea_data)
            
            # Create and set grain
            if motor_type == MotorType.SOLID:
                propellant = PropellantProperties(
                    name=self.prop_name_edit.text(),
                    density=self.prop_density_spin.value(),
                    a_coefficient=self.prop_a_coef_spin.value(),
                    n_exponent=self.prop_n_exp_spin.value()
                )
                
                grain_type_str = self.grain_type_combo.currentText()
                
                if grain_type_str == "BATES":
                    geometry = BatesGrain(
                        length=self.grain_length_spin.value(),
                        outer_diameter=self.grain_od_spin.value(),
                        core_diameter=self.grain_id_spin.value()
                    )
                elif grain_type_str == "Star":
                    geometry = StarGrain(
                        length=self.grain_length_spin.value(),
                        outer_diameter=self.grain_od_spin.value(),
                        core_diameter=self.grain_id_spin.value(),
                        number_of_points=5  # Default
                    )
                else:  # End Burner
                    geometry = EndBurnerGrain(
                        length=self.grain_length_spin.value(),
                        outer_diameter=self.grain_od_spin.value()
                    )
                    
                grain = MotorGrain(geometry, propellant)
                self.motor_design.set_grain(grain)
            
            # Create and set case
            case = MotorCase(
                material=self.case_material_combo.currentText(),
                inner_diameter=self.grain_od_spin.value() + 0.002,  # 2mm clearance
                wall_thickness=self.case_wall_spin.value(),
                length=self.grain_length_spin.value() * 1.1,  # 10% extra
                safety_factor=self.case_sf_spin.value()
            )
            self.motor_design.set_case(case)
            
            # Create and set nozzle
            nozzle = Nozzle(
                throat_diameter=self.nozzle_throat_spin.value(),
                expansion_ratio=self.nozzle_er_spin.value(),
                contour_type=self.nozzle_type_combo.currentText()
            )
            self.motor_design.set_nozzle(nozzle)
            
            # Calculate performance
            self.motor_design.calculate_performance()
            
            # Update display
            self._update_motor_display()
            
            # Enable save button
            self.save_button.setEnabled(True)
            
            # Emit signal
            self.motor_designed.emit(self.motor_design)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error generating motor design: {str(e)}")
            
    def _update_motor_display(self):
        """Update the motor display with current design."""
        if self.motor_design is None:
            return
            
        # Update thrust curve
        self._plot_thrust_curve()
        
        # Update summary
        self._update_summary()
        
    def _plot_thrust_curve(self):
        """Plot the motor thrust curve."""
        if self.motor_design is None or not hasattr(self.motor_design, '_performance') or not self.motor_design._performance:
            return
            
        # Get performance data
        perf = self.motor_design._performance
        
        # Plot thrust curve
        self.thrust_curve_fig.clear()
        ax = self.thrust_curve_fig.add_subplot(111)
        
        # Plot thrust vs time
        if 'time' in perf and 'thrust' in perf:
            ax.plot(perf['time'], perf['thrust'], 'b-', linewidth=2)
            
            # Add average thrust line
            if 'average_thrust' in perf:
                ax.axhline(y=perf['average_thrust'], color='r', linestyle='--', 
                          label=f"Average: {perf['average_thrust']:.1f} N")
            
            # Add total impulse annotation
            if 'total_impulse' in perf:
                total_impulse = perf['total_impulse']
                if total_impulse >= 1000:
                    impulse_text = f"Total Impulse: {total_impulse/1000:.2f} kN·s"
                else:
                    impulse_text = f"Total Impulse: {total_impulse:.1f} N·s"
                    
                ax.text(0.95, 0.95, impulse_text, transform=ax.transAxes, 
                       ha='right', va='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add labels and grid
        ax.set_title(f"{self.motor_design.name} Thrust Curve")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Thrust (N)")
        ax.grid(True)
        ax.legend()
        
        # Redraw canvas
        self.thrust_curve_fig.tight_layout()
        self.thrust_curve_canvas.draw()
        
    def _update_summary(self):
        """Update the motor summary display."""
        if self.motor_design is None:
            return
            
        # Get summary
        summary = self.motor_design.get_summary()
        
        # Format summary text
        html = f"""<html>
        <style>
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
        </style>
        <h2>{summary['name']} - {summary['type']}</h2>
        """
        
        # Grain section
        if 'grain' in summary:
            html += """
            <h3>Grain Configuration</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Type</td><td>{0}</td></tr>
                <tr><td>Length</td><td>{1}</td></tr>
                <tr><td>Diameter</td><td>{2}</td></tr>
                <tr><td>Volume</td><td>{3}</td></tr>
                <tr><td>Mass</td><td>{4}</td></tr>
                <tr><td>Propellant</td><td>{5}</td></tr>
            </table>
            """.format(
                summary['grain']['type'],
                summary['grain']['length'],
                summary['grain']['diameter'],
                summary['grain']['volume'],
                summary['grain']['mass'],
                summary['grain']['propellant']
            )
        
        # Case section
        if 'case' in summary:
            html += """
            <h3>Motor Case</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Material</td><td>{0}</td></tr>
                <tr><td>Inner Diameter</td><td>{1}</td></tr>
                <tr><td>Wall Thickness</td><td>{2}</td></tr>
                <tr><td>Length</td><td>{3}</td></tr>
                <tr><td>Mass</td><td>{4}</td></tr>
                <tr><td>Max Pressure</td><td>{5}</td></tr>
            </table>
            """.format(
                summary['case']['material'],
                summary['case']['inner_diameter'],
                summary['case']['wall_thickness'],
                summary['case']['length'],
                summary['case']['mass'],
                summary['case']['max_pressure']
            )
        
        # Nozzle section
        if 'nozzle' in summary:
            html += """
            <h3>Nozzle Design</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Type</td><td>{0}</td></tr>
                <tr><td>Throat Diameter</td><td>{1}</td></tr>
                <tr><td>Exit Diameter</td><td>{2}</td></tr>
                <tr><td>Expansion Ratio</td><td>{3}</td></tr>
                <tr><td>Length</td><td>{4}</td></tr>
                <tr><td>Mass</td><td>{5}</td></tr>
            </table>
            """.format(
                summary['nozzle']['type'],
                summary['nozzle']['throat_diameter'],
                summary['nozzle']['exit_diameter'],
                summary['nozzle']['expansion_ratio'],
                summary['nozzle']['length'],
                summary['nozzle']['mass']
            )
        
        # Performance section
        if 'performance' in summary:
            html += """
            <h3>Performance</h3>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Average Thrust</td><td>{0}</td></tr>
                <tr><td>Maximum Thrust</td><td>{1}</td></tr>
                <tr><td>Total Impulse</td><td>{2}</td></tr>
                <tr><td>Burn Time</td><td>{3}</td></tr>
                <tr><td>Specific Impulse</td><td>{4}</td></tr>
            </table>
            """.format(
                summary['performance']['average_thrust'],
                summary['performance']['max_thrust'],
                summary['performance']['total_impulse'],
                summary['performance']['burn_time'],
                summary['performance']['specific_impulse']
            )
        
        html += "</html>"
        
        self.summary_text.setText(html)
        
    def _save_motor(self):
        """Save motor design to file."""
        if self.motor_design is None:
            return
            
        # Get file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Motor Design", "", 
            "Motor Design Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            # Save to file
            success = self.motor_design.save_to_file(file_path)
            
            if success:
                QMessageBox.information(self, "Success", f"Motor design saved to {file_path}")
            else:
                QMessageBox.warning(self, "Warning", "Failed to save motor design")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving motor design: {str(e)}")
            
    def _load_motor(self):
        """Load motor design from file."""
        # Get file path
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Load Motor Design", "", 
            "Motor Design Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            # Load from file
            self.motor_design = MotorDesign.load_from_file(file_path)
            
            if self.motor_design:
                # Update display
                self._update_motor_display()
                
                # Enable save button
                self.save_button.setEnabled(True)
                
                # Emit signal
                self.motor_designed.emit(self.motor_design)
                
                QMessageBox.information(self, "Success", f"Motor design loaded from {file_path}")
            else:
                QMessageBox.warning(self, "Warning", "Failed to load motor design")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading motor design: {str(e)}")
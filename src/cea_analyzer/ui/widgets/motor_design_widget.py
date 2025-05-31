"""
Motor Design Widget
----------------

Widget for designing and analyzing rocket motors with various grain configurations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QComboBox, QLineEdit, QDoubleSpinBox, QCheckBox, QPushButton,
    QFileDialog, QTabWidget, QSplitter, QMessageBox, QRadioButton,
    QButtonGroup, QGridLayout, QSpacerItem, QSizePolicy, QFormLayout,
    QScrollArea, QFrame, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from ...propulsion.motor.design import MotorDesign
from ...propulsion.motor.types import MotorType
from ...propulsion.motor.components import MotorCase, Nozzle
from ...propulsion.grain import MotorGrain, GrainType, PropellantProperties
from ...propulsion.grain import BatesGrain, StarGrain, EndBurnerGrain
from ...propulsion.grain import CSlotGrain, FinocylGrain, WagonWheelGrain
from ...propulsion.grain import (
    GrainRegressionSimulation,
    generate_grain_cross_section,
    visualize_grain_regression,
    create_3d_grain_model
)
from ...propulsion.grain.propellants import get_propellant_by_name, get_available_propellants


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
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top panel: Design controls
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Tab widget for different design components
        self.design_tabs = QTabWidget()
        
        # 1. Basic Motor Tab
        basic_tab_content = QWidget()
        basic_layout = QVBoxLayout(basic_tab_content)
        
        # Create a scroll area for the tab content
        basic_tab = QScrollArea()
        basic_tab.setWidgetResizable(True)
        basic_tab.setWidget(basic_tab_content)
        basic_tab.setFrameShape(QFrame.Shape.NoFrame)
        
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
        grain_tab_content = QWidget()
        grain_layout = QVBoxLayout(grain_tab_content)
        
        # Create a scroll area for the tab content
        grain_tab = QScrollArea()
        grain_tab.setWidgetResizable(True)
        grain_tab.setWidget(grain_tab_content)
        grain_tab.setFrameShape(QFrame.Shape.NoFrame)
        
        # Grain type
        grain_type_group = QGroupBox("Grain Configuration")
        grain_type_layout = QFormLayout()
        
        self.grain_type_combo = QComboBox()
        self.grain_type_combo.addItems(["BATES", "Star", "End Burner", "C-Slot", "Finocyl", "Wagon Wheel"])
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
        
        # Add inhibited ends checkbox
        self.inhibited_ends_check = QCheckBox("Inhibited Ends")
        self.inhibited_ends_check.setChecked(True)
        self.inhibited_ends_check.setToolTip("When checked, the grain ends will not burn")
        grain_type_layout.addRow("", self.inhibited_ends_check)
        
        # Advanced grain parameters (will be shown/hidden based on grain type)
        # Star grain parameters
        self.star_points_spin = QDoubleSpinBox()
        self.star_points_spin.setDecimals(0)
        self.star_points_spin.setRange(3, 12)
        self.star_points_spin.setValue(5)
        self.star_points_spin.setSingleStep(1)
        self.star_points_spin.setVisible(False)
        grain_type_layout.addRow("Number of Points:", self.star_points_spin)
        
        self.star_point_depth_spin = QDoubleSpinBox()
        self.star_point_depth_spin.setDecimals(3)
        self.star_point_depth_spin.setRange(0.001, 0.1)
        self.star_point_depth_spin.setValue(0.01)
        self.star_point_depth_spin.setSingleStep(0.001)
        self.star_point_depth_spin.setVisible(False)
        grain_type_layout.addRow("Point Depth (m):", self.star_point_depth_spin)
        
        self.star_inner_angle_spin = QDoubleSpinBox()
        self.star_inner_angle_spin.setDecimals(1)
        self.star_inner_angle_spin.setRange(10, 90)
        self.star_inner_angle_spin.setValue(60)
        self.star_inner_angle_spin.setSingleStep(5)
        self.star_inner_angle_spin.setVisible(False)
        grain_type_layout.addRow("Inner Angle (deg):", self.star_inner_angle_spin)
        
        # C-Slot grain parameters
        self.slot_width_spin = QDoubleSpinBox()
        self.slot_width_spin.setDecimals(3)
        self.slot_width_spin.setRange(0.001, 0.1)
        self.slot_width_spin.setValue(0.01)
        self.slot_width_spin.setSingleStep(0.001)
        self.slot_width_spin.setVisible(False)
        grain_type_layout.addRow("Slot Width (m):", self.slot_width_spin)
        
        self.slot_depth_spin = QDoubleSpinBox()
        self.slot_depth_spin.setDecimals(3)
        self.slot_depth_spin.setRange(0.001, 0.1)
        self.slot_depth_spin.setValue(0.015)
        self.slot_depth_spin.setSingleStep(0.001)
        self.slot_depth_spin.setVisible(False)
        grain_type_layout.addRow("Slot Depth (m):", self.slot_depth_spin)
        
        self.slot_angle_spin = QDoubleSpinBox()
        self.slot_angle_spin.setDecimals(1)
        self.slot_angle_spin.setRange(30, 180)
        self.slot_angle_spin.setValue(120)
        self.slot_angle_spin.setSingleStep(10)
        self.slot_angle_spin.setVisible(False)
        grain_type_layout.addRow("Slot Angle (deg):", self.slot_angle_spin)
        
        # Finocyl grain parameters
        self.fin_count_spin = QDoubleSpinBox()
        self.fin_count_spin.setDecimals(0)
        self.fin_count_spin.setRange(3, 12)
        self.fin_count_spin.setValue(6)
        self.fin_count_spin.setSingleStep(1)
        self.fin_count_spin.setVisible(False)
        grain_type_layout.addRow("Number of Fins:", self.fin_count_spin)
        
        self.fin_height_spin = QDoubleSpinBox()
        self.fin_height_spin.setDecimals(3)
        self.fin_height_spin.setRange(0.001, 0.1)
        self.fin_height_spin.setValue(0.015)
        self.fin_height_spin.setSingleStep(0.001)
        self.fin_height_spin.setVisible(False)
        grain_type_layout.addRow("Fin Height (m):", self.fin_height_spin)
        
        self.fin_width_spin = QDoubleSpinBox()
        self.fin_width_spin.setDecimals(3)
        self.fin_width_spin.setRange(0.001, 0.05)
        self.fin_width_spin.setValue(0.008)
        self.fin_width_spin.setSingleStep(0.001)
        self.fin_width_spin.setVisible(False)
        grain_type_layout.addRow("Fin Width (m):", self.fin_width_spin)
        
        # Wagon Wheel grain parameters
        self.spoke_count_spin = QDoubleSpinBox()
        self.spoke_count_spin.setDecimals(0)
        self.spoke_count_spin.setRange(4, 16)
        self.spoke_count_spin.setValue(8)
        self.spoke_count_spin.setSingleStep(1)
        self.spoke_count_spin.setVisible(False)
        grain_type_layout.addRow("Number of Spokes:", self.spoke_count_spin)
        
        self.spoke_width_spin = QDoubleSpinBox()
        self.spoke_width_spin.setDecimals(3)
        self.spoke_width_spin.setRange(0.001, 0.05)
        self.spoke_width_spin.setValue(0.008)
        self.spoke_width_spin.setSingleStep(0.001)
        self.spoke_width_spin.setVisible(False)
        grain_type_layout.addRow("Spoke Width (m):", self.spoke_width_spin)
        
        self.spoke_length_spin = QDoubleSpinBox()
        self.spoke_length_spin.setDecimals(3)
        self.spoke_length_spin.setRange(0.001, 0.1)
        self.spoke_length_spin.setValue(0.02)
        self.spoke_length_spin.setSingleStep(0.001)
        self.spoke_length_spin.setVisible(False)
        grain_type_layout.addRow("Spoke Length (m):", self.spoke_length_spin)
        
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
        
        # Grain Visualization Section
        viz_group = QGroupBox("Grain Visualization")
        viz_layout = QVBoxLayout()
        
        # Create Visualization Button
        self.visualize_button = QPushButton("Visualize Grain")
        self.visualize_button.clicked.connect(self._visualize_grain)
        viz_layout.addWidget(self.visualize_button)
        
        # Visualization Tabs (Cross Section and 3D View)
        self.viz_tabs = QTabWidget()
        
        # Cross Section Tab
        self.cross_section_widget = QWidget()
        cross_section_layout = QVBoxLayout(self.cross_section_widget)
        
        self.cross_section_figure = Figure(figsize=(8, 8), dpi=120)
        self.cross_section_canvas = FigureCanvas(self.cross_section_figure)
        self.cross_section_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.cross_section_canvas.setMinimumHeight(500)  # Set minimum height
        self.cross_section_toolbar = NavigationToolbar(self.cross_section_canvas, self)
        
        cross_section_layout.addWidget(self.cross_section_toolbar)
        cross_section_layout.addWidget(self.cross_section_canvas)
        
        # 3D View Tab
        self.view_3d_widget = QWidget()
        view_3d_layout = QVBoxLayout(self.view_3d_widget)
        
        self.view_3d_figure = Figure(figsize=(8, 8), dpi=120)
        self.view_3d_canvas = FigureCanvas(self.view_3d_figure)
        self.view_3d_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view_3d_canvas.setMinimumHeight(500)  # Set minimum height
        self.view_3d_toolbar = NavigationToolbar(self.view_3d_canvas, self)
        
        view_3d_layout.addWidget(self.view_3d_toolbar)
        view_3d_layout.addWidget(self.view_3d_canvas)
        
        # Add tabs to visualization tab widget
        self.viz_tabs.addTab(self.cross_section_widget, "Cross Section")
        self.viz_tabs.addTab(self.view_3d_widget, "3D View")
        
        # Regression Control
        regression_group = QGroupBox("Grain Regression")
        regression_layout = QFormLayout()
        
        self.web_slider = QDoubleSpinBox()
        self.web_slider.setRange(0.0, 100.0)
        self.web_slider.setValue(0.0)
        self.web_slider.setSingleStep(5.0)
        self.web_slider.setSuffix("%")
        self.web_slider.valueChanged.connect(self._update_grain_regression)
        regression_layout.addRow("Web Burned:", self.web_slider)
        
        regression_group.setLayout(regression_layout)
        
        # Ensure visualization tabs take most of the space
        self.viz_tabs.setMinimumHeight(600)  # Set a minimum height for visualization tabs
        
        # Add to visualization layout
        viz_layout.addWidget(self.viz_tabs, 1)  # Give it a stretch factor of 1
        viz_layout.addWidget(regression_group, 0)  # No stretch factor
        
        viz_group.setLayout(viz_layout)
        
        # Make the visualization group expand to fill available space
        viz_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Add the visualization group to the grain layout
        grain_layout.addWidget(viz_group)
        
        # 3. Case Tab
        case_tab_content = QWidget()
        case_layout = QVBoxLayout(case_tab_content)
        
        # Create a scroll area for the tab content
        case_tab = QScrollArea()
        case_tab.setWidgetResizable(True)
        case_tab.setWidget(case_tab_content)
        case_tab.setFrameShape(QFrame.Shape.NoFrame)
        
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
        nozzle_tab_content = QWidget()
        nozzle_layout = QVBoxLayout(nozzle_tab_content)
        
        # Create a scroll area for the tab content
        nozzle_tab = QScrollArea()
        nozzle_tab.setWidgetResizable(True)
        nozzle_tab.setWidget(nozzle_tab_content)
        nozzle_tab.setFrameShape(QFrame.Shape.NoFrame)
        
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
        
        # 1. Summary Tab
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        # Add text edit for summary display
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setHtml("<p>Motor design summary will appear here.</p>")
        summary_layout.addWidget(self.summary_text)
        
        # Create a scroll area for the tab content
        summary_scroll = QScrollArea()
        summary_scroll.setWidgetResizable(True)
        summary_scroll.setWidget(summary_tab)
        summary_scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        # Add summary tab to results tabs
        results_tabs.addTab(summary_scroll, "Summary")
        
        # 2. Performance Tab
        perf_tab_content = QWidget()
        perf_layout = QVBoxLayout(perf_tab_content)
        
        # Create a scroll area for the tab content
        perf_tab = QScrollArea()
        perf_tab.setWidgetResizable(True)
        perf_tab.setWidget(perf_tab_content)
        perf_tab.setFrameShape(QFrame.Shape.NoFrame)
        
        # Add performance tab to results tabs
        results_tabs.addTab(perf_tab, "Performance")
        
        # Add results tabs to bottom layout
        bottom_layout.addWidget(results_tabs)
        
        # Add panels to splitter
        self.splitter.addWidget(top_panel)
        self.splitter.addWidget(bottom_panel)
        self.splitter.setSizes([300, 700])  # Initial sizes - giving more space to visualizations
        
        # Add splitter to main layout - THIS WAS THE MISSING PART
        main_layout.addWidget(self.splitter)
        
    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable controls."""
        try:
            if hasattr(self, 'design_tabs') and self.design_tabs is not None:
                self.design_tabs.setEnabled(enabled)
            if hasattr(self, 'design_button') and self.design_button is not None:
                self.design_button.setEnabled(enabled)
        except (RuntimeError, AttributeError) as e:
            # Widgets might have been deleted or not fully initialized
            print(f"Warning: Could not set controls enabled state: {e}")
        
    def _update_controls_visibility(self):
        """Update visibility of controls based on selected motor type."""
        try:
            if not hasattr(self, 'motor_type_combo') or not hasattr(self, 'design_tabs'):
                return
                
            motor_type = self.motor_type_combo.currentText()
            
            # Enable/disable tabs based on motor type
            if "Solid" in motor_type:
                self.design_tabs.setTabEnabled(1, True)  # Grain tab
            else:
                self.design_tabs.setTabEnabled(1, False)  # Grain tab
        except (RuntimeError, AttributeError) as e:
            # Widgets might have been deleted or not fully initialized
            print(f"Warning: Could not update controls visibility: {e}")
            
    def _update_grain_controls(self):
        """Update grain controls based on selected grain type."""
        try:
            if not hasattr(self, 'grain_type_combo'):
                return
                
            grain_type = self.grain_type_combo.currentText()
            
            # Check if all required widgets exist
            required_widgets = [
                'star_points_spin', 'star_point_depth_spin', 'star_inner_angle_spin',
                'slot_width_spin', 'slot_depth_spin', 'slot_angle_spin',
                'fin_count_spin', 'fin_height_spin', 'fin_width_spin',
                'spoke_count_spin', 'spoke_width_spin', 'spoke_length_spin',
                'grain_id_spin'
            ]
            
            for widget_name in required_widgets:
                if not hasattr(self, widget_name):
                    print(f"Warning: Widget {widget_name} not found")
                    return
            
            # Hide all advanced parameters first
            self.star_points_spin.setVisible(False)
            self.star_point_depth_spin.setVisible(False)
            self.star_inner_angle_spin.setVisible(False)
            
            self.slot_width_spin.setVisible(False)
            self.slot_depth_spin.setVisible(False)
            self.slot_angle_spin.setVisible(False)
            
            self.fin_count_spin.setVisible(False)
            self.fin_height_spin.setVisible(False)
            self.fin_width_spin.setVisible(False)
            
            self.spoke_count_spin.setVisible(False)
            self.spoke_width_spin.setVisible(False)
            self.spoke_length_spin.setVisible(False)
            
            # Show/hide core diameter for end burner
            if grain_type == "End Burner":
                self.grain_id_spin.setEnabled(False)
            else:
                self.grain_id_spin.setEnabled(True)
            
            # Show specific parameters based on grain type
            if grain_type == "Star":
                self.star_points_spin.setVisible(True)
                self.star_point_depth_spin.setVisible(True)
                self.star_inner_angle_spin.setVisible(True)
            elif grain_type == "C-Slot":
                self.slot_width_spin.setVisible(True)
                self.slot_depth_spin.setVisible(True)
                self.slot_angle_spin.setVisible(True)
            elif grain_type == "Finocyl":
                self.fin_count_spin.setVisible(True)
                self.fin_height_spin.setVisible(True)
                self.fin_width_spin.setVisible(True)
            elif grain_type == "Wagon Wheel":
                self.spoke_count_spin.setVisible(True)
                self.spoke_width_spin.setVisible(True)
                self.spoke_length_spin.setVisible(True)
        except (RuntimeError, AttributeError) as e:
            # Widgets might have been deleted or not fully initialized
            print(f"Warning: Could not update grain controls: {e}")
            
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
                    burn_rate_coefficient=self.prop_a_coef_spin.value(),
                    burn_rate_exponent=self.prop_n_exp_spin.value()
                )
                
                # Use the grain creation method we already have
                geometry = self._create_grain_from_parameters()
                    
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
            import traceback
            traceback.print_exc()
            
    def _update_motor_display(self):
        """Update the motor display with current design."""
        if not self.motor_design:
            return
            
        # Update summary
        self._update_summary()
        
        # Also visualize the grain if we have a solid motor
        if self.motor_design.motor_type == MotorType.SOLID and hasattr(self.motor_design, 'grain'):
            # Create regression simulation for visualization
            grain_geom = self.motor_design.grain.geometry
            propellant = self.motor_design.grain.propellant
            self.regression_sim = GrainRegressionSimulation(grain_geom, propellant)
            
            # Update visualizations
            self._update_cross_section()
            self._update_3d_model()
            
    def set_cea_data(self, cea_data: Dict[str, Any]):
        """Set the CEA data for motor design."""
        if cea_data is None:
            self._set_controls_enabled(False)
            return
            
        # Store CEA data
        self.cea_data = cea_data
        
        # Set expansion ratio from CEA data if available
        try:
            if 'area_ratio' in cea_data and hasattr(self, 'nozzle_er_spin') and self.nozzle_er_spin is not None:
                self.nozzle_er_spin.setValue(cea_data['area_ratio'])
        except (RuntimeError, AttributeError) as e:
            # Widget might have been deleted or not fully initialized
            print(f"Warning: Could not set nozzle expansion ratio: {e}")
            
        # Enable controls
        self._set_controls_enabled(True)
            
    def _update_summary(self):
        """Update the motor summary display."""
        if not self.motor_design:
            return
        
        motor = self.motor_design
        self.summary_text.clear()
        
        self.summary_text.append(f"<h3>Motor: {motor.name}</h3>")
        self.summary_text.append(f"<b>Type:</b> {motor.motor_type.name}")
        
        # Access performance data from the _performance dictionary
        if hasattr(motor, '_performance') and motor._performance:
            self.summary_text.append(f"<b>Total Impulse:</b> {motor._performance.get('total_impulse', 0):.1f} N·s")
            self.summary_text.append(f"<b>Average Thrust:</b> {motor._performance.get('average_thrust', 0):.1f} N")
            self.summary_text.append(f"<b>Burn Time:</b> {motor._performance.get('burn_time', 0):.1f} s")
            self.summary_text.append(f"<b>Specific Impulse:</b> {motor._performance.get('specific_impulse', 0):.1f} s")
            
            # Calculate propellant mass from grain if available
            propellant_mass = motor.grain.mass() if motor.grain else 0
            self.summary_text.append(f"<b>Propellant Mass:</b> {propellant_mass:.3f} kg")
            
            # Calculate total mass (propellant + case + nozzle)
            case_mass = motor.case.mass() if motor.case else 0
            nozzle_mass = motor.nozzle.mass() if motor.nozzle else 0
            total_mass = propellant_mass + case_mass + nozzle_mass
            self.summary_text.append(f"<b>Total Mass:</b> {total_mass:.3f} kg")
            
            # Calculate mass ratio if possible
            if total_mass > 0 and propellant_mass > 0:
                mass_ratio = propellant_mass / total_mass
                self.summary_text.append(f"<b>Mass Ratio:</b> {mass_ratio:.2f}")
        else:
            self.summary_text.append("<i>Performance data not available. Generate the motor design first.</i>")
        
        # Case info
        if motor.case:
            self.summary_text.append(f"<h4>Motor Case</h4>")
            self.summary_text.append(f"<b>Material:</b> {motor.case.material}")
            # Calculate outer diameter from inner diameter and wall thickness
            outer_diameter = motor.case.inner_diameter + 2 * motor.case.wall_thickness
            self.summary_text.append(f"<b>Dimensions:</b> {motor.case.inner_diameter*1000:.1f}mm ID x {outer_diameter*1000:.1f}mm OD x {motor.case.length*1000:.1f}mm length")
            self.summary_text.append(f"<b>Wall Thickness:</b> {motor.case.wall_thickness*1000:.2f} mm")
            self.summary_text.append(f"<b>Case Mass:</b> {motor.case.mass():.3f} kg")
        else:
            self.summary_text.append(f"<h4>Motor Case</h4>")
            self.summary_text.append("<i>No case defined</i>")
        
        # Nozzle info
        if motor.nozzle:
            self.summary_text.append(f"<h4>Nozzle</h4>")
            self.summary_text.append(f"<b>Throat Diameter:</b> {motor.nozzle.throat_diameter*1000:.2f} mm")
            self.summary_text.append(f"<b>Exit Diameter:</b> {motor.nozzle.exit_diameter()*1000:.2f} mm")
            self.summary_text.append(f"<b>Expansion Ratio:</b> {motor.nozzle.expansion_ratio:.2f}")
            self.summary_text.append(f"<b>Nozzle Mass:</b> {motor.nozzle.mass():.3f} kg")
        else:
            self.summary_text.append(f"<h4>Nozzle</h4>")
            self.summary_text.append("<i>No nozzle defined</i>")
        
    def _save_motor(self):
        """Save motor design to file."""
        if not self.motor_design:
            QMessageBox.warning(self, "No Motor Design", "Please generate a motor design first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Motor Design", "", "Motor Design Files (*.json);;All Files (*)"
        )
        
        if file_path:
            try:
                self.motor_design.save_to_file(file_path)
                QMessageBox.information(self, "Motor Saved", f"Motor design saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving motor design: {str(e)}")
                
    def _create_grain_from_parameters(self):
        """Create a grain geometry based on the current UI parameters."""
        # Get common parameters
        length = self.grain_length_spin.value()
        outer_diameter = self.grain_od_spin.value()
        core_diameter = self.grain_id_spin.value()
        inhibited_ends = self.inhibited_ends_check.isChecked()
        
        # Get grain type index
        grain_type_str = self.grain_type_combo.currentText()
        
        # Create grain based on type
        if grain_type_str == "BATES":
            return BatesGrain(
                length=length,
                outer_diameter=outer_diameter,
                core_diameter=core_diameter,
                inhibited_ends=inhibited_ends,
                inhibited_outer_surface=True
            )
        elif grain_type_str == "Star":
            num_points = int(self.star_points_spin.value())
            point_depth = self.star_point_depth_spin.value()
            inner_angle = self.star_inner_angle_spin.value()
            
            return StarGrain(
                length=length,
                outer_diameter=outer_diameter,
                core_diameter=core_diameter,
                inhibited_ends=inhibited_ends,
                inhibited_outer_surface=True,
                number_of_points=num_points,
                point_depth=point_depth,
                inner_angle=inner_angle
            )
        elif grain_type_str == "End Burner":
            return EndBurnerGrain(
                length=length,
                outer_diameter=outer_diameter,
                inhibited_outer_surface=True
            )
        elif grain_type_str == "C-Slot":
            slot_width = self.slot_width_spin.value()
            slot_depth = self.slot_depth_spin.value()
            slot_angle = self.slot_angle_spin.value()
            
            return CSlotGrain(
                length=length,
                outer_diameter=outer_diameter,
                core_diameter=core_diameter,
                inhibited_ends=inhibited_ends,
                inhibited_outer_surface=True,
                slot_width=slot_width,
                slot_depth=slot_depth,
                slot_angle=slot_angle
            )
        elif grain_type_str == "Finocyl":
            fin_count = int(self.fin_count_spin.value())
            fin_height = self.fin_height_spin.value()
            fin_width = self.fin_width_spin.value()
            
            return FinocylGrain(
                length=length,
                outer_diameter=outer_diameter,
                core_diameter=core_diameter,
                inhibited_ends=inhibited_ends,
                inhibited_outer_surface=True,
                number_of_fins=fin_count,
                fin_height=fin_height,
                fin_width=fin_width
            )
        elif grain_type_str == "Wagon Wheel":
            spoke_count = int(self.spoke_count_spin.value())
            spoke_width = self.spoke_width_spin.value()
            spoke_length = self.spoke_length_spin.value()
            
            return WagonWheelGrain(
                length=length,
                outer_diameter=outer_diameter,
                core_diameter=core_diameter,
                inhibited_ends=inhibited_ends,
                inhibited_outer_surface=True,
                number_of_spokes=spoke_count,
                spoke_width=spoke_width,
                spoke_length=spoke_length
            )
        
        # Default to BATES if no valid type selected
        return BatesGrain(
            length=length,
            outer_diameter=outer_diameter,
            core_diameter=core_diameter,
            inhibited_ends=inhibited_ends,
            inhibited_outer_surface=True
        )

    def _visualize_grain(self):
        """Visualize the current grain configuration."""
        try:
            # Create grain geometry
            grain = self._create_grain_from_parameters()
            
            # Create propellant properties
            propellant = PropellantProperties(
                name=self.prop_name_edit.text(),
                density=self.prop_density_spin.value(),
                burn_rate_coefficient=self.prop_a_coef_spin.value(),
                burn_rate_exponent=self.prop_n_exp_spin.value()
            )
            
            # Create regression simulation
            self.regression_sim = GrainRegressionSimulation(grain, propellant)
            
            # Update visualizations
            self._update_cross_section()
            self._update_3d_model()
            
            # Reset web slider range
            max_web = (grain.outer_diameter - (grain.core_diameter if hasattr(grain, 'core_diameter') else 0)) / 2
            self.web_slider.setMaximum(100)  # Percentage of max web
            self.web_slider.setValue(0)  # Reset to 0%
            
        except Exception as e:
            QMessageBox.critical(self, "Visualization Error", f"Error visualizing grain: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _update_grain_regression(self):
        """Update grain regression based on web burned percentage."""
        if not hasattr(self, 'regression_sim'):
            return
        
        # Get the percentage of web burned
        percent_burned = self.web_slider.value() / 100.0
        
        # Update visualizations with the regression percentage
        self._update_cross_section(percent_burned)
        self._update_3d_model(percent_burned)
    
    def _update_cross_section(self, percent_burned=0.0):
        """Update the cross-section visualization."""
        if not hasattr(self, 'regression_sim'):
            return
        
        # Clear the figure
        self.cross_section_figure.clear()
        ax = self.cross_section_figure.add_subplot(111)
        
        try:
            # Configure the axes for better visibility
            ax.set_aspect('equal')  # Equal aspect ratio
            ax.grid(True, linestyle='--', alpha=0.7)  # Add grid
            
            # Set title with grain info
            grain_type = type(self.regression_sim.grain).__name__
            ax.set_title(f"{grain_type} Cross-Section\n{percent_burned*100:.1f}% Web Burned", fontsize=14)
            
            # Generate grain cross-section
            generate_grain_cross_section(
                self.regression_sim.grain,
                ax,
                percent_burned=percent_burned
            )
            
            # Add labels and improve aesthetics
            ax.set_xlabel('Diameter (m)', fontsize=12)
            ax.set_ylabel('Diameter (m)', fontsize=12)
            ax.tick_params(labelsize=10)
            
            # Add colorbar explanation
            legend_elements = [
                plt.Line2D([0], [0], color='lightgray', lw=10, label='Propellant'),
                plt.Line2D([0], [0], color='white', lw=10, label='Burned Area/Port')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            # Ensure tight layout
            self.cross_section_figure.tight_layout()
            
            # Update the canvas
            self.cross_section_canvas.draw()
            
        except Exception as e:
            print(f"Error updating cross-section: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _update_3d_model(self, percent_burned=0.0):
        """Update the 3D model visualization."""
        if not hasattr(self, 'regression_sim'):
            return
        
        # Clear the figure
        self.view_3d_figure.clear()
        ax = self.view_3d_figure.add_subplot(111, projection='3d')
        
        try:
            # Configure the axes for better visibility
            ax.set_box_aspect([1, 1, 2])  # Better aspect ratio for grain
            
            # Set title with grain info
            grain_type = type(self.regression_sim.grain).__name__
            ax.set_title(f"{grain_type} 3D Model\n{percent_burned*100:.1f}% Web Burned", fontsize=14)
            
            # Generate 3D grain model
            create_3d_grain_model(
                self.regression_sim.grain,
                ax,
                percent_burned=percent_burned
            )
            
            # Add labels and improve aesthetics
            ax.set_xlabel('X (m)', fontsize=12)
            ax.set_ylabel('Y (m)', fontsize=12)
            ax.set_zlabel('Length (m)', fontsize=12)
            ax.tick_params(labelsize=10)
            
            # Set optimal initial view angle
            ax.view_init(elev=30, azim=45)
            
            # Ensure tight layout
            self.view_3d_figure.tight_layout()
            
            # Update the canvas
            self.view_3d_canvas.draw()
            
        except Exception as e:
            print(f"Error updating 3D model: {str(e)}")
            import traceback
            traceback.print_exc()
            
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
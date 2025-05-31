"""
Grain Visualization Widget
-----------------------

This module provides a PyQt widget for visualizing grain geometries
and their regression over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import time

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QComboBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QTabWidget, QMessageBox, QFormLayout, QSpinBox, QRadioButton,
    QButtonGroup, QSlider, QFrame, QSplitter, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot, QTimer

from ...propulsion.grain.base import GrainGeometry, PropellantProperties
from ...propulsion.grain.geometries import BatesGrain, StarGrain
from ...propulsion.grain.advanced_geometries import CSlotGrain, FinocylGrain, WagonWheelGrain
from ...propulsion.grain.regression import (
    GrainRegressionSimulation, generate_grain_cross_section,
    visualize_grain_regression, create_3d_grain_model
)
from ...propulsion.grain.propellants import get_propellant_names, get_propellant, PROPELLANT_LIBRARY
from ...core.logger import get_logger

# Setup logger
logger = get_logger(__name__)


class GrainVisualizationWidget(QWidget):
    """Widget for visualizing grain geometries and their regression."""
    
    # Signal emitted when visualization is updated
    visualization_updated = pyqtSignal()
    
    def __init__(self, parent=None):
        """Initialize the grain visualization widget."""
        super().__init__(parent)
        
        # Initialize grain and propellant with default values
        self.grain = BatesGrain(
            length=0.2,
            outer_diameter=0.08,
            core_diameter=0.03,
            inhibited_ends=True,
            inhibited_outer_surface=True,
            number_of_segments=1,
            segment_spacing=0.005
        )
        
        self.propellant = PropellantProperties(
            name="KNDX",
            density=1800.0,
            burn_rate_coefficient=0.006,
            burn_rate_exponent=0.4,
            temperature_sensitivity=0.5,
            reference_temperature=298.15
        )
        
        self.simulation = None
        self.current_web_distance = 0.0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._animation_step)
        self.animation_frame_index = 0
        
        # Setup UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Create a splitter for the main panels
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top panel: Controls
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Grain selection group
        grain_group = QGroupBox("Grain Configuration")
        grain_layout = QFormLayout()
        
        # Grain type selection
        self.grain_type_combo = QComboBox()
        self.grain_type_combo.addItems([
            "BATES", "Star", "C-Slot", "Finocyl", "Wagon Wheel"
        ])
        self.grain_type_combo.currentIndexChanged.connect(self._on_grain_type_changed)
        grain_layout.addRow("Grain Type:", self.grain_type_combo)
        
        # Common grain parameters
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.05, 1.0)
        self.length_spin.setValue(0.2)
        self.length_spin.setSingleStep(0.01)
        self.length_spin.setDecimals(3)
        self.length_spin.setSuffix(" m")
        self.length_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Length:", self.length_spin)
        
        self.outer_diameter_spin = QDoubleSpinBox()
        self.outer_diameter_spin.setRange(0.02, 0.3)
        self.outer_diameter_spin.setValue(0.08)
        self.outer_diameter_spin.setSingleStep(0.005)
        self.outer_diameter_spin.setDecimals(3)
        self.outer_diameter_spin.setSuffix(" m")
        self.outer_diameter_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Outer Diameter:", self.outer_diameter_spin)
        
        self.core_diameter_spin = QDoubleSpinBox()
        self.core_diameter_spin.setRange(0.005, 0.1)
        self.core_diameter_spin.setValue(0.03)
        self.core_diameter_spin.setSingleStep(0.001)
        self.core_diameter_spin.setDecimals(3)
        self.core_diameter_spin.setSuffix(" m")
        self.core_diameter_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Core Diameter:", self.core_diameter_spin)
        
        # BATES specific parameters
        self.segments_spin = QSpinBox()
        self.segments_spin.setRange(1, 10)
        self.segments_spin.setValue(1)
        self.segments_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Number of Segments:", self.segments_spin)
        
        self.segment_spacing_spin = QDoubleSpinBox()
        self.segment_spacing_spin.setRange(0.0, 0.05)
        self.segment_spacing_spin.setValue(0.005)
        self.segment_spacing_spin.setSingleStep(0.001)
        self.segment_spacing_spin.setDecimals(3)
        self.segment_spacing_spin.setSuffix(" m")
        self.segment_spacing_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Segment Spacing:", self.segment_spacing_spin)
        
        # End inhibition
        self.inhibit_ends_check = QCheckBox()
        self.inhibit_ends_check.setChecked(True)
        self.inhibit_ends_check.stateChanged.connect(self._update_grain)
        grain_layout.addRow("Inhibit Ends:", self.inhibit_ends_check)
        
        # Star grain parameters
        self.star_points_spin = QSpinBox()
        self.star_points_spin.setRange(3, 12)
        self.star_points_spin.setValue(5)
        self.star_points_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Number of Points:", self.star_points_spin)
        
        self.star_point_depth_spin = QDoubleSpinBox()
        self.star_point_depth_spin.setRange(0.001, 0.05)
        self.star_point_depth_spin.setValue(0.01)
        self.star_point_depth_spin.setSingleStep(0.001)
        self.star_point_depth_spin.setDecimals(3)
        self.star_point_depth_spin.setSuffix(" m")
        self.star_point_depth_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Point Depth:", self.star_point_depth_spin)
        
        self.star_inner_angle_spin = QDoubleSpinBox()
        self.star_inner_angle_spin.setRange(10.0, 90.0)
        self.star_inner_angle_spin.setValue(60.0)
        self.star_inner_angle_spin.setSingleStep(1.0)
        self.star_inner_angle_spin.setSuffix("°")
        self.star_inner_angle_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Inner Angle:", self.star_inner_angle_spin)
        
        # C-Slot grain parameters
        self.slot_width_spin = QDoubleSpinBox()
        self.slot_width_spin.setRange(0.001, 0.05)
        self.slot_width_spin.setValue(0.01)
        self.slot_width_spin.setSingleStep(0.001)
        self.slot_width_spin.setDecimals(3)
        self.slot_width_spin.setSuffix(" m")
        self.slot_width_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Slot Width:", self.slot_width_spin)
        
        self.slot_depth_spin = QDoubleSpinBox()
        self.slot_depth_spin.setRange(0.001, 0.05)
        self.slot_depth_spin.setValue(0.015)
        self.slot_depth_spin.setSingleStep(0.001)
        self.slot_depth_spin.setDecimals(3)
        self.slot_depth_spin.setSuffix(" m")
        self.slot_depth_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Slot Depth:", self.slot_depth_spin)
        
        self.slot_angle_spin = QDoubleSpinBox()
        self.slot_angle_spin.setRange(30.0, 270.0)
        self.slot_angle_spin.setValue(120.0)
        self.slot_angle_spin.setSingleStep(5.0)
        self.slot_angle_spin.setSuffix("°")
        self.slot_angle_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Slot Angle:", self.slot_angle_spin)
        
        # Finocyl grain parameters
        self.fin_count_spin = QSpinBox()
        self.fin_count_spin.setRange(3, 12)
        self.fin_count_spin.setValue(6)
        self.fin_count_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Number of Fins:", self.fin_count_spin)
        
        self.fin_height_spin = QDoubleSpinBox()
        self.fin_height_spin.setRange(0.001, 0.05)
        self.fin_height_spin.setValue(0.01)
        self.fin_height_spin.setSingleStep(0.001)
        self.fin_height_spin.setDecimals(3)
        self.fin_height_spin.setSuffix(" m")
        self.fin_height_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Fin Height:", self.fin_height_spin)
        
        self.fin_width_spin = QDoubleSpinBox()
        self.fin_width_spin.setRange(0.001, 0.05)
        self.fin_width_spin.setValue(0.005)
        self.fin_width_spin.setSingleStep(0.001)
        self.fin_width_spin.setDecimals(3)
        self.fin_width_spin.setSuffix(" m")
        self.fin_width_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Fin Width:", self.fin_width_spin)
        
        # Wagon Wheel grain parameters
        self.spoke_count_spin = QSpinBox()
        self.spoke_count_spin.setRange(3, 12)
        self.spoke_count_spin.setValue(6)
        self.spoke_count_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Number of Spokes:", self.spoke_count_spin)
        
        self.spoke_width_spin = QDoubleSpinBox()
        self.spoke_width_spin.setRange(1.0, 45.0)
        self.spoke_width_spin.setValue(15.0)
        self.spoke_width_spin.setSingleStep(1.0)
        self.spoke_width_spin.setSuffix("°")
        self.spoke_width_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Spoke Width:", self.spoke_width_spin)
        
        self.spoke_length_spin = QDoubleSpinBox()
        self.spoke_length_spin.setRange(0.001, 0.05)
        self.spoke_length_spin.setValue(0.02)
        self.spoke_length_spin.setSingleStep(0.001)
        self.spoke_length_spin.setDecimals(3)
        self.spoke_length_spin.setSuffix(" m")
        self.spoke_length_spin.valueChanged.connect(self._update_grain)
        grain_layout.addRow("Spoke Length:", self.spoke_length_spin)
        
        grain_group.setLayout(grain_layout)
        
        # Propellant properties group
        propellant_group = QGroupBox("Propellant Properties")
        propellant_layout = QFormLayout()
        
        # Propellant selection
        self.propellant_combo = QComboBox()
        self.propellant_combo.addItems(get_propellant_names())
        self.propellant_combo.addItem("Custom")
        self.propellant_combo.currentIndexChanged.connect(self._on_propellant_changed)
        propellant_layout.addRow("Propellant Type:", self.propellant_combo)
        
        # Propellant density
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(1000.0, 2500.0)
        self.density_spin.setValue(1800.0)
        self.density_spin.setSingleStep(10.0)
        self.density_spin.setSuffix(" kg/m³")
        self.density_spin.valueChanged.connect(self._update_propellant)
        propellant_layout.addRow("Density:", self.density_spin)
        
        # Burn rate coefficient
        self.burn_rate_coef_spin = QDoubleSpinBox()
        self.burn_rate_coef_spin.setRange(0.001, 0.05)
        self.burn_rate_coef_spin.setValue(0.006)
        self.burn_rate_coef_spin.setSingleStep(0.001)
        self.burn_rate_coef_spin.setDecimals(4)
        self.burn_rate_coef_spin.setSuffix(" m/s/(MPa^n)")
        self.burn_rate_coef_spin.valueChanged.connect(self._update_propellant)
        propellant_layout.addRow("Burn Rate Coefficient:", self.burn_rate_coef_spin)
        
        # Burn rate exponent
        self.burn_rate_exp_spin = QDoubleSpinBox()
        self.burn_rate_exp_spin.setRange(0.1, 0.8)
        self.burn_rate_exp_spin.setValue(0.4)
        self.burn_rate_exp_spin.setSingleStep(0.01)
        self.burn_rate_exp_spin.setDecimals(2)
        self.burn_rate_exp_spin.valueChanged.connect(self._update_propellant)
        propellant_layout.addRow("Burn Rate Exponent:", self.burn_rate_exp_spin)
        
        # Temperature sensitivity
        self.temp_sens_spin = QDoubleSpinBox()
        self.temp_sens_spin.setRange(0.0, 2.0)
        self.temp_sens_spin.setValue(0.5)
        self.temp_sens_spin.setSingleStep(0.1)
        self.temp_sens_spin.setDecimals(2)
        self.temp_sens_spin.setSuffix(" %/K")
        self.temp_sens_spin.valueChanged.connect(self._update_propellant)
        propellant_layout.addRow("Temperature Sensitivity:", self.temp_sens_spin)
        
        propellant_group.setLayout(propellant_layout)
        
        # Visualization controls group
        viz_controls_group = QGroupBox("Visualization Controls")
        viz_controls_layout = QFormLayout()
        
        # Web distance slider
        self.web_slider = QSlider(Qt.Orientation.Horizontal)
        self.web_slider.setRange(0, 100)
        self.web_slider.setValue(0)
        self.web_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.web_slider.setTickInterval(10)
        self.web_slider.valueChanged.connect(self._on_web_slider_changed)
        
        # Web distance value display
        self.web_value_label = QLabel("0.000 m")
        
        # Create a layout for the slider and label
        web_layout = QHBoxLayout()
        web_layout.addWidget(self.web_slider)
        web_layout.addWidget(self.web_value_label)
        
        viz_controls_layout.addRow("Web Distance:", web_layout)
        
        # Animation controls
        animation_layout = QHBoxLayout()
        
        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self._on_play_clicked)
        
        self.stop_button = QPushButton("⏹ Stop")
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.stop_button.setEnabled(False)
        
        self.reset_button = QPushButton("⟲ Reset")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        
        animation_layout.addWidget(self.play_button)
        animation_layout.addWidget(self.stop_button)
        animation_layout.addWidget(self.reset_button)
        
        viz_controls_layout.addRow("Animation:", animation_layout)
        
        # Animation speed
        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1x", "2x", "5x"])
        self.speed_combo.setCurrentIndex(2)  # Default to 1x speed
        viz_controls_layout.addRow("Animation Speed:", self.speed_combo)
        
        # Buttons for simulation and analysis
        button_layout = QHBoxLayout()
        
        self.run_sim_button = QPushButton("Run Simulation")
        self.run_sim_button.clicked.connect(self._run_simulation)
        
        self.analyze_button = QPushButton("Analyze Burn Profile")
        self.analyze_button.clicked.connect(self._analyze_burn_profile)
        self.analyze_button.setEnabled(False)
        
        button_layout.addWidget(self.run_sim_button)
        button_layout.addWidget(self.analyze_button)
        
        viz_controls_layout.addRow("", button_layout)
        
        viz_controls_group.setLayout(viz_controls_layout)
        
        # Add groups to top layout
        top_layout.addWidget(grain_group)
        top_layout.addWidget(propellant_group)
        top_layout.addWidget(viz_controls_group)
        
        # Bottom panel: Visualization
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        # Visualization tabs
        self.viz_tabs = QTabWidget()
        
        # 1. Cross-section tab
        cross_section_tab = QWidget()
        cross_section_layout = QVBoxLayout(cross_section_tab)
        
        self.cross_section_fig = Figure(figsize=(6, 6))
        self.cross_section_canvas = FigureCanvas(self.cross_section_fig)
        self.cross_section_toolbar = NavigationToolbar(self.cross_section_canvas, self)
        
        cross_section_layout.addWidget(self.cross_section_toolbar)
        cross_section_layout.addWidget(self.cross_section_canvas)
        
        # 2. 3D Model tab
        model_tab = QWidget()
        model_layout = QVBoxLayout(model_tab)
        
        self.model_fig = Figure(figsize=(6, 6))
        self.model_canvas = FigureCanvas(self.model_fig)
        self.model_toolbar = NavigationToolbar(self.model_canvas, self)
        
        model_layout.addWidget(self.model_toolbar)
        model_layout.addWidget(self.model_canvas)
        
        # 3. Regression tab
        regression_tab = QWidget()
        regression_layout = QVBoxLayout(regression_tab)
        
        self.regression_fig = Figure(figsize=(8, 6))
        self.regression_canvas = FigureCanvas(self.regression_fig)
        self.regression_toolbar = NavigationToolbar(self.regression_canvas, self)
        
        regression_layout.addWidget(self.regression_toolbar)
        regression_layout.addWidget(self.regression_canvas)
        
        # 4. Burn Profile tab
        burn_profile_tab = QWidget()
        burn_profile_layout = QVBoxLayout(burn_profile_tab)
        
        self.burn_profile_fig = Figure(figsize=(8, 6))
        self.burn_profile_canvas = FigureCanvas(self.burn_profile_fig)
        self.burn_profile_toolbar = NavigationToolbar(self.burn_profile_canvas, self)
        
        burn_profile_layout.addWidget(self.burn_profile_toolbar)
        burn_profile_layout.addWidget(self.burn_profile_canvas)
        
        # Add tabs with scroll areas for resizing
        self.viz_tabs.addTab(cross_section_tab, "Cross-section")
        self.viz_tabs.addTab(model_tab, "3D Model")
        self.viz_tabs.addTab(regression_tab, "Regression")
        self.viz_tabs.addTab(burn_profile_tab, "Burn Profile")
        
        bottom_layout.addWidget(self.viz_tabs)
        
        # Add panels to splitter
        self.splitter.addWidget(top_panel)
        self.splitter.addWidget(bottom_panel)
        self.splitter.setSizes([400, 600])  # Initial sizes
        
        # Add splitter to main layout
        main_layout.addWidget(self.splitter)
        
        # Show/hide specific controls based on initial grain type
        self._on_grain_type_changed(0)  # BATES is initially selected (index 0)
        
        # Initialize visualizations
        self._update_grain()
        self._update_cross_section()
        self._update_3d_model()
        self._create_empty_plots()
        
        # Set initial propellant (KNDX is a common choice)
        initial_propellant = "KNDX"
        index = self.propellant_combo.findText(initial_propellant)
        if index >= 0:
            self.propellant_combo.setCurrentIndex(index)
        self._on_propellant_changed(self.propellant_combo.currentIndex())
        
    def _on_grain_type_changed(self, index):
        """Handle grain type selection change."""
        # Hide all grain-specific controls
        self.segments_spin.setVisible(False)
        self.segment_spacing_spin.setVisible(False)
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
        
        # Show grain-specific controls based on selected type
        if index == 0:  # BATES
            self.segments_spin.setVisible(True)
            self.segment_spacing_spin.setVisible(True)
        elif index == 1:  # Star
            self.star_points_spin.setVisible(True)
            self.star_point_depth_spin.setVisible(True)
            self.star_inner_angle_spin.setVisible(True)
        elif index == 2:  # C-Slot
            self.slot_width_spin.setVisible(True)
            self.slot_depth_spin.setVisible(True)
            self.slot_angle_spin.setVisible(True)
        elif index == 3:  # Finocyl
            self.fin_count_spin.setVisible(True)
            self.fin_height_spin.setVisible(True)
            self.fin_width_spin.setVisible(True)
        elif index == 4:  # Wagon Wheel
            self.spoke_count_spin.setVisible(True)
            self.spoke_width_spin.setVisible(True)
            self.spoke_length_spin.setVisible(True)
        
        # Update grain with new type
        self._update_grain()
    
    def _update_grain(self, *args):
        """Update grain with current parameters."""
        grain_type = self.grain_type_combo.currentIndex()
        
        # Common parameters
        length = self.length_spin.value()
        outer_diameter = self.outer_diameter_spin.value()
        core_diameter = self.core_diameter_spin.value()
        inhibited_ends = self.inhibit_ends_check.isChecked()
        
        try:
            # Create appropriate grain type
            if grain_type == 0:  # BATES
                segments = self.segments_spin.value()
                segment_spacing = self.segment_spacing_spin.value()
                
                self.grain = BatesGrain(
                    length=length,
                    outer_diameter=outer_diameter,
                    core_diameter=core_diameter,
                    inhibited_ends=inhibited_ends,
                    inhibited_outer_surface=True,  # Always inhibited for simulation purposes
                    number_of_segments=segments,
                    segment_spacing=segment_spacing
                )
            elif grain_type == 1:  # Star
                points = self.star_points_spin.value()
                point_depth = self.star_point_depth_spin.value()
                inner_angle = self.star_inner_angle_spin.value()
                
                self.grain = StarGrain(
                    length=length,
                    outer_diameter=outer_diameter,
                    core_diameter=core_diameter,
                    inhibited_ends=inhibited_ends,
                    inhibited_outer_surface=True,
                    number_of_points=points,
                    point_depth=point_depth,
                    inner_angle=inner_angle
                )
            elif grain_type == 2:  # C-Slot
                slot_width = self.slot_width_spin.value()
                slot_depth = self.slot_depth_spin.value()
                slot_angle = self.slot_angle_spin.value()
                
                self.grain = CSlotGrain(
                    length=length,
                    outer_diameter=outer_diameter,
                    core_diameter=core_diameter,
                    inhibited_ends=inhibited_ends,
                    inhibited_outer_surface=True,
                    slot_width=slot_width,
                    slot_depth=slot_depth,
                    slot_angle=slot_angle
                )
            elif grain_type == 3:  # Finocyl
                fin_count = self.fin_count_spin.value()
                fin_height = self.fin_height_spin.value()
                fin_width = self.fin_width_spin.value()
                
                self.grain = FinocylGrain(
                    length=length,
                    outer_diameter=outer_diameter,
                    core_diameter=core_diameter,
                    inhibited_ends=inhibited_ends,
                    inhibited_outer_surface=True,
                    number_of_fins=fin_count,
                    fin_height=fin_height,
                    fin_width=fin_width
                )
            elif grain_type == 4:  # Wagon Wheel
                spoke_count = self.spoke_count_spin.value()
                spoke_width = self.spoke_width_spin.value()
                spoke_length = self.spoke_length_spin.value()
                
                self.grain = WagonWheelGrain(
                    length=length,
                    outer_diameter=outer_diameter,
                    core_diameter=core_diameter,
                    inhibited_ends=inhibited_ends,
                    inhibited_outer_surface=True,
                    number_of_spokes=spoke_count,
                    spoke_width=spoke_width,
                    spoke_length=spoke_length
                )
            
            # Update visualizations
            self._update_cross_section()
            self._update_3d_model()
            
            # Reset web slider range
            max_web = (outer_diameter - core_diameter) / 2
            self.web_slider.setMaximum(100)  # Percentage of max web
            self.web_value_label.setText(f"0.000 m of {max_web:.3f} m")
            
            # Reset simulation data
            self.simulation = None
            self.analyze_button.setEnabled(False)
            
        except Exception as e:
            logger.error(f"Error updating grain: {e}")
            QMessageBox.warning(self, "Error", f"Failed to update grain: {e}")
    
    def _on_propellant_changed(self, index):
        """Handle propellant selection change."""
        propellant_name = self.propellant_combo.currentText()
        
        # Enable/disable property controls based on selection
        is_custom = (propellant_name == "Custom")
        self.density_spin.setEnabled(is_custom)
        self.burn_rate_coef_spin.setEnabled(is_custom)
        self.burn_rate_exp_spin.setEnabled(is_custom)
        self.temp_sens_spin.setEnabled(is_custom)
        
        if not is_custom:
            # Load selected propellant properties
            propellant = get_propellant(propellant_name)
            if propellant:
                self.density_spin.setValue(propellant.density)
                self.burn_rate_coef_spin.setValue(propellant.burn_rate_coefficient)
                self.burn_rate_exp_spin.setValue(propellant.burn_rate_exponent)
                self.temp_sens_spin.setValue(propellant.temperature_sensitivity)
        
        self._update_propellant()
    
    def _update_propellant(self, *args):
        """Update propellant properties."""
        try:
            propellant_name = self.propellant_combo.currentText()
            
            if propellant_name != "Custom" and propellant_name in PROPELLANT_LIBRARY:
                # Use predefined propellant
                self.propellant = PROPELLANT_LIBRARY[propellant_name]
            else:
                # Use custom propellant values
                density = self.density_spin.value()
                burn_rate_coef = self.burn_rate_coef_spin.value()
                burn_rate_exp = self.burn_rate_exp_spin.value()
                temp_sens = self.temp_sens_spin.value()
                
                self.propellant = PropellantProperties(
                    name="Custom",
                    density=density,
                    burn_rate_coefficient=burn_rate_coef,
                    burn_rate_exponent=burn_rate_exp,
                    temperature_sensitivity=temp_sens,
                    reference_temperature=298.15  # Ambient temperature (K)
                )
            
            # Reset simulation data
            self.simulation = None
            self.analyze_button.setEnabled(False)
            
        except Exception as e:
            logger.error(f"Error updating propellant: {e}")
            QMessageBox.warning(self, "Error", f"Failed to update propellant: {e}")
    
    def _update_cross_section(self):
        """Update cross-section visualization."""
        try:
            # Clear figure
            self.cross_section_fig.clear()
            ax = self.cross_section_fig.add_subplot(111, aspect='equal')
            
            # Get web distance from slider (as percentage of max)
            max_web = (self.grain.outer_diameter - self.grain.core_diameter) / 2
            web_fraction = self.web_slider.value() / 100.0
            web_distance = web_fraction * max_web
            
            # Generate and plot cross-section
            generate_grain_cross_section(
                grain=self.grain,
                web_distance=web_distance,
                ax=ax
            )
            
            # Add title
            ax.set_title(f"Grain Cross-section at Web = {web_distance:.3f} m")
            self.cross_section_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating cross-section: {e}")
            self.cross_section_fig.clear()
            ax = self.cross_section_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            self.cross_section_canvas.draw()
    
    def _update_3d_model(self):
        """Update 3D model visualization."""
        try:
            # Clear figure
            self.model_fig.clear()
            ax = self.model_fig.add_subplot(111, projection='3d')
            
            # Get web distance from slider (as percentage of max)
            max_web = (self.grain.outer_diameter - self.grain.core_diameter) / 2
            web_fraction = self.web_slider.value() / 100.0
            web_distance = web_fraction * max_web
            
            # Generate and plot 3D model
            create_3d_grain_model(
                grain=self.grain,
                web_distance=web_distance,
                ax=ax
            )
            
            # Add title
            ax.set_title(f"3D Grain Model at Web = {web_distance:.3f} m")
            self.model_canvas.draw()
            
        except Exception as e:
            logger.error(f"Error updating 3D model: {e}")
            self.model_fig.clear()
            ax = self.model_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            self.model_canvas.draw()
    
    def _create_empty_plots(self):
        """Create empty plots for regression and burn profile tabs."""
        # Regression plot
        self.regression_fig.clear()
        ax1 = self.regression_fig.add_subplot(121)
        ax1.set_title("Burn Area vs. Web Distance")
        ax1.set_xlabel("Web Distance (m)")
        ax1.set_ylabel("Burn Area (m²)")
        ax1.grid(True)
        
        ax2 = self.regression_fig.add_subplot(122)
        ax2.set_title("Grain Volume vs. Web Distance")
        ax2.set_xlabel("Web Distance (m)")
        ax2.set_ylabel("Grain Volume (m³)")
        ax2.grid(True)
        
        self.regression_canvas.draw()
        
        # Burn profile plot
        self.burn_profile_fig.clear()
        ax1 = self.burn_profile_fig.add_subplot(221)
        ax1.set_title("Thrust vs. Time")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Thrust (N)")
        ax1.grid(True)
        
        ax2 = self.burn_profile_fig.add_subplot(222)
        ax2.set_title("Chamber Pressure vs. Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Pressure (MPa)")
        ax2.grid(True)
        
        ax3 = self.burn_profile_fig.add_subplot(223)
        ax3.set_title("Burn Rate vs. Time")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Burn Rate (mm/s)")
        ax3.grid(True)
        
        ax4 = self.burn_profile_fig.add_subplot(224)
        ax4.set_title("Specific Impulse vs. Time")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Specific Impulse (s)")
        ax4.grid(True)
        
        self.burn_profile_fig.tight_layout()
        self.burn_profile_canvas.draw()
    
    def _on_web_slider_changed(self, value):
        """Handle web slider value change."""
        max_web = (self.grain.outer_diameter - self.grain.core_diameter) / 2
        web_fraction = value / 100.0
        web_distance = web_fraction * max_web
        
        # Update label
        self.web_value_label.setText(f"{web_distance:.3f} m of {max_web:.3f} m")
        
        # Update visualizations
        self._update_cross_section()
        self._update_3d_model()
    
    def _on_play_clicked(self):
        """Start animation of grain regression."""
        if not self.simulation:
            QMessageBox.warning(self, "No Simulation", "Please run a simulation first.")
            return
        
        # Setup animation timer
        self.animation_frame_index = 0
        
        # Get animation speed
        speed_text = self.speed_combo.currentText()
        speed_factor = float(speed_text.strip("x"))
        
        # Set timer interval based on speed (100ms for 1x)
        interval = int(100 / speed_factor)
        self.animation_timer.setInterval(interval)
        self.animation_timer.start()
        
        # Update button states
        self.play_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.run_sim_button.setEnabled(False)
    
    def _on_stop_clicked(self):
        """Stop animation of grain regression."""
        self.animation_timer.stop()
        
        # Update button states
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.run_sim_button.setEnabled(True)
    
    def _on_reset_clicked(self):
        """Reset web distance to zero."""
        self.animation_timer.stop()
        self.web_slider.setValue(0)
        
        # Update button states
        self.play_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.run_sim_button.setEnabled(True)
    
    def _animation_step(self):
        """Handle animation step."""
        if not self.simulation or self.animation_frame_index >= len(self.simulation.web_distances):
            self.animation_timer.stop()
            self.play_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            return
        
        # Get current web distance
        web_distance = self.simulation.web_distances[self.animation_frame_index]
        
        # Calculate slider position
        max_web = (self.grain.outer_diameter - self.grain.core_diameter) / 2
        slider_value = int(100 * web_distance / max_web) if max_web > 0 else 0
        self.web_slider.setValue(min(slider_value, 100))
        
        # Increment frame index
        self.animation_frame_index += 1
    
    def _run_simulation(self):
        """Run grain regression simulation."""
        try:
            # Get chamber pressure from propellant and nozzle parameters
            # In a real implementation, these would come from the motor design
            # For now, we'll use a fixed value for testing
            chamber_pressure = 5.0  # MPa
            
            # Create simulation object
            self.simulation = GrainRegressionSimulation(
                grain=self.grain,
                propellant=self.propellant,
                chamber_pressure=chamber_pressure,
                time_step=0.1,  # seconds
                max_time=10.0    # seconds
            )
            
            # Run simulation
            self.simulation.run()
            
            # Update regression plots
            self._update_regression_plots()
            
            # Enable analyze button
            self.analyze_button.setEnabled(True)
            
            # Switch to regression tab
            self.viz_tabs.setCurrentIndex(2)
            
            # Show success message
            QMessageBox.information(
                self, "Simulation Complete", 
                f"Simulation completed successfully.\n"
                f"Total burn time: {self.simulation.burn_time:.2f} s\n"
                f"Max thrust: {max(self.simulation.thrust):.2f} N"
            )
            
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            QMessageBox.warning(self, "Error", f"Failed to run simulation: {e}")
    
    def _update_regression_plots(self):
        """Update regression plots with simulation results."""
        if not self.simulation:
            return
        
        # Clear figure
        self.regression_fig.clear()
        
        # Burn area vs. web distance
        ax1 = self.regression_fig.add_subplot(121)
        ax1.plot(self.simulation.web_distances, self.simulation.burn_areas, 'b-')
        ax1.set_title("Burn Area vs. Web Distance")
        ax1.set_xlabel("Web Distance (m)")
        ax1.set_ylabel("Burn Area (m²)")
        ax1.grid(True)
        
        # Grain volume vs. web distance
        ax2 = self.regression_fig.add_subplot(122)
        ax2.plot(self.simulation.web_distances, self.simulation.grain_volumes, 'r-')
        ax2.set_title("Grain Volume vs. Web Distance")
        ax2.set_xlabel("Web Distance (m)")
        ax2.set_ylabel("Grain Volume (m³)")
        ax2.grid(True)
        
        self.regression_fig.tight_layout()
        self.regression_canvas.draw()
    
    def _analyze_burn_profile(self):
        """Analyze burn profile and visualize results."""
        if not self.simulation:
            QMessageBox.warning(self, "No Simulation", "Please run a simulation first.")
            return
        
        try:
            # Determine burn profile type
            profile_type = self.simulation.determine_burn_profile_type()
            
            # Update burn profile plots
            self._update_burn_profile_plots(profile_type)
            
            # Switch to burn profile tab
            self.viz_tabs.setCurrentIndex(3)
            
        except Exception as e:
            logger.error(f"Error analyzing burn profile: {e}")
            QMessageBox.warning(self, "Error", f"Failed to analyze burn profile: {e}")
    
    def _update_burn_profile_plots(self, profile_type):
        """Update burn profile plots with simulation results."""
        if not self.simulation:
            return
        
        # Clear figure
        self.burn_profile_fig.clear()
        
        # Thrust vs. time
        ax1 = self.burn_profile_fig.add_subplot(221)
        ax1.plot(self.simulation.times, self.simulation.thrust, 'b-')
        ax1.set_title("Thrust vs. Time")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Thrust (N)")
        ax1.grid(True)
        
        # Chamber pressure vs. time
        ax2 = self.burn_profile_fig.add_subplot(222)
        ax2.plot(self.simulation.times, self.simulation.chamber_pressures, 'r-')
        ax2.set_title("Chamber Pressure vs. Time")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Pressure (MPa)")
        ax2.grid(True)
        
        # Burn rate vs. time
        ax3 = self.burn_profile_fig.add_subplot(223)
        ax3.plot(self.simulation.times, [r * 1000 for r in self.simulation.burn_rates], 'g-')
        ax3.set_title("Burn Rate vs. Time")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Burn Rate (mm/s)")
        ax3.grid(True)
        
        # Specific impulse vs. time (if available)
        ax4 = self.burn_profile_fig.add_subplot(224)
        if hasattr(self.simulation, 'specific_impulses'):
            ax4.plot(self.simulation.times, self.simulation.specific_impulses, 'm-')
        else:
            # Placeholder for specific impulse (could be calculated in a real implementation)
            isp = [220 for _ in self.simulation.times]  # Example value
            ax4.plot(self.simulation.times, isp, 'm-')
        ax4.set_title(f"Burn Profile: {profile_type}")
        ax4.set_xlabel("Time (s)")
        ax4.set_ylabel("Specific Impulse (s)")
        ax4.grid(True)
        
        self.burn_profile_fig.tight_layout()
        self.burn_profile_canvas.draw()

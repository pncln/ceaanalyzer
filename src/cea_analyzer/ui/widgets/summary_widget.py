"""
Summary Widget
--------------

Widget for displaying summary information about CEA analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox,
    QScrollArea, QTableWidget, QTableWidgetItem, QSizePolicy,
    QHeaderView, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont


class SummaryWidget(QWidget):
    """
    Widget for displaying summary information about CEA analysis.
    """
    
    def __init__(self, parent=None):
        """Initialize the summary widget."""
        super().__init__(parent)
        
        # Data
        self.cea_data = None
        
        # Setup UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Title
        title_label = QLabel("CEA Analysis Summary")
        font = QFont("Arial", 14)
        font.setWeight(QFont.Weight.Bold)
        title_label.setFont(font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Scroll area for the summary content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Content widget
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        scroll_area.setWidget(content_widget)
        
        # Propellant group
        self.propellant_group = QGroupBox("Propellant")
        propellant_layout = QVBoxLayout()
        
        self.propellant_label = QLabel("No propellant data available.")
        self.propellant_label.setWordWrap(True)
        propellant_layout.addWidget(self.propellant_label)
        
        self.propellant_group.setLayout(propellant_layout)
        
        # Performance group
        self.performance_group = QGroupBox("Performance")
        performance_layout = QVBoxLayout()
        
        self.performance_table = QTableWidget(0, 2)
        self.performance_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.performance_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.performance_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.performance_table.verticalHeader().setVisible(False)
        
        performance_layout.addWidget(self.performance_table)
        self.performance_group.setLayout(performance_layout)
        
        # Nozzle group
        self.nozzle_group = QGroupBox("Nozzle Geometry")
        nozzle_layout = QVBoxLayout()
        
        self.nozzle_table = QTableWidget(0, 2)
        self.nozzle_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.nozzle_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.nozzle_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.nozzle_table.verticalHeader().setVisible(False)
        
        nozzle_layout.addWidget(self.nozzle_table)
        self.nozzle_group.setLayout(nozzle_layout)
        
        # Thermodynamics group
        self.thermo_group = QGroupBox("Thermodynamic Properties")
        thermo_layout = QVBoxLayout()
        
        self.thermo_table = QTableWidget(0, 2)
        self.thermo_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.thermo_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.thermo_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.thermo_table.verticalHeader().setVisible(False)
        
        thermo_layout.addWidget(self.thermo_table)
        self.thermo_group.setLayout(thermo_layout)
        
        # Add groups to content layout
        content_layout.addWidget(self.propellant_group)
        content_layout.addWidget(self.performance_group)
        content_layout.addWidget(self.nozzle_group)
        content_layout.addWidget(self.thermo_group)
        content_layout.addStretch()
        
        # Add widgets to main layout
        main_layout.addWidget(title_label)
        main_layout.addWidget(scroll_area)
        
        # Initial state
        self._update_display()
        
    def set_data(self, df: pd.DataFrame, cea_data: Optional[Dict[str, Any]] = None):
        """Set the data for summary display (interface consistent with other widgets)."""
        # If cea_data is not provided, create a minimal dict with just the dataframe
        if cea_data is None:
            cea_data = {"data": df}
        else:
            # Ensure the dataframe is in the cea_data
            cea_data["data"] = df
            
        # Call the detailed method
        self.set_cea_data(cea_data)
        
    def set_cea_data(self, cea_data: Dict[str, Any]):
        """Set the CEA data and update the display."""
        # Store data
        self.cea_data = cea_data
        
        # Update display
        self._update_display()
        
    def _update_display(self):
        """Update the display with current CEA data."""
        if self.cea_data is None:
            # Clear tables
            self._clear_tables()
            self.propellant_label.setText("No propellant data available.")
            return
        
        # Update propellant information
        self._update_propellant_info()
        
        # Update performance table
        self._update_performance_table()
        
        # Update nozzle table
        self._update_nozzle_table()
        
        # Update thermodynamics table
        self._update_thermo_table()
        
    def _clear_tables(self):
        """Clear all tables."""
        self.performance_table.setRowCount(0)
        self.nozzle_table.setRowCount(0)
        self.thermo_table.setRowCount(0)
        
    def _update_propellant_info(self):
        """Update propellant information."""
        if 'propellant' not in self.cea_data:
            self.propellant_label.setText("No propellant data available.")
            return
        
        propellant = self.cea_data['propellant']
        
        if isinstance(propellant, dict):
            # Format propellant information as HTML
            html = "<b>Propellant:</b><br>"
            
            if 'fuel' in propellant:
                html += f"<b>Fuel:</b> {propellant['fuel']}<br>"
            
            if 'oxidizer' in propellant:
                html += f"<b>Oxidizer:</b> {propellant['oxidizer']}<br>"
            
            if 'o_f' in self.cea_data:
                html += f"<b>O/F Ratio:</b> {self.cea_data['o_f']:.2f}<br>"
            
            self.propellant_label.setText(html)
        else:
            self.propellant_label.setText(f"<b>Propellant:</b> {propellant}")
        
    def _update_performance_table(self):
        """Update performance table with CEA data."""
        # Clear table
        self.performance_table.setRowCount(0)
        
        # Performance parameters to display
        performance_params = [
            ('isp', 'Specific Impulse (s)'),
            ('isp_vac', 'Vacuum Specific Impulse (s)'),
            ('thrust_coefficient', 'Thrust Coefficient'),
            ('thrust', 'Thrust (N)'),
            ('c_star', 'Characteristic Velocity (m/s)'),
            ('mach_exit', 'Exit Mach Number'),
            ('area_ratio', 'Area Ratio (Ae/At)'),
            ('pressure_ratio', 'Pressure Ratio (Pe/Pc)')
        ]
        
        # Add rows to table
        for i, (key, label) in enumerate(performance_params):
            if key in self.cea_data:
                self.performance_table.insertRow(i)
                
                # Parameter name
                name_item = QTableWidgetItem(label)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.performance_table.setItem(i, 0, name_item)
                
                # Parameter value
                value = self.cea_data[key]
                if isinstance(value, (int, float)):
                    value_text = f"{value:.4g}"
                else:
                    value_text = str(value)
                    
                value_item = QTableWidgetItem(value_text)
                value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.performance_table.setItem(i, 1, value_item)
        
    def _update_nozzle_table(self):
        """Update nozzle table with CEA data."""
        # Clear table
        self.nozzle_table.setRowCount(0)
        
        # Nozzle parameters to display
        nozzle_params = [
            ('throat_diameter', 'Throat Diameter (m)'),
            ('exit_diameter', 'Exit Diameter (m)'),
            ('expansion_ratio', 'Expansion Ratio'),
            ('contraction_ratio', 'Contraction Ratio'),
            ('nozzle_length', 'Nozzle Length (m)'),
            ('divergence_angle', 'Divergence Angle (deg)'),
            ('throat_radius_curvature', 'Throat Radius of Curvature (m)')
        ]
        
        # Add rows to table
        for i, (key, label) in enumerate(nozzle_params):
            if key in self.cea_data:
                self.nozzle_table.insertRow(i)
                
                # Parameter name
                name_item = QTableWidgetItem(label)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.nozzle_table.setItem(i, 0, name_item)
                
                # Parameter value
                value = self.cea_data[key]
                if isinstance(value, (int, float)):
                    value_text = f"{value:.4g}"
                else:
                    value_text = str(value)
                    
                value_item = QTableWidgetItem(value_text)
                value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.nozzle_table.setItem(i, 1, value_item)
        
    def _update_thermo_table(self):
        """Update thermodynamics table with CEA data."""
        # Clear table
        self.thermo_table.setRowCount(0)
        
        # Thermodynamic parameters to display
        thermo_params = [
            ('p_chamber', 'Chamber Pressure (bar)'),
            ('t_chamber', 'Chamber Temperature (K)'),
            ('p_exit', 'Exit Pressure (bar)'),
            ('t_exit', 'Exit Temperature (K)'),
            ('gamma', 'Specific Heat Ratio (γ)'),
            ('mw', 'Molecular Weight (g/mol)'),
            ('rho', 'Density (kg/m³)'),
            ('viscosity', 'Viscosity (Pa·s)')
        ]
        
        # Add rows to table
        for i, (key, label) in enumerate(thermo_params):
            if key in self.cea_data:
                self.thermo_table.insertRow(i)
                
                # Parameter name
                name_item = QTableWidgetItem(label)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.thermo_table.setItem(i, 0, name_item)
                
                # Parameter value
                value = self.cea_data[key]
                if isinstance(value, (int, float)):
                    value_text = f"{value:.4g}"
                else:
                    value_text = str(value)
                    
                value_item = QTableWidgetItem(value_text)
                value_item.setFlags(value_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.thermo_table.setItem(i, 1, value_item)
    
    def update_nozzle(self, nozzle_data: Dict[str, Any]):
        """Update the nozzle design data in the summary."""
        if nozzle_data is None:
            return
            
        # Store the nozzle data in the cea_data dictionary
        if self.cea_data is None:
            self.cea_data = {}
            
        self.cea_data["nozzle"] = nozzle_data
        
        # Update the nozzle table
        self._update_nozzle_table()
        
    def update_motor(self, motor_design: Any):
        """Update the motor design data in the summary."""
        if motor_design is None:
            return
            
        # Store the motor design data in the cea_data dictionary
        if self.cea_data is None:
            self.cea_data = {}
            
        self.cea_data["motor"] = motor_design
        
        # If there's a specific method to update motor data in the UI, call it here
        # For now, we'll just update the performance table which might contain some motor stats
        self._update_performance_table()

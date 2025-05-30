"""
Data Table Widget
---------------

Widget for displaying and interacting with tabular CEA data.
"""

import pandas as pd
from typing import Dict, List, Optional

from PyQt5.QtWidgets import (
    QWidget, QTableView, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QLineEdit, QPushButton, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal

from ...core.models import PandasModel


class DataTableWidget(QWidget):
    """
    Widget for displaying CEA analysis data in a table with filtering capabilities.
    """
    
    # Signal emitted when data is filtered/updated
    data_updated = pyqtSignal(pd.DataFrame)
    
    def __init__(self, parent=None):
        """Initialize the data table widget."""
        super().__init__(parent)
        
        # Data
        self.df_full = None
        self.df = None
        self.model = None
        
        # Setup UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Filter controls
        filter_group = QGroupBox("Data Filters")
        filter_layout = QFormLayout()
        
        # O/F filter
        self.of_min = QLineEdit()
        self.of_max = QLineEdit()
        filter_layout.addRow("O/F Minimum:", self.of_min)
        filter_layout.addRow("O/F Maximum:", self.of_max)
        
        # Pc filter
        self.pc_min = QLineEdit()
        self.pc_max = QLineEdit()
        filter_layout.addRow("Pc (bar) Minimum:", self.pc_min)
        filter_layout.addRow("Pc (bar) Maximum:", self.pc_max)
        
        # Isp filter
        self.isp_min = QLineEdit()
        self.isp_max = QLineEdit()
        filter_layout.addRow("Isp (s) Minimum:", self.isp_min)
        filter_layout.addRow("Isp (s) Maximum:", self.isp_max)
        
        # Filter buttons
        button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply Filters")
        self.apply_button.clicked.connect(self._apply_filters)
        self.reset_button = QPushButton("Reset Filters")
        self.reset_button.clicked.connect(self._reset_filters)
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.reset_button)
        
        filter_layout.addRow(button_layout)
        filter_group.setLayout(filter_layout)
        
        # Table view
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        
        # Info label
        self.info_label = QLabel("No data loaded")
        
        # Add to main layout
        main_layout.addWidget(filter_group)
        main_layout.addWidget(self.table_view)
        main_layout.addWidget(self.info_label)
        
        # Disable controls until data is loaded
        self._set_controls_enabled(False)
        
    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable filter controls."""
        self.of_min.setEnabled(enabled)
        self.of_max.setEnabled(enabled)
        self.pc_min.setEnabled(enabled)
        self.pc_max.setEnabled(enabled)
        self.isp_min.setEnabled(enabled)
        self.isp_max.setEnabled(enabled)
        self.apply_button.setEnabled(enabled)
        self.reset_button.setEnabled(enabled)
        
    def set_data(self, df: pd.DataFrame):
        """Set the data for the table view."""
        if df is None or df.empty:
            self._set_controls_enabled(False)
            self.info_label.setText("No data loaded")
            return
            
        # Store data
        self.df_full = self.df = df
        
        # Create and set model
        self.model = PandasModel(df)
        self.table_view.setModel(self.model)
        
        # Resize columns to content
        self.table_view.resizeColumnsToContents()
        
        # Enable controls
        self._set_controls_enabled(True)
        
        # Update info label
        self.info_label.setText(f"Showing {len(df)} records")
        
        # Emit signal
        self.data_updated.emit(df)
        
    def _apply_filters(self):
        """Apply filters to the data."""
        if self.df_full is None:
            return
            
        # Start with full dataset
        filtered_df = self.df_full
        
        # Apply O/F filter
        if self.of_min.text():
            try:
                min_val = float(self.of_min.text())
                filtered_df = filtered_df[filtered_df["O/F"] >= min_val]
            except ValueError:
                pass
                
        if self.of_max.text():
            try:
                max_val = float(self.of_max.text())
                filtered_df = filtered_df[filtered_df["O/F"] <= max_val]
            except ValueError:
                pass
                
        # Apply Pc filter
        if self.pc_min.text():
            try:
                min_val = float(self.pc_min.text())
                filtered_df = filtered_df[filtered_df["Pc (bar)"] >= min_val]
            except ValueError:
                pass
                
        if self.pc_max.text():
            try:
                max_val = float(self.pc_max.text())
                filtered_df = filtered_df[filtered_df["Pc (bar)"] <= max_val]
            except ValueError:
                pass
                
        # Apply Isp filter
        if self.isp_min.text():
            try:
                min_val = float(self.isp_min.text())
                filtered_df = filtered_df[filtered_df["Isp (s)"] >= min_val]
            except ValueError:
                pass
                
        if self.isp_max.text():
            try:
                max_val = float(self.isp_max.text())
                filtered_df = filtered_df[filtered_df["Isp (s)"] <= max_val]
            except ValueError:
                pass
                
        # Update table
        self.df = filtered_df
        self.model = PandasModel(filtered_df)
        self.table_view.setModel(self.model)
        
        # Resize columns to content
        self.table_view.resizeColumnsToContents()
        
        # Update info label
        self.info_label.setText(f"Showing {len(filtered_df)} of {len(self.df_full)} records")
        
        # Emit signal
        self.data_updated.emit(filtered_df)
        
    def _reset_filters(self):
        """Reset all filters."""
        self.of_min.clear()
        self.of_max.clear()
        self.pc_min.clear()
        self.pc_max.clear()
        self.isp_min.clear()
        self.isp_max.clear()
        
        # Reset to full dataset
        if self.df_full is not None:
            self.df = self.df_full
            self.model = PandasModel(self.df_full)
            self.table_view.setModel(self.model)
            
            # Resize columns to content
            self.table_view.resizeColumnsToContents()
            
            # Update info label
            self.info_label.setText(f"Showing {len(self.df_full)} records")
            
            # Emit signal
            self.data_updated.emit(self.df_full)

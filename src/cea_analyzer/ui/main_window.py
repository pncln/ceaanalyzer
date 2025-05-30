"""
Main Window Module for CEA Analyzer
---------------------------------

This module implements the main application window following modern UI design principles.
It uses a clean, professional interface tailored for engineering applications.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QMainWindow, QApplication, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QToolBar, QStatusBar, QProgressBar, QFileDialog,
    QMessageBox, QLabel, QComboBox, QDockWidget, QTableView, QMenu
)
from PyQt6.QtCore import Qt, QSize, QSettings, pyqtSignal, QThread
from PyQt6.QtGui import QIcon, QPixmap, QFont, QAction
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices

from ..core.config import CONFIG, CONFIG_PATH
from ..core.logger import get_logger
from ..core.models import PandasModel
from ..analysis.cea_parser import parse_cea_output, extract_thermo_data
from ..analysis import compute_system
from ..utils.threads import ParserThread
from ..utils.plots import create_graphs, create_optimization_plot
from ..utils.export import export_csv, export_excel, export_pdf
from ..propulsion import nozzle
from ..propulsion.motor.design import MotorDesign
from ..propulsion.motor.types import MotorType

# Import UI components
from .widgets.data_table_widget import DataTableWidget
from .widgets.plotting_widget import PlottingWidget
from .widgets.nozzle_design_widget import NozzleDesignWidget
from .widgets.motor_design_widget import MotorDesignWidget
from .widgets.optimization_widget import OptimizationWidget
from .widgets.summary_widget import SummaryWidget
from .dialogs.settings_dialog import SettingsDialog
from .dialogs.about_dialog import AboutDialog

# Setup logger
logger = get_logger(__name__)


class MainWindow(QMainWindow):
    """
    Main application window for CEA Analyzer.
    
    This window provides a modern, professional interface for analyzing rocket propulsion
    data from NASA-CEA output files. It includes advanced features for nozzle design,
    motor design, performance optimization, and data visualization.
    """
    
    def __init__(self):
        """Initialize the main window with a modern UI layout."""
        super().__init__()
        
        # Setup window properties
        self.setWindowTitle("CEA Analyzer")
        self.resize(1200, 800)
        self.setMinimumSize(800, 600)
        
        # Initialize data containers
        self.df_full = None  # Full dataset
        self.df = None       # Filtered dataset
        self.cea_data = None  # Extracted thermochemical data
        self.motor_design = None  # Current motor design
        
        # Create UI components
        self._create_actions()
        self._create_menu_bar()
        self._create_toolbars()
        self._create_status_bar()
        self._create_central_widget()
        self._create_dock_widgets()
        
        # Connect signals and slots
        self._connect_signals()
        
        # Load settings
        self._load_settings()
        
        # Show welcome message
        self.statusBar().showMessage("Welcome to CEA Analyzer. Open a CEA output file to begin analysis.")
        
    def _create_actions(self):
        """Create application actions with modern icons and shortcuts."""
        # File actions
        self.open_action = QAction("&Open CEA File", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.setStatusTip("Open a NASA-CEA output file")
        self.open_action.triggered.connect(self.open_file)
        
        self.save_project_action = QAction("&Save Project", self)
        self.save_project_action.setShortcut("Ctrl+S")
        self.save_project_action.setStatusTip("Save current project")
        self.save_project_action.triggered.connect(self.save_project)
        
        self.load_project_action = QAction("&Load Project", self)
        self.load_project_action.setShortcut("Ctrl+L")
        self.load_project_action.setStatusTip("Load a saved project")
        self.load_project_action.triggered.connect(self.load_project)
        
        # Export actions
        self.export_csv_action = QAction("Export to &CSV", self)
        self.export_csv_action.setStatusTip("Export data to CSV file")
        self.export_csv_action.triggered.connect(self.export_csv)
        
        self.export_excel_action = QAction("Export to &Excel", self)
        self.export_excel_action.setStatusTip("Export data to Excel file")
        self.export_excel_action.triggered.connect(self.export_excel)
        
        self.export_pdf_action = QAction("Export to &PDF", self)
        self.export_pdf_action.setStatusTip("Export report to PDF file")
        self.export_pdf_action.triggered.connect(self.export_pdf)
        
        # Settings actions
        self.settings_action = QAction("Se&ttings", self)
        self.settings_action.setStatusTip("Configure application settings")
        self.settings_action.triggered.connect(self.open_settings)
        
        # Help actions
        self.about_action = QAction("&About", self)
        self.about_action.setStatusTip("Show information about CEA Analyzer")
        self.about_action.triggered.connect(self.show_about)
        
        self.help_action = QAction("&Help", self)
        self.help_action.setShortcut("F1")
        self.help_action.setStatusTip("Open user manual")
        self.help_action.triggered.connect(self.open_help)
        
        # Design actions
        self.create_motor_action = QAction("Create &Motor Design", self)
        self.create_motor_action.setStatusTip("Create a new motor design")
        self.create_motor_action.triggered.connect(self.create_motor_design)
        
        self.optimize_action = QAction("&Optimize Design", self)
        self.optimize_action.setStatusTip("Optimize current design parameters")
        self.optimize_action.triggered.connect(self.optimize_design)
        
    def _create_menu_bar(self):
        """Create the application menu bar with modern organization."""
        # File menu
        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.save_project_action)
        file_menu.addAction(self.load_project_action)
        file_menu.addSeparator()
        
        # Export submenu
        export_menu = file_menu.addMenu("&Export")
        export_menu.addAction(self.export_csv_action)
        export_menu.addAction(self.export_excel_action)
        export_menu.addAction(self.export_pdf_action)
        
        file_menu.addSeparator()
        file_menu.addAction(self.settings_action)
        file_menu.addSeparator()
        file_menu.addAction(QAction("E&xit", self, triggered=self.close))
        
        # Design menu
        design_menu = self.menuBar().addMenu("&Design")
        design_menu.addAction(self.create_motor_action)
        design_menu.addAction(self.optimize_action)
        
        # Analysis menu
        analysis_menu = self.menuBar().addMenu("&Analysis")
        analysis_menu.addAction(QAction("Performance &Analysis", self, 
                                        triggered=self.show_performance_analysis))
        analysis_menu.addAction(QAction("&Sensitivity Analysis", self,
                                       triggered=self.show_sensitivity_analysis))
        
        # Help menu
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.help_action)
        help_menu.addAction(self.about_action)
        
    def _create_toolbars(self):
        """Create the application toolbars with modern icons."""
        # Main toolbar
        self.main_toolbar = QToolBar("Main", self)
        self.main_toolbar.setObjectName("main_toolbar")
        self.main_toolbar.setMovable(False)
        self.main_toolbar.setIconSize(QSize(24, 24))
        self.main_toolbar.addAction(self.open_action)
        self.main_toolbar.addAction(self.save_project_action)
        self.main_toolbar.addAction(self.load_project_action)
        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self.create_motor_action)
        self.main_toolbar.addAction(self.optimize_action)
        self.main_toolbar.addSeparator()
        self.main_toolbar.addAction(self.help_action)
        
        self.addToolBar(self.main_toolbar)
        
    def _create_status_bar(self):
        """Create a modern status bar with progress indicator."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMaximumHeight(15)
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
        
    def _create_central_widget(self):
        """Create the central widget with tabbed interface."""
        self.central_tabs = QTabWidget()
        self.central_tabs.setTabPosition(QTabWidget.TabPosition.North)
        self.central_tabs.setMovable(True)
        self.central_tabs.setDocumentMode(True)
        
        # Create tab widgets
        self.data_widget = DataTableWidget()
        self.plotting_widget = PlottingWidget()
        self.nozzle_widget = NozzleDesignWidget()
        self.motor_widget = MotorDesignWidget()
        self.optimization_widget = OptimizationWidget()
        self.summary_widget = SummaryWidget()
        
        # Add tabs
        self.central_tabs.addTab(self.data_widget, "Data")
        self.central_tabs.addTab(self.plotting_widget, "Plots")
        self.central_tabs.addTab(self.nozzle_widget, "Nozzle Design")
        self.central_tabs.addTab(self.motor_widget, "Motor Design")
        self.central_tabs.addTab(self.optimization_widget, "Optimization")
        self.central_tabs.addTab(self.summary_widget, "Summary")
        
        self.setCentralWidget(self.central_tabs)
        
    def _create_dock_widgets(self):
        """Create dockable widgets for filters and properties."""
        # Filter dock widget
        self.filter_dock = QDockWidget("Filters", self)
        self.filter_dock.setObjectName("filter_dock")
        self.filter_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        
        # Create filter widget (will be implemented in widgets package)
        filter_widget = QWidget()
        self.filter_dock.setWidget(filter_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.filter_dock)
        
        # Properties dock widget
        self.properties_dock = QDockWidget("Properties", self)
        self.properties_dock.setObjectName("properties_dock")
        self.properties_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        
        # Create properties widget (will be implemented in widgets package)
        properties_widget = QWidget()
        self.properties_dock.setWidget(properties_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.properties_dock)
        
    def _connect_signals(self):
        """Connect UI signals to slots."""
        # Connect any signals from widgets to methods in this class
        self.data_widget.data_updated.connect(self.on_data_updated)
        self.nozzle_widget.nozzle_designed.connect(self.on_nozzle_designed)
        self.motor_widget.motor_designed.connect(self.on_motor_designed)
        
    def _load_settings(self):
        """Load application settings."""
        settings = QSettings("CEA-Analyzer", "CEA-Analyzer")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = settings.value("windowState")
        if state:
            self.restoreState(state)
            
    def _save_settings(self):
        """Save application settings."""
        settings = QSettings("CEA-Analyzer", "CEA-Analyzer")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        
    def closeEvent(self, event):
        """Handle window close event."""
        self._save_settings()
        super().closeEvent(event)
        
    def open_file(self, path=None):
        """Open a CEA output file for analysis."""
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Open CEA Output File", "", "CEA Output (*.out);;All Files (*)"
            )
            
        if not path:
            return
            
        self.statusBar().showMessage(f"Opening file: {path}")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Create background thread for file parsing
        self.parser_thread = ParserThread(path)
        self.parser_thread.progress.connect(self.progress_bar.setValue)
        self.parser_thread.finished.connect(self.on_parsing_finished)
        self.parser_thread.start()
        
    def on_parsing_finished(self, df):
        """Handle completion of file parsing."""
        if df is None or df.empty:
            self.statusBar().showMessage("Error parsing file")
            QMessageBox.critical(self, "Error", "Could not parse the selected file.")
            self.progress_bar.setVisible(False)
            return
            
        self.df_full = self.df = df
        self.cea_data = extract_thermo_data(df)
        
        # Update UI components with data
        self.data_widget.set_data(df)
        self.plotting_widget.set_data(df)
        self.nozzle_widget.set_cea_data(self.cea_data)
        self.motor_widget.set_cea_data(self.cea_data)
        self.optimization_widget.set_data(df)
        self.summary_widget.set_data(df, self.cea_data)
        
        # Switch to data tab
        self.central_tabs.setCurrentWidget(self.data_widget)
        
        # Enable actions that require data
        self.save_project_action.setEnabled(True)
        self.export_csv_action.setEnabled(True)
        self.export_excel_action.setEnabled(True)
        self.export_pdf_action.setEnabled(True)
        self.create_motor_action.setEnabled(True)
        self.optimize_action.setEnabled(True)
        
        # Show success message
        num_records = len(df)
        self.statusBar().showMessage(f"Successfully parsed file with {num_records} records")
        self.progress_bar.setVisible(False)
        
        # Find best Isp and show info
        if "Isp (s)" in df.columns:
            best_isp_idx = df["Isp (s)"].idxmax()
            best_isp = df.loc[best_isp_idx]["Isp (s)"]
            best_of = df.loc[best_isp_idx]["O/F"]
            logger.info(f"Best Isp: {best_isp:.2f} s at O/F = {best_of:.2f}")
        
    def on_data_updated(self, df):
        """Handle updates to the dataset (e.g., from filters)."""
        self.df = df
        
        # Update plots with filtered data
        self.plotting_widget.set_data(df)
        self.optimization_widget.set_data(df)
        
    def on_nozzle_designed(self, nozzle_data):
        """Handle completion of nozzle design."""
        # Update motor design with nozzle data if available
        if self.motor_design:
            self.motor_widget.update_nozzle(nozzle_data)
            
        # Update summary
        self.summary_widget.update_nozzle(nozzle_data)
        
    def on_motor_designed(self, motor_design):
        """Handle completion of motor design."""
        self.motor_design = motor_design
        
        # Update summary
        self.summary_widget.update_motor(motor_design)
        
    def save_project(self):
        """Save current project to file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "CEA Analyzer Project (*.ceaproj);;All Files (*)"
        )
        
        if not path:
            return
            
        # Implement project saving logic
        self.statusBar().showMessage(f"Project saved to {path}")
        
    def load_project(self):
        """Load project from file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "CEA Analyzer Project (*.ceaproj);;All Files (*)"
        )
        
        if not path:
            return
            
        # Implement project loading logic
        self.statusBar().showMessage(f"Project loaded from {path}")
        
    def export_csv(self):
        """Export data to CSV file."""
        if self.df is None:
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Export to CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        
        if not path:
            return
            
        success = export_csv(self.df, path)
        if success:
            self.statusBar().showMessage(f"Data exported to {path}")
        else:
            self.statusBar().showMessage("Error exporting data to CSV")
            
    def export_excel(self):
        """Export data to Excel file."""
        if self.df is None:
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Export to Excel", "", "Excel Files (*.xlsx);;All Files (*)"
        )
        
        if not path:
            return
            
        success = export_excel(self.df, path)
        if success:
            self.statusBar().showMessage(f"Data exported to {path}")
        else:
            self.statusBar().showMessage("Error exporting data to Excel")
            
    def export_pdf(self):
        """Export report to PDF file."""
        if self.df is None:
            return
            
        path, _ = QFileDialog.getSaveFileName(
            self, "Export to PDF", "", "PDF Files (*.pdf);;All Files (*)"
        )
        
        if not path:
            return
            
        success = export_pdf(self.df, path, self.cea_data)
        if success:
            self.statusBar().showMessage(f"Report exported to {path}")
        else:
            self.statusBar().showMessage("Error exporting report to PDF")
            
    def open_settings(self):
        """Open settings dialog."""
        dialog = SettingsDialog(self)
        dialog.exec()
        
    def show_about(self):
        """Show about dialog."""
        dialog = AboutDialog(self)
        dialog.exec()
        
    def open_help(self):
        """Open user manual."""
        # Implement help functionality
        self.statusBar().showMessage("Opening user manual...")
        
    def create_motor_design(self):
        """Create a new motor design."""
        if self.cea_data is None:
            QMessageBox.warning(self, "Warning", "Load CEA data first to create a motor design.")
            return
            
        # Switch to motor design tab
        self.central_tabs.setCurrentWidget(self.motor_widget)
        
    def optimize_design(self):
        """Optimize current design parameters."""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Load CEA data first to perform optimization.")
            return
            
        # Switch to optimization tab
        self.central_tabs.setCurrentWidget(self.optimization_widget)
        
    def show_performance_analysis(self):
        """Show performance analysis view."""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Load CEA data first to analyze performance.")
            return
            
        # Implement performance analysis view
        self.statusBar().showMessage("Showing performance analysis...")
        
    def show_sensitivity_analysis(self):
        """Show sensitivity analysis view."""
        if self.df is None:
            QMessageBox.warning(self, "Warning", "Load CEA data first to analyze sensitivity.")
            return
            
        # Implement sensitivity analysis view
        self.statusBar().showMessage("Showing sensitivity analysis...")

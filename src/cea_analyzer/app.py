"""
CEA Analyzer Application
-----------------------

Main application entry point for the modern CEA Analyzer GUI.
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QSplashScreen, QMessageBox, 
    QFileDialog, QDialog, QProgressDialog
)
from PyQt5.QtCore import Qt, QSettings, QTimer
from PyQt5.QtGui import QPixmap, QIcon

from .ui.main_window import MainWindow
from .ui.widgets.data_table_widget import DataTableWidget
from .ui.widgets.plotting_widget import PlottingWidget
from .ui.widgets.nozzle_design_widget import NozzleDesignWidget
from .ui.widgets.motor_design_widget import MotorDesignWidget
from .ui.widgets.optimization_widget import OptimizationWidget
from .analysis import compute_system, ambient_pressure
from .utils.logger import setup_logger


class CEAAnalyzerApp:
    """
    Main application class for CEA Analyzer.
    """
    
    def __init__(self):
        """Initialize the application."""
        # Setup logger
        self.logger = setup_logger("cea_analyzer")
        self.logger.info("Starting CEA Analyzer Application")
        
        # Create application
        self.app = QApplication(sys.argv)
        self.app.setApplicationName("CEA Analyzer")
        self.app.setOrganizationName("Aerospace Engineering")
        self.app.setOrganizationDomain("aero.edu")
        
        # Set application style
        self.app.setStyle("Fusion")
        
        # Load and set app icon
        self._set_application_icon()
        
        # Show splash screen
        self._show_splash_screen()
        
        # Load settings
        self.settings = QSettings()
        
        # Create main window
        self.main_window = MainWindow()
        
        # Initialize UI
        self._init_ui()
        
        # Connect signals and slots
        self._connect_signals()
        
        # Hide splash and show main window
        QTimer.singleShot(1500, self._finish_loading)
        
    def _set_application_icon(self):
        """Set the application icon."""
        # Try to load icon from resources
        icon_path = os.path.join(os.path.dirname(__file__), "resources", "icon.png")
        if os.path.exists(icon_path):
            self.app.setWindowIcon(QIcon(icon_path))
        
    def _show_splash_screen(self):
        """Show the application splash screen."""
        # Try to load splash image from resources
        splash_path = os.path.join(os.path.dirname(__file__), "resources", "splash.png")
        if os.path.exists(splash_path):
            pixmap = QPixmap(splash_path)
        else:
            # Create a default splash if image not found
            from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont
            pixmap = QPixmap(600, 400)
            pixmap.fill(QColor(0, 20, 40))
            painter = QPainter(pixmap)
            painter.setFont(QFont("Arial", 40))
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, "CEA Analyzer")
            painter.setFont(QFont("Arial", 20))
            painter.drawText(pixmap.rect().adjusted(0, 80, 0, 0), Qt.AlignCenter, "Modern Rocket Design Tools")
            painter.end()
            
        self.splash = QSplashScreen(pixmap)
        self.splash.show()
        self.app.processEvents()
        
    def _init_ui(self):
        """Initialize the user interface."""
        # Create widgets
        self.data_table_widget = DataTableWidget()
        self.plotting_widget = PlottingWidget()
        self.nozzle_design_widget = NozzleDesignWidget()
        self.motor_design_widget = MotorDesignWidget()
        self.optimization_widget = OptimizationWidget()
        
        # Add widgets to main window
        self.main_window.add_central_widget("Data View", self.data_table_widget)
        self.main_window.add_central_widget("Plotting", self.plotting_widget)
        self.main_window.add_central_widget("Nozzle Design", self.nozzle_design_widget)
        self.main_window.add_central_widget("Motor Design", self.motor_design_widget)
        self.main_window.add_central_widget("Optimization", self.optimization_widget)
        
        # Connect main window signals
        self.main_window.file_loaded.connect(self._handle_file_loaded)
        self.main_window.cea_data_updated.connect(self._update_widgets_with_cea_data)
        
        # Set window title
        self.main_window.setWindowTitle("CEA Analyzer - Modern Rocket Design Tools")
        
        # Restore window geometry from settings
        geometry = self.settings.value("MainWindow/Geometry")
        if geometry:
            self.main_window.restoreGeometry(geometry)
        else:
            # Default size
            self.main_window.resize(1200, 800)
            
        # Restore window state from settings
        state = self.settings.value("MainWindow/State")
        if state:
            self.main_window.restoreState(state)
            
    def _connect_signals(self):
        """Connect signals and slots between widgets."""
        # Connect nozzle design widget to motor design widget
        self.nozzle_design_widget.nozzle_designed.connect(self.motor_design_widget.update_nozzle)
        
        # Connect data table widget to plotting widget
        self.data_table_widget.data_selected.connect(self.plotting_widget.plot_data)
        
        # Connect CEA data updated signal from main window
        self.main_window.cea_data_updated.connect(self.data_table_widget.set_cea_data)
        self.main_window.cea_data_updated.connect(self.plotting_widget.set_cea_data)
        self.main_window.cea_data_updated.connect(self.nozzle_design_widget.set_cea_data)
        self.main_window.cea_data_updated.connect(self.motor_design_widget.set_cea_data)
        self.main_window.cea_data_updated.connect(self.optimization_widget.set_cea_data)
        
        # Connect application exit
        self.app.aboutToQuit.connect(self._save_settings)
        
    def _finish_loading(self):
        """Finish the loading process and show the main window."""
        self.splash.finish(self.main_window)
        self.main_window.show()
        self.main_window.status_message("Ready")
        
    def _save_settings(self):
        """Save application settings."""
        self.settings.setValue("MainWindow/Geometry", self.main_window.saveGeometry())
        self.settings.setValue("MainWindow/State", self.main_window.saveState())
        
    def _handle_file_loaded(self, file_path):
        """Handle a file being loaded."""
        try:
            # Extract file extension
            ext = Path(file_path).suffix.lower()
            
            if ext == '.csv':
                # Load CSV file
                df = pd.read_csv(file_path)
                
                # Compute system parameters
                cea_data = compute_system(df)
                
                # Update main window with CEA data
                self.main_window.set_cea_data(cea_data)
                
                # Update status
                self.main_window.status_message(f"Loaded and processed {file_path}")
                
            else:
                QMessageBox.warning(
                    self.main_window,
                    "Unsupported File Type",
                    f"The file type '{ext}' is not supported. Please load a CSV file."
                )
                
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "Error Loading File",
                f"Error loading file: {str(e)}"
            )
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            
    def _update_widgets_with_cea_data(self, cea_data):
        """Update all widgets with CEA data."""
        # Already handled by individual connections
        pass
        
    def run(self):
        """Run the application."""
        return self.app.exec_()


def main():
    """Main entry point for the application."""
    app = CEAAnalyzerApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())

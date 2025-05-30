"""
CEA Analyzer Application
-----------------------

Main application entry point for the modern CEA Analyzer GUI.
"""

import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QSplashScreen, QMessageBox, 
    QFileDialog, QDialog, QProgressDialog
)
from PyQt6.QtCore import Qt, QSettings, QTimer
from PyQt6.QtGui import QPixmap, QIcon, QPainter, QColor, QFont, QLinearGradient

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
        
        # Show splash screen - This needs to be shown before any windows
        self._show_splash_screen()
        
        # Force application to process events to make splash visible
        self.app.processEvents()
        
        # Add delay to make the splash screen visible longer
        # This is a synchronous delay to ensure the splash screen stays visible
        time.sleep(1.0)  # Show splash for at least 1 second
        
        # Load settings
        self.settings = QSettings()
        
        # Create main window but don't show it yet
        self.main_window = MainWindow()
        
        # Initialize UI
        self._init_ui()
        
        # Connect signals and slots
        self._connect_signals()
        
        # Use a longer timer for finishing loading
        # This ensures the splash is visible for a reasonable time
        QTimer.singleShot(2000, self._finish_loading)
        
    def _set_application_icon(self):
        """Set the application icon."""
        # Try to load icon from resources
        icon_path = os.path.join(os.path.dirname(__file__), "resources", "icon.png")
        if os.path.exists(icon_path):
            self.app.setWindowIcon(QIcon(icon_path))
        
    def _show_splash_screen(self):
        """Show the application splash screen."""
        # Create a pixmap for the splash screen
        pixmap = QPixmap(600, 400)
        pixmap.fill(QColor(0, 30, 60))  # Dark blue background
        
        # Paint on the pixmap
        painter = QPainter(pixmap)
        
        # Set a gradient background
        gradient = QLinearGradient(0, 0, 0, 400)
        gradient.setColorAt(0, QColor(0, 40, 80))   # Dark blue at top
        gradient.setColorAt(1, QColor(0, 10, 30))   # Darker blue at bottom
        painter.fillRect(pixmap.rect(), gradient)
        
        # Draw the application name
        font = QFont("Arial", 40)
        font.setWeight(QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255))  # White color
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "CEA Analyzer")
        
        # Draw the subtitle
        painter.setFont(QFont("Arial", 20))
        painter.setPen(QColor(200, 200, 255))  # Light blue color
        painter.drawText(pixmap.rect().adjusted(0, 80, 0, 0), 
                         Qt.AlignmentFlag.AlignCenter, "Modern Rocket Design Tools")
        
        # Draw the version
        try:
            from cea_analyzer import __version__
        except ImportError:
            __version__ = "2.0.0"
            
        painter.setFont(QFont("Arial", 12))
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(pixmap.rect().adjusted(0, 0, -20, -20), 
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom, 
                         f"Version {__version__}")
        
        # End painting
        painter.end()
        
        # Create splash screen
        self.splash = QSplashScreen(pixmap)
        
        # Show the splash screen
        self.splash.show()
        
        # Display a message
        self.splash.showMessage("Loading CEA Analyzer...", 
                               Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, 
                               Qt.GlobalColor.white)
        
        # Force processing of events to make splash visible
        self.app.processEvents()
        
        # Make sure it stays visible for a bit
        time.sleep(1.0)  # Force the splash to display for at least 1 second
        
    def _init_ui(self):
        """Initialize the user interface."""
        # The MainWindow class already creates all the necessary widgets internally
        # and organizes them in a tabbed interface. We don't need to manually create and add them.
        
        # We just need to initialize our main window and let it handle the UI setup
        # The main window will internally create instances of:
        # - DataTableWidget
        # - PlottingWidget
        # - NozzleDesignWidget
        # - MotorDesignWidget
        # - OptimizationWidget
        # - SummaryWidget
        
        # MainWindow class already handles these signals internally
        # No need to connect them here
        
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
        # The MainWindow class already connects most widget signals internally
        # We only need to connect application exit signals
        
        # Connect application exit
        self.app.aboutToQuit.connect(self._save_settings)
        
    def _finish_loading(self):
        """Finish the loading process and show the main window."""
        self.splash.finish(self.main_window)
        self.main_window.show()
        self.main_window.statusBar().showMessage("Ready")
        
    def _save_settings(self):
        """Save application settings."""
        self.settings.setValue("MainWindow/Geometry", self.main_window.saveGeometry())
        self.settings.setValue("MainWindow/State", self.main_window.saveState())
        
    def _handle_file_loaded(self, file_path):
        """Handle a file being loaded."""
        # This method is now handled directly by the MainWindow class
        # and is no longer needed in this class
        pass
            
    def _update_widgets_with_cea_data(self, cea_data):
        """Update all widgets with CEA data."""
        # Already handled by individual connections
        pass
        
    def run(self):
        """Run the application."""
        return self.app.exec()


def main():
    """Main entry point for the application."""
    app = CEAAnalyzerApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())

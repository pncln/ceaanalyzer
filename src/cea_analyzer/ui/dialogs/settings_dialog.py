"""
Settings Dialog
--------------

Dialog for configuring application settings.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton,
    QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QGroupBox, QFormLayout, QFileDialog, QDialogButtonBox,
    QColorDialog, QFontDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QSettings, QSize
from PyQt6.QtGui import QFont, QColor

import os
import logging

# Set up logger
logger = logging.getLogger("cea_analyzer.settings_dialog")


class SettingsDialog(QDialog):
    """
    Dialog for configuring application settings.
    """
    
    def __init__(self, parent=None):
        """Initialize the settings dialog."""
        super().__init__(parent)
        
        # Set window properties
        self.setWindowTitle("Settings")
        self.resize(600, 400)
        
        # Load settings
        self.settings = QSettings()
        
        # Setup UI
        self._init_ui()
        
        # Load current settings
        self._load_settings()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Tab widget for different settings categories
        self.tab_widget = QTabWidget()
        
        # 1. General tab
        general_tab = QWidget()
        general_layout = QFormLayout(general_tab)
        
        # Theme selection
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["System", "Light", "Dark", "High Contrast"])
        general_layout.addRow("Theme:", self.theme_combo)
        
        # Save data on exit
        self.save_on_exit_check = QCheckBox("Automatically save data on exit")
        general_layout.addRow("", self.save_on_exit_check)
        
        # Recent files count
        self.recent_files_spin = QSpinBox()
        self.recent_files_spin.setRange(0, 20)
        self.recent_files_spin.setValue(10)
        general_layout.addRow("Number of recent files:", self.recent_files_spin)
        
        # Default save location
        self.default_save_edit = QLineEdit()
        self.default_save_edit.setReadOnly(True)
        
        browse_button = QPushButton("Browse...")
        browse_button.clicked.connect(self._browse_save_location)
        
        save_layout = QHBoxLayout()
        save_layout.addWidget(self.default_save_edit)
        save_layout.addWidget(browse_button)
        
        general_layout.addRow("Default save location:", save_layout)
        
        # 2. Analysis tab
        analysis_tab = QWidget()
        analysis_layout = QFormLayout(analysis_tab)
        
        # Default ambient pressure
        self.ambient_pressure_spin = QDoubleSpinBox()
        self.ambient_pressure_spin.setRange(0.001, 10.0)
        self.ambient_pressure_spin.setValue(1.0)
        self.ambient_pressure_spin.setDecimals(3)
        self.ambient_pressure_spin.setSuffix(" bar")
        analysis_layout.addRow("Default ambient pressure:", self.ambient_pressure_spin)
        
        # Default expansion ratio
        self.expansion_ratio_spin = QDoubleSpinBox()
        self.expansion_ratio_spin.setRange(1.1, 100.0)
        self.expansion_ratio_spin.setValue(8.0)
        self.expansion_ratio_spin.setDecimals(1)
        analysis_layout.addRow("Default expansion ratio:", self.expansion_ratio_spin)
        
        # Default units
        self.units_combo = QComboBox()
        self.units_combo.addItems(["SI", "Imperial", "Mixed"])
        analysis_layout.addRow("Default units:", self.units_combo)
        
        # Significant digits
        self.sig_digits_spin = QSpinBox()
        self.sig_digits_spin.setRange(2, 8)
        self.sig_digits_spin.setValue(4)
        analysis_layout.addRow("Significant digits:", self.sig_digits_spin)
        
        # 3. Display tab
        display_tab = QWidget()
        display_layout = QFormLayout(display_tab)
        
        # Font settings
        self.font_combo = QFontComboBox()
        self.font_combo.setCurrentFont(QFont("Arial"))
        
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(10)
        
        font_layout = QHBoxLayout()
        font_layout.addWidget(self.font_combo)
        font_layout.addWidget(self.font_size_spin)
        
        display_layout.addRow("Font:", font_layout)
        
        # Plot settings
        self.plot_color_button = QPushButton("Choose Color...")
        self.plot_color_button.clicked.connect(self._choose_plot_color)
        self.plot_color = QColor(0, 114, 189)  # Default MATLAB blue
        
        self.plot_line_width_spin = QDoubleSpinBox()
        self.plot_line_width_spin.setRange(0.5, 5.0)
        self.plot_line_width_spin.setValue(1.5)
        self.plot_line_width_spin.setDecimals(1)
        
        plot_layout = QHBoxLayout()
        plot_layout.addWidget(self.plot_color_button)
        plot_layout.addWidget(QLabel("Line width:"))
        plot_layout.addWidget(self.plot_line_width_spin)
        
        display_layout.addRow("Plot appearance:", plot_layout)
        
        # Table settings
        self.alternate_row_colors_check = QCheckBox("Use alternate row colors")
        self.alternate_row_colors_check.setChecked(True)
        display_layout.addRow("Tables:", self.alternate_row_colors_check)
        
        # 4. Advanced tab
        advanced_tab = QWidget()
        advanced_layout = QFormLayout(advanced_tab)
        
        # Logging level
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.log_level_combo.setCurrentText("INFO")
        advanced_layout.addRow("Logging level:", self.log_level_combo)
        
        # Log file location
        self.log_file_edit = QLineEdit()
        self.log_file_edit.setReadOnly(True)
        
        log_browse_button = QPushButton("Browse...")
        log_browse_button.clicked.connect(self._browse_log_file)
        
        log_layout = QHBoxLayout()
        log_layout.addWidget(self.log_file_edit)
        log_layout.addWidget(log_browse_button)
        
        advanced_layout.addRow("Log file location:", log_layout)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(general_tab, "General")
        self.tab_widget.addTab(analysis_tab, "Analysis")
        self.tab_widget.addTab(display_tab, "Display")
        self.tab_widget.addTab(advanced_tab, "Advanced")
        
        # Dialog buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel | QDialogButtonBox.Apply | QDialogButtonBox.Reset
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.Apply).clicked.connect(self._apply_settings)
        self.button_box.button(QDialogButtonBox.Reset).clicked.connect(self._reset_settings)
        
        # Add widgets to main layout
        main_layout.addWidget(self.tab_widget)
        main_layout.addWidget(self.button_box)
        
    def _load_settings(self):
        """Load settings from QSettings."""
        # General settings
        self.theme_combo.setCurrentText(self.settings.value("General/Theme", "System"))
        self.save_on_exit_check.setChecked(self.settings.value("General/SaveOnExit", True, type=bool))
        self.recent_files_spin.setValue(self.settings.value("General/RecentFilesCount", 10, type=int))
        self.default_save_edit.setText(self.settings.value("General/DefaultSaveLocation", os.path.expanduser("~")))
        
        # Analysis settings
        self.ambient_pressure_spin.setValue(self.settings.value("Analysis/AmbientPressure", 1.0, type=float))
        self.expansion_ratio_spin.setValue(self.settings.value("Analysis/ExpansionRatio", 8.0, type=float))
        self.units_combo.setCurrentText(self.settings.value("Analysis/Units", "SI"))
        self.sig_digits_spin.setValue(self.settings.value("Analysis/SigDigits", 4, type=int))
        
        # Display settings
        font_family = self.settings.value("Display/FontFamily", "Arial")
        font_size = self.settings.value("Display/FontSize", 10, type=int)
        self.font_combo.setCurrentFont(QFont(font_family))
        self.font_size_spin.setValue(font_size)
        
        plot_color = self.settings.value("Display/PlotColor", QColor(0, 114, 189).name())
        self.plot_color = QColor(plot_color)
        
        self.plot_line_width_spin.setValue(self.settings.value("Display/PlotLineWidth", 1.5, type=float))
        self.alternate_row_colors_check.setChecked(self.settings.value("Display/AlternateRowColors", True, type=bool))
        
        # Advanced settings
        self.log_level_combo.setCurrentText(self.settings.value("Advanced/LogLevel", "INFO"))
        self.log_file_edit.setText(self.settings.value("Advanced/LogFile", ""))
        
    def _apply_settings(self):
        """Apply current settings."""
        # General settings
        self.settings.setValue("General/Theme", self.theme_combo.currentText())
        self.settings.setValue("General/SaveOnExit", self.save_on_exit_check.isChecked())
        self.settings.setValue("General/RecentFilesCount", self.recent_files_spin.value())
        self.settings.setValue("General/DefaultSaveLocation", self.default_save_edit.text())
        
        # Analysis settings
        self.settings.setValue("Analysis/AmbientPressure", self.ambient_pressure_spin.value())
        self.settings.setValue("Analysis/ExpansionRatio", self.expansion_ratio_spin.value())
        self.settings.setValue("Analysis/Units", self.units_combo.currentText())
        self.settings.setValue("Analysis/SigDigits", self.sig_digits_spin.value())
        
        # Display settings
        self.settings.setValue("Display/FontFamily", self.font_combo.currentFont().family())
        self.settings.setValue("Display/FontSize", self.font_size_spin.value())
        self.settings.setValue("Display/PlotColor", self.plot_color.name())
        self.settings.setValue("Display/PlotLineWidth", self.plot_line_width_spin.value())
        self.settings.setValue("Display/AlternateRowColors", self.alternate_row_colors_check.isChecked())
        
        # Advanced settings
        self.settings.setValue("Advanced/LogLevel", self.log_level_combo.currentText())
        self.settings.setValue("Advanced/LogFile", self.log_file_edit.text())
        
        # Log the changes
        logger.info("Settings applied")
        
        # Emit signal if parent window is available
        if self.parent():
            # If the parent has a settings_changed signal, emit it
            if hasattr(self.parent(), 'settings_changed'):
                self.parent().settings_changed.emit()
        
    def _reset_settings(self):
        """Reset settings to defaults."""
        # Ask for confirmation
        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to default values?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Reset QSettings
            self.settings.clear()
            
            # Reload default settings
            self._load_settings()
            
            # Log the reset
            logger.info("Settings reset to defaults")
        
    def _browse_save_location(self):
        """Browse for default save location."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Default Save Location",
            self.default_save_edit.text() or os.path.expanduser("~")
        )
        
        if folder:
            self.default_save_edit.setText(folder)
        
    def _browse_log_file(self):
        """Browse for log file location."""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Log File Location",
            self.log_file_edit.text() or os.path.join(os.path.expanduser("~"), "cea_analyzer.log"),
            "Log Files (*.log);;All Files (*)"
        )
        
        if file_path:
            self.log_file_edit.setText(file_path)
        
    def _choose_plot_color(self):
        """Choose plot color."""
        color = QColorDialog.getColor(
            self.plot_color, self, "Select Plot Color"
        )
        
        if color.isValid():
            self.plot_color = color
        
    def accept(self):
        """Handle dialog acceptance."""
        # Apply settings first
        self._apply_settings()
        
        # Call parent accept
        super().accept()
        
    def sizeHint(self):
        """Preferred size for the dialog."""
        return QSize(600, 400)

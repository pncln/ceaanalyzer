"""
About Dialog
-----------

Dialog for displaying information about the application.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QWidget, QTextEdit, QScrollArea
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap, QFont, QIcon

import os
from ... import __version__


class AboutDialog(QDialog):
    """
    Dialog for displaying information about the CEA Analyzer application.
    """
    
    def __init__(self, parent=None):
        """Initialize the about dialog."""
        super().__init__(parent)
        
        # Set window properties
        self.setWindowTitle("About CEA Analyzer")
        self.resize(500, 400)
        self.setModal(True)
        
        # Setup UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Header layout
        header_layout = QHBoxLayout()
        
        # Logo
        logo_label = QLabel()
        # Try to load logo from resources
        logo_path = os.path.join(os.path.dirname(__file__), "..", "..", "resources", "icon.png")
        
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            logo_label.setPixmap(pixmap.scaled(64, 64, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            # Display a placeholder text if icon not found
            logo_label.setText("CEA")
            font = QFont("Arial", 20)
            font.setWeight(QFont.Weight.Bold)
            logo_label.setFont(font)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_label.setFixedSize(64, 64)
            logo_label.setStyleSheet("background-color: #003366; color: white; border-radius: 8px;")
        
        # Application info
        info_layout = QVBoxLayout()
        
        app_name_label = QLabel("CEA Analyzer")
        font = QFont("Arial", 16)
        font.setWeight(QFont.Weight.Bold)
        app_name_label.setFont(font)
        
        version_label = QLabel(f"Version {__version__}")
        version_label.setFont(QFont("Arial", 10))
        
        info_layout.addWidget(app_name_label)
        info_layout.addWidget(version_label)
        info_layout.addStretch()
        
        # Add to header layout
        header_layout.addWidget(logo_label)
        header_layout.addLayout(info_layout)
        header_layout.addStretch()
        
        # Tab widget for different information
        tab_widget = QTabWidget()
        
        # 1. About tab
        about_tab = QWidget()
        about_layout = QVBoxLayout(about_tab)
        
        about_text = QTextEdit()
        about_text.setReadOnly(True)
        about_text.setHtml("""
        <h3>CEA Analyzer</h3>
        <p>A modern tool for analyzing rocket propulsion with NASA-CEA data.</p>
        
        <p>The CEA Analyzer is designed to help aerospace engineers and students analyze
        and visualize rocket engine performance using NASA's Chemical Equilibrium with 
        Applications (CEA) code. This application provides a user-friendly interface for 
        working with CEA data, designing nozzles, and optimizing propulsion systems.</p>
        
        <p>Key features include:</p>
        <ul>
            <li>Analysis of NASA-CEA output files</li>
            <li>Interactive visualization of thermochemical properties</li>
            <li>Nozzle design and analysis tools</li>
            <li>Motor design capabilities</li>
            <li>Performance optimization</li>
        </ul>
        """)
        
        about_layout.addWidget(about_text)
        
        # 2. Credits tab
        credits_tab = QWidget()
        credits_layout = QVBoxLayout(credits_tab)
        
        credits_text = QTextEdit()
        credits_text.setReadOnly(True)
        credits_text.setHtml("""
        <h3>Credits</h3>
        <p>CEA Analyzer was developed as part of AE442 Rocket Engineering course.</p>
        
        <h4>Developers</h4>
        <ul>
            <li>Aerospace Engineering Department</li>
        </ul>
        
        <h4>Acknowledgements</h4>
        <ul>
            <li>NASA Chemical Equilibrium with Applications (CEA) Program</li>
            <li>Python Software Foundation</li>
            <li>PyQt and Qt Project</li>
            <li>NumPy, SciPy, and Matplotlib Projects</li>
        </ul>
        """)
        
        credits_layout.addWidget(credits_text)
        
        # 3. License tab
        license_tab = QWidget()
        license_layout = QVBoxLayout(license_tab)
        
        license_text = QTextEdit()
        license_text.setReadOnly(True)
        license_text.setHtml("""
        <h3>License</h3>
        <p>Copyright (c) 2025 Aerospace Engineering Department</p>
        
        <p>Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:</p>
        
        <p>The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.</p>
        
        <p>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
        OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
        SOFTWARE.</p>
        """)
        
        license_layout.addWidget(license_text)
        
        # Add tabs to tab widget
        tab_widget.addTab(about_tab, "About")
        tab_widget.addTab(credits_tab, "Credits")
        tab_widget.addTab(license_tab, "License")
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        
        # Add widgets to main layout
        main_layout.addLayout(header_layout)
        main_layout.addWidget(tab_widget)
        main_layout.addWidget(close_button, alignment=Qt.AlignRight)
        
    def sizeHint(self):
        """Preferred size for the dialog."""
        return QSize(500, 400)

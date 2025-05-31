#!/usr/bin/env python3
"""
Simple test script to verify the grain visualization widget works independently.
"""
import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from cea_analyzer.ui.widgets.grain_visualization_widget import GrainVisualizationWidget

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QMainWindow()
    window.setWindowTitle("Grain Visualization Test")
    window.resize(1000, 800)
    
    # Create and set the grain visualization widget as central widget
    grain_widget = GrainVisualizationWidget()
    window.setCentralWidget(grain_widget)
    
    # Show the window
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

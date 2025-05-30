#!/usr/bin/env python3
"""
CEA Analyzer - Entry Point
--------------------------

This is the main entry point for the CEA Analyzer application.
It initializes the GUI and starts the application.
"""

import sys
import argparse
from pathlib import Path

from PyQt5.QtWidgets import QApplication, QMessageBox

from .app import CEAAnalyzerApp
from . import __version__
from .utils.logger import setup_logger


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="CEA Analyzer - A tool for analyzing rocket propulsion with NASA-CEA data"
    )
    parser.add_argument(
        "file", nargs="?", help="CEA output file to open at startup"
    )
    parser.add_argument(
        "--version", action="version", version=f"CEA Analyzer v{__version__}"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )
    parser.add_argument(
        "--log-file", help="Custom log file path"
    )
    
    return parser.parse_args()


def main():
    """
    Main entry point for the application.
    Initializes the GUI and handles command line arguments.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set up logging
    log_level = "DEBUG" if args.debug else "INFO"
    logger = setup_logger("cea_analyzer", level=log_level, log_file=args.log_file)
    logger.info(f"Starting CEA Analyzer v{__version__}")
    
    try:
        # Create and run the application
        analyzer_app = CEAAnalyzerApp()
        
        # If a file path was provided, open it after app initialization
        if args.file:
            file_path = Path(args.file)
            if file_path.exists():
                logger.info(f"Opening file from command line: {file_path}")
                # Use QTimer to load the file after the main window is shown
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(500, lambda: analyzer_app.main_window.open_file(str(file_path)))
            else:
                logger.warning(f"File not found: {file_path}")
                QMessageBox.warning(None, "File Not Found", f"The specified file was not found: {file_path}")
        
        # Run the application
        return analyzer_app.run()
        
    except Exception as e:
        logger.error(f"Error starting application: {str(e)}")
        QMessageBox.critical(None, "Application Error", 
                          f"An error occurred while starting the application:\n{str(e)}")
        return 1


if __name__ == "__main__":
    main()

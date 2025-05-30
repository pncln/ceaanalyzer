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

from PyQt5.QtWidgets import QApplication

from gui import MainWindow
from __init__ import __version__
from logger import configure_logging, get_logger


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
    
    # Configure logging based on arguments
    log_level = "DEBUG" if args.debug else "INFO"
    configure_logging(log_level=log_level, log_file=args.log_file)
    
    # Get logger for main module
    logger = get_logger("main")
    logger.info(f"Starting CEA Analyzer v{__version__}")
    
    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("CEA Analyzer")
    app.setApplicationVersion(__version__)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show main window
    win = MainWindow()
    win.show()
    
    # If a file path was provided, open it
    if args.file:
        file_path = args.file
        logger.info(f"Opening file from command line: {file_path}")
        win.open_file(file_path)
    
    # Start event loop
    return sys.exit(app.exec_())


if __name__ == "__main__":
    main()

"""
Logging Configuration Module
---------------------------

This module provides centralized logging configuration for the CEA Analyzer application.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional


def configure_logging(log_level: int = logging.INFO,
                     log_file: Optional[str] = None,
                     console: bool = True) -> None:
    """
    Configure logging for the CEA Analyzer application.
    
    Parameters
    ----------
    log_level : int, optional
        Logging level (default: logging.INFO)
    log_file : str, optional
        Path to log file, if None logs will be stored in ~/.cea_analyzer/cea_analyzer.log
    console : bool, optional
        Whether to output logs to console (default: True)
    """
    # Create root logger
    logger = logging.getLogger('cea_analyzer')
    logger.setLevel(log_level)
    
    # Clear any existing handlers
    logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # File handler
    if log_file is None:
        log_dir = Path.home() / ".cea_analyzer"
        log_dir.mkdir(exist_ok=True)
        log_file = str(log_dir / "cea_analyzer.log")
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    logger.debug(f"Logging configured. Log file: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for a specific module.
    
    Parameters
    ----------
    name : str
        Name of the module requesting the logger
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    return logging.getLogger(f'cea_analyzer.{name}')

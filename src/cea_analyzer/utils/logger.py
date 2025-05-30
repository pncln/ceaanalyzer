"""
Logging Utility
--------------

Provides standardized logging setup for the CEA Analyzer application.
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name, level="INFO", log_file=None):
    """
    Set up and configure a logger with the given name.
    
    Parameters
    ----------
    name : str
        Name of the logger
    level : str
        Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    log_file : str, optional
        Path to the log file. If None, logs to a default location.
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified or use default
    if log_file is None:
        # Create logs directory if it doesn't exist
        logs_dir = Path.home() / ".cea_analyzer" / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Use date in log filename
        today = datetime.now().strftime("%Y-%m-%d")
        log_file = logs_dir / f"cea_analyzer_{today}.log"
    else:
        log_file = Path(log_file)
        # Create parent directories if they don't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    # Log a startup message
    logger.info(f"Logger '{name}' initialized with level {level}")
    
    return logger

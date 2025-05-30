"""
Configuration module for CEA Analyzer
------------------------------------

This module provides configuration management for the CEA Analyzer application.
It handles loading, saving, and accessing configuration settings.
"""

import os
import json
import logging
from typing import Dict, Any

# Standard gravitational acceleration (m/s²)
G0: float = 9.80665

# Configuration file path
CONFIG_PATH: str = os.path.expanduser("~/.cea_analyzer_config.json")

# Default configuration settings
DEFAULT_CONFIG: Dict[str, Any] = {
    "pdf_report_title": "CEA Analysis Report",
    "default_throat_radius": 0.05,  # Default throat radius in meters
    "default_chamber_pressure": 50.0,  # Default chamber pressure in bar
    "default_area_ratio": 8.0,  # Default nozzle area ratio
}


def load_config() -> Dict[str, Any]:
    """
    Load configuration from file or create default if not exists.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing configuration settings
    """
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError, OSError) as e:
            logging.warning(f"Failed to load config file: {e}")
    
    # Create default config
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
    except (IOError, OSError) as e:
        logging.warning(f"Failed to create default config file: {e}")
    
    return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]) -> bool:
    """
    Save configuration to file.
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary to save
        
    Returns
    -------
    bool
        True if save was successful, False otherwise
    """
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except (IOError, OSError) as e:
        logging.error(f"Failed to save config file: {e}")
        return False


# Global configuration instance
CONFIG: Dict[str, Any] = load_config()

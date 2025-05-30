"""
Analysis Package
---------------

This package provides functions for analyzing rocket propulsion systems
and parsing NASA-CEA output data.
"""

# Import from performance module
from .performance import compute_system, ambient_pressure

# Import from cea_parser module
from .cea_parser import parse_cea_output, extract_thermo_data

__all__ = [
    # Performance analysis functions
    'compute_system',
    'ambient_pressure',
    
    # CEA parsing functions
    'parse_cea_output',
    'extract_thermo_data'
]
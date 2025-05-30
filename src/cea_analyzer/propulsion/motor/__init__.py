"""
Rocket Motor Design Package
---------------------------

This package provides classes and functions for designing and analyzing
rocket motor components and integrated motor designs.
"""

# Import motor types
from .types import MotorType

# Import motor components
from .components import MotorCase, Nozzle

# Import motor design
from .design import MotorDesign

__all__ = [
    # Enumerations
    'MotorType',
    
    # Component classes
    'MotorCase',
    'Nozzle',
    
    # Motor design class
    'MotorDesign'
]
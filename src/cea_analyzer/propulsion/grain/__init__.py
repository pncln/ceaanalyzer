"""
Rocket Motor Grain Design Package
------------------------------

This package provides classes and functions for designing and analyzing
solid rocket motor grain geometries and propellant properties.
"""

# Import grain types
from .types import GrainType

# Import base classes
from .base import GrainGeometry, PropellantProperties

# Import geometry implementations
from .geometries import BatesGrain, StarGrain, EndBurnerGrain

# Import motor grain
from .motor_grain import MotorGrain

__all__ = [
    # Enumerations
    'GrainType',
    
    # Base classes
    'GrainGeometry',
    'PropellantProperties',
    
    # Grain geometry implementations
    'BatesGrain',
    'StarGrain',
    'EndBurnerGrain',
    
    # Motor grain class
    'MotorGrain'
]
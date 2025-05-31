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
from .advanced_geometries import CSlotGrain, FinocylGrain, WagonWheelGrain

# Import motor grain
from .motor_grain import MotorGrain

# Import regression simulation
from .regression import (
    GrainRegressionSimulation,
    generate_grain_cross_section,
    visualize_grain_regression,
    create_3d_grain_model
)

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
    
    # Advanced grain geometries
    'CSlotGrain',
    'FinocylGrain', 
    'WagonWheelGrain',
    
    # Regression simulation
    'GrainRegressionSimulation',
    'generate_grain_cross_section',
    'visualize_grain_regression',
    'create_3d_grain_model',
    
    # Motor grain class
    'MotorGrain'
]
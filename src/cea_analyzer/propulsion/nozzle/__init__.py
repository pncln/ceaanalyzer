"""
Rocket Nozzle Design Package
---------------------------

This package provides various methods for designing rocket nozzle contours.
It includes implementations for conical, bell, Rao optimum, method of characteristics (MOC),
and truncated ideal contour (TIC) nozzles.
"""

# Import base utilities
from .base import (
    get_throat_properties,
    add_inlet_section,
    export_nozzle_coordinates,
    plot_nozzle_contour
)

# Import performance calculation
from .performance import calculate_performance

# Import nozzle design functions
from .conical import conical_nozzle
from .bell import bell_nozzle
from .rao import rao_optimum_nozzle
from .moc_nozzle import moc_nozzle
from .tic import truncated_ideal_contour

# Import MOC utilities
from .moc import (
    prandtl_meyer,
    inverse_prandtl_meyer,
    mach_from_area_ratio,
    generate_moc_contour
)

__all__ = [
    # Base utilities
    'get_throat_properties',
    'add_inlet_section',
    'export_nozzle_coordinates',
    'plot_nozzle_contour',
    'calculate_performance',
    
    # Nozzle design functions
    'conical_nozzle',
    'bell_nozzle',
    'rao_optimum_nozzle',
    'moc_nozzle',
    'truncated_ideal_contour',
    
    # MOC utilities
    'prandtl_meyer',
    'inverse_prandtl_meyer',
    'mach_from_area_ratio',
    'generate_moc_contour'
]
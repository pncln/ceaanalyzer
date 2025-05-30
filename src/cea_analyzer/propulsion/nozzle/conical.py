"""
Conical Nozzle Design Module for CEA Analyzer
--------------------------------------------

This module provides functionality for designing conical nozzle contours.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
import pandas as pd

from .base import get_throat_properties


def conical_nozzle(cea_data, half_angle=15, R_throat=None, N=100):
    """
    Generate a conical nozzle contour following standard aerospace engineering practices.
    
    The conical nozzle is the simplest supersonic nozzle design, consisting of a
    straight-walled cone attached to the throat. While not as efficient as contoured
    nozzles, it provides a baseline design that is easy to manufacture.
    
    Parameters
    ----------
    cea_data : dict or pandas.Series
        CEA data containing at minimum: area_ratio
    half_angle : float, optional
        Half-angle of the nozzle cone in degrees, default 15°
        Standard values range from 12° to 18°
    R_throat : float, optional
        Throat radius in meters, if None it will be calculated from CEA data
    N : int, optional
        Number of points to generate for the contour
        
    Returns
    -------
    tuple
        (x_coordinates, r_coordinates) for the nozzle contour
    """
    # Extract area ratio from CEA data
    props = get_throat_properties(cea_data)
    area_ratio = props['area_ratio']
    
    # Calculate throat radius if not provided
    if R_throat is None:
        if 'At' in cea_data:
            R_throat = np.sqrt(cea_data['At'] / np.pi)
        else:
            R_throat = 0.05  # Default 5cm throat radius
    
    # Calculate exit radius based on area ratio
    R_exit = R_throat * np.sqrt(area_ratio)
    
    # Calculate nozzle length based on half-angle
    half_angle_rad = np.radians(half_angle)
    L_nozzle = (R_exit - R_throat) / np.tan(half_angle_rad)
    
    # Simple straight-line conical nozzle
    # Generate a straight line from throat to exit with proper half-angle
    x = np.linspace(0, L_nozzle, N)
    r = np.zeros(N)
    
    # Set throat radius at x=0
    r[0] = R_throat
    
    # Create a pure conical expansion with the correct half-angle
    for i in range(1, N):
        r[i] = R_throat + x[i] * np.tan(half_angle_rad)
    
    # Ensure exit radius exactly matches the required area ratio
    r[-1] = R_exit
    
    return x, r

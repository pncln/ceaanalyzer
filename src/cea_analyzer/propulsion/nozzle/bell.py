"""
Bell Nozzle Design Module for CEA Analyzer
-----------------------------------------

This module provides functionality for designing bell nozzle contours,
which are industry standard for most rocket engines.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
import pandas as pd

from .base import get_throat_properties


def bell_nozzle(cea_data, R_throat=None, N=100, percent_bell=80):
    """
    Generate a Bell nozzle contour following the Rao method.
    
    Bell nozzles are the industry standard for rocket engines,
    providing better performance than conical nozzles while being shorter in length.
    The standard "80% bell" refers to a nozzle that is 80% the length of a 15° conical
    nozzle with the same area ratio.
    
    Parameters
    ----------
    cea_data : dict or pandas.Series
        CEA data containing at minimum: area_ratio
    R_throat : float, optional
        Throat radius in meters, if None it will be calculated from CEA data
    N : int, optional
        Number of points to generate for the contour
    percent_bell : float, optional
        Percentage of the equivalent 15° conical nozzle length, default 80%
        
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
    
    # Calculate length of an equivalent 15° conical nozzle
    half_angle_rad = np.radians(15)
    L_conical = (R_exit - R_throat) / np.tan(half_angle_rad)
    
    # Bell nozzle length based on percentage
    L_bell = L_conical * (percent_bell / 100.0)
    
    # Initial and final angles for the bell curve
    # Standard values for 80% bell from NASA literature:
    # - Initial angle at throat: 30° to 45°
    # - Exit angle: 7° to 12°
    if percent_bell <= 60:
        theta_initial = np.radians(40)  # More aggressive for shorter bells
        theta_exit = np.radians(15)     # Larger exit angle for shorter bells
    elif percent_bell <= 75:
        theta_initial = np.radians(35)  # Standard for 60-75% bells
        theta_exit = np.radians(12)     # Standard for 60-75% bells
    elif percent_bell <= 85:
        theta_initial = np.radians(30)  # Standard for 80% bell
        theta_exit = np.radians(8)      # Standard for 80% bell
    else:
        theta_initial = np.radians(25)  # Less aggressive for longer bells
        theta_exit = np.radians(5)      # Smaller exit angle for longer bells
    
    # Define the parabolic contour
    # Using a parabola: r(x) = a*x^2 + b*x + c
    
    # Constraints:
    # 1. r(0) = R_throat                            (throat constraint)
    # 2. r(L_bell) = R_exit                         (exit constraint)
    # 3. r'(0) = tan(theta_initial)                 (initial angle constraint)
    # 4. r'(L_bell) = tan(theta_exit)               (exit angle constraint)
    
    # Solve for the parabola coefficients
    # For a cubic polynomial: r(x) = a*x^3 + b*x^2 + c*x + d
    A = np.array([
        [0, 0, 0, 1],                                    # r(0) = d
        [L_bell**3, L_bell**2, L_bell, 1],               # r(L_bell) = ax^3 + bx^2 + cx + d
        [0, 0, 1, 0],                                   # r'(0) = c
        [3*L_bell**2, 2*L_bell, 1, 0]                    # r'(L_bell) = 3ax^2 + 2bx + c
    ])
    
    b = np.array([
        R_throat,
        R_exit,
        np.tan(theta_initial),
        np.tan(theta_exit)
    ])
    
    # Solve the linear system
    coefs = np.linalg.solve(A, b)
    a, b, c, d = coefs
    
    # Generate points along the contour
    x = np.linspace(0, L_bell, N)
    r = a*x**3 + b*x**2 + c*x + d
    
    # Ensure the first point is exactly at the throat
    r[0] = R_throat
    
    # Ensure the last point is exactly at the exit
    r[-1] = R_exit
    
    return x, r

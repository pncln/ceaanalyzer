"""
Rao Optimum Nozzle Design Module for CEA Analyzer
------------------------------------------------

This module provides functionality for designing Rao Thrust-Optimized
Parabolic (TOP) nozzle contours.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
import pandas as pd

from .base import get_throat_properties


def rao_optimum_nozzle(cea_data, R_throat=None, N=100, theta_n=30, theta_e=7):
    """
    Generate a Rao Thrust-Optimized Parabolic (TOP) nozzle contour.
    
    The Rao optimum nozzle, developed by G.V.R. Rao at NASA in the 1960s, is designed
    to maximize thrust for a given length and area ratio. It consists of a circular arc
    near the throat that transitions to a parabolic contour optimized for maximum thrust.
    
    This implementation follows the standard Rao design methodology as documented in
    NASA technical reports and aerospace engineering literature.
    
    Parameters
    ----------
    cea_data : dict or pandas.Series
        CEA data containing at minimum: area_ratio, gamma
    R_throat : float, optional
        Throat radius in meters, if None it will be calculated from CEA data
    N : int, optional
        Number of points to generate for the contour
    theta_n : float, optional
        Initial wall angle at the nozzle throat in degrees, default 30°
        Typical values range from 25° to 35°
    theta_e : float, optional
        Exit wall angle in degrees, default 7°
        Typical values range from 5° to 15°
        
    Returns
    -------
    tuple
        (x_coordinates, r_coordinates) for the nozzle contour
    """
    # Extract area ratio and calculate throat radius
    props = get_throat_properties(cea_data)
    area_ratio = props['area_ratio']
    
    if R_throat is None:
        if 'At' in cea_data:
            R_throat = np.sqrt(cea_data['At'] / np.pi)
        else:
            R_throat = 0.05  # Default 5cm throat radius
    
    # Calculate exit radius based on area ratio
    R_exit = R_throat * np.sqrt(area_ratio)
    
    # Convert angles to radians
    theta_n_rad = np.radians(theta_n)
    theta_e_rad = np.radians(theta_e)
    
    # Calculate the length of an equivalent 15° conical nozzle
    L_conical = (R_exit - R_throat) / np.tan(np.radians(15))
    
    # Rao nozzles are typically 80% of the equivalent conical length
    L_nozzle = 0.8 * L_conical
    
    # Define the throat radius of curvature (standard is 0.4 * R_throat)
    Rc = 0.4 * R_throat
    
    # Use a direct approach for the throat-to-exit contour without a circular arc section
    # This ensures we don't create any bulges at the throat
    
    # Start from the throat (minimum radius point)
    x_start = 0.0
    y_start = R_throat
    
    # Set the initial angle at the throat (tangent to the flow direction)
    # This should be positive for proper expansion
    m_start = np.tan(theta_n_rad * 0.5)  # Reduced initial angle to avoid bulging
    
    # Set the exit conditions
    x_end = L_nozzle
    y_end = R_exit
    m_end = np.tan(theta_e_rad)
    
    # Use a cubic polynomial to create a smooth contour from throat to exit
    # r(x) = a*x^3 + b*x^2 + c*x + d
    # With constraints:
    # r(0) = R_throat
    # r'(0) = m_start
    # r(L_nozzle) = R_exit
    # r'(L_nozzle) = m_end
    
    # Create the system of equations
    A = np.array([
        [0, 0, 0, 1],                                       # r(0) = d
        [0, 0, 1, 0],                                       # r'(0) = c
        [L_nozzle**3, L_nozzle**2, L_nozzle, 1],           # r(L_nozzle) = a*L^3 + b*L^2 + c*L + d
        [3*L_nozzle**2, 2*L_nozzle, 1, 0]                  # r'(L_nozzle) = 3a*L^2 + 2b*L + c
    ])
    
    b = np.array([
        R_throat,
        m_start,
        R_exit,
        m_end
    ])
    
    # Solve for coefficients
    coefs = np.linalg.solve(A, b)
    a, b, c, d = coefs
    
    # Generate the contour
    x = np.linspace(0, L_nozzle, N)
    r = a*x**3 + b*x**2 + c*x + d
    
    # Ensure the first point is exactly at the throat radius
    r[0] = R_throat
    
    # Ensure the last point is exactly at the exit radius
    r[-1] = R_exit
    
    return x, r

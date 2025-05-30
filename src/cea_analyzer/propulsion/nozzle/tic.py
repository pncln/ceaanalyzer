"""
Truncated Ideal Contour (TIC) Nozzle Design Module
-------------------------------------------------

This module provides functionality for designing Truncated Ideal Contour nozzles,
which are optimized for minimum length while maintaining high performance.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
import pandas as pd

from .base import get_throat_properties
from .moc import mach_from_area_ratio, generate_moc_contour


def truncated_ideal_contour(cea_data, R_throat=None, N=100, truncation_factor=0.8):
    """
    Generate a Truncated Ideal Contour (TIC) nozzle.
    
    The Truncated Ideal Contour (TIC) nozzle is derived from an ideal bell nozzle
    but truncated to a practical length. This approach was developed for applications
    where weight and length are critical factors but high performance is still required.
    
    Parameters
    ----------
    cea_data : dict or pandas.Series
        CEA data containing at minimum: area_ratio, gamma
    R_throat : float, optional
        Throat radius in meters, if None it will be calculated from CEA data
    N : int, optional
        Number of points to generate for the contour
    truncation_factor : float, optional
        Factor to truncate the ideal contour length (0-1), default 0.8
        Typical values range from 0.6 to 0.9
        
    Returns
    -------
    tuple
        (x_coordinates, r_coordinates) for the nozzle contour
    """
    # Extract area ratio and gamma from CEA data
    props = get_throat_properties(cea_data)
    area_ratio = props['area_ratio']
    gamma = props['gamma']
    
    # Calculate throat radius if not provided
    if R_throat is None:
        if 'At' in cea_data:
            R_throat = np.sqrt(cea_data['At'] / np.pi)
        else:
            R_throat = 0.05  # Default 5cm throat radius
    
    # Calculate exit radius based on area ratio
    R_exit = R_throat * np.sqrt(area_ratio)
    
    # Generate ideal contour using Method of Characteristics
    # This will be our baseline ideal contour
    x_ideal, r_ideal = generate_moc_contour(area_ratio, gamma, N=N, R_throat=R_throat)
    
    # Ideal contour length
    L_ideal = x_ideal[-1]
    
    # Calculate truncated length
    L_truncated = L_ideal * truncation_factor
    
    # Create a new x-coordinate array for the truncated contour
    x_truncated = np.linspace(0, L_truncated, N)
    
    # Interpolate r-coordinates from the ideal contour
    # Use a cubic spline interpolation for smooth results
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x_ideal, r_ideal)
    r_truncated = cs(x_truncated)
    
    # The interpolation might not give exactly the right exit radius
    # so we'll adjust the last point to match the area ratio
    r_truncated[-1] = R_exit
    
    # Ensure the throat radius is exact
    r_truncated[0] = R_throat
    
    return x_truncated, r_truncated

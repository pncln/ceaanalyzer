"""
Method of Characteristics (MOC) Nozzle Design Module
--------------------------------------------------

This module provides functionality for designing rocket nozzle contours
using the Method of Characteristics (MOC).
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union, Any
import pandas as pd

from .base import get_throat_properties
from .moc import generate_moc_contour, prandtl_meyer, inverse_prandtl_meyer, mach_from_area_ratio


def moc_nozzle(cea_data, R_throat=None, N=30, nu_max=None):
    """
    Generate a Method of Characteristics (MOC) nozzle contour.
    
    The Method of Characteristics is the most accurate approach for designing supersonic
    nozzles. It uses the method of characteristics to solve the inviscid, irrotational
    supersonic flow equations, producing a nozzle contour that provides uniform, parallel
    flow at the exit plane.
    
    This implementation uses the standard aerospace engineering approach for axisymmetric
    nozzle design via MOC as described in Anderson's "Modern Compressible Flow" and
    Zucrow & Hoffman's "Gas Dynamics" textbooks.
    
    Parameters
    ----------
    cea_data : dict or pandas.Series
        CEA data containing at minimum: area_ratio, gamma
    R_throat : float, optional
        Throat radius in meters, if None it will be calculated from CEA data
    N : int, optional
        Number of characteristic lines (higher values give more accurate contours)
    nu_max : float, optional
        Maximum Prandtl-Meyer angle in degrees, if None it will be calculated from area_ratio
        
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
    
    # Calculate the maximum Prandtl-Meyer angle if not provided
    if nu_max is None:
        # Calculate exit Mach number from area ratio
        M_exit = mach_from_area_ratio(area_ratio, gamma)
        # Calculate corresponding Prandtl-Meyer angle
        nu_max_rad = prandtl_meyer(M_exit, gamma)
    else:
        # Convert from degrees to radians if provided
        nu_max_rad = np.radians(nu_max)
    
    # Use the MOC algorithm to generate the contour
    x, r = generate_moc_contour(area_ratio, gamma, N=N, R_throat=R_throat)
    
    return x, r

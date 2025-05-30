"""
Utility Functions Module
----------------------

This module provides various utility functions for the CEA Analyzer application,
including atmospheric and fluid dynamics calculations.
"""

import logging
import numpy as np
from typing import Callable, Union, Optional, Tuple, List


# Standard atmospheric model constants
ISA_PRESSURE_SEA_LEVEL: float = 101325.0  # Pa
ISA_TEMPERATURE_SEA_LEVEL: float = 288.15  # K
ISA_LAPSE_RATE_TROPOSPHERE: float = 0.0065  # K/m
ISA_GRAVITY: float = 9.80665  # m/s²
ISA_GAS_CONSTANT: float = 287.05  # J/(kg·K)
ISA_TROPOPAUSE_ALTITUDE: float = 11000.0  # m
ISA_TROPOPAUSE_TEMPERATURE: float = 216.65  # K

# Other useful constants
R_UNIVERSAL: float = 8314.46  # Universal gas constant J/(kmol·K)


def ambient_pressure(alt_m: float) -> float:
    """
    Calculate ambient pressure at a given altitude using the International 
    Standard Atmosphere (ISA) model.
    
    Parameters
    ----------
    alt_m : float
        Altitude in meters above mean sea level
        
    Returns
    -------
    float
        Ambient pressure in Pascals (Pa)
        
    Notes
    -----
    Uses the ISA model for troposphere (up to 11 km) and 
    a simplified model for stratosphere.
    """
    if not isinstance(alt_m, (int, float)):
        raise TypeError("Altitude must be a numeric value")
    
    # Define ISA constants
    P0 = ISA_PRESSURE_SEA_LEVEL
    T0 = ISA_TEMPERATURE_SEA_LEVEL
    L = ISA_LAPSE_RATE_TROPOSPHERE
    g = ISA_GRAVITY
    R = ISA_GAS_CONSTANT
    
    if alt_m <= ISA_TROPOPAUSE_ALTITUDE:
        # Troposphere (below 11 km) - temperature decreases linearly with altitude
        return P0 * (1 - L * alt_m / T0) ** (g / (R * L))
    else:
        # Simplified stratosphere (above 11 km) - constant temperature
        # Pressure ratio at tropopause
        pressure_ratio_tropopause = (ISA_TROPOPAUSE_TEMPERATURE / T0) ** (g / (R * L))
        # Exponential decay from tropopause
        return P0 * pressure_ratio_tropopause * np.exp(-g * 
                                                    (alt_m - ISA_TROPOPAUSE_ALTITUDE) / 
                                                    (R * ISA_TROPOPAUSE_TEMPERATURE))


def solve_mach(p_ratio: float, gamma: float = 1.4) -> float:
    """
    Numerically solve for Mach number from total-to-static pressure ratio.
    
    Parameters
    ----------
    p_ratio : float
        Ratio of static pressure to total pressure (p/p0)
    gamma : float, optional
        Specific heat ratio, default is 1.4 for air
        
    Returns
    -------
    float
        Mach number corresponding to the given pressure ratio
        
    Notes
    -----
    Uses a binary search algorithm to find the Mach number.
    The isentropic relation is: p/p0 = (1 + 0.5*(gamma-1)*M²)^(-gamma/(gamma-1))
    """
    if not 0 < p_ratio <= 1.0:
        logging.warning(f"Invalid pressure ratio: {p_ratio}, should be between 0 and 1")
        if p_ratio > 1.0:
            return 0.0  # Subsonic
        if p_ratio <= 0.0:
            return 50.0  # Very high Mach number
    
    if gamma <= 1.0:
        raise ValueError(f"Invalid gamma value: {gamma}, should be greater than 1.0")
    
    # Binary search bounds
    lo, hi = 1e-6, 50.0
    
    # Isentropic relation function
    def f(M: float) -> float:
        return (1 + 0.5*(gamma-1)*M*M) ** (-gamma/(gamma-1))
    
    # Binary search
    tolerance = 1e-8
    while hi - lo > tolerance:
        mid = 0.5*(lo + hi)
        if f(mid) > p_ratio:
            lo = mid
        else:
            hi = mid
    
    return 0.5*(lo + hi)


def mach_from_area_ratio(area_ratio: float, gamma: float = 1.4) -> float:
    """
    Calculate Mach number from area ratio (A/A*) for isentropic flow.
    
    Parameters
    ----------
    area_ratio : float
        Ratio of area to throat area (A/A*)
    gamma : float, optional
        Specific heat ratio, default is 1.4
        
    Returns
    -------
    float
        Mach number corresponding to the given area ratio
    """
    if area_ratio < 1.0:
        raise ValueError(f"Area ratio must be >= 1.0, got {area_ratio}")
    
    # Initial guess for Mach number based on area ratio
    if area_ratio > 5.0:
        M = 1.0 + 0.8 * np.log10(area_ratio)  # Better initial guess for high area ratios
    else:
        M = 1.0 + 0.5 * (area_ratio - 1.0)     # Linear approximation for smaller ratios
    
    # Newton-Raphson iteration
    max_iterations = 50
    tolerance = 1e-8
    
    for i in range(max_iterations):
        # Area ratio function from isentropic flow equations
        term = (1.0 + 0.5 * (gamma - 1.0) * M * M)
        f = (1.0 / M) * (2.0 / (gamma + 1.0) * term) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))) - area_ratio
        
        # Derivative of the area ratio function
        df = (1.0 / M**2) * (2.0 / (gamma + 1.0) * term) ** ((gamma + 1.0) / (2.0 * (gamma - 1.0)))
        df = df * (1.0 - (gamma + 1.0) / (2.0) * M**2 / term)
        
        # Update using Newton-Raphson
        M_new = M - f / df
        
        if abs(M_new - M) < tolerance:
            return M_new
            
        M = M_new
    
    logging.warning(f"Mach calculation did not converge after {max_iterations} iterations")
    return M  # Return best estimate


def pressure_ratio_from_mach(mach: float, gamma: float = 1.4) -> float:
    """
    Calculate the pressure ratio (p/p0) from Mach number for isentropic flow.
    
    Parameters
    ----------
    mach : float
        Mach number
    gamma : float, optional
        Specific heat ratio, default is 1.4 for air
        
    Returns
    -------
    float
        Pressure ratio (p/p0)
    """
    return (1.0 + 0.5 * (gamma - 1.0) * mach**2) ** (-gamma / (gamma - 1.0))

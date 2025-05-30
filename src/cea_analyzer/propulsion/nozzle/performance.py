"""
Nozzle Performance Calculation Module
-----------------------------------

This module provides functions to calculate rocket nozzle performance parameters
based on contour geometry and thermochemical data.
"""

import numpy as np
from typing import Dict, Tuple, Union, List, Any

from .base import get_throat_properties


def calculate_performance(cea_data: Dict[str, Any], nozzle_coordinates: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Any]:
    """
    Calculate performance parameters for the designed nozzle following aerospace engineering standards.
    
    This function calculates key performance metrics for rocket nozzles based on the
    contour geometry and CEA thermochemical data. The calculations follow standard
    aerospace engineering methods as documented in Sutton's "Rocket Propulsion Elements"
    and NASA technical publications.
    
    Parameters
    ----------
    cea_data : dict or pandas.Series
        CEA thermochemical data containing at minimum: gamma, area_ratio, and chamber pressure
    nozzle_coordinates : tuple
        (x, r) coordinates of the nozzle contour
        
    Returns
    -------
    dict
        Dictionary of performance parameters including area ratio, thrust coefficient,
        divergence loss factor, and other metrics
    """
    # Extract properties from CEA data
    props = get_throat_properties(cea_data)
    gamma = props['gamma']        # Specific heat ratio
    p_c = props['p_c']            # Chamber pressure (Pa)
    m_exit = props['m_exit']      # Exit Mach number
    
    # Extract nozzle coordinates
    x, r = nozzle_coordinates
    
    # Find the throat location (minimum radius)
    throat_idx = np.argmin(r)
    r_throat = r[throat_idx]
    r_exit = r[-1]
    x_throat = x[throat_idx]
    x_exit = x[-1]
    
    # Ensure positive values to prevent division by zero
    r_throat = max(r_throat, 1e-6)
    
    # Calculate areas
    A_throat = np.pi * r_throat**2  # Throat area (m²)
    A_exit = np.pi * r_exit**2      # Exit area (m²)
    
    # Calculate area ratio
    area_ratio = A_exit / A_throat
    
    # Calculate exit pressure using isentropic flow relations
    # p_e/p_c = (1 + (gamma-1)/2 * M²)^(-gamma/(gamma-1))
    p_ratio = (1 + (gamma-1)/2 * m_exit**2)**(-gamma/(gamma-1))
    p_exit = p_c * p_ratio  # Exit pressure (Pa)
    
    # Calculate ideal thrust coefficient
    # C_f = sqrt((2*gamma²/(gamma-1)) * (2/(gamma+1))^((gamma+1)/(gamma-1)) * (1-(p_e/p_c)^((gamma-1)/gamma)))
    C_f_ideal = np.sqrt(((2*gamma**2)/(gamma-1)) * (2/(gamma+1))**((gamma+1)/(gamma-1)) * 
                       (1-(p_exit/p_c)**((gamma-1)/gamma)))
    
    # Calculate divergence half-angle at the exit
    # Find the last 10% of the nozzle for a better approximation of the exit angle
    exit_section_start = int(0.9 * len(x))
    if exit_section_start < len(x) - 2:  # Ensure we have enough points
        # Use linear regression to find the average slope of the exit section
        exit_x = x[exit_section_start:]
        exit_r = r[exit_section_start:]
        slope, _ = np.polyfit(exit_x, exit_r, 1)
        divergence_angle = np.arctan(slope)  # radians
    else:
        # Fallback to using the last two points
        if len(x) >= 2:
            slope = (r[-1] - r[-2]) / (x[-1] - x[-2])
            divergence_angle = np.arctan(slope)  # radians
        else:
            # Default if we don't have enough points
            divergence_angle = np.radians(15)  # Default 15° in radians
    
    # Calculate divergence loss factor (λ)
    # λ = (1 + cos(α))/2 where α is the divergence half-angle
    lambda_d = 0.5 * (1 + np.cos(divergence_angle))
    
    # Apply divergence loss to the thrust coefficient
    C_f = C_f_ideal * lambda_d
    
    # Calculate discharge coefficient (C_d)
    # For well-designed nozzles, C_d is typically 0.95-0.99
    # More accurate calculation would require boundary layer analysis
    C_d = 0.98
    
    # Calculate nozzle surface area (for heat transfer and weight estimation)
    surface_area = 0
    for i in range(1, len(x)):
        dx = x[i] - x[i-1]
        dr = r[i] - r[i-1]
        ds = np.sqrt(dx**2 + dr**2)  # Length of the segment
        r_avg = (r[i] + r[i-1]) / 2   # Average radius of the segment
        surface_area += 2 * np.pi * r_avg * ds  # Surface area of truncated cone
    
    # Calculate length to throat radius ratio (important design parameter)
    length_to_throat_ratio = (x_exit - x_throat) / r_throat
    
    # Calculate nozzle efficiency
    # This is a simplified approach; full analysis would include boundary layer effects
    # Typical values range from 0.85 to 0.98 depending on nozzle type
    if area_ratio < 10:
        base_efficiency = 0.92
    elif area_ratio < 50:
        base_efficiency = 0.94
    else:
        base_efficiency = 0.96
        
    # Adjust for nozzle type
    nozzle_type = "unknown"
    if "nozzle_type" in cea_data:
        nozzle_type = cea_data["nozzle_type"]
    
    if "conical" in nozzle_type.lower():
        efficiency_factor = 0.98  # Conical nozzles are less efficient
    elif "bell" in nozzle_type.lower() or "rao" in nozzle_type.lower():
        efficiency_factor = 1.01  # Bell nozzles are more efficient
    elif "moc" in nozzle_type.lower():
        efficiency_factor = 1.02  # MOC nozzles are most efficient
    else:
        efficiency_factor = 1.0   # Default
    
    nozzle_efficiency = min(0.99, base_efficiency * efficiency_factor * lambda_d)
    
    # Return comprehensive performance metrics
    return {
        'area_ratio': area_ratio,
        'pressure_ratio': p_c / p_exit,
        'thrust_coefficient': C_f,
        'ideal_thrust_coefficient': C_f_ideal,
        'discharge_coefficient': C_d,
        'surface_area': surface_area,
        'nozzle_efficiency': nozzle_efficiency,
        'divergence_loss_factor': lambda_d,
        'divergence_angle_deg': np.degrees(divergence_angle),
        'length_to_throat_ratio': length_to_throat_ratio,
        'exit_mach_number': m_exit
    }

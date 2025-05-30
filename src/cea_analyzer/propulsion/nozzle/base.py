"""
Base Nozzle Design Module for CEA Analyzer
------------------------------------------

This module provides base functionality for rocket nozzle design
including common utilities and interfaces.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Dict, Tuple, Optional, Union, Any
import pandas as pd


def get_throat_properties(cea_data):
    """
    Extract relevant throat properties from CEA data.
    
    Parameters
    ----------
    cea_data : dict or pandas.Series
        CEA data containing at minimum: gamma, Pc, area_ratio
        
    Returns
    -------
    dict
        Dictionary of throat properties
    """
    if hasattr(cea_data, 'to_dict'):  # If it's a pandas Series
        cea_data = cea_data.to_dict()
    
    # Extract or calculate throat properties
    gamma = cea_data.get('gamma', 1.2)  # Default to 1.2 if not available
    if 'gamma' not in cea_data:
        # CEA data typically provides gamma as specific heat ratio
        if 'GAMMAs' in cea_data:
            gamma = cea_data['GAMMAs']
    
    # Get chamber pressure in Pa
    p_c = cea_data.get('Pc (bar)', 50) * 1e5  # Convert bar to Pa
    
    # Get area ratio (exit area / throat area)
    area_ratio = cea_data.get('Ae/At', 8.0)
    
    # Temperature at chamber
    t_c = cea_data.get('T_chamber (K)', 3500)
    
    # Calculate exit Mach number from area ratio
    from .moc import mach_from_area_ratio
    m_exit = mach_from_area_ratio(area_ratio, gamma)
    
    # Return dictionary of properties
    return {
        'gamma': gamma,
        'p_c': p_c,
        'area_ratio': area_ratio,
        't_c': t_c,
        'm_exit': m_exit
    }


def add_inlet_section(x, r, R_throat, chamber_radius_ratio=2.5, chamber_length_ratio=3.0, N_inlet=40):
    """
    Add an inlet section (combustion chamber and converging section) to the nozzle contour.
    
    This function adds a properly designed combustion chamber and converging section to
    a supersonic nozzle contour. The design follows standard aerospace engineering practices
    for liquid rocket engines, with appropriate contraction ratio and smooth transitions.
    
    Parameters
    ----------
    x : ndarray
        x-coordinates of the nozzle contour, starting at the throat (x=0)
    r : ndarray
        r-coordinates of the nozzle contour
    R_throat : float
        Throat radius in meters
    chamber_radius_ratio : float, optional
        Ratio of chamber radius to throat radius, default 2.5
        Typical values range from 2.0 to 3.0 for liquid rocket engines
    chamber_length_ratio : float, optional
        Ratio of chamber length to throat radius, default 3.0
        Typical values range from 2.0 to 5.0 depending on engine type
    N_inlet : int, optional
        Number of points to generate for the inlet section
        
    Returns
    -------
    tuple
        (x_full, r_full) full nozzle contour including inlet
    """
    # Calculate chamber radius and length
    R_chamber = R_throat * chamber_radius_ratio
    L_chamber = R_throat * chamber_length_ratio
    
    # Create the chamber section (cylindrical)
    chamber_length = L_chamber  # Length of the cylindrical chamber section
    x_chamber = np.linspace(-chamber_length, -chamber_length/2, N_inlet//4)
    r_chamber = np.ones_like(x_chamber) * R_chamber
    
    # Create converging section (cubic curve from chamber to throat)
    x_converging = np.linspace(-chamber_length/2, 0, N_inlet)
    
    # Normalized position (0 at chamber, 1 at throat)
    s = (x_converging + chamber_length/2) / (chamber_length/2)
    
    # Use a cubic function to create a smooth converging section
    # This satisfies: r(0) = R_chamber, r(1) = R_throat, r'(0) = 0, r'(1) = 0
    r_converging = R_chamber * (1 - 3*s**2 + 2*s**3) + R_throat * (3*s**2 - 2*s**3)
    
    # Combine inlet with nozzle contour
    x_full = np.concatenate([x_chamber, x_converging, x])
    r_full = np.concatenate([r_chamber, r_converging, r])
    
    return x_full, r_full


def export_nozzle_coordinates(x, r, filename, include_header=True, format_type='csv'):
    """
    Export nozzle coordinates to a file.
    
    Parameters
    ----------
    x : ndarray
        x-coordinates of the nozzle contour
    r : ndarray
        r-coordinates of the nozzle contour
    filename : str
        Filename for the exported coordinates
    include_header : bool, optional
        Whether to include a header row, default True
    format_type : str, optional
        Export format ('csv', 'txt', or 'dat'), default 'csv'
        
    Returns
    -------
    bool
        True if export was successful
    """
    try:
        # Determine the delimiter based on format type
        if format_type.lower() == 'csv':
            delimiter = ','
        else:
            delimiter = '\t'
        
        # Create a 2D array of coordinates
        coords = np.column_stack((x, r))
        
        # Export based on format type
        if include_header:
            header = "x (m)" + delimiter + "r (m)"
            np.savetxt(filename, coords, delimiter=delimiter, header=header, comments='# ')
        else:
            np.savetxt(filename, coords, delimiter=delimiter)
        
        return True
    except Exception as e:
        print(f"Error exporting coordinates: {e}")
        return False


def plot_nozzle_contour(x, r, title="Rocket Nozzle Contour", show_grid=True, 
                       show_dimensions=True, equal_aspect=True):
    """
    Plot the nozzle contour.
    
    Parameters
    ----------
    x : ndarray
        x-coordinates of the nozzle contour
    r : ndarray
        r-coordinates of the nozzle contour
    title : str, optional
        Plot title, default "Rocket Nozzle Contour"
    show_grid : bool, optional
        Whether to show grid lines, default True
    show_dimensions : bool, optional
        Whether to show key dimensions, default True
    equal_aspect : bool, optional
        Whether to use equal aspect ratio, default True
        
    Returns
    -------
    tuple
        (fig, ax) the figure and axis objects
    """
    # Create a figure
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Plot the upper contour
    ax.plot(x, r, 'b-', linewidth=2)
    
    # Mirror the contour to show full nozzle (if r is not all zeros)
    if not np.all(r == 0):
        ax.plot(x, -r, 'b-', linewidth=2)
    
    # Fill the nozzle area for better visualization
    ax.fill_between(x, r, -r, color='lightblue', alpha=0.3)
    
    # Add centerline
    ax.axhline(y=0, color='k', linestyle='-', linewidth=1)
    
    # Set labels and title
    ax.set_xlabel('Axial Position (m)')
    ax.set_ylabel('Radial Position (m)')
    ax.set_title(title)
    
    # Add grid
    if show_grid:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Equal aspect ratio for proper visualization
    if equal_aspect:
        ax.set_aspect('equal')
    
    # Add key dimensions if requested
    if show_dimensions:
        # Throat location
        throat_idx = np.argmin(r)
        throat_x, throat_r = x[throat_idx], r[throat_idx]
        
        # Exit dimensions
        exit_x, exit_r = x[-1], r[-1]
        
        # Annotate throat and exit
        ax.plot([throat_x, throat_x], [0, throat_r], 'r--', linewidth=1)
        ax.plot([exit_x, exit_x], [0, exit_r], 'r--', linewidth=1)
        
        # Add dimension text
        ax.text(throat_x, throat_r*1.1, f'Throat\nR={throat_r:.3f}m', 
                ha='center', va='bottom', color='red')
        ax.text(exit_x, exit_r*1.1, f'Exit\nR={exit_r:.3f}m', 
                ha='center', va='bottom', color='red')
        
        # Add nozzle length dimension
        if len(x) > 1:
            nozzle_length = exit_x - throat_x
            ax.annotate(f'Length: {nozzle_length:.3f}m', 
                       xy=(throat_x + nozzle_length/2, -max(r)*0.2),
                       xytext=(throat_x + nozzle_length/2, -max(r)*0.4),
                       arrowprops=dict(arrowstyle='<->'),
                       ha='center', va='center')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig, ax


def calculate_performance(cea_data, nozzle_coordinates):
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
    # Extract nozzle coordinates
    x, r = nozzle_coordinates
    
    # Get throat properties
    props = get_throat_properties(cea_data)
    gamma = props['gamma']
    p_c = props['p_c']
    m_exit = props['m_exit']
    
    # Find throat and exit indices
    throat_idx = np.argmin(r)
    exit_idx = len(r) - 1
    
    # Calculate actual throat and exit area
    A_throat = np.pi * r[throat_idx]**2
    A_exit = np.pi * r[exit_idx]**2
    
    # Calculate actual area ratio
    area_ratio = A_exit / A_throat
    
    # Calculate ideal thrust coefficient (1D theory)
    p_e_p_c_ratio = (1 + (gamma-1)/2 * m_exit**2)**(-gamma/(gamma-1))
    C_F_ideal = np.sqrt((2*gamma**2)/(gamma-1) * (2/(gamma+1))**((gamma+1)/(gamma-1)) * 
                       (1 - p_e_p_c_ratio)) + area_ratio * p_e_p_c_ratio
    
    # Calculate divergence loss factor (approximation)
    # Estimate angle at exit
    if len(x) > 5:
        # Use last few points to estimate exit angle
        x_end = x[-5:]
        r_end = r[-5:]
        
        # Linear fit to get slope
        slope = np.polyfit(x_end, r_end, 1)[0]
        exit_angle_rad = np.arctan(slope)
        exit_angle_deg = np.degrees(exit_angle_rad)
        
        # Divergence loss factor lambda = (1 + cos(alpha))/2
        divergence_factor = (1 + np.cos(exit_angle_rad)) / 2
    else:
        # Default estimate if not enough points
        exit_angle_deg = 12  # Typical exit angle for bell nozzles
        exit_angle_rad = np.radians(exit_angle_deg)
        divergence_factor = (1 + np.cos(exit_angle_rad)) / 2
    
    # Calculate actual thrust coefficient
    C_F_actual = C_F_ideal * divergence_factor
    
    # Calculate characteristic velocity c* (in m/s)
    c_star = np.sqrt((gamma * 8314.46 / cea_data.get('mol_weight', 20)) * 
                    props['t_c'] / gamma) * ((gamma+1)/2)**((gamma+1)/(2*(gamma-1)))
    
    # Calculate specific impulse (in seconds)
    Isp = C_F_actual * c_star / 9.80665
    
    # Nozzle length from throat to exit
    nozzle_length = x[exit_idx] - x[throat_idx]
    
    # Surface area for heat transfer calculations
    # Approximate using trapezoidal rule along contour
    if len(x) > 1:
        segment_lengths = np.sqrt(np.diff(x)**2 + np.diff(r)**2)
        avg_radii = (r[:-1] + r[1:]) / 2
        surface_area = 2 * np.pi * np.sum(avg_radii * segment_lengths)
    else:
        surface_area = 0
    
    # Nozzle expansion ratio (exit pressure / ambient pressure)
    p_exit = p_c * p_e_p_c_ratio  # Exit pressure
    p_ambient = 101325  # Standard sea level pressure (Pa)
    expansion_ratio = p_exit / p_ambient
    
    # Compile results
    performance = {
        'throat_radius': r[throat_idx],
        'exit_radius': r[exit_idx],
        'area_ratio': area_ratio,
        'nozzle_length': nozzle_length,
        'surface_area': surface_area,
        'exit_angle_deg': exit_angle_deg,
        'divergence_factor': divergence_factor,
        'CF_ideal': C_F_ideal,
        'CF_actual': C_F_actual,
        'c_star': c_star,
        'Isp': Isp,
        'expansion_ratio': expansion_ratio,
        'p_exit': p_exit
    }
    
    return performance

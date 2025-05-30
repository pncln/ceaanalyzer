"""
Rocket System Analysis Module
----------------------------

This module provides functions for analyzing rocket propulsion systems
based on NASA-CEA data. It includes calculations for rocket performance
parameters, such as thrust, mass flow rate, and delta-V.
"""

import logging
from typing import Dict, List, Tuple, Union, Any, Optional

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from ..core.config import G0
from ..utils.utilities import ambient_pressure

# Universal gas constant
R_UNIVERSAL = 8.31446261815324  # J/(mol·K)

# Configure module logger
logger = logging.getLogger(__name__)


def compute_system(df: pd.DataFrame, 
                  vehicle_mass: float = 1000.0,
                  propellant_mass: float = 100.0,
                  initial_mass: Optional[float] = None,
                  mol_weight: float = 0.022) -> Dict[str, Any]:
    """
    Compute rocket nozzle and system parameters from CEA data.
    
    This function calculates key rocket system parameters including thrust,
    specific impulse, nozzle dimensions, and mission characteristics based
    on the best performing mixture ratio from CEA data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing CEA analysis results
    vehicle_mass : float, optional
        Total vehicle mass in kg, default 1000 kg
    propellant_mass : float, optional
        Propellant mass in kg, default 100 kg
    initial_mass : float, optional
        Initial mass for delta-V calculation in kg, default is vehicle_mass
    mol_weight : float, optional
        Molecular weight of exhaust gases in kg/mol, default 0.022 kg/mol
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing the following keys:
        - 'best': Series with the best performance data row
        - 'At': Throat area in m²
        - 'Ae': Exit area in m²
        - 'alts': Array of altitudes for performance evaluation
        - 'Fs': Array of thrust values at different altitudes
        - 'mdot': Propellant mass flow rate in kg/s
        - 'dv': Delta-V capability in m/s
        - 'tb': Burn time in seconds
    
    Raises
    ------
    ValueError
        If required data is missing from the DataFrame
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty")
    
    if "Isp (s)" not in df.columns:
        raise ValueError("DataFrame missing 'Isp (s)' column")
    
    # Set default initial mass if not provided
    if initial_mass is None:
        initial_mass = vehicle_mass
    
    try:
        # 1) Find the best-Isp row
        best = df.loc[df["Isp (s)"].idxmax()]
        logger.info(f"Best Isp: {best['Isp (s)']:.2f} s at O/F = {best['O/F']:.2f}")

        # 2) Extract core parameters
        isp_s = best["Isp (s)"]               # Isp in seconds
        pc = best["Pc (bar)"] * 1e5           # chamber pressure in Pa
        t_ch = best["T_chamber (K)"]          # chamber temperature in K
        area_ratio = best["Expansion Ratio"]   # Ae/At
        
        if pd.isna(area_ratio) or area_ratio <= 0:
            logger.warning("Invalid area ratio, using default value of 8.0")
            area_ratio = 8.0

        # 3) System assumptions
        gamma = best.get("gamma", 1.2)         # specific heat ratio
        r_specific = R_UNIVERSAL / mol_weight  # specific gas constant [J/(kg·K)]

        # 4) Thrust & mass flow
        thrust_design = vehicle_mass * G0       # assume hover thrust [N]
        mdot = thrust_design / (isp_s * G0)     # mass flow [kg/s]

        # 5) Choked-flow throat area calculation
        # Formula: mdot = At · Pc/√Tch · √(γ/R) · (2/(γ+1))^((γ+1)/(2(γ−1)))
        choke_term = (2.0/(gamma+1.0))**((gamma+1.0)/(2.0*(gamma-1.0)))
        at = mdot * np.sqrt(t_ch) / (pc * np.sqrt(gamma/r_specific) * choke_term)

        # 6) Exit area
        ae = at * area_ratio
        
        # 7) Calculate throat and exit diameters
        d_t = 2 * np.sqrt(at / np.pi)  # throat diameter [m]
        d_e = 2 * np.sqrt(ae / np.pi)  # exit diameter [m]

        # 8) Altitude sweep for performance evaluation
        alt_max = 10000.0  # meters
        alt_points = 20    # number of altitude data points
        altitudes = np.linspace(0, alt_max, alt_points)
        
        thrust_values = []
        isp_values = []
        
        for altitude in altitudes:
            # Get ambient pressure at this altitude
            p_ambient = ambient_pressure(altitude)
            
            # Calculate nozzle thrust: mdot·Isp·g0 + pressure thrust
            thrust = mdot * isp_s * G0 + (pc/area_ratio - p_ambient) * ae
            thrust_values.append(thrust)
            
            # Calculate effective Isp at this altitude
            isp_eff = thrust / (mdot * G0)
            isp_values.append(isp_eff)

        # 9) Calculate burn time and delta-V
        burn_time = propellant_mass / mdot
        delta_v = isp_s * G0 * np.log(initial_mass / (initial_mass - propellant_mass))

        # 10) Calculate nozzle performance parameters
        # Ideal thrust coefficient
        ideal_cf = np.sqrt((2 * gamma**2) / (gamma - 1) * 
                        (2 / (gamma + 1))**((gamma + 1) / (gamma - 1)) * 
                        (1 - (1 / area_ratio)**((gamma - 1) / gamma)))
        
        # Divergence loss factor (simplified estimate based on cone half-angle)
        divergence_angle_deg = 15.0  # Default conical nozzle half-angle
        divergence_loss_factor = 0.5 * (1 + np.cos(np.radians(divergence_angle_deg)))
        
        # Actual thrust coefficient
        thrust_coefficient = ideal_cf * divergence_loss_factor
        
        # Exit Mach number (estimated)
        exit_mach_number = 2.2  # Typical value for rockets, would be calculated more precisely
        if area_ratio > 4:
            exit_mach_number = 2.5 + 0.5 * np.log(area_ratio / 4.0)
        
        # Nozzle length to throat diameter ratio (simplified formula)
        length_to_throat_ratio = 0.5 * (np.sqrt(area_ratio) - 1) / np.tan(np.radians(divergence_angle_deg))
        
        # Nozzle surface area (simplified conical approximation)
        nozzle_length = length_to_throat_ratio * d_t
        surface_area = np.pi * (d_t + d_e) * np.sqrt((d_e - d_t)**2 / 4 + nozzle_length**2) / 2
        
        # 11) Compile results
        results = {
            "best": best,
            "At": at,
            "Ae": ae,
            "dt": d_t,
            "de": d_e,
            "alts": altitudes,
            "Fs": thrust_values,
            "Isps": isp_values,
            "mdot": mdot,
            "dv": delta_v,
            "tb": burn_time,
            # Add nozzle performance parameters
            "area_ratio": area_ratio,
            "thrust_coefficient": thrust_coefficient,
            "ideal_thrust_coefficient": ideal_cf,
            "divergence_loss_factor": divergence_loss_factor,
            "divergence_angle_deg": divergence_angle_deg,
            "nozzle_efficiency": divergence_loss_factor,  # Simplified, same as divergence loss in this model
            "length_to_throat_ratio": length_to_throat_ratio,
            "surface_area": surface_area,
            "exit_mach_number": exit_mach_number
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Error in compute_system: {e}")
        raise


def create_performance_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary DataFrame with key performance metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing CEA analysis results
        
    Returns
    -------
    pd.DataFrame
        Summary DataFrame with key performance metrics
    """
    if df.empty:
        return pd.DataFrame()
    
    # Find optimal O/F for each pressure
    optimal = df.loc[df.groupby('Pc (bar)')['Isp (s)'].idxmax()]
    
    # Extract key columns
    summary = optimal[['Pc (bar)', 'O/F', 'Isp (s)', 'T_chamber (K)', 'Delta_H (kJ/kg)']].copy()
    
    # Add column descriptions
    descriptions = {
        'Pc (bar)': 'Chamber pressure',
        'O/F': 'Optimal oxidizer to fuel ratio',
        'Isp (s)': 'Specific impulse',
        'T_chamber (K)': 'Chamber temperature',
        'Delta_H (kJ/kg)': 'Enthalpy change'
    }
    
    # Create a formatted summary
    formatted_summary = []
    for col in summary.columns:
        for _, row in summary.iterrows():
            formatted_summary.append({
                'Parameter': f"{descriptions.get(col, col)} at {row['Pc (bar)']} bar",
                'Value': f"{row[col]:.4g}"
            })
    
    return pd.DataFrame(formatted_summary)


def create_altitude_performance_plot(results: Dict[str, Any]) -> Figure:
    """
    Create a plot of thrust and Isp vs altitude.
    
    Parameters
    ----------
    results : Dict[str, Any]
        Results dictionary from compute_system
        
    Returns
    -------
    Figure
        Matplotlib Figure with the altitude performance plot
    """
    from matplotlib.figure import Figure
    
    # Create figure and axes
    fig = Figure(figsize=(10, 6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    
    # Extract data
    altitudes = results["alts"]
    thrusts = results["Fs"]
    isps = results["Isps"]
    
    # Plot data
    ax1.plot(altitudes, thrusts, 'b-', linewidth=2, label='Thrust')
    ax2.plot(altitudes, isps, 'r-', linewidth=2, label='Isp')
    
    # Set labels and title
    ax1.set_xlabel('Altitude (m)')
    ax1.set_ylabel('Thrust (N)', color='b')
    ax2.set_ylabel('Specific Impulse (s)', color='r')
    fig.suptitle('Rocket Performance vs. Altitude')
    
    # Set tick colors
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='r')
    
    # Add legends and grid
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    ax1.grid(True)
    
    fig.tight_layout()
    
    return fig

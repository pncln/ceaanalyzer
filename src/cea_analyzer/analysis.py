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

from config import G0
from util import ambient_pressure

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

        # 10) Compile results
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
            "tb": burn_time
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
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Plot thrust vs altitude
    ax1.plot(results['alts'] / 1000, np.array(results['Fs']) / 1000, 'b-', linewidth=2)
    ax1.set_ylabel('Thrust (kN)')
    ax1.set_title('Rocket Performance vs Altitude')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Isp vs altitude
    ax2.plot(results['alts'] / 1000, results['Isps'], 'r-', linewidth=2)
    ax2.set_xlabel('Altitude (km)')
    ax2.set_ylabel('Specific Impulse (s)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    return fig

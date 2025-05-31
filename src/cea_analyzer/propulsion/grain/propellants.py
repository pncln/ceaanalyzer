"""
Propellant Definitions
---------------------

This module provides standard propellant definitions for common solid rocket propellants.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any

from .base import PropellantProperties


# Common solid propellant definitions
PROPELLANT_LIBRARY = {
    "KNDX": PropellantProperties(
        name="KNDX",
        density=1800.0,  # kg/m³
        burn_rate_coefficient=0.006,  # m/s at 1 MPa
        burn_rate_exponent=0.4,
        temperature_sensitivity=0.5,  # %/K
        reference_temperature=298.15  # K
    ),
    
    "KNSU": PropellantProperties(
        name="KNSU",
        density=1850.0,  # kg/m³
        burn_rate_coefficient=0.008,  # m/s at 1 MPa
        burn_rate_exponent=0.3,
        temperature_sensitivity=0.6,  # %/K
        reference_temperature=298.15  # K
    ),
    
    "AeroTech-Blue-Thunder": PropellantProperties(
        name="AeroTech Blue Thunder",
        density=1730.0,  # kg/m³
        burn_rate_coefficient=0.0071,  # m/s at 1 MPa
        burn_rate_exponent=0.32,
        temperature_sensitivity=0.3,  # %/K
        reference_temperature=298.15  # K
    ),
    
    "APCP-Ammonium-Perchlorate": PropellantProperties(
        name="APCP (Ammonium Perchlorate Composite)",
        density=1750.0,  # kg/m³
        burn_rate_coefficient=0.005,  # m/s at 1 MPa
        burn_rate_exponent=0.38,
        temperature_sensitivity=0.25,  # %/K
        reference_temperature=298.15  # K
    ),
    
    "RP-1-High-Performance": PropellantProperties(
        name="RP-1 High Performance",
        density=1780.0,  # kg/m³
        burn_rate_coefficient=0.0055,  # m/s at 1 MPa
        burn_rate_exponent=0.36,
        temperature_sensitivity=0.15,  # %/K
        reference_temperature=298.15  # K
    ),
    
    "HTPB-AP-AL": PropellantProperties(
        name="HTPB/AP/AL (Shuttle SRB)",
        density=1770.0,  # kg/m³
        burn_rate_coefficient=0.0068,  # m/s at 1 MPa
        burn_rate_exponent=0.35,
        temperature_sensitivity=0.08,  # %/K
        reference_temperature=298.15  # K
    ),
}


def get_propellant(name: str) -> Optional[PropellantProperties]:
    """
    Get a propellant by name.
    
    Args:
        name: The name of the propellant
        
    Returns:
        The propellant properties or None if not found
    """
    return PROPELLANT_LIBRARY.get(name)


def get_propellant_names() -> List[str]:
    """
    Get a list of all available propellant names.
    
    Returns:
        List of propellant names
    """
    return list(PROPELLANT_LIBRARY.keys())


# Aliases for compatibility with grain visualization widget
def get_propellant_by_name(name: str) -> Optional[PropellantProperties]:
    """
    Get a propellant by name (alias for get_propellant).
    
    Args:
        name: The name of the propellant
        
    Returns:
        The propellant properties or None if not found
    """
    return get_propellant(name)


def get_available_propellants() -> List[str]:
    """
    Get a list of all available propellant names (alias for get_propellant_names).
    
    Returns:
        List of propellant names
    """
    return get_propellant_names()


def add_custom_propellant(
    name: str,
    density: float,
    burn_rate_coefficient: float,
    burn_rate_exponent: float,
    temperature_sensitivity: float,
    reference_temperature: float = 298.15
) -> PropellantProperties:
    """
    Add a custom propellant to the library.
    
    Args:
        name: Name of the propellant
        density: Density in kg/m³
        burn_rate_coefficient: Burn rate coefficient in m/s at 1 MPa
        burn_rate_exponent: Burn rate pressure exponent
        temperature_sensitivity: Temperature sensitivity in %/K
        reference_temperature: Reference temperature in K
        
    Returns:
        The created propellant properties
    """
    prop = PropellantProperties(
        name=name,
        density=density,
        burn_rate_coefficient=burn_rate_coefficient,
        burn_rate_exponent=burn_rate_exponent,
        temperature_sensitivity=temperature_sensitivity,
        reference_temperature=reference_temperature
    )
    
    # Add to library
    PROPELLANT_LIBRARY[name] = prop
    
    return prop

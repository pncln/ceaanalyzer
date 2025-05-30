"""
Base Grain Geometry Module
------------------------

This module provides the base classes for defining rocket motor grain geometries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Callable
import numpy as np
from abc import ABC, abstractmethod

from ...core.logger import get_logger

# Setup logger
logger = get_logger(__name__)


@dataclass
class GrainGeometry(ABC):
    """Base class for grain geometry parameters."""
    length: float = 0.0  # m
    outer_diameter: float = 0.0  # m
    
    @abstractmethod
    def volume(self) -> float:
        """Calculate the total grain volume in cubic meters."""
        pass
    
    @abstractmethod
    def burn_area(self, web_burned: float) -> float:
        """
        Calculate the burn surface area in square meters at a given web distance.
        
        Args:
            web_burned: The distance burned perpendicular to the burning surface (m)
            
        Returns:
            Burn surface area in square meters
        """
        pass
    
    @abstractmethod
    def web_distance(self) -> float:
        """
        Calculate the maximum web distance (thickness) in meters.
        
        The web distance is the maximum distance that can be burned,
        typically from the initial burning surface to the case wall
        or from the initial burning surface to another burning surface.
        
        Returns:
            Maximum web distance in meters
        """
        pass


@dataclass
class PropellantProperties:
    """Class representing propellant thermochemical and physical properties."""
    
    name: str
    density: float  # kg/m³
    burn_rate_coefficient: float  # m/s/(MPa^n)
    burn_rate_exponent: float  # dimensionless
    temperature_sensitivity: float = 0.0  # %/K
    reference_temperature: float = 298.15  # K
    
    def burn_rate(self, pressure: float, temperature: Optional[float] = None) -> float:
        """
        Calculate the propellant burn rate.
        
        Args:
            pressure: Chamber pressure in MPa
            temperature: Propellant temperature in K (optional)
            
        Returns:
            Burn rate in m/s
        """
        # Base burn rate at reference temperature: r = a * P^n
        rate = self.burn_rate_coefficient * (pressure ** self.burn_rate_exponent)
        
        # Apply temperature correction if temperature is provided
        if temperature is not None and self.temperature_sensitivity != 0:
            # Temperature sensitivity typically expressed as percent change per Kelvin
            temperature_factor = 1.0 + (self.temperature_sensitivity / 100.0) * (temperature - self.reference_temperature)
            rate *= temperature_factor
            
        return rate
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PropellantProperties':
        """Create a PropellantProperties instance from a dictionary."""
        return cls(
            name=data.get('name', 'Unknown'),
            density=data.get('density', 1800.0),
            burn_rate_coefficient=data.get('burn_rate_coefficient', 0.005),
            burn_rate_exponent=data.get('burn_rate_exponent', 0.4),
            temperature_sensitivity=data.get('temperature_sensitivity', 0.0),
            reference_temperature=data.get('reference_temperature', 298.15)
        )
    
    def to_dict(self) -> Dict:
        """Convert the propellant properties to a dictionary."""
        return {
            'name': self.name,
            'density': self.density,
            'burn_rate_coefficient': self.burn_rate_coefficient,
            'burn_rate_exponent': self.burn_rate_exponent,
            'temperature_sensitivity': self.temperature_sensitivity,
            'reference_temperature': self.reference_temperature
        }

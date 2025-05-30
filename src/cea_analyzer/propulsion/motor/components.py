"""
Motor Components Module
--------------------

This module provides classes for various rocket motor components
such as the motor case and nozzle.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import matplotlib.pyplot as plt

from ...core.logger import get_logger

# Setup logger
logger = get_logger(__name__)


@dataclass
class MotorCase:
    """Class representing a motor case design."""
    material: str
    inner_diameter: float  # m
    wall_thickness: float  # m
    length: float  # m
    density: float = 7800.0  # kg/m³ (default: steel)
    tensile_strength: float = 500.0  # MPa (default: steel)
    safety_factor: float = 1.5
    
    def max_pressure(self) -> float:
        """Calculate maximum allowable pressure based on hoop stress (MPa)."""
        # Thin-walled pressure vessel equation
        return self.tensile_strength * self.wall_thickness / (self.inner_diameter * self.safety_factor)
    
    def mass(self) -> float:
        """Calculate case mass in kg."""
        outer_diameter = self.inner_diameter + 2 * self.wall_thickness
        case_volume = np.pi * self.length / 4 * (outer_diameter**2 - self.inner_diameter**2)
        return case_volume * self.density


@dataclass
class Nozzle:
    """Class representing a nozzle design."""
    throat_diameter: float  # m
    expansion_ratio: float
    contour_type: str = "Conical"
    half_angle: float = 15.0  # degrees (for conical)
    percentage_bell: float = 80.0  # percentage (for bell nozzles)
    divergence_efficiency: float = 0.98
    material: str = "Graphite"
    material_density: float = 1850.0  # kg/m³ (default: graphite)
    
    def exit_diameter(self) -> float:
        """Calculate exit diameter in meters."""
        return self.throat_diameter * np.sqrt(self.expansion_ratio)
    
    def throat_area(self) -> float:
        """Calculate throat area in square meters."""
        return np.pi * (self.throat_diameter / 2)**2
    
    def exit_area(self) -> float:
        """Calculate exit area in square meters."""
        return self.throat_area() * self.expansion_ratio
    
    def length(self) -> float:
        """Approximate nozzle length in meters."""
        if self.contour_type == "Conical":
            # Simplified conical nozzle length
            return (self.exit_diameter() - self.throat_diameter) / (2 * np.tan(np.radians(self.half_angle)))
        elif "Bell" in self.contour_type or "Rao" in self.contour_type:
            # Approximation for bell nozzle length
            conical_length = (self.exit_diameter() - self.throat_diameter) / (2 * np.tan(np.radians(15)))
            return conical_length * (self.percentage_bell / 100)
        else:
            # Default approximation
            return (self.exit_diameter() - self.throat_diameter) / 2
    
    def mass(self) -> float:
        """Approximate nozzle mass in kg (very simplified)."""
        # Simplified as a truncated cone with average thickness
        average_diameter = (self.exit_diameter() + self.throat_diameter) / 2
        average_thickness = self.throat_diameter * 0.15  # Approximate thickness as 15% of throat diameter
        volume = np.pi * average_thickness * self.length() * average_diameter
        return volume * self.material_density
    
    def generate_contour(self, cea_data: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Generate nozzle contour coordinates."""
        from ...propulsion.nozzle import conical_nozzle, bell_nozzle, rao_optimum_nozzle
        
        if cea_data is None:
            # Default values if CEA data not provided
            cea_data = {
                'gamma': 1.2,
                'area_ratio': self.expansion_ratio
            }
        
        # Use existing nozzle design functions
        if self.contour_type == "Conical":
            # Create simplified conical nozzle parameters
            cone_params = {
                'half_angle': self.half_angle,
                'area_ratio': self.expansion_ratio
            }
            x, r = conical_nozzle(cea_data, R_throat=self.throat_diameter/2, **cone_params)
            
        elif self.contour_type == "Bell" or self.contour_type == "80% Bell":
            # Create bell nozzle parameters
            bell_params = {
                'percent_bell': self.percentage_bell
            }
            x, r = bell_nozzle(cea_data, R_throat=self.throat_diameter/2, **bell_params)
            
        elif self.contour_type == "Rao Optimum":
            x, r = rao_optimum_nozzle(cea_data, R_throat=self.throat_diameter/2)
            
        else:
            # Default to conical if unsupported type
            logger.warning(f"Unsupported nozzle type: {self.contour_type}. Using conical.")
            x, r = conical_nozzle(cea_data, R_throat=self.throat_diameter/2)
        
        return x, r

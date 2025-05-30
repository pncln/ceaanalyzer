"""
Motor Grain Module
---------------

This module provides the main MotorGrain class that combines grain geometry
with propellant properties for analyzing solid rocket motor performance.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path

from .base import GrainGeometry, PropellantProperties
from .types import GrainType
from .geometries import BatesGrain, StarGrain, EndBurnerGrain
from ...core.logger import get_logger

# Setup logger
logger = get_logger(__name__)


class MotorGrain:
    """Class representing a complete motor grain configuration."""
    
    def __init__(self, geometry: GrainGeometry, propellant: PropellantProperties):
        """
        Initialize a motor grain with geometry and propellant properties.
        
        Args:
            geometry: A GrainGeometry instance defining the grain shape
            propellant: PropellantProperties for the grain material
        """
        self.geometry = geometry
        self.propellant = propellant
    
    def mass(self) -> float:
        """Calculate the grain mass in kg."""
        return self.geometry.volume() * self.propellant.density
    
    def regression_rate(self, pressure: float, temperature: float = None) -> float:
        """Calculate the grain regression rate in m/s at given pressure and temperature."""
        return self.propellant.burn_rate(pressure, temperature)
    
    def simulate_burn(self, chamber_pressure: float, time_step: float = 0.01, 
                     max_time: float = 10.0, temperature: float = None) -> Dict:
        """
        Simulate grain regression over time.
        
        Args:
            chamber_pressure: Chamber pressure in MPa (can be a function of time)
            time_step: Time step for simulation in seconds
            max_time: Maximum simulation time in seconds
            temperature: Propellant temperature in K
            
        Returns:
            Dictionary containing time, web burned, burn area, and mass flow rate arrays
        """
        # Initialize arrays
        time_points = np.arange(0, max_time, time_step)
        web_burned = np.zeros_like(time_points)
        burn_area = np.zeros_like(time_points)
        mass_flow = np.zeros_like(time_points)
        
        # Get initial burn area
        burn_area[0] = self.geometry.burn_area(0)
        
        # Check if pressure is a function or a constant
        if callable(chamber_pressure):
            pressure_func = chamber_pressure
        else:
            pressure_func = lambda t: chamber_pressure
        
        # Simulate time steps
        for i in range(1, len(time_points)):
            t = time_points[i]
            p = pressure_func(t)
            
            # Calculate regression rate at this time step
            r_dot = self.regression_rate(p, temperature)
            
            # Update web burned
            web_burned[i] = web_burned[i-1] + r_dot * time_step
            
            # Update burn area
            burn_area[i] = self.geometry.burn_area(web_burned[i])
            
            # Calculate mass flow rate
            mass_flow[i] = burn_area[i] * r_dot * self.propellant.density
            
            # Stop if fully burned
            if web_burned[i] >= self.geometry.web_distance():
                # Truncate arrays
                time_points = time_points[:i+1]
                web_burned = web_burned[:i+1]
                burn_area = burn_area[:i+1]
                mass_flow = mass_flow[:i+1]
                break
        
        return {
            'time': time_points,
            'web_burned': web_burned,
            'burn_area': burn_area,
            'mass_flow': mass_flow
        }
    
    def plot_burn_area_progression(self, num_steps: int = 20):
        """
        Plot the burn area as a function of web distance.
        
        Args:
            num_steps: Number of points to calculate
            
        Returns:
            Matplotlib figure and axes
        """
        max_web = self.geometry.web_distance()
        web_distances = np.linspace(0, max_web, num_steps)
        areas = [self.geometry.burn_area(web) for web in web_distances]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(web_distances * 1000, areas, 'b-', linewidth=2)  # Convert to mm for display
        
        ax.set_xlabel('Web Distance (mm)')
        ax.set_ylabel('Burn Area (m²)')
        ax.set_title(f'Burn Area vs. Web Distance')
        ax.grid(True)
        
        return fig, ax
    
    @classmethod
    def create_grain(cls, grain_type: GrainType, parameters: Dict, propellant: PropellantProperties):
        """Factory method to create a grain of the specified type."""
        try:
            if grain_type == GrainType.BATES:
                geometry = BatesGrain(
                    length=parameters.get('length', 0.1),
                    outer_diameter=parameters.get('outer_diameter', 0.05),
                    core_diameter=parameters.get('core_diameter', 0.02),
                    inhibited_ends=parameters.get('inhibited_ends', False),
                    inhibited_outer_surface=parameters.get('inhibited_outer_surface', True),
                    number_of_segments=parameters.get('number_of_segments', 1),
                    segment_spacing=parameters.get('segment_spacing', 0.005)
                )
            elif grain_type == GrainType.STAR:
                geometry = StarGrain(
                    length=parameters.get('length', 0.1),
                    outer_diameter=parameters.get('outer_diameter', 0.05),
                    core_diameter=parameters.get('core_diameter', 0.01),
                    number_of_points=parameters.get('number_of_points', 5),
                    star_point_depth=parameters.get('star_point_depth', 0.01),
                    star_inner_angle=parameters.get('star_inner_angle', 60),
                    inhibited_ends=parameters.get('inhibited_ends', True),
                    inhibited_outer_surface=parameters.get('inhibited_outer_surface', True)
                )
            elif grain_type == GrainType.ENDBURNER:
                geometry = EndBurnerGrain(
                    length=parameters.get('length', 0.1),
                    outer_diameter=parameters.get('outer_diameter', 0.05),
                    inhibited_outer_surface=parameters.get('inhibited_outer_surface', True)
                )
            else:
                logger.error(f"Unsupported grain type: {grain_type}")
                raise ValueError(f"Unsupported grain type: {grain_type}")
                
            return cls(geometry, propellant)
            
        except Exception as e:
            logger.error(f"Error creating grain: {e}")
            raise

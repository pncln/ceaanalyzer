"""
Grain Geometry Implementations Module
----------------------------------

This module provides concrete implementations of various grain geometries
for solid rocket motors.
"""

from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

from .base import GrainGeometry


@dataclass
class BatesGrain(GrainGeometry):
    """BATES (Circular Port) grain geometry."""
    core_diameter: float = 0.0  # m
    inhibited_ends: bool = False
    inhibited_outer_surface: bool = True
    number_of_segments: int = 1
    segment_spacing: float = 0.005  # m
    
    def volume(self) -> float:
        """Calculate the total grain volume in cubic meters."""
        # Calculate single segment volume
        segment_length = (self.length - (self.number_of_segments - 1) * self.segment_spacing) / self.number_of_segments
        grain_outer_radius = self.outer_diameter / 2
        core_radius = self.core_diameter / 2
        
        segment_volume = np.pi * segment_length * (grain_outer_radius**2 - core_radius**2)
        total_volume = segment_volume * self.number_of_segments
        
        return total_volume
    
    def burn_area(self, web_burned: float) -> float:
        """Calculate the burn surface area at a given web distance."""
        if web_burned < 0:
            raise ValueError(f"Web burned distance cannot be negative: {web_burned}")
            
        # Check if completely burned
        if web_burned >= self.web_distance():
            return 0.0
            
        # Calculate segment dimensions
        segment_length = (self.length - (self.number_of_segments - 1) * self.segment_spacing) / self.number_of_segments
        grain_outer_radius = self.outer_diameter / 2
        core_radius = self.core_diameter / 2
        
        # Calculate current inner radius
        current_inner_radius = core_radius + web_burned
        
        # Calculate current surface areas
        inner_surface_area = 2 * np.pi * current_inner_radius * segment_length * self.number_of_segments
        
        # End faces
        end_area = 0.0
        if not self.inhibited_ends:
            end_area = np.pi * (grain_outer_radius**2 - current_inner_radius**2) * 2 * self.number_of_segments
            
        # Outer surface
        outer_area = 0.0
        if not self.inhibited_outer_surface:
            outer_area = 2 * np.pi * grain_outer_radius * segment_length * self.number_of_segments
            
        total_area = inner_surface_area + end_area + outer_area
        return total_area
    
    def web_distance(self) -> float:
        """Calculate the maximum web distance (thickness) in meters."""
        # For BATES, it's the distance from core to outer wall
        # Unless ends are uninhibited, in which case it's the smaller of:
        # - radial distance: (outer_diameter - core_diameter) / 2
        # - axial distance: segment_length / 2
        
        grain_outer_radius = self.outer_diameter / 2
        core_radius = self.core_diameter / 2
        radial_web = grain_outer_radius - core_radius
        
        if not self.inhibited_ends:
            segment_length = (self.length - (self.number_of_segments - 1) * self.segment_spacing) / self.number_of_segments
            axial_web = segment_length / 2
            return min(radial_web, axial_web)
        else:
            return radial_web


@dataclass
class StarGrain(GrainGeometry):
    """Star-shaped grain geometry."""
    core_diameter: float = 0.0  # m
    number_of_points: int = 5
    star_point_depth: float = 0.0  # m
    star_inner_angle: float = 60.0  # degrees
    inhibited_ends: bool = True
    inhibited_outer_surface: bool = True
    
    def volume(self) -> float:
        """Approximate the total grain volume in cubic meters."""
        # Star grain volume calculation is complex, we'll use an approximation
        grain_outer_radius = self.outer_diameter / 2
        core_radius = self.core_diameter / 2
        
        # Simplified approach - calculate the equivalent circular core volume
        # and subtract from the cylinder volume
        circular_area = np.pi * grain_outer_radius**2
        
        # Approximate star core area
        theta = 2 * np.pi / self.number_of_points  # angle between points
        half_angle = np.radians(self.star_inner_angle / 2)
        
        # Area of the star points
        r_star = core_radius + self.star_point_depth
        point_area = self.number_of_points * (
            0.5 * r_star**2 * np.sin(theta) - 
            0.5 * core_radius**2 * np.sin(theta)
        )
        
        # Approximate core area with star points
        core_area = np.pi * core_radius**2 + point_area
        
        # Net area
        net_area = circular_area - core_area
        
        # Total volume
        volume = net_area * self.length
        
        return volume
    
    def burn_area(self, web_burned: float) -> float:
        """
        Approximate the burn surface area at a given web distance.
        
        This is a simplified calculation that doesn't account for the 
        complex burning surface evolution in a star grain.
        """
        if web_burned < 0:
            raise ValueError(f"Web burned distance cannot be negative: {web_burned}")
            
        # Check if completely burned
        if web_burned >= self.web_distance():
            return 0.0
            
        grain_outer_radius = self.outer_diameter / 2
        core_radius = self.core_diameter / 2
        
        # Calculate current inner profile
        # This is a complex calculation for star grains
        # We'll use an approximation based on the current web distance
        
        # For a simplified star, we'll assume the burning progresses uniformly
        current_core_radius = core_radius + web_burned
        current_star_depth = max(0, self.star_point_depth - web_burned)
        
        # Calculate perimeter of the current star shape
        theta = 2 * np.pi / self.number_of_points  # angle between points
        half_angle = np.radians(self.star_inner_angle / 2)
        
        # Simplified perimeter calculation
        star_perimeter = self.number_of_points * (
            2 * current_star_depth / np.sin(half_angle) + 
            theta * current_core_radius
        )
        
        # Calculate burning surface area
        inner_surface_area = star_perimeter * self.length
        
        # End faces
        end_area = 0.0
        if not self.inhibited_ends:
            end_area = 2 * np.pi * (grain_outer_radius**2 - current_core_radius**2)
            
        # Outer surface
        outer_area = 0.0
        if not self.inhibited_outer_surface:
            outer_area = 2 * np.pi * grain_outer_radius * self.length
            
        total_area = inner_surface_area + end_area + outer_area
        return total_area
    
    def web_distance(self) -> float:
        """Calculate the maximum web distance in meters."""
        grain_outer_radius = self.outer_diameter / 2
        core_radius = self.core_diameter / 2
        
        # For a star grain, the web is the shorter of:
        # 1. Distance from the core to the outer wall
        # 2. Distance from the star valley to the adjacent star point
        radial_web = grain_outer_radius - core_radius
        
        # The second distance depends on the star geometry
        # For a simplified calculation:
        theta = 2 * np.pi / self.number_of_points  # angle between points
        valley_to_point = self.star_point_depth
        
        web = min(radial_web, valley_to_point)
        
        # If ends are uninhibited, also consider axial burning
        if not self.inhibited_ends:
            axial_web = self.length / 2
            web = min(web, axial_web)
            
        return web


@dataclass
class EndBurnerGrain(GrainGeometry):
    """End-burning grain geometry."""
    inhibited_outer_surface: bool = True
    
    def volume(self) -> float:
        """Calculate the total grain volume in cubic meters."""
        grain_radius = self.outer_diameter / 2
        return np.pi * grain_radius**2 * self.length
    
    def burn_area(self, web_burned: float) -> float:
        """Calculate the burn surface area at a given web distance."""
        if web_burned < 0:
            raise ValueError(f"Web burned distance cannot be negative: {web_burned}")
            
        # Check if completely burned
        if web_burned >= self.web_distance():
            return 0.0
            
        grain_radius = self.outer_diameter / 2
        
        # For end burner, only the end(s) burn unless outer surface is uninhibited
        end_area = np.pi * grain_radius**2
        
        # Outer cylindrical surface
        outer_area = 0.0
        if not self.inhibited_outer_surface:
            current_length = self.length - web_burned
            if current_length > 0:
                outer_area = 2 * np.pi * grain_radius * current_length
                
        return end_area + outer_area
    
    def web_distance(self) -> float:
        """Calculate the maximum web distance in meters."""
        # For an end burner, the web is the grain length
        return self.length

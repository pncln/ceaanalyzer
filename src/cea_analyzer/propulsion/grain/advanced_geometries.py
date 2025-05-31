"""
Advanced Grain Geometry Implementations Module
----------------------------------------

This module provides implementations of more complex grain geometries
for solid rocket motors, extending the basic geometries.
"""

from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union

from .base import GrainGeometry


@dataclass
class CSlotGrain(GrainGeometry):
    """
    C-slot (Moon Burner) grain geometry.
    
    This grain has a circular core with a C-shaped slot extending from it,
    providing progressive-regressive or neutral burning characteristics.
    """
    core_diameter: float = 0.0  # m
    slot_width: float = 0.0  # m
    slot_depth: float = 0.0  # m
    slot_angle: float = 120.0  # degrees
    inhibited_ends: bool = True
    inhibited_outer_surface: bool = True
    
    def volume(self) -> float:
        """Calculate the total grain volume in cubic meters."""
        grain_outer_radius = self.outer_diameter / 2
        core_radius = self.core_diameter / 2
        
        # Calculate the circular cylinder volume
        cylinder_volume = np.pi * grain_outer_radius**2 * self.length
        
        # Calculate the volume of the central core
        core_volume = np.pi * core_radius**2 * self.length
        
        # Calculate the volume of the C-slot
        # Approximate as a partial cylinder
        slot_angle_rad = np.radians(self.slot_angle)
        slot_radius = core_radius + self.slot_depth
        
        # Area of the sector minus the core overlap
        sector_area = 0.5 * slot_angle_rad * slot_radius**2
        core_overlap = 0.5 * slot_angle_rad * core_radius**2
        
        # Additional rectangular area of the slot extension
        rect_area = self.slot_width * (slot_radius - core_radius)
        
        # Total slot area
        slot_area = sector_area - core_overlap + rect_area
        slot_volume = slot_area * self.length
        
        # Total grain volume
        grain_volume = cylinder_volume - core_volume - slot_volume
        
        return grain_volume
    
    def burn_area(self, web_burned: float) -> float:
        """
        Calculate the burn surface area at a given web distance.
        
        This is an approximate calculation that models the C-slot as
        a combination of a circular core and a radial slot.
        """
        if web_burned < 0:
            raise ValueError(f"Web burned distance cannot be negative: {web_burned}")
            
        # Check if completely burned
        if web_burned >= self.web_distance():
            return 0.0
            
        grain_outer_radius = self.outer_diameter / 2
        core_radius = self.core_diameter / 2
        
        # Calculate current dimensions
        current_core_radius = core_radius + web_burned
        current_slot_width = self.slot_width + 2 * web_burned
        current_slot_depth = max(0, self.slot_depth - web_burned)
        
        # Calculate the perimeter of the burning surface
        slot_angle_rad = np.radians(self.slot_angle)
        
        # Core perimeter (portion not intersecting with slot)
        core_perimeter = 2 * np.pi * current_core_radius - slot_angle_rad * current_core_radius
        
        # Slot perimeter (approximation)
        slot_perimeter = 0
        if current_slot_depth > 0:
            # Outer arc of the slot
            outer_radius = current_core_radius + current_slot_depth
            outer_arc = slot_angle_rad * outer_radius
            
            # Add straight sections of the slot (both sides)
            straight_sections = 2 * current_slot_depth
            
            slot_perimeter = outer_arc + straight_sections
        
        # Total burning perimeter
        total_perimeter = core_perimeter + slot_perimeter
        
        # Inner surface area
        inner_surface_area = total_perimeter * self.length
        
        # End faces
        end_area = 0.0
        if not self.inhibited_ends:
            # Approximate as circular area minus core
            end_area = np.pi * (grain_outer_radius**2 - current_core_radius**2)
            # Subtract approximate slot area
            slot_area = 0.5 * slot_angle_rad * (current_core_radius + current_slot_depth)**2 - 0.5 * slot_angle_rad * current_core_radius**2
            end_area -= slot_area
            end_area *= 2  # Both ends
            
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
        
        # For a C-slot grain, the web is the minimum of:
        # 1. Distance from core to outer wall: grain_outer_radius - core_radius
        # 2. Distance from slot end to outer wall: grain_outer_radius - (core_radius + slot_depth)
        # 3. Half the minimum thickness between slot sides
        
        radial_web_core = grain_outer_radius - core_radius
        radial_web_slot = grain_outer_radius - (core_radius + self.slot_depth)
        
        # Calculate minimum thickness between slot sides
        # This is complex to calculate exactly
        # Approximate using slot width
        slot_side_thickness = (2 * np.pi * core_radius - self.slot_width) / 2
        
        web = min(radial_web_core, radial_web_slot, slot_side_thickness)
        
        # If ends are uninhibited, also consider axial burning
        if not self.inhibited_ends:
            axial_web = self.length / 2
            web = min(web, axial_web)
            
        return web


@dataclass
class FinocylGrain(GrainGeometry):
    """
    Finocyl (Fins-on-Cylinder) grain geometry.
    
    This grain has a central circular port with radial fins extending outward,
    providing high initial burning area and progressive burning characteristics.
    """
    core_diameter: float = 0.0  # m
    number_of_fins: int = 6
    fin_height: float = 0.0  # m
    fin_width: float = 0.0  # m
    inhibited_ends: bool = True
    inhibited_outer_surface: bool = True
    
    def volume(self) -> float:
        """Calculate the total grain volume in cubic meters."""
        grain_outer_radius = self.outer_diameter / 2
        core_radius = self.core_diameter / 2
        
        # Calculate the circular cylinder volume
        cylinder_volume = np.pi * grain_outer_radius**2 * self.length
        
        # Calculate the volume of the central core
        core_volume = np.pi * core_radius**2 * self.length
        
        # Calculate the volume of each fin
        fin_volume = self.fin_height * self.fin_width * self.length
        total_fin_volume = fin_volume * self.number_of_fins
        
        # Total grain volume
        grain_volume = cylinder_volume - core_volume - total_fin_volume
        
        return grain_volume
    
    def burn_area(self, web_burned: float) -> float:
        """
        Calculate the burn surface area at a given web distance.
        
        This calculation models the Finocyl as a central core with
        rectangular fins extending radially.
        """
        if web_burned < 0:
            raise ValueError(f"Web burned distance cannot be negative: {web_burned}")
            
        # Check if completely burned
        if web_burned >= self.web_distance():
            return 0.0
            
        grain_outer_radius = self.outer_diameter / 2
        core_radius = self.core_diameter / 2
        
        # Calculate current dimensions
        current_core_radius = core_radius + web_burned
        current_fin_height = max(0, self.fin_height - web_burned)
        current_fin_width = self.fin_width + 2 * web_burned
        
        # Calculate the perimeter of the burning surface
        
        # Core perimeter
        core_perimeter = 2 * np.pi * current_core_radius
        
        # Fin perimeter (considering burning on three sides of each fin)
        fin_perimeter = 0
        if current_fin_height > 0:
            # Three sides of each fin burn: two sides and the end
            fin_perimeter = self.number_of_fins * (2 * current_fin_height + current_fin_width)
        
        # Total burning perimeter
        total_perimeter = core_perimeter + fin_perimeter
        
        # Inner surface area
        inner_surface_area = total_perimeter * self.length
        
        # End faces
        end_area = 0.0
        if not self.inhibited_ends:
            # Approximate as circular area minus core
            end_area = np.pi * (grain_outer_radius**2 - current_core_radius**2)
            # Subtract approximate fin area
            fin_area = self.number_of_fins * current_fin_height * current_fin_width
            end_area -= fin_area
            end_area *= 2  # Both ends
            
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
        
        # For a Finocyl grain, the web is the minimum of:
        # 1. Radial distance from core to outer wall: grain_outer_radius - core_radius
        # 2. Distance from fin tip to outer wall or adjacent fin
        
        radial_web_core = grain_outer_radius - core_radius
        
        # Calculate minimum thickness between fins
        # Approximate using circular spacing
        fin_spacing = 2 * np.pi * core_radius / self.number_of_fins
        fin_web = min(self.fin_height, fin_spacing / 2)
        
        web = min(radial_web_core, fin_web)
        
        # If ends are uninhibited, also consider axial burning
        if not self.inhibited_ends:
            axial_web = self.length / 2
            web = min(web, axial_web)
            
        return web


@dataclass
class WagonWheelGrain(GrainGeometry):
    """
    Wagon Wheel grain geometry.
    
    This grain has a central circular port with radial slots/spokes
    extending outward, creating a design similar to a wagon wheel.
    """
    core_diameter: float = 0.0  # m
    number_of_spokes: int = 8
    spoke_width: float = 0.0  # m
    spoke_length: float = 0.0  # m
    inhibited_ends: bool = True
    inhibited_outer_surface: bool = True
    
    def volume(self) -> float:
        """Calculate the total grain volume in cubic meters."""
        grain_outer_radius = self.outer_diameter / 2
        core_radius = self.core_diameter / 2
        
        # Calculate the circular cylinder volume
        cylinder_volume = np.pi * grain_outer_radius**2 * self.length
        
        # Calculate the volume of the central core
        core_volume = np.pi * core_radius**2 * self.length
        
        # Calculate the volume of each spoke slot
        spoke_volume = self.spoke_width * self.spoke_length * self.length
        total_spoke_volume = spoke_volume * self.number_of_spokes
        
        # Total grain volume
        grain_volume = cylinder_volume - core_volume - total_spoke_volume
        
        return grain_volume
    
    def burn_area(self, web_burned: float) -> float:
        """
        Calculate the burn surface area at a given web distance.
        
        This calculation models the Wagon Wheel as a central core with
        radial slot spokes.
        """
        if web_burned < 0:
            raise ValueError(f"Web burned distance cannot be negative: {web_burned}")
            
        # Check if completely burned
        if web_burned >= self.web_distance():
            return 0.0
            
        grain_outer_radius = self.outer_diameter / 2
        core_radius = self.core_diameter / 2
        
        # Calculate current dimensions
        current_core_radius = core_radius + web_burned
        current_spoke_length = max(0, self.spoke_length - web_burned)
        current_spoke_width = self.spoke_width + 2 * web_burned
        
        # Calculate the perimeter of the burning surface
        
        # Core perimeter (portion not intersecting with spokes)
        # Subtract the parts where spokes connect to core
        core_arc_angle = 2 * np.pi - self.number_of_spokes * (current_spoke_width / current_core_radius)
        core_perimeter = core_arc_angle * current_core_radius
        
        # Spoke perimeter (considering burning on both sides and end of each spoke)
        spoke_perimeter = 0
        if current_spoke_length > 0:
            # Two sides and end of each spoke
            spoke_perimeter = self.number_of_spokes * (2 * current_spoke_length + current_spoke_width)
        
        # Total burning perimeter
        total_perimeter = core_perimeter + spoke_perimeter
        
        # Inner surface area
        inner_surface_area = total_perimeter * self.length
        
        # End faces
        end_area = 0.0
        if not self.inhibited_ends:
            # Approximate as circular area minus core
            end_area = np.pi * (grain_outer_radius**2 - current_core_radius**2)
            # Subtract approximate spoke area
            spoke_area = self.number_of_spokes * current_spoke_width * current_spoke_length
            end_area -= spoke_area
            end_area *= 2  # Both ends
            
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
        
        # For a Wagon Wheel grain, the web is the minimum of:
        # 1. Radial distance from core to outer wall: grain_outer_radius - core_radius
        # 2. Distance from spoke end to outer wall
        # 3. Half the minimum thickness between adjacent spokes
        
        radial_web_core = grain_outer_radius - core_radius
        radial_web_spoke = grain_outer_radius - (core_radius + self.spoke_length)
        
        # Calculate minimum thickness between spokes
        # Approximate using circular spacing at outer radius of spokes
        spoke_outer_radius = core_radius + self.spoke_length
        circumference_at_spoke_end = 2 * np.pi * spoke_outer_radius
        spoke_spacing = (circumference_at_spoke_end / self.number_of_spokes) - self.spoke_width
        
        web = min(radial_web_core, radial_web_spoke, spoke_spacing / 2)
        
        # If ends are uninhibited, also consider axial burning
        if not self.inhibited_ends:
            axial_web = self.length / 2
            web = min(web, axial_web)
            
        return web

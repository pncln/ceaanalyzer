"""
Grain Regression Simulation Module
----------------------------------

This module provides classes and functions for simulating the regression
of solid rocket motor grain geometries over time.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Dict, List, Tuple, Optional, Union, Any
import time
from dataclasses import dataclass, field

from .base import GrainGeometry, PropellantProperties
from ...core.logger import get_logger

# Setup logger
logger = get_logger(__name__)


@dataclass
class GrainRegressionSimulation:
    """Class for simulating grain regression over time."""
    
    grain: GrainGeometry
    propellant: PropellantProperties
    chamber_pressure: float = 7.0  # MPa, default chamber pressure
    pressure_function: Optional[callable] = None
    time_step: float = 0.05  # seconds
    max_time: float = 10.0  # seconds
    temperature: float = 298.15  # K
    
    # Results storage
    times: List[float] = field(default_factory=list)
    web_distances: List[float] = field(default_factory=list)
    burn_areas: List[float] = field(default_factory=list)
    burn_rates: List[float] = field(default_factory=list)
    mass_flow_rates: List[float] = field(default_factory=list)
    thrust: List[float] = field(default_factory=list)  # Renamed from thrust_values for widget compatibility
    chamber_pressures: List[float] = field(default_factory=list)  # Renamed from pressures for widget compatibility
    grain_volumes: List[float] = field(default_factory=list)  # Renamed from volumes for widget compatibility
    remaining_mass: List[float] = field(default_factory=list)
    burn_time: float = 0.0  # Total burn time
    
    def __post_init__(self):
        """Initialize the simulation with default pressure function if needed."""
        if self.pressure_function is None:
            # Default to constant pressure
            self.pressure_function = lambda t, web, burn_area: self.chamber_pressure
    
    def run(self):
        """Run the grain regression simulation."""
        return self.run_simulation()
        
    def run_simulation(self):
        """Run the grain regression simulation."""
        # Reset results
        self.times = []
        self.web_distances = []
        self.burn_areas = []
        self.burn_rates = []
        self.mass_flow_rates = []
        self.thrust_values = []
        self.pressures = []
        self.volumes = []
        self.remaining_mass = []
        
        # Initialize
        current_time = 0.0
        current_web = 0.0
        max_web = self.grain.web_distance()
        initial_volume = self.grain.volume()
        initial_mass = initial_volume * self.propellant.density
        
        while current_time <= self.max_time and current_web < max_web:
            # Calculate burn area at current web distance
            burn_area = self.grain.burn_area(current_web)
            
            # Get pressure (could depend on time, web distance, and burn area)
            pressure = self.pressure_function(current_time, current_web, burn_area)
            
            # Calculate burn rate
            burn_rate = self.propellant.burn_rate(pressure, self.temperature)
            
            # Calculate volume and mass
            remaining_volume = self.grain.volume() - self.calculate_burned_volume(current_web)
            remaining_mass = remaining_volume * self.propellant.density
            
            # Calculate mass flow rate
            mass_flow_rate = burn_area * burn_rate * self.propellant.density
            
            # Calculate thrust (simplified)
            # F = mdot * Isp * g0
            isp = 200.0  # Default value, should be calculated from CEA data
            g0 = 9.81
            thrust = mass_flow_rate * isp * g0
            
            # Store results
            self.times.append(current_time)
            self.web_distances.append(current_web)
            self.burn_areas.append(burn_area)
            self.burn_rates.append(burn_rate)
            self.mass_flow_rates.append(mass_flow_rate)
            self.thrust.append(thrust)
            self.chamber_pressures.append(pressure)
            self.grain_volumes.append(remaining_volume)
            self.remaining_mass.append(remaining_mass)
            
            # Increment time and web distance
            current_time += self.time_step
            current_web += burn_rate * self.time_step
            
        # Store final burn time
        if len(self.times) > 0:
            self.burn_time = self.times[-1]
        
        logger.info(f"Simulation completed: {len(self.times)} time steps, final web = {current_web:.3f} m, burn time = {self.burn_time:.2f} s")
        return self
    
    def calculate_burned_volume(self, web_distance: float) -> float:
        """
        Calculate the volume of propellant burned at a given web distance.
        
        This is a simplified approach that depends on the grain geometry.
        For complex grain geometries, this would need to be implemented 
        specifically for each type.
        """
        # For simple geometries, we can approximate
        # For BATES, it's the volume difference
        if hasattr(self.grain, 'core_diameter'):
            grain_outer_radius = self.grain.outer_diameter / 2
            core_radius = self.grain.core_diameter / 2
            current_inner_radius = core_radius + web_distance
            
            # Calculate the current volume
            if current_inner_radius > grain_outer_radius:
                # Completely burned
                return self.grain.volume()
            
            # For simple cylindrical grains
            if hasattr(self.grain, 'number_of_segments'):
                # BATES-like geometry
                segment_length = (self.grain.length - (self.grain.number_of_segments - 1) * self.grain.segment_spacing) / self.grain.number_of_segments
                burned_volume = np.pi * (current_inner_radius**2 - core_radius**2) * segment_length * self.grain.number_of_segments
                
                # Add end burning if applicable
                if not getattr(self.grain, 'inhibited_ends', True):
                    end_burn_depth = min(web_distance, segment_length / 2)
                    end_area = np.pi * (grain_outer_radius**2 - current_inner_radius**2)
                    end_burned_volume = end_area * end_burn_depth * 2 * self.grain.number_of_segments
                    burned_volume += end_burned_volume
                
                return burned_volume
            
            # For other geometries, need specific implementations
            # Default approximation
            initial_volume = self.grain.volume()
            current_volume = np.pi * (grain_outer_radius**2 - current_inner_radius**2) * self.grain.length
            return initial_volume - current_volume
        
        # Fallback for unknown geometries
        # Approximate as percentage of web distance
        return (web_distance / self.grain.web_distance()) * self.grain.volume()
    
    def determine_burn_profile_type(self) -> str:
        """
        Determine the burn profile type (progressive, neutral, regressive).
        
        Returns:
            str: 'Progressive', 'Neutral', or 'Regressive'
        """
        if len(self.thrust) < 3:
            return "Unknown"
        
        # Calculate the average change in thrust
        thrust_changes = np.diff(self.thrust)
        avg_change = np.mean(thrust_changes)
        
        # Calculate the magnitude of average thrust for better relative comparison
        avg_thrust = np.mean(self.thrust)
        relative_change = avg_change / avg_thrust if avg_thrust > 0 else 0
        
        # Determine profile type based on relative change
        if abs(relative_change) < 0.01:  # Threshold for neutral (1% change)
            return "Neutral"
        elif relative_change > 0:
            return "Progressive"
        else:
            return "Regressive"
    
    def get_burn_profile_type(self) -> str:
        """Legacy method, kept for backward compatibility."""
        return self.determine_burn_profile_type()
    
    def plot_regression_results(self):
        """Plot the key results from the regression simulation."""
        if not self.times:
            logger.warning("No simulation data to plot")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Thrust vs. Time
        axes[0, 0].plot(self.times, self.thrust, 'b-', linewidth=2)
        axes[0, 0].set_title('Thrust vs. Time')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Thrust (N)')
        axes[0, 0].grid(True)
        
        # Burn Area vs. Time
        axes[0, 1].plot(self.times, self.burn_areas, 'r-', linewidth=2)
        axes[0, 1].set_title('Burn Area vs. Time')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Burn Area (m²)')
        axes[0, 1].grid(True)
        
        # Pressure vs. Time
        axes[1, 0].plot(self.times, self.pressures, 'g-', linewidth=2)
        axes[1, 0].set_title('Pressure vs. Time')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Pressure (MPa)')
        axes[1, 0].grid(True)
        
        # Remaining Mass vs. Time
        axes[1, 1].plot(self.times, self.remaining_mass, 'k-', linewidth=2)
        axes[1, 1].set_title('Remaining Mass vs. Time')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Mass (kg)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Add burn profile type as text
        profile_type = self.get_burn_profile_type()
        burn_profile_text = f"Burn Profile: {profile_type.capitalize()}"
        fig.text(0.5, 0.01, burn_profile_text, ha='center', fontsize=12, 
                 bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        return fig


def generate_grain_cross_section(grain: GrainGeometry, web_distance: float = 0.0, 
                              ax=None, resolution: int = 100):
    """
    Generate a 2D cross-section of the grain at a given web distance and plot it.
    
    Args:
        grain: The grain geometry object
        web_distance: The web distance burned (m)
        ax: Matplotlib axis to plot on (if None, one will be created)
        resolution: The resolution of the cross-section plot
        
    Returns:
        The matplotlib axis with the plot
    """
    # Create axis if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
    
    # Get grain parameters
    outer_radius = grain.outer_diameter / 2
    core_radius = grain.core_diameter / 2
    current_core_radius = core_radius + web_distance
    
    # Ensure current_core_radius doesn't exceed outer_radius
    current_core_radius = min(current_core_radius, outer_radius)
    
    # Create the outer circle (grain perimeter)
    circle_outer = plt.Circle((0, 0), outer_radius, fill=True, color='lightgray', alpha=0.7)
    ax.add_patch(circle_outer)
    
    # Different grain geometries have different port shapes
    grain_type = type(grain).__name__
    
    if grain_type == 'BatesGrain':
        # Simple circular port
        circle_inner = plt.Circle((0, 0), current_core_radius, fill=True, color='white')
        ax.add_patch(circle_inner)
        
    elif grain_type == 'StarGrain':
        # Star grain with points
        num_points = grain.number_of_points
        point_depth = grain.point_depth
        inner_angle = np.radians(grain.inner_angle)
        
        # Calculate star points
        theta = np.linspace(0, 2*np.pi, num_points+1)[:-1]  # Angles for points
        
        # Inner and outer radii for the star points
        r_inner = current_core_radius - point_depth * (1 - web_distance/point_depth if web_distance < point_depth else 0)
        r_outer = current_core_radius
        
        # Create star polygon
        star_points = []
        for i in range(num_points):
            # Point at valley (inner radius)
            valley_angle = theta[i] - inner_angle/2
            star_points.append((r_inner * np.cos(valley_angle), r_inner * np.sin(valley_angle)))
            
            # Point at peak (outer radius)
            star_points.append((r_outer * np.cos(theta[i]), r_outer * np.sin(theta[i])))
            
            # Point at next valley
            valley_angle = theta[i] + inner_angle/2
            star_points.append((r_inner * np.cos(valley_angle), r_inner * np.sin(valley_angle)))
        
        # Create star polygon
        star = plt.Polygon(star_points, closed=True, fill=True, color='white')
        ax.add_patch(star)
        
    elif grain_type == 'CSlotGrain':
        # C-slot/moon burner
        slot_width = grain.slot_width
        slot_depth = grain.slot_depth
        slot_angle = np.radians(grain.slot_angle)
        
        # Draw the core circle
        circle_inner = plt.Circle((0, 0), current_core_radius, fill=True, color='white')
        ax.add_patch(circle_inner)
        
        # Draw the slot if it's not fully burned away
        if web_distance < slot_depth:
            # Calculate remaining slot depth
            remaining_depth = slot_depth - web_distance
            
            # Create points for the slot arc
            theta_start = -slot_angle/2
            theta_end = slot_angle/2
            theta = np.linspace(theta_start, theta_end, 100)
            
            # Inner radius for slot
            r_slot = current_core_radius + remaining_depth
            
            # Arc points
            x_arc = r_slot * np.cos(theta)
            y_arc = r_slot * np.sin(theta)
            
            # Add straight lines to complete the shape
            x_start = current_core_radius * np.cos(theta_start)
            y_start = current_core_radius * np.sin(theta_start)
            x_end = current_core_radius * np.cos(theta_end)
            y_end = current_core_radius * np.sin(theta_end)
            
            # Create polygon for the slot
            x = np.append(np.append([x_start], x_arc), [x_end])
            y = np.append(np.append([y_start], y_arc), [y_end])
            slot = plt.Polygon(np.column_stack([x, y]), closed=True, fill=True, color='white')
            ax.add_patch(slot)
        
    elif grain_type == 'FinocylGrain':
        # Finocyl (fins on cylinder)
        num_fins = grain.number_of_fins
        fin_length = grain.fin_length
        fin_width = grain.fin_width
        
        # Draw the core circle
        circle_inner = plt.Circle((0, 0), current_core_radius, fill=True, color='white')
        ax.add_patch(circle_inner)
        
        # Draw fins if they're not fully burned away
        if web_distance < fin_length:
            # Calculate remaining fin length
            remaining_length = fin_length - web_distance
            
            # Create fins
            for i in range(num_fins):
                angle = 2 * np.pi * i / num_fins
                
                # Calculate fin corners
                fin_points = []
                
                # Inner corners at core radius
                r_inner = current_core_radius
                angle_half_width = np.arcsin(fin_width / (2 * r_inner)) if r_inner > 0 else 0
                
                fin_points.append((r_inner * np.cos(angle - angle_half_width),
                                  r_inner * np.sin(angle - angle_half_width)))
                fin_points.append((r_inner * np.cos(angle + angle_half_width),
                                  r_inner * np.sin(angle + angle_half_width)))
                
                # Outer corners at core radius + fin length
                r_outer = current_core_radius + remaining_length
                angle_half_width = np.arcsin(fin_width / (2 * r_outer)) if r_outer > 0 else 0
                
                fin_points.append((r_outer * np.cos(angle + angle_half_width),
                                  r_outer * np.sin(angle + angle_half_width)))
                fin_points.append((r_outer * np.cos(angle - angle_half_width),
                                  r_outer * np.sin(angle - angle_half_width)))
                
                # Create fin polygon
                fin = plt.Polygon(fin_points, closed=True, fill=True, color='white')
                ax.add_patch(fin)
    
    elif grain_type == 'WagonWheelGrain':
        # Wagon wheel with spokes
        num_spokes = grain.number_of_spokes
        spoke_width = np.radians(grain.spoke_width)
        
        # Draw the core circle
        circle_inner = plt.Circle((0, 0), current_core_radius, fill=True, color='white')
        ax.add_patch(circle_inner)
        
        # Draw spokes
        for i in range(num_spokes):
            angle = 2 * np.pi * i / num_spokes
            
            # Spoke start and end angles
            start_angle = angle - spoke_width/2
            end_angle = angle + spoke_width/2
            
            # Create wedge for spoke
            wedge = plt.Wedge((0, 0), outer_radius, np.degrees(start_angle), np.degrees(end_angle), width=0, fill=True, color='white')
            ax.add_patch(wedge)
    
    else:
        # Default to circular port for unknown geometries
        circle_inner = plt.Circle((0, 0), current_core_radius, fill=True, color='white')
        ax.add_patch(circle_inner)
    
    # Set axis limits and labels
    ax.set_xlim(-outer_radius*1.1, outer_radius*1.1)
    ax.set_ylim(-outer_radius*1.1, outer_radius*1.1)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('Distance (m)')
    ax.set_ylabel('Distance (m)')
    
    # Add burned web annotation
    ax.annotate(f'Web burned: {web_distance:.3f} m', xy=(0.05, 0.05), xycoords='axes fraction',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))
    
    return ax


def visualize_grain_regression(grain: GrainGeometry, web_distances: List[float], 
                           num_frames: int = 4, resolution: int = 100):
    """
    Visualize the grain regression at various web distances.
    
    Args:
        grain: The grain geometry
        web_distances: List of web distances to visualize
        num_frames: Number of frames to show (evenly spaced)
        resolution: Resolution of the cross-section
    
    Returns:
        The matplotlib figure with the visualization
    """
    # Create a new figure with subplots
    fig, axes = plt.subplots(1, num_frames, figsize=(4 * num_frames, 4))
    
    # If only one frame, ensure axes is a list
    if num_frames == 1:
        axes = [axes]
    
    # Select evenly spaced web distances
    if len(web_distances) <= num_frames:
        selected_web_distances = web_distances
    else:
        indices = np.linspace(0, len(web_distances) - 1, num_frames, dtype=int)
        selected_web_distances = [web_distances[i] for i in indices]
    
    # Generate and plot cross-sections
    for i, web_distance in enumerate(selected_web_distances):
        ax = axes[i]
        generate_grain_cross_section(grain, web_distance, ax, resolution)
        ax.set_title(f'Web = {web_distance:.3f} m')
    
    fig.tight_layout()
    return fig


def create_3d_grain_model(grain: GrainGeometry, web_distance: float = 0.0, 
                     ax=None, resolution: int = 30, length_segments: int = 10):
    """
    Create a 3D model of the grain geometry for visualization.
    
    Args:
        grain: The grain geometry
        web_distance: The web distance burned (m)
        ax: Matplotlib 3D axis to plot on (if None, one will be created)
        resolution: Angular resolution for circular components
        length_segments: Number of segments along the length
        
    Returns:
        The matplotlib 3D axis with the plot
    """
    # Create 3D axis if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Get grain parameters
    length = grain.length
    outer_radius = grain.outer_diameter / 2
    core_radius = grain.core_diameter / 2
    current_core_radius = core_radius + web_distance
    
    # Ensure current_core_radius doesn't exceed outer_radius
    current_core_radius = min(current_core_radius, outer_radius)
    
    # Create a cylinder mesh for the outer grain
    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0, length, length_segments)
    theta_grid, z_grid = np.meshgrid(theta, z)
    
    # Outer surface coordinates
    x_outer = outer_radius * np.cos(theta_grid)
    y_outer = outer_radius * np.sin(theta_grid)
    
    # Plot outer surface with light gray
    ax.plot_surface(x_outer, y_outer, z_grid, color='lightgray', alpha=0.7)
    
    # Inner surface coordinates based on grain type
    grain_type = type(grain).__name__
    
    if grain_type == 'BatesGrain' and hasattr(grain, 'number_of_segments') and grain.number_of_segments > 1:
        # Multiple segments with spacing
        segment_length = (length - (grain.number_of_segments - 1) * grain.segment_spacing) / grain.number_of_segments
        
        for i in range(grain.number_of_segments):
            segment_start = i * (segment_length + grain.segment_spacing)
            segment_end = segment_start + segment_length
            
            # Create segment mesh
            z_segment = np.linspace(segment_start, segment_end, length_segments // grain.number_of_segments + 2)
            theta_segment, z_segment_grid = np.meshgrid(theta, z_segment)
            
            # Inner surface
            x_inner = current_core_radius * np.cos(theta_segment)
            y_inner = current_core_radius * np.sin(theta_segment)
            
            # Plot inner surface with white color
            ax.plot_surface(x_inner, y_inner, z_segment_grid, color='white', alpha=1.0)
            
            # Plot end faces if not inhibited
            if not grain.inhibited_ends:
                # Create circular end faces
                r = np.linspace(current_core_radius, outer_radius, 10)
                r_grid, theta_grid_end = np.meshgrid(r, theta)
                
                # Front end face
                x_end = r_grid * np.cos(theta_grid_end)
                y_end = r_grid * np.sin(theta_grid_end)
                z_end = np.ones_like(x_end) * segment_start
                ax.plot_surface(x_end, y_end, z_end, color='lightgray', alpha=0.7)
                
                # Back end face
                z_end = np.ones_like(x_end) * segment_end
                ax.plot_surface(x_end, y_end, z_end, color='lightgray', alpha=0.7)
    
    else:  # Single segment or other grain types
        # Inner surface based on port shape
        if grain_type in ['BatesGrain', 'CSlotGrain', 'FinocylGrain', 'WagonWheelGrain']:
            # Simplified visualization - just show the core for these types
            # A more accurate visualization would require complex shape generation for each type
            
            # Inner cylindrical port
            x_inner = current_core_radius * np.cos(theta_grid)
            y_inner = current_core_radius * np.sin(theta_grid)
            
            # Plot inner surface with white color
            ax.plot_surface(x_inner, y_inner, z_grid, color='white', alpha=1.0)
            
            # For C-Slot, add a simplified slot representation
            if grain_type == 'CSlotGrain' and web_distance < grain.slot_depth:
                slot_depth = grain.slot_depth - web_distance
                slot_angle = np.radians(grain.slot_angle)
                slot_width = grain.slot_width
                
                # Create points for the slot
                theta_slot = np.linspace(-slot_angle/2, slot_angle/2, int(resolution * slot_angle / (2*np.pi)))
                r_slot = current_core_radius + slot_depth
                
                for z_val in z:
                    x_slot = r_slot * np.cos(theta_slot)
                    y_slot = r_slot * np.sin(theta_slot)
                    z_slot = np.ones_like(x_slot) * z_val
                    
                    ax.plot(x_slot, y_slot, z_slot, 'w-', linewidth=2)
            
            # For Finocyl, add simplified fin representations
            elif grain_type == 'FinocylGrain' and web_distance < grain.fin_length:
                fin_length = grain.fin_length - web_distance
                num_fins = grain.number_of_fins
                fin_width = grain.fin_width
                
                for i in range(num_fins):
                    angle = 2 * np.pi * i / num_fins
                    
                    # Calculate fin corners
                    r_outer = current_core_radius + fin_length
                    angle_half_width = np.arcsin(fin_width / (2 * r_outer)) if r_outer > 0 else 0
                    
                    # Create fin lines at different z positions
                    for z_val in z:
                        # Draw fin outline
                        x_fin = [current_core_radius * np.cos(angle - angle_half_width),
                                r_outer * np.cos(angle - angle_half_width),
                                r_outer * np.cos(angle + angle_half_width),
                                current_core_radius * np.cos(angle + angle_half_width)]
                        y_fin = [current_core_radius * np.sin(angle - angle_half_width),
                                r_outer * np.sin(angle - angle_half_width),
                                r_outer * np.sin(angle + angle_half_width),
                                current_core_radius * np.sin(angle + angle_half_width)]
                        z_fin = [z_val, z_val, z_val, z_val]
                        
                        ax.plot(x_fin + [x_fin[0]], y_fin + [y_fin[0]], z_fin + [z_fin[0]], 'w-', linewidth=2)
            
            # For Wagon Wheel, add simplified spoke representations
            elif grain_type == 'WagonWheelGrain':
                num_spokes = grain.number_of_spokes
                spoke_width = np.radians(grain.spoke_width)
                
                for i in range(num_spokes):
                    angle = 2 * np.pi * i / num_spokes
                    
                    # Spoke start and end angles
                    start_angle = angle - spoke_width/2
                    end_angle = angle + spoke_width/2
                    
                    # Create spoke lines at different z positions
                    for z_val in z:
                        # Draw spoke edges
                        theta_spoke = np.linspace(start_angle, end_angle, 10)
                        x_spoke = outer_radius * np.cos(theta_spoke)
                        y_spoke = outer_radius * np.sin(theta_spoke)
                        z_spoke = np.ones_like(x_spoke) * z_val
                        
                        ax.plot(x_spoke, y_spoke, z_spoke, 'w-', linewidth=2)
        
        elif grain_type == 'StarGrain':
            # For star grain, create a simple visualization
            num_points = grain.number_of_points
            point_depth = grain.point_depth
            inner_angle = np.radians(grain.inner_angle)
            
            # Calculate star points
            star_angles = np.linspace(0, 2*np.pi, num_points+1)[:-1]  # Angles for points
            
            # Inner and outer radii for the star points
            r_inner = current_core_radius - point_depth * (1 - web_distance/point_depth if web_distance < point_depth else 0)
            r_outer = current_core_radius
            
            # Create star profile at different z values
            for z_val in z:
                x_star = []
                y_star = []
                
                for i in range(num_points):
                    # Point at valley (inner radius)
                    valley_angle = star_angles[i] - inner_angle/2
                    x_star.append(r_inner * np.cos(valley_angle))
                    y_star.append(r_inner * np.sin(valley_angle))
                    
                    # Point at peak (outer radius)
                    x_star.append(r_outer * np.cos(star_angles[i]))
                    y_star.append(r_outer * np.sin(star_angles[i]))
                    
                    # Point at next valley
                    valley_angle = star_angles[i] + inner_angle/2
                    x_star.append(r_inner * np.cos(valley_angle))
                    y_star.append(r_inner * np.sin(valley_angle))
                
                # Close the loop
                x_star.append(x_star[0])
                y_star.append(y_star[0])
                
                # Create z array for this slice
                z_star = np.ones_like(x_star) * z_val
                
                # Plot star profile
                ax.plot(x_star, y_star, z_star, 'w-', linewidth=2)
    
    # Set axis properties
    ax.set_box_aspect([1, 1, length/(2*outer_radius)])  # Adjust aspect ratio
    ax.set_xlim(-outer_radius*1.2, outer_radius*1.2)
    ax.set_ylim(-outer_radius*1.2, outer_radius*1.2)
    ax.set_zlim(-length*0.1, length*1.1)
    
    # Set labels
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Add title with web distance
    ax.set_title(f'Grain 3D Model - Web Distance: {web_distance:.3f} m')
    
    return ax
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    grain_outer_radius = grain.outer_diameter / 2
    
    # Generate points for the outer cylinder
    theta = np.linspace(0, 2*np.pi, resolution)
    z = np.linspace(0, grain.length, length_segments)
    Theta, Z = np.meshgrid(theta, z)
    X = grain_outer_radius * np.cos(Theta)
    Y = grain_outer_radius * np.sin(Theta)
    
    # Plot outer surface as wireframe
    ax.plot_surface(X, Y, Z, color='gray', alpha=0.3, edgecolor='k', linewidth=0.1)
    
    # Handle different grain types for the inner geometry
    if hasattr(grain, 'core_diameter'):
        core_radius = grain.core_diameter / 2
        current_inner_radius = core_radius + web_burned
        
        # Create inner core
        X_inner = current_inner_radius * np.cos(Theta)
        Y_inner = current_inner_radius * np.sin(Theta)
        
        # Plot inner surface
        ax.plot_surface(X_inner, Y_inner, Z, color='red', alpha=0.5, edgecolor='k', linewidth=0.1)
        
        # For star grains
        if hasattr(grain, 'number_of_points') and hasattr(grain, 'star_point_depth'):
            n_points = grain.number_of_points
            point_depth = grain.star_point_depth
            inner_angle_rad = np.radians(grain.star_inner_angle)
            
            # Star points
            for i in range(n_points):
                angle = i * 2*np.pi / n_points
                
                # Calculate star point coordinates
                r_outer = core_radius + point_depth
                half_angle = inner_angle_rad / 2
                
                # Create a wedge for the star point
                theta_wedge = np.linspace(angle - half_angle, angle + half_angle, resolution//n_points)
                
                for z_val in np.linspace(0, grain.length, length_segments):
                    x_wedge = np.array([core_radius * np.cos(angle - half_angle), 
                                      r_outer * np.cos(angle), 
                                      core_radius * np.cos(angle + half_angle)])
                    y_wedge = np.array([core_radius * np.sin(angle - half_angle), 
                                      r_outer * np.sin(angle), 
                                      core_radius * np.sin(angle + half_angle)])
                    z_wedge = np.array([z_val, z_val, z_val])
                    
                    # Plot the star point
                    ax.plot_trisurf(x_wedge, y_wedge, z_wedge, color='red', alpha=0.5)
        
        # For C-slot grains
        elif hasattr(grain, 'slot_width') and hasattr(grain, 'slot_depth'):
            slot_depth = grain.slot_depth
            slot_angle_rad = np.radians(grain.slot_angle)
            
            # Create C-slot
            slot_outer = core_radius + slot_depth
            half_angle = slot_angle_rad / 2
            
            # Create points for the C-slot
            theta_slot = np.linspace(-half_angle, half_angle, resolution//4)
            for z_val in np.linspace(0, grain.length, length_segments):
                x_slot = np.array([slot_outer * np.cos(-half_angle),
                                  slot_outer * np.cos(0),
                                  slot_outer * np.cos(half_angle)])
                y_slot = np.array([slot_outer * np.sin(-half_angle),
                                  slot_outer * np.sin(0),
                                  slot_outer * np.sin(half_angle)])
                z_slot = np.array([z_val, z_val, z_val])
                
                # Plot the C-slot
                ax.plot_trisurf(x_slot, y_slot, z_slot, color='red', alpha=0.5)
        
        # For Finocyl grains
        elif hasattr(grain, 'fin_count') and hasattr(grain, 'fin_length'):
            fin_count = grain.fin_count
            fin_length = grain.fin_length
            fin_width = grain.fin_width
            
            # Create fins
            for i in range(fin_count):
                angle = i * 2*np.pi / fin_count
                
                # Fin corners in polar coordinates
                fin_inner = core_radius
                fin_outer = fin_inner + fin_length
                half_width = np.radians(fin_width / 2)
                
                # Create fin for each z value
                for z_val in np.linspace(0, grain.length, length_segments):
                    # Create the four corners of the fin
                    corners_x = np.array([
                        fin_inner * np.cos(angle - half_width),
                        fin_outer * np.cos(angle - half_width),
                        fin_outer * np.cos(angle + half_width),
                        fin_inner * np.cos(angle + half_width)
                    ])
                    corners_y = np.array([
                        fin_inner * np.sin(angle - half_width),
                        fin_outer * np.sin(angle - half_width),
                        fin_outer * np.sin(angle + half_width),
                        fin_inner * np.sin(angle + half_width)
                    ])
                    corners_z = np.array([z_val, z_val, z_val, z_val])
                    
                    # Create two triangles for the fin face
                    ax.plot_trisurf(corners_x[:3], corners_y[:3], corners_z[:3], color='blue', alpha=0.5)
                    ax.plot_trisurf(corners_x[2:] + corners_x[:1], corners_y[2:] + corners_y[:1], 
                                  corners_z[2:] + corners_z[:1], color='blue', alpha=0.5)
        
        # For wagon wheel grains
        elif hasattr(grain, 'spoke_count') and hasattr(grain, 'spoke_width'):
            spoke_count = grain.spoke_count
            spoke_width = grain.spoke_width
            
            # Create spokes
            for i in range(spoke_count):
                angle = i * 2*np.pi / spoke_count
                
                # Spoke boundaries
                half_width = np.radians(spoke_width / 2)
                
                # Create spoke for each z value
                for z_val in np.linspace(0, grain.length, length_segments):
                    # Create the four corners of the spoke (as a rectangular tunnel)
                    corners_x = np.array([
                        core_radius * np.cos(angle - half_width),
                        grain_outer_radius * np.cos(angle - half_width),
                        grain_outer_radius * np.cos(angle + half_width),
                        core_radius * np.cos(angle + half_width)
                    ])
                    corners_y = np.array([
                        core_radius * np.sin(angle - half_width),
                        grain_outer_radius * np.sin(angle - half_width),
                        grain_outer_radius * np.sin(angle + half_width),
                        core_radius * np.sin(angle + half_width)
                    ])
                    corners_z = np.array([z_val, z_val, z_val, z_val])
                    
                    # Create two triangles for the spoke face
                    ax.plot_trisurf(corners_x[:3], corners_y[:3], corners_z[:3], color='blue', alpha=0.5)
                    ax.plot_trisurf(corners_x[2:] + corners_x[:1], corners_y[2:] + corners_y[:1], 
                                  corners_z[2:] + corners_z[:1], color='blue', alpha=0.5)
    
    # Set axis labels and limits
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    
    # Set equal aspect ratio
    max_range = grain.outer_diameter * 0.6
    mid_x = 0
    mid_y = 0
    mid_z = grain.length / 2
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add a title
    grain_type = type(grain).__name__
    ax.set_title(f'3D Model of {grain_type} (Web Burned: {web_burned:.3f} m)')
    
    return fig, ax

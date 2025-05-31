"""
Motor Design Module
---------------

This module provides the main MotorDesign class that integrates all 
rocket motor components into a complete rocket motor design.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
import json
from pathlib import Path
import pandas as pd

from .types import MotorType
from .components import MotorCase, Nozzle
from ..grain import MotorGrain
from ...analysis.performance import compute_system
from ...utils.utilities import ambient_pressure
from ...core.logger import get_logger

# Setup logger
logger = get_logger(__name__)


class MotorDesign:
    """Main class for integrated rocket motor design."""
    
    def __init__(self, 
                 name: str, 
                 motor_type: MotorType = MotorType.SOLID, 
                 cea_data: Optional[Dict] = None):
        """
        Initialize a motor design.
        
        Args:
            name: Motor designation
            motor_type: Type of motor (solid, liquid, hybrid)
            cea_data: Thermochemical data from CEA analysis
        """
        self.name = name
        self.motor_type = motor_type
        self.cea_data = cea_data
        
        # Components (to be set later)
        self.grain = None
        self.case = None
        self.nozzle = None
        
        # Performance parameters
        self._performance = {}
    
    def set_grain(self, grain: MotorGrain) -> None:
        """Set the motor grain."""
        self.grain = grain
        
        # Auto-size case if grain is set
        if self.grain and not self.case:
            self._auto_size_case()
    
    def set_case(self, case: MotorCase) -> None:
        """Set the motor case."""
        self.case = case
    
    def set_nozzle(self, nozzle: Nozzle) -> None:
        """Set the nozzle."""
        self.nozzle = nozzle
    
    def _auto_size_case(self) -> None:
        """Automatically size the case based on grain dimensions."""
        if not self.grain:
            return
        
        # Get grain outer dimensions
        grain_outer_diameter = self.grain.geometry.outer_diameter
        grain_length = self.grain.geometry.length
        
        # Add clearance
        case_inner_diameter = grain_outer_diameter + 0.005  # 5mm clearance
        case_length = grain_length * 1.1  # 10% extra length
        
        # Estimate wall thickness based on expected pressure
        expected_pressure = 7.0  # MPa (typical for amateur/small motors)
        safety_factor = 2.0
        tensile_strength = 500.0  # MPa (typical steel)
        
        wall_thickness = (expected_pressure * case_inner_diameter * safety_factor) / (2 * tensile_strength)
        wall_thickness = max(wall_thickness, 0.003)  # Minimum 3mm thickness
        
        # Create case
        self.case = MotorCase(
            material="Steel",
            inner_diameter=case_inner_diameter,
            wall_thickness=wall_thickness,
            length=case_length
        )
        
        logger.info(f"Auto-sized case: ID={case_inner_diameter:.3f}m, thickness={wall_thickness:.3f}m, length={case_length:.3f}m")
    
    def _auto_size_nozzle(self) -> None:
        """Automatically size the nozzle based on grain and expected performance."""
        if not self.grain or not self.case:
            logger.warning("Cannot auto-size nozzle without grain and case")
            return
        
        # Estimate design chamber pressure (70% of max case pressure)
        p_chamber = 0.7 * self.case.max_pressure()  # MPa
        
        # Calculate throat area based on grain burn area and burn rate
        # Assuming steady-state operation: mdot = A_burn * rho * r_burn = A_throat * P_c / c*
        burn_area = self.grain.geometry.burn_area(0)  # Initial burn area
        burn_rate = self.grain.propellant.burn_rate(p_chamber)  # At design pressure
        
        # Estimate throat area using simplified rocket equation:
        # A_t = (A_b * rho * r_burn) / (Cf * P_c)
        # where Cf is thrust coefficient, typically around 1.4
        density = self.grain.propellant.density
        thrust_coefficient = 1.4  # Approximate
        
        throat_area = (burn_area * density * burn_rate) / (thrust_coefficient * p_chamber * 1e6)
        throat_diameter = 2 * np.sqrt(throat_area / np.pi)
        
        # Determine appropriate expansion ratio based on expected operating conditions
        # For amateur motors, typical values range from 4 to 12
        if self.cea_data and 'Expansion Ratio' in self.cea_data:
            expansion_ratio = self.cea_data['Expansion Ratio']
        else:
            expansion_ratio = 8.0  # Default for amateur motors
        
        # Create nozzle
        self.nozzle = Nozzle(
            throat_diameter=throat_diameter,
            expansion_ratio=expansion_ratio,
            contour_type="Bell",  # Bell nozzles are a good default
            percentage_bell=80.0  # 80% bell is standard
        )
        
        logger.info(f"Auto-sized nozzle: throat diameter={throat_diameter:.3f}m, expansion ratio={expansion_ratio:.1f}")
    
    def calculate_performance(self, time_step: float = 0.05, max_time: float = 10.0, 
                             initial_pressure: float = 7.0, altitude: float = 0.0) -> Dict:
        """
        Calculate motor performance over time.
        
        Args:
            time_step: Simulation time step in seconds
            max_time: Maximum simulation time in seconds
            initial_pressure: Initial chamber pressure in MPa
            altitude: Launch altitude in meters
            
        Returns:
            Dictionary of performance parameters over time
        """
        if not self.grain or not self.nozzle:
            logger.error("Cannot calculate performance without grain and nozzle")
            return {}
        
        # Auto-size components if needed
        if not self.case:
            self._auto_size_case()
        
        # Set up the simulation
        time_points = np.arange(0, max_time, time_step)
        num_points = len(time_points)
        
        # Arrays to store results
        pressure = np.zeros(num_points)
        thrust = np.zeros(num_points)
        isp = np.zeros(num_points)
        mass_flow = np.zeros(num_points)
        mass_remaining = np.zeros(num_points)
        
        # Get initial mass
        initial_mass = self.grain.mass()
        mass_remaining[0] = initial_mass
        
        # Initial pressure
        pressure[0] = initial_pressure  # MPa
        
        # Get ambient pressure at the specified altitude
        p_ambient = ambient_pressure(altitude) / 1e6  # Convert Pa to MPa
        
        # Get throat and exit areas
        a_throat = self.nozzle.throat_area()
        a_exit = self.nozzle.exit_area()
        
        # Simulation loop
        for i in range(num_points):
            if i > 0:
                # Skip first iteration which already has initial values
                
                # Get current burn area and burn rate
                # Add safety check to prevent division by zero
                initial_burn_area = self.grain.geometry.burn_area(0)
                if initial_burn_area <= 1e-10:  # Small threshold to avoid division by zero
                    initial_burn_area = 1e-10  # Use a small positive value instead of zero
                    
                web_distance = np.sum([mass_flow[j] * time_step / (self.grain.propellant.density * initial_burn_area) 
                                      for j in range(i) if mass_flow[j] > 0])  # Skip zero mass flow
                burn_area = self.grain.geometry.burn_area(web_distance)
                
                # If completely burned, stop simulation
                if burn_area <= 0:
                    # Truncate arrays
                    time_points = time_points[:i]
                    pressure = pressure[:i]
                    thrust = thrust[:i]
                    isp = isp[:i]
                    mass_flow = mass_flow[:i]
                    mass_remaining = mass_remaining[:i]
                    break
                
                # Calculate burn rate at current pressure
                burn_rate = self.grain.propellant.burn_rate(pressure[i-1])
                
                # Mass flow rate from burning surface
                burn_mass_flow = burn_area * burn_rate * self.grain.propellant.density
                
                # Mass flow rate through throat (using simplified rocket equation)
                # mdot = (A_t * P_c) / c*
                # where c* is characteristic velocity
                
                # Estimate c* from CEA data or use a typical value
                if self.cea_data and 'c_star' in self.cea_data:
                    c_star = self.cea_data['c_star']
                else:
                    c_star = 1500  # Typical value in m/s
                
                # Calculate new pressure using mass balance
                # For steady state: mdot_in = mdot_out
                # mdot_in = A_b * rho * r_burn = f(P_c)
                # mdot_out = (A_t * P_c) / c*
                
                # Iterate to find pressure that balances mass flow
                # Simplified approach for now:
                # Add safety check to prevent division by zero
                if a_throat <= 1e-10:  # Small threshold to avoid division by zero
                    a_throat_safe = 1e-10  # Use a small positive value
                else:
                    a_throat_safe = a_throat
                    
                p_new = (burn_mass_flow * c_star) / a_throat_safe
                pressure[i] = p_new / 1e6  # Convert to MPa
                
                # Calculate thrust using nozzle thrust equation
                # F = mdot * v_e + (p_e - p_a) * A_e
                
                # Exit velocity estimation from c* and expansion ratio
                if self.cea_data and 'gamma' in self.cea_data:
                    gamma = self.cea_data['gamma']
                else:
                    gamma = 1.2  # Typical value
                
                # Simplified thrust calculation
                cf = self.nozzle.divergence_efficiency * np.sqrt((2*gamma**2)/(gamma-1) * 
                                                                (2/(gamma+1))**((gamma+1)/(gamma-1)) * 
                                                                (1 - (p_ambient/pressure[i])**((gamma-1)/gamma)))
                
                thrust[i] = cf * pressure[i] * 1e6 * a_throat  # Convert MPa to Pa for thrust calculation
                
                # Update mass flow and remaining mass
                mass_flow[i] = burn_mass_flow
                mass_remaining[i] = mass_remaining[i-1] - mass_flow[i-1] * time_step
                
                # Update Isp
                isp[i] = thrust[i] / (mass_flow[i] * 9.81)  # s
        
        # Store results with safety checks for empty arrays and potential numerical issues
        # Check if arrays are empty or contain only zeros
        non_zero_thrust = thrust[thrust > 0]
        non_zero_isp = isp[isp > 0]
        
        self._performance = {
            'time': time_points,
            'pressure': pressure,
            'thrust': thrust,
            'isp': isp,
            'mass_flow': mass_flow,
            'mass_remaining': mass_remaining,
            'total_impulse': np.trapz(thrust, time_points) if len(time_points) > 1 else 0.0,
            'average_thrust': np.mean(non_zero_thrust) if len(non_zero_thrust) > 0 else 0.0,
            'max_thrust': np.max(thrust) if len(thrust) > 0 else 0.0,
            'burn_time': time_points[-1] if len(time_points) > 0 else 0.0,
            'specific_impulse': np.mean(non_zero_isp) if len(non_zero_isp) > 0 else 0.0,
            'initial_mass': initial_mass
        }
        
        return self._performance
    
    def plot_thrust_curve(self):
        """Plot the thrust curve."""
        if not self._performance:
            logger.warning("No performance data available. Run calculate_performance() first.")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self._performance['time'], self._performance['thrust'], 'b-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Thrust (N)')
        ax.set_title(f'{self.name} Thrust Curve')
        ax.grid(True)
        
        # Add horizontal line for average thrust
        ax.axhline(y=self._performance['average_thrust'], color='r', linestyle='--', 
                   label=f'Average: {self._performance["average_thrust"]:.1f} N')
        
        # Add total impulse annotation
        total_impulse = self._performance['total_impulse']
        if total_impulse >= 1000:
            impulse_text = f'Total Impulse: {total_impulse/1000:.2f} kN·s'
        else:
            impulse_text = f'Total Impulse: {total_impulse:.1f} N·s'
        
        ax.text(0.95, 0.95, impulse_text, transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.legend()
        fig.tight_layout()
        
        return fig
    
    def plot_pressure_curve(self):
        """Plot the chamber pressure curve."""
        if not self._performance:
            logger.warning("No performance data available. Run calculate_performance() first.")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self._performance['time'], self._performance['pressure'], 'g-', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Chamber Pressure (MPa)')
        ax.set_title(f'{self.name} Chamber Pressure')
        ax.grid(True)
        
        # Add max pressure
        max_pressure = np.max(self._performance['pressure'])
        ax.axhline(y=max_pressure, color='r', linestyle='--', 
                   label=f'Maximum: {max_pressure:.2f} MPa')
        
        # Add case limit if available
        if self.case:
            case_limit = self.case.max_pressure()
            ax.axhline(y=case_limit, color='orange', linestyle='-.', 
                      label=f'Case Limit: {case_limit:.2f} MPa')
            
            # Calculate safety margin
            margin = (case_limit / max_pressure - 1) * 100
            ax.text(0.95, 0.95, f'Safety Margin: {margin:.1f}%', transform=ax.transAxes, 
                    ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        ax.legend()
        fig.tight_layout()
        
        return fig
    
    def get_summary(self) -> Dict:
        """Get a summary of the motor design and performance."""
        summary = {
            'name': self.name,
            'type': self.motor_type.value
        }
        
        # Add grain info if available
        if self.grain:
            summary['grain'] = {
                'type': self.grain.geometry.__class__.__name__,
                'length': f"{self.grain.geometry.length:.3f} m",
                'diameter': f"{self.grain.geometry.outer_diameter:.3f} m",
                'volume': f"{self.grain.geometry.volume() * 1e6:.1f} cm³",
                'mass': f"{self.grain.mass():.3f} kg",
                'propellant': self.grain.propellant.name
            }
        
        # Add case info if available
        if self.case:
            summary['case'] = {
                'material': self.case.material,
                'inner_diameter': f"{self.case.inner_diameter:.3f} m",
                'wall_thickness': f"{self.case.wall_thickness * 1000:.2f} mm",
                'length': f"{self.case.length:.3f} m",
                'mass': f"{self.case.mass():.3f} kg",
                'max_pressure': f"{self.case.max_pressure():.2f} MPa"
            }
        
        # Add nozzle info if available
        if self.nozzle:
            summary['nozzle'] = {
                'type': self.nozzle.contour_type,
                'throat_diameter': f"{self.nozzle.throat_diameter * 1000:.2f} mm",
                'exit_diameter': f"{self.nozzle.exit_diameter() * 1000:.2f} mm",
                'expansion_ratio': f"{self.nozzle.expansion_ratio:.1f}",
                'length': f"{self.nozzle.length():.3f} m",
                'mass': f"{self.nozzle.mass():.3f} kg"
            }
        
        # Add performance data if available
        if self._performance:
            summary['performance'] = {
                'average_thrust': f"{self._performance['average_thrust']:.1f} N",
                'max_thrust': f"{self._performance['max_thrust']:.1f} N",
                'total_impulse': f"{self._performance['total_impulse']:.1f} N·s",
                'burn_time': f"{self._performance['burn_time']:.2f} s",
                'specific_impulse': f"{self._performance['specific_impulse']:.1f} s"
            }
        
        return summary
    
    def to_dict(self) -> Dict:
        """Convert the motor design to a dictionary for serialization."""
        data = {
            'name': self.name,
            'type': self.motor_type.value
        }
        
        # Add grain data if available
        if self.grain:
            data['grain'] = {
                'type': self.grain.geometry.__class__.__name__,
                'geometry': {
                    'length': self.grain.geometry.length,
                    'outer_diameter': self.grain.geometry.outer_diameter
                },
                'propellant': self.grain.propellant.to_dict()
            }
            
            # Add geometry-specific parameters
            if hasattr(self.grain.geometry, 'core_diameter'):
                data['grain']['geometry']['core_diameter'] = self.grain.geometry.core_diameter
                
            if hasattr(self.grain.geometry, 'inhibited_ends'):
                data['grain']['geometry']['inhibited_ends'] = self.grain.geometry.inhibited_ends
                
            if hasattr(self.grain.geometry, 'inhibited_outer_surface'):
                data['grain']['geometry']['inhibited_outer_surface'] = self.grain.geometry.inhibited_outer_surface
                
            if hasattr(self.grain.geometry, 'number_of_segments'):
                data['grain']['geometry']['number_of_segments'] = self.grain.geometry.number_of_segments
                
            if hasattr(self.grain.geometry, 'segment_spacing'):
                data['grain']['geometry']['segment_spacing'] = self.grain.geometry.segment_spacing
        
        # Add case data if available
        if self.case:
            data['case'] = {
                'material': self.case.material,
                'inner_diameter': self.case.inner_diameter,
                'wall_thickness': self.case.wall_thickness,
                'length': self.case.length,
                'density': self.case.density,
                'tensile_strength': self.case.tensile_strength,
                'safety_factor': self.case.safety_factor
            }
        
        # Add nozzle data if available
        if self.nozzle:
            data['nozzle'] = {
                'throat_diameter': self.nozzle.throat_diameter,
                'expansion_ratio': self.nozzle.expansion_ratio,
                'contour_type': self.nozzle.contour_type,
                'half_angle': self.nozzle.half_angle,
                'percentage_bell': self.nozzle.percentage_bell,
                'divergence_efficiency': self.nozzle.divergence_efficiency,
                'material': self.nozzle.material,
                'material_density': self.nozzle.material_density
            }
        
        # Add CEA data if available
        if self.cea_data:
            data['cea_data'] = self.cea_data
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MotorDesign':
        """Create a MotorDesign instance from a dictionary."""
        from ..grain import MotorGrain, PropellantProperties, GrainType
        from ..grain import BatesGrain, StarGrain, EndBurnerGrain
        
        # Create the base motor design
        motor_type = MotorType[data['type'].split()[0].upper()] if 'type' in data else MotorType.SOLID
        design = cls(name=data.get('name', 'Unnamed Motor'), motor_type=motor_type)
        
        # Set CEA data if available
        if 'cea_data' in data:
            design.cea_data = data['cea_data']
        
        # Create and set grain if data available
        if 'grain' in data:
            grain_data = data['grain']
            geom_data = grain_data.get('geometry', {})
            
            # Create propellant
            propellant = PropellantProperties.from_dict(grain_data.get('propellant', {}))
            
            # Determine grain type and create geometry
            grain_type_str = grain_data.get('type', 'BatesGrain')
            
            if grain_type_str == 'BatesGrain':
                geometry = BatesGrain(
                    length=geom_data.get('length', 0.1),
                    outer_diameter=geom_data.get('outer_diameter', 0.05),
                    core_diameter=geom_data.get('core_diameter', 0.02),
                    inhibited_ends=geom_data.get('inhibited_ends', False),
                    inhibited_outer_surface=geom_data.get('inhibited_outer_surface', True),
                    number_of_segments=geom_data.get('number_of_segments', 1),
                    segment_spacing=geom_data.get('segment_spacing', 0.005)
                )
                grain_type = GrainType.BATES
            elif grain_type_str == 'StarGrain':
                geometry = StarGrain(
                    length=geom_data.get('length', 0.1),
                    outer_diameter=geom_data.get('outer_diameter', 0.05),
                    core_diameter=geom_data.get('core_diameter', 0.01),
                    number_of_points=geom_data.get('number_of_points', 5),
                    star_point_depth=geom_data.get('star_point_depth', 0.01),
                    star_inner_angle=geom_data.get('star_inner_angle', 60),
                    inhibited_ends=geom_data.get('inhibited_ends', True),
                    inhibited_outer_surface=geom_data.get('inhibited_outer_surface', True)
                )
                grain_type = GrainType.STAR
            elif grain_type_str == 'EndBurnerGrain':
                geometry = EndBurnerGrain(
                    length=geom_data.get('length', 0.1),
                    outer_diameter=geom_data.get('outer_diameter', 0.05),
                    inhibited_outer_surface=geom_data.get('inhibited_outer_surface', True)
                )
                grain_type = GrainType.ENDBURNER
            else:
                # Default to BATES if unknown
                geometry = BatesGrain(
                    length=geom_data.get('length', 0.1),
                    outer_diameter=geom_data.get('outer_diameter', 0.05)
                )
                grain_type = GrainType.BATES
            
            # Create grain
            grain = MotorGrain(geometry, propellant)
            design.set_grain(grain)
        
        # Create and set case if data available
        if 'case' in data:
            case_data = data['case']
            case = MotorCase(
                material=case_data.get('material', 'Steel'),
                inner_diameter=case_data.get('inner_diameter', 0.05),
                wall_thickness=case_data.get('wall_thickness', 0.003),
                length=case_data.get('length', 0.2),
                density=case_data.get('density', 7800.0),
                tensile_strength=case_data.get('tensile_strength', 500.0),
                safety_factor=case_data.get('safety_factor', 1.5)
            )
            design.set_case(case)
        
        # Create and set nozzle if data available
        if 'nozzle' in data:
            nozzle_data = data['nozzle']
            nozzle = Nozzle(
                throat_diameter=nozzle_data.get('throat_diameter', 0.01),
                expansion_ratio=nozzle_data.get('expansion_ratio', 8.0),
                contour_type=nozzle_data.get('contour_type', 'Conical'),
                half_angle=nozzle_data.get('half_angle', 15.0),
                percentage_bell=nozzle_data.get('percentage_bell', 80.0),
                divergence_efficiency=nozzle_data.get('divergence_efficiency', 0.98),
                material=nozzle_data.get('material', 'Graphite'),
                material_density=nozzle_data.get('material_density', 1850.0)
            )
            design.set_nozzle(nozzle)
        
        return design
    
    def save_to_file(self, file_path: str) -> bool:
        """Save the motor design to a JSON file."""
        try:
            # Convert to dictionary
            data = self.to_dict()
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            logger.info(f"Motor design saved to {file_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving motor design: {e}")
            return False
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'MotorDesign':
        """Load a motor design from a JSON file."""
        try:
            # Load from file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Create from dictionary
            design = cls.from_dict(data)
            
            logger.info(f"Motor design loaded from {file_path}")
            return design
        
        except Exception as e:
            logger.error(f"Error loading motor design: {e}")
            raise

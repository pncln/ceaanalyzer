"""
Optimization Module
-----------------

Provides optimization methods for rocket propulsion parameters.
"""

import numpy as np
from enum import Enum
from typing import Dict, List, Tuple, Callable, Any, Optional
import logging

# Set up logger
logger = logging.getLogger("cea_analyzer.optimization")


class OptimizationMethod(Enum):
    """Enumeration of available optimization methods."""
    GRID_SEARCH = "grid_search"
    GOLDEN_SECTION = "golden_section"
    GRADIENT_DESCENT = "gradient_descent"
    PARTICLE_SWARM = "particle_swarm"


def optimize_parameter(
    target_function: Optional[Callable] = None,
    params: Dict[str, Any] = None,
    bounds: Tuple[float, float] = (0.0, 1.0),
    method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    progress_callback: Callable[[int], None] = None,
    iteration_callback: Callable[[Dict], None] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Optimize a parameter within given bounds to maximize/minimize an objective.
    
    Parameters
    ----------
    target_function : Callable, optional
        Function that calculates the objective value for a given parameter value.
        If None, the function will be constructed based on params.
    params : Dict[str, Any]
        Dictionary containing the CEA data, parameter name, objective, and constraints.
    bounds : Tuple[float, float]
        Lower and upper bounds for the parameter.
    method : OptimizationMethod
        Method to use for optimization.
    max_iterations : int
        Maximum number of iterations for the optimization algorithm.
    tolerance : float
        Convergence tolerance.
    progress_callback : Callable[[int], None], optional
        Callback function to report progress (0-100).
    iteration_callback : Callable[[Dict], None], optional
        Callback function to report iteration data.
        
    Returns
    -------
    Tuple[Dict[str, Any], List[Dict[str, Any]]]
        Results dictionary and list of iteration data.
    """
    logger.info(f"Starting optimization with method: {method.name}")
    logger.info(f"Parameter bounds: {bounds}")
    
    # Extract parameters
    cea_data = params.get('cea_data', None)
    parameter_name = params.get('parameter', None)
    objective_name = params.get('objective', None)
    constraints = params.get('constraints', {})
    
    if cea_data is None or parameter_name is None or objective_name is None:
        raise ValueError("Missing required parameters: cea_data, parameter, or objective")
    
    # Define whether we're maximizing or minimizing
    maximize = objective_name in ['isp', 'thrust', 'thrust_to_weight']
    
    # Define the objective function if not provided
    if target_function is None:
        target_function = lambda x: _evaluate_objective(
            x, cea_data, parameter_name, objective_name, constraints
        )
    
    # Initialize iteration history
    iteration_history = []
    
    # Choose optimization method
    if method == OptimizationMethod.GRID_SEARCH:
        results = _grid_search(
            target_function, bounds, max_iterations, 
            maximize, progress_callback, iteration_callback, 
            iteration_history
        )
    elif method == OptimizationMethod.GOLDEN_SECTION:
        results = _golden_section(
            target_function, bounds, max_iterations, tolerance,
            maximize, progress_callback, iteration_callback, 
            iteration_history
        )
    elif method == OptimizationMethod.GRADIENT_DESCENT:
        results = _gradient_descent(
            target_function, bounds, max_iterations, tolerance,
            maximize, progress_callback, iteration_callback, 
            iteration_history
        )
    elif method == OptimizationMethod.PARTICLE_SWARM:
        results = _particle_swarm(
            target_function, bounds, max_iterations, 
            maximize, progress_callback, iteration_callback, 
            iteration_history
        )
    else:
        raise ValueError(f"Unsupported optimization method: {method}")
    
    logger.info(f"Optimization complete. Optimal value: {results['optimal_value']}")
    logger.info(f"Optimal parameter: {results['optimal_parameter']}")
    
    return results, iteration_history


def _evaluate_objective(
    parameter_value: float,
    cea_data: Dict[str, Any],
    parameter_name: str,
    objective_name: str,
    constraints: Dict[str, float]
) -> float:
    """
    Evaluate the objective function for a given parameter value.
    
    Parameters
    ----------
    parameter_value : float
        Value of the parameter to evaluate.
    cea_data : Dict[str, Any]
        CEA data dictionary.
    parameter_name : str
        Name of the parameter being optimized.
    objective_name : str
        Name of the objective being optimized.
    constraints : Dict[str, float]
        Dictionary of constraints.
        
    Returns
    -------
    float
        Objective value for the given parameter value.
    """
    # Create a copy of the CEA data to avoid modifying the original
    data = cea_data.copy()
    
    # Update the parameter value in the CEA data
    if parameter_name == "expansion_ratio":
        data['area_ratio'] = parameter_value
    elif parameter_name == "chamber_pressure":
        data['p_chamber'] = parameter_value
    elif parameter_name == "mixture_ratio":
        data['o_f'] = parameter_value
    elif parameter_name == "throat_diameter":
        data['throat_diameter'] = parameter_value
    elif parameter_name == "nozzle_length":
        data['nozzle_length'] = parameter_value
    else:
        raise ValueError(f"Unsupported parameter: {parameter_name}")
    
    # TODO: In a real implementation, this would recompute the rocket
    # performance based on the updated parameter. For this example, we'll
    # simulate the calculation with a simple function.
    
    # Simulate calculation of new performance metrics
    result = _simulate_performance(data, parameter_value, parameter_name)
    
    # Extract the objective value
    if objective_name == "isp":
        objective_value = result.get('isp', 0.0)
    elif objective_name == "thrust":
        objective_value = result.get('thrust', 0.0)
    elif objective_name == "mass":
        objective_value = -result.get('mass', 0.0)  # Negate for minimization
    elif objective_name == "thrust_to_weight":
        objective_value = result.get('thrust_to_weight', 0.0)
    elif objective_name == "length":
        objective_value = -result.get('length', 0.0)  # Negate for minimization
    else:
        raise ValueError(f"Unsupported objective: {objective_name}")
    
    # Apply constraints as penalties
    for constraint_name, constraint_value in constraints.items():
        if constraint_name == "max_length":
            if result.get('length', 0.0) > constraint_value:
                objective_value = float('-inf') if objective_name != "mass" and objective_name != "length" else float('inf')
        elif constraint_name == "max_mass":
            if result.get('mass', 0.0) > constraint_value:
                objective_value = float('-inf') if objective_name != "mass" and objective_name != "length" else float('inf')
        elif constraint_name == "min_isp":
            if result.get('isp', 0.0) < constraint_value:
                objective_value = float('-inf') if objective_name != "mass" and objective_name != "length" else float('inf')
    
    return objective_value


def _simulate_performance(
    data: Dict[str, Any],
    parameter_value: float,
    parameter_name: str
) -> Dict[str, float]:
    """
    Simulate calculation of rocket performance for optimization.
    
    Note: This is a simplified model for demonstration. In a real implementation,
    this would call the actual performance calculation functions.
    
    Parameters
    ----------
    data : Dict[str, Any]
        CEA data dictionary.
    parameter_value : float
        Value of the parameter being optimized.
    parameter_name : str
        Name of the parameter being optimized.
        
    Returns
    -------
    Dict[str, float]
        Dictionary of calculated performance metrics.
    """
    # Get baseline values from the data
    base_isp = data.get('isp', 300.0)
    base_thrust = data.get('thrust', 5000.0)
    base_mass = data.get('mass', 100.0)
    base_length = data.get('length', 1.0)
    
    # Simplified models for how parameters affect performance
    if parameter_name == "expansion_ratio":
        # Increasing expansion ratio increases Isp to a point, then decreases
        er_optimal = 20.0
        isp_factor = 1.0 - abs(parameter_value - er_optimal) / er_optimal * 0.1
        
        # Increasing expansion ratio increases length and mass
        length_factor = 0.5 + parameter_value / 30.0
        mass_factor = 0.8 + parameter_value / 50.0
        
        # Thrust calculation (simplified)
        thrust_factor = isp_factor
        
    elif parameter_name == "chamber_pressure":
        # Higher chamber pressure generally means higher Isp
        isp_factor = 0.8 + parameter_value / 10000.0 * 0.2
        
        # Higher pressure requires stronger chamber (more mass)
        mass_factor = 0.7 + parameter_value / 5000.0 * 0.3
        
        # Length is less affected by chamber pressure
        length_factor = 1.0
        
        # Thrust increases with chamber pressure
        thrust_factor = 0.5 + parameter_value / 5000.0 * 0.5
        
    elif parameter_name == "mixture_ratio":
        # Mixture ratio has an optimal point for Isp
        o_f_optimal = 2.1  # Typical optimal O/F for some propellants
        isp_factor = 1.0 - abs(parameter_value - o_f_optimal) / 3.0
        
        # Mass and length are less affected by mixture ratio
        mass_factor = 1.0
        length_factor = 1.0
        
        # Thrust follows Isp
        thrust_factor = isp_factor
        
    elif parameter_name == "throat_diameter":
        # Throat diameter affects thrust directly
        thrust_factor = (parameter_value / 0.05) ** 2  # Area is proportional to diameter squared
        
        # Isp is not strongly affected by throat diameter
        isp_factor = 1.0
        
        # Larger throat means larger engine
        mass_factor = (parameter_value / 0.05) ** 1.5
        length_factor = (parameter_value / 0.05) ** 0.5
        
    elif parameter_name == "nozzle_length":
        # Longer nozzle generally means better expansion and Isp
        isp_factor = 0.9 + parameter_value / 2.0 * 0.1
        
        # Length directly affected
        length_factor = parameter_value
        
        # Mass increases with length
        mass_factor = 0.8 + parameter_value / 1.0 * 0.2
        
        # Thrust follows Isp
        thrust_factor = isp_factor
        
    else:
        # Default factors
        isp_factor = 1.0
        thrust_factor = 1.0
        mass_factor = 1.0
        length_factor = 1.0
    
    # Calculate performance metrics
    isp = base_isp * isp_factor
    thrust = base_thrust * thrust_factor
    mass = base_mass * mass_factor
    length = base_length * length_factor
    
    # Derived metrics
    thrust_to_weight = thrust / (mass * 9.81)
    
    return {
        'isp': isp,
        'thrust': thrust,
        'mass': mass,
        'length': length,
        'thrust_to_weight': thrust_to_weight,
        'parameter': parameter_value
    }


def _grid_search(
    target_function: Callable[[float], float],
    bounds: Tuple[float, float],
    num_points: int,
    maximize: bool = True,
    progress_callback: Callable[[int], None] = None,
    iteration_callback: Callable[[Dict], None] = None,
    iteration_history: List[Dict] = None
) -> Dict[str, Any]:
    """
    Perform a grid search optimization.
    
    Parameters
    ----------
    target_function : Callable[[float], float]
        Function to optimize.
    bounds : Tuple[float, float]
        Lower and upper bounds for the parameter.
    num_points : int
        Number of points to evaluate.
    maximize : bool
        Whether to maximize (True) or minimize (False) the function.
    progress_callback : Callable[[int], None], optional
        Callback function to report progress (0-100).
    iteration_callback : Callable[[Dict], None], optional
        Callback function to report iteration data.
    iteration_history : List[Dict]
        List to store iteration history.
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary.
    """
    lower, upper = bounds
    step = (upper - lower) / (num_points - 1)
    
    best_value = float('-inf') if maximize else float('inf')
    best_parameter = lower
    best_index = 0
    
    # Evaluate the function at each grid point
    for i in range(num_points):
        # Calculate progress percentage
        progress = int(100 * i / (num_points - 1))
        if progress_callback:
            progress_callback(progress)
        
        # Calculate parameter value
        parameter = lower + i * step
        
        # Evaluate function
        value = target_function(parameter)
        
        # Extract performance data (assuming target_function returns a dict)
        performance = None
        if hasattr(target_function, '__closure__') and target_function.__closure__:
            # This is a lambda function with captured variables
            performance = _simulate_performance(
                {}, parameter, 
                target_function.__closure__[2].cell_contents  # Extract parameter_name
            )
        
        # Store iteration data
        iteration_data = {
            'iteration': i,
            'parameter': parameter,
            'value': value,
            'performance': performance
        }
        
        if iteration_callback:
            iteration_callback(iteration_data)
        
        if iteration_history is not None:
            iteration_history.append(iteration_data)
        
        # Update best value
        if (maximize and value > best_value) or (not maximize and value < best_value):
            best_value = value
            best_parameter = parameter
            best_index = i
    
    # Final results
    results = {
        'optimal_parameter': best_parameter,
        'optimal_value': best_value,
        'iterations': num_points,
        'converged': True,
        'best_iteration': best_index
    }
    
    if iteration_history and best_index < len(iteration_history) and 'performance' in iteration_history[best_index]:
        results['performance'] = iteration_history[best_index]['performance']
    
    return results


def _golden_section(
    target_function: Callable[[float], float],
    bounds: Tuple[float, float],
    max_iterations: int,
    tolerance: float,
    maximize: bool = True,
    progress_callback: Callable[[int], None] = None,
    iteration_callback: Callable[[Dict], None] = None,
    iteration_history: List[Dict] = None
) -> Dict[str, Any]:
    """
    Perform a golden section search optimization.
    
    Parameters
    ----------
    target_function : Callable[[float], float]
        Function to optimize.
    bounds : Tuple[float, float]
        Lower and upper bounds for the parameter.
    max_iterations : int
        Maximum number of iterations.
    tolerance : float
        Convergence tolerance.
    maximize : bool
        Whether to maximize (True) or minimize (False) the function.
    progress_callback : Callable[[int], None], optional
        Callback function to report progress (0-100).
    iteration_callback : Callable[[Dict], None], optional
        Callback function to report iteration data.
    iteration_history : List[Dict]
        List to store iteration history.
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary.
    """
    # Golden ratio
    gr = (np.sqrt(5) - 1) / 2
    
    a, b = bounds
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    
    fc = target_function(c)
    fd = target_function(d)
    
    # Store initial evaluations
    iteration_data_c = {
        'iteration': 0,
        'parameter': c,
        'value': fc,
        'performance': None  # Would need to be computed
    }
    
    iteration_data_d = {
        'iteration': 1,
        'parameter': d,
        'value': fd,
        'performance': None  # Would need to be computed
    }
    
    if iteration_callback:
        iteration_callback(iteration_data_c)
        iteration_callback(iteration_data_d)
    
    if iteration_history is not None:
        iteration_history.append(iteration_data_c)
        iteration_history.append(iteration_data_d)
    
    best_value = max(fc, fd) if maximize else min(fc, fd)
    best_parameter = c if ((fc > fd) == maximize) else d
    best_index = 0 if ((fc > fd) == maximize) else 1
    
    # Main iteration loop
    for i in range(2, max_iterations):
        # Calculate progress percentage
        progress = int(100 * i / max_iterations)
        if progress_callback:
            progress_callback(progress)
        
        # Check convergence
        if abs(b - a) < tolerance:
            break
        
        # Update interval
        if ((fc > fd) == maximize):
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = target_function(c)
            
            # Store iteration data
            iteration_data = {
                'iteration': i,
                'parameter': c,
                'value': fc,
                'performance': None  # Would need to be computed
            }
        else:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = target_function(d)
            
            # Store iteration data
            iteration_data = {
                'iteration': i,
                'parameter': d,
                'value': fd,
                'performance': None  # Would need to be computed
            }
        
        if iteration_callback:
            iteration_callback(iteration_data)
        
        if iteration_history is not None:
            iteration_history.append(iteration_data)
        
        # Update best value
        if (maximize and iteration_data['value'] > best_value) or (not maximize and iteration_data['value'] < best_value):
            best_value = iteration_data['value']
            best_parameter = iteration_data['parameter']
            best_index = i
    
    # Final results
    results = {
        'optimal_parameter': best_parameter,
        'optimal_value': best_value,
        'iterations': min(max_iterations, i + 1),
        'converged': abs(b - a) < tolerance,
        'best_iteration': best_index
    }
    
    if iteration_history and best_index < len(iteration_history) and 'performance' in iteration_history[best_index]:
        results['performance'] = iteration_history[best_index]['performance']
    
    return results


def _gradient_descent(
    target_function: Callable[[float], float],
    bounds: Tuple[float, float],
    max_iterations: int,
    tolerance: float,
    maximize: bool = True,
    progress_callback: Callable[[int], None] = None,
    iteration_callback: Callable[[Dict], None] = None,
    iteration_history: List[Dict] = None
) -> Dict[str, Any]:
    """
    Perform a gradient descent optimization.
    
    Parameters
    ----------
    target_function : Callable[[float], float]
        Function to optimize.
    bounds : Tuple[float, float]
        Lower and upper bounds for the parameter.
    max_iterations : int
        Maximum number of iterations.
    tolerance : float
        Convergence tolerance.
    maximize : bool
        Whether to maximize (True) or minimize (False) the function.
    progress_callback : Callable[[int], None], optional
        Callback function to report progress (0-100).
    iteration_callback : Callable[[Dict], None], optional
        Callback function to report iteration data.
    iteration_history : List[Dict]
        List to store iteration history.
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary.
    """
    # Start at the middle of the bounds
    lower, upper = bounds
    parameter = (lower + upper) / 2
    
    # Initial step size and learning rate
    step_size = (upper - lower) / 20
    learning_rate = 0.1
    
    # Initial evaluation
    value = target_function(parameter)
    
    # Extract performance data (assuming target_function returns a dict)
    performance = None
    if hasattr(target_function, '__closure__') and target_function.__closure__:
        # This is a lambda function with captured variables
        performance = _simulate_performance(
            {}, parameter, 
            target_function.__closure__[2].cell_contents  # Extract parameter_name
        )
    
    # Store initial evaluation
    iteration_data = {
        'iteration': 0,
        'parameter': parameter,
        'value': value,
        'performance': performance
    }
    
    if iteration_callback:
        iteration_callback(iteration_data)
    
    if iteration_history is not None:
        iteration_history.append(iteration_data)
    
    best_value = value
    best_parameter = parameter
    best_index = 0
    
    # Main iteration loop
    for i in range(1, max_iterations):
        # Calculate progress percentage
        progress = int(100 * i / max_iterations)
        if progress_callback:
            progress_callback(progress)
        
        # Compute gradient (central difference)
        h = step_size * 0.01
        forward = target_function(parameter + h)
        backward = target_function(parameter - h)
        gradient = (forward - backward) / (2 * h)
        
        # Invert gradient if minimizing
        if not maximize:
            gradient = -gradient
        
        # Update parameter with gradient ascent/descent
        parameter_new = parameter + learning_rate * gradient
        
        # Ensure parameter stays within bounds
        parameter_new = max(lower, min(upper, parameter_new))
        
        # Evaluate new parameter
        value_new = target_function(parameter_new)
        
        # Extract performance data
        performance = None
        if hasattr(target_function, '__closure__') and target_function.__closure__:
            performance = _simulate_performance(
                {}, parameter_new, 
                target_function.__closure__[2].cell_contents
            )
        
        # Store iteration data
        iteration_data = {
            'iteration': i,
            'parameter': parameter_new,
            'value': value_new,
            'performance': performance
        }
        
        if iteration_callback:
            iteration_callback(iteration_data)
        
        if iteration_history is not None:
            iteration_history.append(iteration_data)
        
        # Check for improvement
        improved = (maximize and value_new > value) or (not maximize and value_new < value)
        
        # Update best value
        if improved and ((maximize and value_new > best_value) or (not maximize and value_new < best_value)):
            best_value = value_new
            best_parameter = parameter_new
            best_index = i
        
        # Check convergence
        if abs(parameter_new - parameter) < tolerance or abs(value_new - value) < tolerance:
            parameter = parameter_new
            value = value_new
            break
        
        # Update for next iteration
        parameter = parameter_new
        value = value_new
        
        # Adaptive learning rate: reduce if not improving
        if not improved:
            learning_rate *= 0.8
    
    # Final results
    results = {
        'optimal_parameter': best_parameter,
        'optimal_value': best_value,
        'iterations': min(max_iterations, i + 1),
        'converged': i < max_iterations,
        'best_iteration': best_index
    }
    
    if iteration_history and best_index < len(iteration_history) and 'performance' in iteration_history[best_index]:
        results['performance'] = iteration_history[best_index]['performance']
    
    return results


def _particle_swarm(
    target_function: Callable[[float], float],
    bounds: Tuple[float, float],
    max_iterations: int,
    maximize: bool = True,
    progress_callback: Callable[[int], None] = None,
    iteration_callback: Callable[[Dict], None] = None,
    iteration_history: List[Dict] = None
) -> Dict[str, Any]:
    """
    Perform a particle swarm optimization.
    
    Parameters
    ----------
    target_function : Callable[[float], float]
        Function to optimize.
    bounds : Tuple[float, float]
        Lower and upper bounds for the parameter.
    max_iterations : int
        Maximum number of iterations.
    maximize : bool
        Whether to maximize (True) or minimize (False) the function.
    progress_callback : Callable[[int], None], optional
        Callback function to report progress (0-100).
    iteration_callback : Callable[[Dict], None], optional
        Callback function to report iteration data.
    iteration_history : List[Dict]
        List to store iteration history.
        
    Returns
    -------
    Dict[str, Any]
        Results dictionary.
    """
    # PSO parameters
    num_particles = 10
    inertia_weight = 0.5
    cognitive_weight = 1.5
    social_weight = 1.5
    
    # Initialize particles
    lower, upper = bounds
    positions = np.random.uniform(lower, upper, num_particles)
    velocities = np.random.uniform(-0.1 * (upper - lower), 0.1 * (upper - lower), num_particles)
    
    # Evaluate initial positions
    values = np.array([target_function(p) for p in positions])
    
    # Initialize personal and global best
    personal_best_positions = positions.copy()
    personal_best_values = values.copy()
    
    if maximize:
        global_best_idx = np.argmax(values)
    else:
        global_best_idx = np.argmin(values)
        
    global_best_position = positions[global_best_idx]
    global_best_value = values[global_best_idx]
    
    # Store initial best
    best_parameter = global_best_position
    best_value = global_best_value
    best_index = 0
    
    # Extract performance data
    performance = None
    if hasattr(target_function, '__closure__') and target_function.__closure__:
        performance = _simulate_performance(
            {}, global_best_position, 
            target_function.__closure__[2].cell_contents
        )
    
    # Store initial iteration data
    iteration_data = {
        'iteration': 0,
        'parameter': global_best_position,
        'value': global_best_value,
        'performance': performance
    }
    
    if iteration_callback:
        iteration_callback(iteration_data)
    
    if iteration_history is not None:
        iteration_history.append(iteration_data)
    
    # Main iteration loop
    for i in range(1, max_iterations):
        # Calculate progress percentage
        progress = int(100 * i / max_iterations)
        if progress_callback:
            progress_callback(progress)
        
        # Update velocities and positions
        for j in range(num_particles):
            # Random components
            r1 = np.random.random()
            r2 = np.random.random()
            
            # Update velocity
            velocities[j] = (
                inertia_weight * velocities[j] +
                cognitive_weight * r1 * (personal_best_positions[j] - positions[j]) +
                social_weight * r2 * (global_best_position - positions[j])
            )
            
            # Update position
            positions[j] += velocities[j]
            
            # Ensure position stays within bounds
            positions[j] = max(lower, min(upper, positions[j]))
            
            # Evaluate new position
            values[j] = target_function(positions[j])
            
            # Update personal best
            if (maximize and values[j] > personal_best_values[j]) or (not maximize and values[j] < personal_best_values[j]):
                personal_best_positions[j] = positions[j]
                personal_best_values[j] = values[j]
                
                # Update global best
                if (maximize and values[j] > global_best_value) or (not maximize and values[j] < global_best_value):
                    global_best_position = positions[j]
                    global_best_value = values[j]
        
        # Extract performance data for current global best
        performance = None
        if hasattr(target_function, '__closure__') and target_function.__closure__:
            performance = _simulate_performance(
                {}, global_best_position, 
                target_function.__closure__[2].cell_contents
            )
        
        # Store iteration data
        iteration_data = {
            'iteration': i,
            'parameter': global_best_position,
            'value': global_best_value,
            'performance': performance
        }
        
        if iteration_callback:
            iteration_callback(iteration_data)
        
        if iteration_history is not None:
            iteration_history.append(iteration_data)
        
        # Update best overall
        if (maximize and global_best_value > best_value) or (not maximize and global_best_value < best_value):
            best_value = global_best_value
            best_parameter = global_best_position
            best_index = i
    
    # Final results
    results = {
        'optimal_parameter': best_parameter,
        'optimal_value': best_value,
        'iterations': max_iterations,
        'converged': True,
        'best_iteration': best_index
    }
    
    if iteration_history and best_index < len(iteration_history) and 'performance' in iteration_history[best_index]:
        results['performance'] = iteration_history[best_index]['performance']
    
    return results

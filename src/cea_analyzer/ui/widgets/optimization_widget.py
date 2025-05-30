"""
Optimization Widget
----------------

Widget for optimizing rocket propulsion parameters.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Callable

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, 
    QComboBox, QDoubleSpinBox, QCheckBox, QPushButton,
    QRadioButton, QButtonGroup, QGridLayout, QFormLayout,
    QTabWidget, QMessageBox, QSplitter, QSpinBox, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

# Import optimization utilities
from ...analysis.optimization import OptimizationMethod, optimize_parameter


class OptimizationWorker(QThread):
    """
    Worker thread for running optimization algorithms without blocking the UI.
    """
    # Signals for progress and results
    progress_update = pyqtSignal(int)
    iteration_update = pyqtSignal(dict)
    optimization_complete = pyqtSignal(dict)
    optimization_error = pyqtSignal(str)
    
    def __init__(self, target_function: Callable, params: Dict, bounds: Tuple, 
                 method: OptimizationMethod, max_iterations: int = 100):
        """Initialize the optimization worker."""
        super().__init__()
        self.target_function = target_function
        self.params = params
        self.bounds = bounds
        self.method = method
        self.max_iterations = max_iterations
        self.results = None
        self.is_running = False
        
    def run(self):
        """Run the optimization process."""
        self.is_running = True
        try:
            # Call optimization function
            results, iterations = optimize_parameter(
                self.target_function,
                self.params,
                self.bounds,
                self.method,
                self.max_iterations,
                progress_callback=self._progress_callback,
                iteration_callback=self._iteration_callback
            )
            
            self.results = results
            self.optimization_complete.emit(results)
            
        except Exception as e:
            self.optimization_error.emit(str(e))
        finally:
            self.is_running = False
            
    def _progress_callback(self, progress: int):
        """Callback for reporting optimization progress."""
        self.progress_update.emit(progress)
        
    def _iteration_callback(self, iteration_data: Dict):
        """Callback for reporting iteration data."""
        self.iteration_update.emit(iteration_data)
        
    def stop(self):
        """Stop the optimization process."""
        if self.is_running:
            self.terminate()
            self.is_running = False


class OptimizationWidget(QWidget):
    """
    Widget for optimizing rocket propulsion parameters.
    """
    # Signal emitted when optimization is complete
    optimization_complete = pyqtSignal(dict)
    
    def __init__(self, parent=None):
        """Initialize the optimization widget."""
        super().__init__(parent)
        
        # Data
        self.cea_data = None
        self.target_function = None
        self.optimization_worker = None
        self.optimization_results = None
        self.iteration_history = []
        
        # Setup UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the UI components."""
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        # Create a splitter for the main panels
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top panel: Optimization controls
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        # Optimization parameters group
        params_group = QGroupBox("Optimization Parameters")
        params_layout = QFormLayout()
        
        # Parameter to optimize
        self.param_combo = QComboBox()
        self.param_combo.addItems([
            "Expansion Ratio", 
            "Chamber Pressure", 
            "Mixture Ratio", 
            "Throat Diameter",
            "Nozzle Length"
        ])
        self.param_combo.currentIndexChanged.connect(self._update_bounds)
        params_layout.addRow("Parameter to Optimize:", self.param_combo)
        
        # Objective function
        self.objective_combo = QComboBox()
        self.objective_combo.addItems([
            "Maximize Specific Impulse",
            "Maximize Thrust",
            "Minimize Mass",
            "Maximize Thrust-to-Weight Ratio",
            "Minimize Length"
        ])
        params_layout.addRow("Objective:", self.objective_combo)
        
        # Optimization method
        self.method_combo = QComboBox()
        self.method_combo.addItems([
            "Grid Search",
            "Golden Section",
            "Gradient Descent",
            "Particle Swarm"
        ])
        self.method_combo.currentIndexChanged.connect(self._update_method_controls)
        params_layout.addRow("Method:", self.method_combo)
        
        # Lower bound
        self.lower_bound_spin = QDoubleSpinBox()
        self.lower_bound_spin.setRange(0.01, 10000.0)
        self.lower_bound_spin.setValue(1.0)
        self.lower_bound_spin.setDecimals(2)
        params_layout.addRow("Lower Bound:", self.lower_bound_spin)
        
        # Upper bound
        self.upper_bound_spin = QDoubleSpinBox()
        self.upper_bound_spin.setRange(0.01, 10000.0)
        self.upper_bound_spin.setValue(20.0)
        self.upper_bound_spin.setDecimals(2)
        params_layout.addRow("Upper Bound:", self.upper_bound_spin)
        
        # Max iterations
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(10, 1000)
        self.max_iter_spin.setValue(100)
        self.max_iter_spin.setSingleStep(10)
        params_layout.addRow("Max Iterations:", self.max_iter_spin)
        
        # Constraints group
        self.constraints_group = QGroupBox("Constraints")
        constraints_layout = QFormLayout()
        
        # Maximum length constraint
        self.max_length_check = QCheckBox()
        self.max_length_check.setChecked(False)
        self.max_length_spin = QDoubleSpinBox()
        self.max_length_spin.setRange(0.01, 10.0)
        self.max_length_spin.setValue(1.0)
        self.max_length_spin.setDecimals(2)
        self.max_length_spin.setEnabled(False)
        self.max_length_check.toggled.connect(self.max_length_spin.setEnabled)
        
        max_length_widget = QWidget()
        max_length_layout = QHBoxLayout(max_length_widget)
        max_length_layout.setContentsMargins(0, 0, 0, 0)
        max_length_layout.addWidget(self.max_length_check)
        max_length_layout.addWidget(self.max_length_spin)
        constraints_layout.addRow("Maximum Length (m):", max_length_widget)
        
        # Maximum mass constraint
        self.max_mass_check = QCheckBox()
        self.max_mass_check.setChecked(False)
        self.max_mass_spin = QDoubleSpinBox()
        self.max_mass_spin.setRange(0.1, 1000.0)
        self.max_mass_spin.setValue(100.0)
        self.max_mass_spin.setDecimals(1)
        self.max_mass_spin.setEnabled(False)
        self.max_mass_check.toggled.connect(self.max_mass_spin.setEnabled)
        
        max_mass_widget = QWidget()
        max_mass_layout = QHBoxLayout(max_mass_widget)
        max_mass_layout.setContentsMargins(0, 0, 0, 0)
        max_mass_layout.addWidget(self.max_mass_check)
        max_mass_layout.addWidget(self.max_mass_spin)
        constraints_layout.addRow("Maximum Mass (kg):", max_mass_widget)
        
        # Minimum performance constraint
        self.min_isp_check = QCheckBox()
        self.min_isp_check.setChecked(False)
        self.min_isp_spin = QDoubleSpinBox()
        self.min_isp_spin.setRange(50.0, 500.0)
        self.min_isp_spin.setValue(200.0)
        self.min_isp_spin.setDecimals(1)
        self.min_isp_spin.setEnabled(False)
        self.min_isp_check.toggled.connect(self.min_isp_spin.setEnabled)
        
        min_isp_widget = QWidget()
        min_isp_layout = QHBoxLayout(min_isp_widget)
        min_isp_layout.setContentsMargins(0, 0, 0, 0)
        min_isp_layout.addWidget(self.min_isp_check)
        min_isp_layout.addWidget(self.min_isp_spin)
        constraints_layout.addRow("Minimum Isp (s):", min_isp_widget)
        
        # Set layouts for groups
        params_group.setLayout(params_layout)
        self.constraints_group.setLayout(constraints_layout)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        
        self.run_button = QPushButton("Run Optimization")
        self.run_button.clicked.connect(self._run_optimization)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self._stop_optimization)
        self.stop_button.setEnabled(False)
        
        self.export_button = QPushButton("Export Results")
        self.export_button.clicked.connect(self._export_results)
        self.export_button.setEnabled(False)
        
        buttons_layout.addWidget(self.run_button)
        buttons_layout.addWidget(self.stop_button)
        buttons_layout.addWidget(self.export_button)
        
        # Add to top layout
        top_layout.addWidget(params_group)
        top_layout.addWidget(self.constraints_group)
        top_layout.addWidget(self.progress_bar)
        top_layout.addLayout(buttons_layout)
        
        # Bottom panel: Results visualization
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        # Results tabs
        results_tabs = QTabWidget()
        
        # 1. Convergence Tab
        convergence_tab = QWidget()
        convergence_layout = QVBoxLayout(convergence_tab)
        
        self.convergence_fig = Figure(figsize=(8, 4))
        self.convergence_canvas = FigureCanvas(self.convergence_fig)
        self.convergence_toolbar = NavigationToolbar(self.convergence_canvas, self)
        
        convergence_layout.addWidget(self.convergence_toolbar)
        convergence_layout.addWidget(self.convergence_canvas)
        
        # 2. Results Tab
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        self.results_label = QLabel("No optimization results available.")
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("font-family: monospace;")
        
        results_layout.addWidget(self.results_label)
        
        # Add tabs to results_tabs
        results_tabs.addTab(convergence_tab, "Convergence")
        results_tabs.addTab(results_tab, "Results Summary")
        
        # Add results_tabs to bottom layout
        bottom_layout.addWidget(results_tabs)
        
        # Add panels to splitter
        self.splitter.addWidget(top_panel)
        self.splitter.addWidget(bottom_panel)
        self.splitter.setSizes([300, 300])  # Initial sizes
        
        # Add splitter to main layout
        main_layout.addWidget(self.splitter)
        
        # Initialize with empty plot
        self._create_empty_plot()
        
        # Update bounds based on selected parameter
        self._update_bounds()
        
        # Update method controls
        self._update_method_controls()
        
        # Disable controls until CEA data is loaded
        self._set_controls_enabled(False)
        
    def _set_controls_enabled(self, enabled: bool):
        """Enable or disable controls."""
        self.param_combo.setEnabled(enabled)
        self.objective_combo.setEnabled(enabled)
        self.method_combo.setEnabled(enabled)
        self.lower_bound_spin.setEnabled(enabled)
        self.upper_bound_spin.setEnabled(enabled)
        self.max_iter_spin.setEnabled(enabled)
        self.constraints_group.setEnabled(enabled)
        self.run_button.setEnabled(enabled)
        
    def _create_empty_plot(self):
        """Create an empty convergence plot with placeholder text."""
        self.convergence_fig.clear()
        ax = self.convergence_fig.add_subplot(111)
        ax.set_title("Optimization Convergence")
        ax.text(0.5, 0.5, "No optimization data available", 
               ha='center', va='center', fontsize=14, 
               transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        self.convergence_canvas.draw()
        
    def _update_bounds(self):
        """Update the bounds based on selected parameter."""
        param = self.param_combo.currentText()
        
        if param == "Expansion Ratio":
            self.lower_bound_spin.setRange(1.1, 100.0)
            self.lower_bound_spin.setValue(4.0)
            self.upper_bound_spin.setRange(1.1, 100.0)
            self.upper_bound_spin.setValue(20.0)
            self.lower_bound_spin.setSingleStep(0.5)
            self.upper_bound_spin.setSingleStep(0.5)
            self.lower_bound_spin.setDecimals(1)
            self.upper_bound_spin.setDecimals(1)
            
        elif param == "Chamber Pressure":
            self.lower_bound_spin.setRange(100.0, 35000.0)
            self.lower_bound_spin.setValue(1000.0)
            self.upper_bound_spin.setRange(100.0, 35000.0)
            self.upper_bound_spin.setValue(7000.0)
            self.lower_bound_spin.setSingleStep(100.0)
            self.upper_bound_spin.setSingleStep(100.0)
            self.lower_bound_spin.setDecimals(1)
            self.upper_bound_spin.setDecimals(1)
            
        elif param == "Mixture Ratio":
            self.lower_bound_spin.setRange(0.2, 10.0)
            self.lower_bound_spin.setValue(1.0)
            self.upper_bound_spin.setRange(0.2, 10.0)
            self.upper_bound_spin.setValue(6.0)
            self.lower_bound_spin.setSingleStep(0.1)
            self.upper_bound_spin.setSingleStep(0.1)
            self.lower_bound_spin.setDecimals(2)
            self.upper_bound_spin.setDecimals(2)
            
        elif param == "Throat Diameter":
            self.lower_bound_spin.setRange(0.001, 1.0)
            self.lower_bound_spin.setValue(0.01)
            self.upper_bound_spin.setRange(0.001, 1.0)
            self.upper_bound_spin.setValue(0.1)
            self.lower_bound_spin.setSingleStep(0.001)
            self.upper_bound_spin.setSingleStep(0.001)
            self.lower_bound_spin.setDecimals(3)
            self.upper_bound_spin.setDecimals(3)
            
        elif param == "Nozzle Length":
            self.lower_bound_spin.setRange(0.01, 5.0)
            self.lower_bound_spin.setValue(0.1)
            self.upper_bound_spin.setRange(0.01, 5.0)
            self.upper_bound_spin.setValue(1.0)
            self.lower_bound_spin.setSingleStep(0.01)
            self.upper_bound_spin.setSingleStep(0.01)
            self.lower_bound_spin.setDecimals(2)
            self.upper_bound_spin.setDecimals(2)
            
    def _update_method_controls(self):
        """Update controls based on optimization method."""
        method = self.method_combo.currentText()
        
        # Enable/disable max iterations based on method
        if method == "Grid Search":
            self.max_iter_spin.setEnabled(True)
            self.max_iter_spin.setToolTip("Number of points to evaluate")
        elif method == "Golden Section":
            self.max_iter_spin.setEnabled(True)
            self.max_iter_spin.setToolTip("Maximum number of iterations")
        elif method == "Gradient Descent":
            self.max_iter_spin.setEnabled(True)
            self.max_iter_spin.setToolTip("Maximum number of iterations")
        elif method == "Particle Swarm":
            self.max_iter_spin.setEnabled(True)
            self.max_iter_spin.setToolTip("Maximum number of iterations")
            
    def set_data(self, df: pd.DataFrame):
        """Set the dataframe for optimization (compatible with MainWindow interface)."""
        # Convert DataFrame to the format expected by set_cea_data
        cea_data = {"data": df}
        self.set_cea_data(cea_data)
    
    def set_cea_data(self, cea_data: Dict[str, Any]):
        """Set the CEA data for optimization."""
        if cea_data is None:
            self._set_controls_enabled(False)
            return
            
        # Store CEA data
        self.cea_data = cea_data
        
        # Enable controls
        self._set_controls_enabled(True)
        
    def _run_optimization(self):
        """Run the optimization process."""
        if self.cea_data is None:
            QMessageBox.warning(self, "Warning", "No CEA data available. Load data first.")
            return
            
        # Get parameters
        param = self.param_combo.currentText()
        objective = self.objective_combo.currentText()
        method_str = self.method_combo.currentText()
        lower_bound = self.lower_bound_spin.value()
        upper_bound = self.upper_bound_spin.value()
        max_iterations = self.max_iter_spin.value()
        
        # Create bounds tuple
        bounds = (lower_bound, upper_bound)
        
        # Map method string to enum
        method_map = {
            "Grid Search": OptimizationMethod.GRID_SEARCH,
            "Golden Section": OptimizationMethod.GOLDEN_SECTION,
            "Gradient Descent": OptimizationMethod.GRADIENT_DESCENT,
            "Particle Swarm": OptimizationMethod.PARTICLE_SWARM
        }
        method = method_map.get(method_str, OptimizationMethod.GRID_SEARCH)
        
        # Get constraints
        constraints = {}
        if self.max_length_check.isChecked():
            constraints['max_length'] = self.max_length_spin.value()
        if self.max_mass_check.isChecked():
            constraints['max_mass'] = self.max_mass_spin.value()
        if self.min_isp_check.isChecked():
            constraints['min_isp'] = self.min_isp_spin.value()
            
        # Reset data
        self.iteration_history = []
        self.optimization_results = None
        
        # Create target function based on parameter and objective
        params = {
            'cea_data': self.cea_data,
            'parameter': param.lower().replace(' ', '_'),
            'objective': objective.lower().replace(' ', '_').split('_')[1],
            'constraints': constraints
        }
        
        # Create worker thread
        self.optimization_worker = OptimizationWorker(
            target_function=None,  # Will be set in optimize_parameter
            params=params,
            bounds=bounds,
            method=method,
            max_iterations=max_iterations
        )
        
        # Connect signals
        self.optimization_worker.progress_update.connect(self._update_progress)
        self.optimization_worker.iteration_update.connect(self._update_iteration)
        self.optimization_worker.optimization_complete.connect(self._optimization_complete)
        self.optimization_worker.optimization_error.connect(self._optimization_error)
        
        # Update UI
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.export_button.setEnabled(False)
        
        # Start optimization
        self.optimization_worker.start()
        
    def _stop_optimization(self):
        """Stop the optimization process."""
        if self.optimization_worker and self.optimization_worker.isRunning():
            self.optimization_worker.stop()
            
        # Update UI
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
    def _update_progress(self, progress: int):
        """Update the progress bar."""
        self.progress_bar.setValue(progress)
        
    def _update_iteration(self, iteration_data: Dict):
        """Update with data from the current iteration."""
        # Store iteration data
        self.iteration_history.append(iteration_data)
        
        # Update convergence plot
        self._update_convergence_plot()
        
    def _optimization_complete(self, results: Dict):
        """Handle completion of optimization."""
        # Store results
        self.optimization_results = results
        
        # Update UI
        self.progress_bar.setValue(100)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(True)
        
        # Update results display
        self._update_results_display()
        
        # Update convergence plot
        self._update_convergence_plot()
        
        # Emit signal
        self.optimization_complete.emit(results)
        
        # Show completion message
        QMessageBox.information(self, "Optimization Complete", 
                              f"Optimization completed successfully. Optimal value: {results['optimal_value']:.4f}")
        
    def _optimization_error(self, error_msg: str):
        """Handle optimization error."""
        # Update UI
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        
        # Show error message
        QMessageBox.critical(self, "Optimization Error", f"Error during optimization: {error_msg}")
        
    def _update_convergence_plot(self):
        """Update the convergence plot with current iteration history."""
        if not self.iteration_history:
            return
            
        # Extract iteration data
        iterations = list(range(1, len(self.iteration_history) + 1))
        values = [data['value'] for data in self.iteration_history]
        parameters = [data['parameter'] for data in self.iteration_history]
        
        # Clear figure
        self.convergence_fig.clear()
        
        # Create two subplots
        ax1 = self.convergence_fig.add_subplot(211)
        ax2 = self.convergence_fig.add_subplot(212, sharex=ax1)
        
        # Plot objective value convergence
        ax1.plot(iterations, values, 'b-', marker='o', markersize=4)
        ax1.set_ylabel('Objective Value')
        ax1.set_title('Optimization Convergence')
        ax1.grid(True)
        
        # Plot parameter convergence
        ax2.plot(iterations, parameters, 'r-', marker='s', markersize=4)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Parameter Value')
        ax2.grid(True)
        
        # Highlight best point if optimization is complete
        if self.optimization_results:
            best_iter = self.optimization_results.get('best_iteration', 0)
            if 0 <= best_iter < len(iterations):
                ax1.plot([iterations[best_iter]], [values[best_iter]], 
                        'go', markersize=8, label='Optimal')
                ax2.plot([iterations[best_iter]], [parameters[best_iter]], 
                        'go', markersize=8)
                ax1.legend()
        
        # Adjust layout
        self.convergence_fig.tight_layout()
        self.convergence_canvas.draw()
        
    def _update_results_display(self):
        """Update the results display with optimization results."""
        if not self.optimization_results:
            self.results_label.setText("No optimization results available.")
            return
            
        # Get results
        results = self.optimization_results
        param = self.param_combo.currentText()
        objective = self.objective_combo.currentText()
        
        # Format results text
        html = f"""<html>
        <style>
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:hover {{ background-color: #f5f5f5; }}
        </style>
        <h2>Optimization Results</h2>
        <p><b>Parameter:</b> {param}</p>
        <p><b>Objective:</b> {objective}</p>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Optimal Value</td><td>{results['optimal_parameter']:.4f}</td></tr>
            <tr><td>Objective Value</td><td>{results['optimal_value']:.4f}</td></tr>
            <tr><td>Number of Iterations</td><td>{results['iterations']}</td></tr>
            <tr><td>Convergence Status</td><td>{results['converged']}</td></tr>
        """
        
        # Add performance metrics if available
        if 'performance' in results:
            perf = results['performance']
            html += """
            <tr><th colspan="2">Performance at Optimal Point</th></tr>
            """
            
            for key, value in perf.items():
                if isinstance(value, (int, float)):
                    html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value:.4f}</td></tr>"
                else:
                    html += f"<tr><td>{key.replace('_', ' ').title()}</td><td>{value}</td></tr>"
        
        html += """
        </table>
        </html>
        """
        
        self.results_label.setText(html)
        
    def _export_results(self):
        """Export optimization results."""
        if not self.optimization_results:
            return
            
        # Get file path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Optimization Results", "", 
            "CSV Files (*.csv);;JSON Files (*.json);;All Files (*)"
        )
        
        if not file_path:
            return
            
        try:
            if file_path.endswith('.csv'):
                # Export as CSV
                df = pd.DataFrame(self.iteration_history)
                df.to_csv(file_path, index=False)
            elif file_path.endswith('.json'):
                # Export as JSON
                import json
                with open(file_path, 'w') as f:
                    json.dump({
                        'results': self.optimization_results,
                        'iterations': self.iteration_history
                    }, f, indent=2)
            else:
                # Default to CSV
                df = pd.DataFrame(self.iteration_history)
                df.to_csv(file_path, index=False)
                
            QMessageBox.information(self, "Success", f"Results exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error exporting results: {str(e)}")

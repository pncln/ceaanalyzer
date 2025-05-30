"""
Plotting Module for CEA Analyzer
------------------------------

This module provides functionality for creating visualizations of CEA analysis results.
"""

from typing import Dict, Optional, List, Tuple
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib import rcParams
from ..core.config import CONFIG


# Set default figure style for consistent appearance
DEFAULT_FIGURE_STYLE = {
    'figure.figsize': (6, 4),
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'figure.edgecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.family': 'sans-serif',
    'font.size': 10,
}

# Apply default style
for key, value in DEFAULT_FIGURE_STYLE.items():
    rcParams[key] = value


def create_graphs(df: pd.DataFrame) -> Dict[str, Figure]:
    """
    Create a set of standard graphs from CEA analysis data.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing CEA analysis results
        
    Returns
    -------
    Dict[str, Figure]
        Dictionary of matplotlib Figure objects for different plots
    """
    if df.empty:
        return {}
        
    figs: Dict[str, Figure] = {}
    pcs = sorted(df["Pc (bar)"].unique())
    
    # Color map for consistent colors across plots
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # Isp vs O/F
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    _plot_series_by_pressure(ax, df, pcs, "O/F", "Isp (s)", 
                            "Specific Impulse vs O/F Ratio", "O/F Ratio", "Isp (s)",
                            marker='o', colors=colors)
    fig.tight_layout()
    figs["Isp"] = fig

    # T_chamber vs O/F
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    _plot_series_by_pressure(ax, df, pcs, "O/F", "T_chamber (K)", 
                            "Chamber Temperature vs O/F Ratio", "O/F Ratio", "Temperature (K)", 
                            marker='s', colors=colors)
    fig.tight_layout()
    figs["Temp"] = fig

    # Pressure Ratio vs O/F
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    _plot_series_by_pressure(ax, df, pcs, "O/F", "Pressure Ratio", 
                            "Pressure Ratio vs O/F Ratio", "O/F Ratio", "P_throat/Pc", 
                            marker='^', colors=colors)
    fig.tight_layout()
    figs["PressureRatio"] = fig

    # Enthalpy Drop vs O/F
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    _plot_series_by_pressure(ax, df, pcs, "O/F", "Delta_H (kJ/kg)", 
                            "Enthalpy Drop vs O/F Ratio", "O/F Ratio", "ΔH (kJ/kg)", 
                            marker='d', colors=colors)
    fig.tight_layout()
    figs["Enthalpy"] = fig
    
    # Add new plot: Area Ratio vs Isp
    if "Expansion Ratio" in df.columns:
        fig = Figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
        _plot_series_by_pressure(ax, df, pcs, "Expansion Ratio", "Isp (s)",
                                "Isp vs Area Ratio", "Area Ratio (Ae/At)", "Isp (s)",
                                marker='*', colors=colors)
        ax.set_xscale('log')
        fig.tight_layout()
        figs["AreaRatio"] = fig

    return figs


def _plot_series_by_pressure(ax, df: pd.DataFrame, pressure_values: List[float], 
                            x_col: str, y_col: str, title: str, xlabel: str, ylabel: str,
                            marker: str = 'o', colors: Optional[List[str]] = None) -> None:
    """
    Helper function to plot data series grouped by pressure values.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    df : pd.DataFrame
        DataFrame containing the data
    pressure_values : List[float]
        List of pressure values to group by
    x_col : str
        Column name for x-axis data
    y_col : str
        Column name for y-axis data
    title : str
        Plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    marker : str, optional
        Marker style for plot points
    colors : List[str], optional
        List of colors for different pressure series
    """
    if colors is None:
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, pc in enumerate(pressure_values):
        color = colors[i % len(colors)]
        sub = df[df["Pc (bar)"] == pc]
        if not sub.empty:
            ax.plot(sub[x_col], sub[y_col], marker=marker, linestyle='-', 
                    label=f'{pc} bar', color=color, markersize=6)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add grid but keep it subtle
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a light box around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#CCCCCC')


def create_optimization_plot(df: pd.DataFrame, target_col: str = "Isp (s)") -> Figure:
    """
    Create an optimization surface plot for a target parameter.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing CEA analysis results
    target_col : str, optional
        Column to optimize, default is "Isp (s)"
        
    Returns
    -------
    Figure
        Matplotlib Figure object containing the optimization plot
    """
    if df.empty or "O/F" not in df.columns or "Pc (bar)" not in df.columns:
        # Create empty figure if no valid data
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Insufficient data for optimization plot", 
                ha='center', va='center', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    # Create a meshgrid for the surface plot
    unique_ofs = sorted(df["O/F"].unique())
    unique_pcs = sorted(df["Pc (bar)"].unique())
    
    if len(unique_ofs) < 2 or len(unique_pcs) < 2:
        # Create empty figure if not enough data points
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Need multiple O/F and Pc values for optimization plot", 
                ha='center', va='center', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        return fig
    
    # Create the 3D plot
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a meshgrid for surface plotting
    OF_mesh, PC_mesh = np.meshgrid(unique_ofs, unique_pcs)
    Z_mesh = np.zeros_like(OF_mesh)
    
    # Fill in Z values from data
    for i, pc in enumerate(unique_pcs):
        for j, of in enumerate(unique_ofs):
            matching = df[(df["Pc (bar)"] == pc) & (df["O/F"] == of)]
            if not matching.empty:
                Z_mesh[i, j] = matching[target_col].values[0]
    
    # Plot the surface
    surf = ax.plot_surface(OF_mesh, PC_mesh, Z_mesh, cmap='viridis', 
                          linewidth=0, antialiased=True, alpha=0.8)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label=target_col)
    
    # Add labels
    ax.set_xlabel('O/F Ratio')
    ax.set_ylabel('Chamber Pressure (bar)')
    ax.set_zlabel(target_col)
    ax.set_title(f'Optimization Surface for {target_col}')
    
    # Find optimal point
    max_idx = np.unravel_index(Z_mesh.argmax(), Z_mesh.shape)
    opt_of = OF_mesh[max_idx]
    opt_pc = PC_mesh[max_idx]
    opt_z = Z_mesh[max_idx]
    
    # Mark optimal point
    ax.scatter([opt_of], [opt_pc], [opt_z], color='red', s=100, marker='*', 
              label=f'Optimal: OF={opt_of:.2f}, Pc={opt_pc:.1f}, {target_col}={opt_z:.1f}')
    
    # Add legend
    ax.legend()
    
    fig.tight_layout()
    return fig

"""
Method of Characteristics (MOC) Module
------------------------------------

This module implements the Method of Characteristics (MOC) for
designing supersonic nozzle contours.
"""

import numpy as np
from scipy.optimize import fsolve

def prandtl_meyer(M, gamma):
    """
    Compute the Prandtl–Meyer function ν(M) [radians].
    ν(M) = sqrt((γ+1)/(γ−1)) * arctan( sqrt((γ−1)/(γ+1)*(M^2−1)) )
           − arctan( sqrt(M^2 −1) )
    
    Parameters
    ----------
    M : float
        Mach number (M > 1)
    gamma : float
        Specific heat ratio
        
    Returns
    -------
    float
        Prandtl-Meyer angle in radians
    """
    term1 = np.sqrt((gamma+1)/(gamma-1))
    nu = term1 * np.arctan(np.sqrt((gamma-1)/(gamma+1)*(M**2 - 1))) \
         - np.arctan(np.sqrt(M**2 - 1))
    return nu

def inverse_prandtl_meyer(nu_target, gamma):
    """
    Invert ν(M) = nu_target → M via a root-finder.
    
    Parameters
    ----------
    nu_target : float
        Target Prandtl-Meyer angle in radians
    gamma : float
        Specific heat ratio
        
    Returns
    -------
    float
        Mach number corresponding to the given Prandtl-Meyer angle
    """
    fn = lambda M: prandtl_meyer(M, gamma) - nu_target
    # initial guess: if nu small, M≈1. else M≈2
    M0 = 1.0 + nu_target/np.pi  
    M, = fsolve(fn, M0)
    return float(M)

def mach_from_area_ratio(AR, gamma):
    """
    Solve A/A* = AR for supersonic Mach M > 1:
    A/A* = (1/M)*[ (2/(γ+1))*(1 + (γ−1)/2*M^2 ) ]^[(γ+1)/(2(γ−1)) ]
    
    Parameters
    ----------
    AR : float
        Area ratio (A/A*)
    gamma : float
        Specific heat ratio
        
    Returns
    -------
    float
        Mach number corresponding to the given area ratio
    """
    def area_eq(M):
        left = (1.0/M) * ( (2.0/(gamma+1))*(1.0 + 0.5*(gamma-1)*M**2) )**((gamma+1)/(2*(gamma-1)))
        return left - AR

    M_exit, = fsolve(area_eq, 2.0)  # start guess M=2
    return float(M_exit)

def generate_moc_contour(area_ratio, gamma, N=25, R_throat=1.0):
    """
    Compute a Method-of-Characteristics wall contour for an axisymmetric nozzle.

    Parameters
    ----------
    area_ratio : float
        Exit area A_e / throat area A*.
    gamma : float
        Specific‐heat ratio of the gas.
    N : int
        Number of characteristic "fan" lines (including the throat and exit).
    R_throat : float
        Physical throat radius (meters or any length unit).

    Returns
    -------
    x_wall : np.ndarray, shape (N,)
    r_wall : np.ndarray, shape (N,)
        Coordinates of the wall contour, starting at the throat (x=0,r=R_throat)
        and ending at the exit lip.
    """
    # 1) find exit Mach from area_ratio
    M_exit = mach_from_area_ratio(area_ratio, gamma)

    # 2) Prandtl-Meyer at exit
    nu_exit = prandtl_meyer(M_exit, gamma)

    # 3) maximum turning angle θ_max = ν_exit / 2
    theta_max = nu_exit / 2.0

    # 4) discretize the fan from 0 → θ_max
    theta = np.linspace(0.0, theta_max, N)

    # 5) for each turning angle, find the local Mach M_i
    nu_i = 2.0 * theta
    M_i = np.array([inverse_prandtl_meyer(nu, gamma) for nu in nu_i])

    # 6) Mach‐angle μ_i = arcsin(1/M_i)
    mu_i = np.arcsin(1.0 / M_i)

    # 7) now step along the wall:
    #    using the small‐angle approximation for characteristic intersection:
    #    Δs_i = R_throat * (θ_i − θ_{i−1}) / tan(μ_i)
    #    then x_i = x_{i−1} + Δs_i * cos(θ_i)
    #         r_i = r_{i−1} + Δs_i * sin(θ_i)
    x_wall = np.zeros(N)
    r_wall = np.zeros(N)
    x_wall[0] = 0.0
    r_wall[0] = R_throat

    for i in range(1, N):
        dtheta = theta[i] - theta[i-1]
        ds = R_throat * dtheta / np.tan(mu_i[i])
        x_wall[i] = x_wall[i-1] + ds * np.cos(theta[i])
        r_wall[i] = r_wall[i-1] + ds * np.sin(theta[i])

    return x_wall, r_wall

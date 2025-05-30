#!/usr/bin/env python3
"""
NASA-CEA Output Parser Module
----------------------------

This module provides functionality to parse NASA Chemical Equilibrium
with Applications (CEA) output files into structured data for analysis.
"""

import re
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union

import pandas as pd

from ..core.config import G0


def parse_cea_output(path: Union[str, Path], progress_cb: Optional[Callable[[int], None]] = None) -> pd.DataFrame:
    """
    Parse a NASA-CEA output file and return a DataFrame with one row per CASE.
    
    Parameters
    ----------
    path : str or Path
        Path to the CEA output file
    progress_cb : Callable[[int], None], optional
        Callback function for progress reporting (0-100)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        'O/F', 'Pc (bar)', 'P_throat (bar)', 'Pressure Ratio', 'Expansion Ratio',
        'T_chamber (K)', 'T_throat (K)', 'H_chamber (kJ/kg)', 'H_throat (kJ/kg)',
        'Delta_H (kJ/kg)', 'Isp (m/s)', 'Isp (s)'
        
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    IOError
        If there are issues reading the file
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CEA output file not found: {path}")
        
    try:
        # Read entire file with robust encoding handling
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        lines = text.splitlines()
    except IOError as e:
        logging.error(f"Error reading CEA output file: {e}")
        raise

    # Find indices of "CASE =" lines
    case_idxs = [i for i, line in enumerate(lines) if line.lstrip().startswith("CASE =")]
    if not case_idxs:
        logging.warning(f"No CASE statements found in {path}")
        return pd.DataFrame()
        
    case_idxs.append(len(lines))  # Add end of file as final boundary

    records: List[Dict[str, Any]] = []
    total = len(case_idxs) - 1

    for idx, (start, end) in enumerate(zip(case_idxs, case_idxs[1:])):
        # Report progress if callback provided
        if progress_cb and total > 0:
            progress_cb(int(100 * idx / total))
            
        # Extract the text block for this case
        block = "\n".join(lines[start:end])

        try:
            # Parse data from the case block
            record = _parse_case_block(block)
            if record:
                records.append(record)
        except Exception as e:
            logging.warning(f"Failed to parse case {idx+1}: {e}")
            continue

    # Build DataFrame
    df = pd.DataFrame(records)
    if df.empty:
        logging.warning("No valid data found in CEA output file")
        return df

    # Sort and reset index
    df.sort_values(["Pc (bar)", "O/F"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def extract_thermo_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Extract key thermodynamic data from a CEA results DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with CEA results, as returned by parse_cea_output
        
    Returns
    -------
    Dict[str, Any]
        Dictionary containing thermodynamic properties:
        - gamma: Specific heat ratio (cp/cv)
        - molecular_weight: Average molecular weight
        - c_star: Characteristic velocity
        - area_ratio: Optimum expansion ratio
        - cf: Thrust coefficient
    """
    if df.empty:
        return {}
    
    # Find row with maximum Isp - typically the optimal O/F ratio
    max_isp_row = df.loc[df['Isp (s)'].idxmax()]
    
    # Extract key data
    thermo_data = {
        'gamma': 1.2,  # Default value if not available
        'molecular_weight': 20.0,  # Default value if not available
        'c_star': max_isp_row.get('Isp (m/s)', 0) / 1.4,  # Approximate from Isp
        'area_ratio': max_isp_row.get('Expansion Ratio', 8.0),
        'cf': 1.4,  # Default value if not available
        'chamber_pressure': max_isp_row.get('Pc (bar)', 0) * 1e5,  # Convert to Pa
        'optimal_of_ratio': max_isp_row.get('O/F', 0)
    }
    
    # Compute gamma if chamber and throat temperatures are available
    if 'T_chamber (K)' in max_isp_row and 'T_throat (K)' in max_isp_row and 'Pressure Ratio' in max_isp_row:
        T_chamber = max_isp_row['T_chamber (K)']
        T_throat = max_isp_row['T_throat (K)']
        P_ratio = max_isp_row['Pressure Ratio']
        
        # Gamma can be estimated from: T_throat/T_chamber = (P_throat/P_chamber)^((gamma-1)/gamma)
        # Solving for gamma: gamma = 1 / (1 - ln(T_throat/T_chamber) / ln(P_ratio))
        try:
            if T_throat > 0 and T_chamber > 0 and P_ratio > 0:
                gamma = 1.0 / (1.0 - (np.log(T_throat/T_chamber) / np.log(P_ratio)))
                # Validate reasonable value
                if 1.1 <= gamma <= 1.7:
                    thermo_data['gamma'] = gamma
        except Exception:
            # Keep default value if calculation fails
            pass
    
    return thermo_data


def _parse_case_block(block: str) -> Optional[Dict[str, float]]:
    """
    Parse a single case block from CEA output.
    
    Parameters
    ----------
    block : str
        Text block containing a single CEA case
        
    Returns
    -------
    Optional[Dict[str, float]]
        Dictionary of extracted values or None if parsing failed
    """
    # 1) Expansion ratio (Ae/At) from PERFORMANCE PARAMETERS
    m_ar = re.search(r"Ae/At\s+([\d\.]+)", block, re.IGNORECASE)
    ar   = float(m_ar.group(1)) if m_ar else 1.0

    # 2) Extract specific heat ratio (gamma)
    m_gamma = re.search(r"GAMMAs\s+([\d\.]+)", block)
    gamma = float(m_gamma.group(1)) if m_gamma else None

    # 3) Core combustion data
    m_of  = re.search(r"O/F=\s*([\d\.]+)", block)
    m_p   = re.search(r"P,\s*BAR\s+([\d\.]+)\s+([\d\.]+)", block)
    m_t   = re.search(r"T,\s*K\s+([\d\.]+)\s+([\d\.]+)", block)
    m_h   = re.search(r"H,\s*KJ/KG\s+([-\d\.]+)\s+([-\d\.]+)", block)
    m_isp = re.search(r"Isp,.*?M/SEC\s+([\d\.]+)", block)

    # Skip if any required field is missing
    if not all([m_of, m_p, m_t, m_h, m_isp]):
        return None

    # 4) Extract numeric values
    of    = float(m_of.group(1))
    pc    = float(m_p.group(1))
    pt    = float(m_p.group(2))
    tch   = float(m_t.group(1))
    tth   = float(m_t.group(2))
    hch   = float(m_h.group(1))
    hth   = float(m_h.group(2))
    isp_m = float(m_isp.group(1))
    isp_s = isp_m / G0

    # 5) Compose result dictionary
    result = {
        "O/F":               of,
        "Pc (bar)":          pc,
        "P_throat (bar)":    pt,
        "Pressure Ratio":    pt/pc,
        "Expansion Ratio":   ar,
        "T_chamber (K)":     tch,
        "T_throat (K)":      tth,
        "H_chamber (kJ/kg)": hch,
        "H_throat (kJ/kg)":  hth,
        "Delta_H (kJ/kg)":   hch - hth,
        "Isp (m/s)":         isp_m,
        "Isp (s)":           isp_s,
    }
    
    # Add gamma if found
    if gamma is not None:
        result["gamma"] = gamma
        
    return result

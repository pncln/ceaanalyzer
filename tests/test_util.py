"""
Unit tests for utility functions.
"""

import os
import sys
import unittest
import numpy as np

# Import from src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.util import (
    ambient_pressure, 
    solve_mach, 
    mach_from_area_ratio,
    pressure_ratio_from_mach
)


class TestUtilFunctions(unittest.TestCase):
    """Test cases for the utility functions in the cea_analyzer.util module."""

    def test_ambient_pressure(self):
        """Test the ambient pressure calculation."""
        # Test sea level pressure
        self.assertAlmostEqual(ambient_pressure(0.0), 101325.0, delta=1.0)
        
        # Test pressure at 5km
        self.assertLess(ambient_pressure(5000.0), ambient_pressure(0.0))
        
        # Test pressure decreases with altitude
        p1 = ambient_pressure(1000.0)
        p2 = ambient_pressure(2000.0)
        self.assertLess(p2, p1)

    def test_solve_mach(self):
        """Test the Mach number solver."""
        # Test known values for gamma = 1.4
        gamma = 1.4
        
        # Subsonic: M=0.5 should give p/p0 = 0.8430
        p_ratio = (1 + 0.5*(gamma-1)*0.5**2) ** (-gamma/(gamma-1))
        mach = solve_mach(p_ratio, gamma)
        self.assertAlmostEqual(mach, 0.5, delta=0.001)
        
        # Supersonic: M=2.0 should give p/p0 = 0.1278
        p_ratio = (1 + 0.5*(gamma-1)*2.0**2) ** (-gamma/(gamma-1))
        mach = solve_mach(p_ratio, gamma)
        self.assertAlmostEqual(mach, 2.0, delta=0.001)

    def test_mach_from_area_ratio(self):
        """Test the Mach number calculation from area ratio."""
        # Test with known values for gamma = 1.4
        gamma = 1.4
        
        # At the throat (AR = 1.0), Mach should be 1.0
        mach = mach_from_area_ratio(1.0, gamma)
        self.assertAlmostEqual(mach, 1.0, delta=0.001)
        
        # Test with higher area ratio (e.g., AR = 4.0)
        mach = mach_from_area_ratio(4.0, gamma)
        self.assertGreater(mach, 1.0)
        
        # Check that area ratios produce unique Mach numbers
        mach1 = mach_from_area_ratio(2.0, gamma)
        mach2 = mach_from_area_ratio(3.0, gamma)
        self.assertNotEqual(mach1, mach2)

    def test_pressure_ratio_from_mach(self):
        """Test the pressure ratio calculation from Mach number."""
        # Test with known values for gamma = 1.4
        gamma = 1.4
        
        # M=0.0 should give p/p0 = 1.0
        self.assertAlmostEqual(pressure_ratio_from_mach(0.0, gamma), 1.0, delta=0.001)
        
        # M=1.0 should give p/p0 = 0.5283 for gamma = 1.4
        expected = (1.0 + 0.5 * 0.4 * 1.0**2) ** (-1.4 / 0.4)
        self.assertAlmostEqual(pressure_ratio_from_mach(1.0, gamma), expected, delta=0.001)


if __name__ == '__main__':
    unittest.main()

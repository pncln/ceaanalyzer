"""
Grain Type Definitions Module
---------------------------

This module provides the basic type definitions and enumerations
for rocket motor grain configurations.
"""

from enum import Enum


class GrainType(Enum):
    """Enumeration of supported grain geometries."""
    BATES = "BATES (Circular Port)"
    STAR = "Star"
    WAGON_WHEEL = "Wagon Wheel"
    FINOCYL = "Finocyl"
    MOON_BURNER = "Moon Burner"
    C_SLOT = "C-Slot"
    ENDBURNER = "End Burner"
    CUSTOM = "Custom"

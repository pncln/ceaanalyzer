"""
Motor Type Definitions Module
--------------------------

This module provides basic enumerations and type definitions for
rocket motor designs.
"""

from enum import Enum


class MotorType(Enum):
    """Enumeration of motor types."""
    SOLID = "Solid Rocket Motor"
    LIQUID = "Liquid Rocket Engine"
    HYBRID = "Hybrid Rocket Motor"

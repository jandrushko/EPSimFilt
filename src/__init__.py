"""
MEP Filter Testing Tool - Core Modules

This package contains the core functionality for MEP signal generation,
noise modeling, filtering, and performance evaluation.
"""

from .signal_generator import MEPGenerator
from .noise_generator import NoiseGenerator
from .filters import MEPFilters
from .metrics import MEPMetrics

__version__ = "1.0.0"
__author__ = "Justin W. Andrushko"
__email__ = "justin.andrushko@northumbria.ac.uk"

__all__ = [
    'MEPGenerator',
    'NoiseGenerator',
    'MEPFilters',
    'MEPMetrics'
]

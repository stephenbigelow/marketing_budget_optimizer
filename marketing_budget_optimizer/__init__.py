"""
Marketing Budget Optimizer package.
"""

from .curves import ResponseCurve, HillCurve
from .optimize import GeneticAlgorithmOptimizer

__all__ = ['ResponseCurve', 'HillCurve', 'GeneticAlgorithmOptimizer'] 
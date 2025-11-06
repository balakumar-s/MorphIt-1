"""
MorphIt: Flexible Spherical Approximation of Robot Morphology

A novel algorithm for approximating robot morphology using spherical primitives
that balances geometric accuracy with computational efficiency.
"""

from .morphit import MorphIt
from .config import get_config, MorphItConfig
from .training import MorphItTrainer
from .losses import MorphItLosses
from .visualization import visualize_packing, MorphItVisualizer

__version__ = "0.1.0"
__author__ = ("Nataliya Nechyporenko, Yutong Zhang, Sean Campbell, "
              "Alessandro Roncone")

__all__ = [
    "MorphIt",
    "get_config",
    "MorphItConfig",
    "MorphItTrainer",
    "MorphItLosses",
    "visualize_packing",
    "MorphItVisualizer",
]
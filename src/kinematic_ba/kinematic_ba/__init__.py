"""Kinematically-constrained bundle adjustment (Ceres) with Python bindings."""

# Import "core" library C++ bindings
from . import _core

# Re-Export modules from the core library
slam = _core.slam

# Remove/hide core namespace
del _core

"""
Phase 15.3 — Coordinated Reservoir Controller Package

Provides deterministic MPC-style and baseline controllers
for coordinated multi-reservoir gate management.
"""

from .baseline_controller import BaselineController
from .mpc_controller import MPCController
from .objective import ObjectiveFunction
from .safety import SafetyLayer

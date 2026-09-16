"""
Phase 15.1 — Deterministic Interconnected Reservoir Network Environment

A generic N-reservoir network simulator for coordinated water management.
This package provides the canonical environment for future controller
integration (Phase 15.2+).

Key design decisions vs. src/simulator/:
  - Generic N-node topology driven entirely by YAML configuration.
  - Explicit provenance classification for every parameter.
  - Configurable routing attenuation with documented mass accounting.
  - No hardcoded reservoir names or cascade ordering in Python source.
"""

from .reservoir_network import ReservoirNetwork, ReservoirNode
from .provenance import Provenance, ProvenanceLevel
from .live_cascade_adapter import LiveCascadeAdapter, LiveReservoirView
from .live_forecast_adapter import LiveForecastAdapter, LiveForecastBundle

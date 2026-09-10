"""
Provenance system for the Reservoir Network Environment.

Every configurable parameter in the network environment carries
a provenance classification indicating how the parameter value
was determined. This prevents silent presentation of assumptions
as verified physical facts.

Levels:
  OBSERVED          — Value taken directly from repository observation data
                      (e.g., maximum observed live_storage from the dataset).
  VERIFIED          — Value cross-checked against multiple independent
                      repository sources (e.g., FRL matched across files).
  ASSUMED_FOR_PROTOTYPE — Value chosen for the prototype demonstration.
                      Not verified against real-world Kerala dam data.
                      Must be clearly flagged in any report or dashboard.
"""

from enum import Enum
from typing import Any, Dict
from dataclasses import dataclass, field


class ProvenanceLevel(Enum):
    OBSERVED = "OBSERVED"
    VERIFIED = "VERIFIED"
    ASSUMED_FOR_PROTOTYPE = "ASSUMED_FOR_PROTOTYPE"


@dataclass
class Provenance:
    """
    A single parameter with provenance metadata.
    """
    name: str
    value: Any
    unit: str
    level: ProvenanceLevel
    source: str   # human-readable description of where the value came from
    notes: str = ""

    def is_assumed(self) -> bool:
        return self.level == ProvenanceLevel.ASSUMED_FOR_PROTOTYPE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "provenance": self.level.value,
            "source": self.source,
            "notes": self.notes,
        }


class ProvenanceRegistry:
    """
    Central registry that collects all provenance-tagged parameters
    for the entire network configuration.  Used by dashboards and
    reports to distinguish verified data from prototype assumptions.
    """

    def __init__(self):
        self._entries: Dict[str, Provenance] = {}

    def register(self, key: str, prov: Provenance) -> None:
        self._entries[key] = prov

    def get(self, key: str) -> Provenance:
        return self._entries[key]

    def all_assumed(self) -> Dict[str, Provenance]:
        return {k: v for k, v in self._entries.items() if v.is_assumed()}

    def all_verified(self) -> Dict[str, Provenance]:
        return {k: v for k, v in self._entries.items() if not v.is_assumed()}

    def summary(self) -> Dict[str, int]:
        counts = {level.value: 0 for level in ProvenanceLevel}
        for prov in self._entries.values():
            counts[prov.level.value] += 1
        return counts

    def all_entries(self) -> Dict[str, Provenance]:
        return dict(self._entries)

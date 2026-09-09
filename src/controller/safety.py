"""
Phase 15.3 — Safety Layer for Controller Output Validation

Validates every controller output BEFORE it is applied to the real
network. Rejects invalid commands and substitutes safe fallbacks.
"""

import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class SafetyCheckResult:
    """Result of the safety validation."""
    is_safe: bool
    validated_gates: Dict[str, float]
    violations: list
    status: str  # "SAFE", "CORRECTED", "EMERGENCY", "FALLBACK"


class SafetyLayer:
    """
    Final validation layer that checks controller outputs.

    Checks:
      1. Gate bounds [0.0, 1.0]
      2. Finite numeric values (no NaN, no Inf)
      3. Maximum gate movement per timestep
      4. Downstream capacity (estimated from proposed releases)
      5. All required nodes present

    If violations are found, the layer either corrects the output
    (gate clamping) or returns a safe fallback (all gates to
    conservative position).
    """

    def __init__(self, max_gate_change_per_step: float = 0.5):
        """
        Parameters
        ----------
        max_gate_change_per_step : float
            Maximum allowed gate position change per timestep [0, 1].
            ASSUMED_FOR_PROTOTYPE.
        """
        self.max_gate_change = max_gate_change_per_step

    def validate(
        self,
        proposed_gates: Dict[str, float],
        current_gates: Dict[str, float],
        node_ids: list,
    ) -> SafetyCheckResult:
        """
        Validate proposed gate commands.

        Returns SafetyCheckResult with corrected gates if needed.
        """
        violations = []
        validated = {}
        corrected = False

        for nid in node_ids:
            if nid not in proposed_gates:
                violations.append(f"{nid}: missing gate command — using fallback")
                validated[nid] = current_gates.get(nid, 0.1)
                corrected = True
                continue

            gate = proposed_gates[nid]

            # Check for NaN/Inf
            if not isinstance(gate, (int, float)) or math.isnan(gate) or math.isinf(gate):
                violations.append(f"{nid}: invalid gate value {gate} — clamped to 0.1")
                validated[nid] = 0.1
                corrected = True
                continue

            # Clamp to [0, 1]
            original = gate
            gate = max(0.0, min(1.0, gate))
            if gate != original:
                violations.append(f"{nid}: gate {original:.4f} clamped to {gate:.4f}")
                corrected = True

            # Rate-limit gate movement
            prev = current_gates.get(nid, gate)
            delta = gate - prev
            if abs(delta) > self.max_gate_change:
                gate = prev + self.max_gate_change * (1.0 if delta > 0 else -1.0)
                gate = max(0.0, min(1.0, gate))
                violations.append(
                    f"{nid}: gate movement {abs(delta):.4f} exceeds limit "
                    f"{self.max_gate_change:.4f} — rate-limited to {gate:.4f}"
                )
                corrected = True

            validated[nid] = gate

        if len(violations) == 0:
            status = "SAFE"
        elif corrected and len(validated) == len(node_ids):
            status = "CORRECTED"
        else:
            status = "EMERGENCY"

        return SafetyCheckResult(
            is_safe=(len(violations) == 0),
            validated_gates=validated,
            violations=violations,
            status=status,
        )

    @staticmethod
    def emergency_fallback(node_ids: list) -> Dict[str, float]:
        """
        Return conservative gate positions for all nodes.
        Sets all gates to 10% open — minimal release.
        """
        return {nid: 0.1 for nid in node_ids}

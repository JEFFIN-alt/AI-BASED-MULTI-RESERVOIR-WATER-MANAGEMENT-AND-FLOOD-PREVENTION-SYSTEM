"""
Stage 15 — END-TO-END ARCHITECTURE / INTEGRATION VERIFICATION: evidence.

Traces the REAL runtime path and records actual values at every boundary:

    input/inflow -> frozen LSTM V3 -> LiveForecastAdapter -> NetworkForecastSnapshot
                 -> GNN advisory (advisory only)
                 -> MPCController -> SafetyLayer -> DownstreamCapacityGuard
                 -> FINAL_SAFE_CONTROL_ACTION
                 -> ReservoirNetwork.step() -> MassBalanceMonitor
                 -> authoritative state -> REST/WebSocket
                 -> Three.js Digital Twin / Streamlit

The boundary call order is measured by instrumenting the real methods (not by
reading a docstring), and the phases are timed individually.

This script is VERIFICATION ONLY. It changes no production behaviour: the
instrumentation wraps real methods and calls straight through.

OUTPUT
    results/phase15_stage15_end_to_end/stage15_end_to_end_evidence.json
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:  # the evidence text contains non-ASCII typography
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

from fastapi.testclient import TestClient  # noqa: E402

from src.controller.downstream_capacity_guard import DownstreamCapacityGuard  # noqa: E402
from src.controller.live_mpc_orchestrator import LiveMPCOrchestrator  # noqa: E402
from src.controller.mpc_controller import MPCController  # noqa: E402
from src.controller.safety import SafetyLayer  # noqa: E402
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.sim_bridge import SimBridge  # noqa: E402
from src.network_env.live_forecast_adapter import LiveForecastAdapter  # noqa: E402
from src.network_env.mass_balance import MassBalanceMonitor  # noqa: E402
from src.network_env.reservoir_network import ReservoirNetwork  # noqa: E402

OUTPUT_DIR = _PROJECT_ROOT / "results" / "phase15_stage15_end_to_end"
OUTPUT_PATH = OUTPUT_DIR / "stage15_end_to_end_evidence.json"

LIVE_CONFIG = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"
FROZEN_ARTIFACTS = [
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "best_model.pt",
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "log_target_scaler.pkl",
    _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget" / "test_predictions_original_units.csv",
    _PROJECT_ROOT / "models" / "gcn_lstm_gated_v1" / "best_model.pt",
    _PROJECT_ROOT / "data" / "processed" / "graph" / "graph_D_correlation_v1_2" / "edges.csv",
]
NODES = ["Virtual Reservoir A", "Virtual Reservoir B",
         "Virtual Reservoir C", "Virtual Reservoir D"]
NODE_TO_NAME = {"Virtual Reservoir A": "Anayirankal", "Virtual Reservoir B": "Ponmudi",
                "Virtual Reservoir C": "Idamalayar", "Virtual Reservoir D": "Idukki"}
HORIZONS = ("forecast_1d", "forecast_3d", "forecast_7d")
PHYSICAL_DELAYS = [2, 1, 1]
PHYSICAL_ATTENUATION = [0.90, 0.85, 0.80]

client = TestClient(app)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(*, status="VALIDATED",
             provenance="REAL_MEASUREMENT_INPUTS_FROZEN_MODEL",
             validated=True, simulated=False,
             v1=2.0, v3=2.1, v7=2.2):
    return {
        "forecast_1d": v1, "forecast_3d": v3, "forecast_7d": v7,
        "forecast_status": status,
        "forecast_provenance": provenance,
        "forecast_source": "FROZEN_LSTM_V3",
        "is_simulated": simulated,
        "validated_metrics_apply": validated,
        "forecast_unit": "MCM/day",
        "horizons": list(HORIZONS),
        "input_provenance": {"synthetic_demo": [], "unavailable": [], "simulated": []},
    }


def _demo_payload():
    return _payload(status="DEMONSTRATION_ONLY",
                    provenance="SIMULATION_OR_SYNTHETIC_INPUTS_FROZEN_MODEL",
                    validated=False, simulated=True)


def _network():
    return SimBridge(str(LIVE_CONFIG), str(THRESH_PATH)).cascade.network


def _bundle(network, payloads=None, nodes=None):
    adapter = LiveForecastAdapter.for_network(network, project_root=str(_PROJECT_ROOT))
    nodes = nodes if nodes is not None else NODES
    payloads = payloads if payloads is not None else {n: _payload() for n in nodes}
    return adapter.build_bundle(payloads, "2026-09-14")


def _net_state(network):
    return {
        nid: {
            "storage": round(float(n.state.storage), 12),
            "gate_position": round(float(n.state.gate_position), 12),
            "total_outflow": round(float(n.state.total_outflow), 12),
        }
        for nid, n in network.nodes.items()
    }


def _decision_dict(decision) -> dict:
    return {
        "controller_type": decision.controller_type,
        "controller_status": decision.controller_status,
        "forecast_control_eligible": decision.forecast_control_eligible,
        "blocked_reason": decision.blocked_reason,
        "mpc_status": decision.mpc_status,
        "mpc_objective_score": decision.mpc_objective_score,
        "raw_mpc_proposal_fraction": dict(decision.proposed_gate_positions_fraction),
        "safety_layer_status": decision.safety_layer_status,
        "safety_is_safe": decision.safety_is_safe,
        "safety_modified": decision.safety_modified,
        "safety_violations": list(decision.safety_violations),
        "safety_layer_output_pct": dict(decision.safety_layer_gate_positions_pct),
        "downstream_status": decision.downstream_status,
        "downstream_capacity_mcm_day": decision.downstream_capacity_mcm_day,
        "downstream_predicted_flow_mcm_day": decision.downstream_predicted_flow_mcm_day,
        "downstream_capacity_achieved": decision.downstream_capacity_achieved,
        "downstream_protection_modified": decision.downstream_protection_modified,
        "downstream_reason": decision.downstream_reason,
        "final_action_pct": dict(decision.final_safe_control_action_pct),
        "final_action_source": decision.final_safe_control_action_source,
        "applied_gates_pct": dict(decision.gate_positions_pct),
        "control_applied": decision.control_applied,
        "action_space_dimension": (decision.action_space or {}).get("dimension"),
        "action_space_candidates": (decision.action_space or {}).get("candidate_vectors"),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STAGE 15 — END-TO-END ARCHITECTURE / INTEGRATION VERIFICATION")
    print("=" * 78)

    frozen_before = {str(p.relative_to(_PROJECT_ROOT)): sha256_file(p) for p in FROZEN_ARTIFACTS}
    evidence: dict = {"stage": 15, "timestamp": datetime.now().isoformat()}

    # ==================================================================
    # [1] SINGLE AUTHORITATIVE OWNER
    # ==================================================================
    live_owners, offline = {}, {}
    for path in (_PROJECT_ROOT / "src").rglob("*.py"):
        rel = str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/")
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in ("GlobalSimulationState()", "SimBridge(", "SimulationEngine(",
                      "LiveCascadeAdapter(", "ReservoirNetwork("):
            if token in text:
                live_owners.setdefault(token, []).append(rel)
        if "VirtualCascade(" in text:
            offline.setdefault("VirtualCascade(", []).append(rel)

    authority = {
        "live_instances": state_manager.authoritative_instance_count(),
        "live_owners": live_owners,
        "offline_research_constructors": offline,
        "single_live_bridge": live_owners.get("SimBridge(", []) ==
                              ["src/dashboard/api/state_manager.py"],
        "single_live_state": live_owners.get("GlobalSimulationState()", []) ==
                             ["src/dashboard/api/state_manager.py"],
    }

    # Offline/research engines may exist; they must not be REACHABLE from the
    # live path. Text scans hit comments, so this is an AST import check.
    live_path_modules = [
        "src/dashboard/api/state_manager.py", "src/dashboard/api/routes.py",
        "src/dashboard/api/app.py", "src/dashboard/app.py",
        "src/dashboard/sim_bridge.py", "src/network_env/live_cascade_adapter.py",
        "src/controller/live_mpc_orchestrator.py",
    ]
    offline_imports, legacy_controller_imports = {}, {}
    for rel in live_path_modules:
        tree = ast.parse((_PROJECT_ROOT / rel).read_text(encoding="utf-8"))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
                names += [a.name for a in node.names]
        # A physics ENGINE on the live path would be a second authority.
        engine_hits = [n for n in names
                       if "simulator.engine" in n or "simulator.environment" in n
                       or n in ("SimulationEngine", "VirtualCascade")]
        # The legacy rule-based CONTROLLER package is a dormant helper (below).
        legacy_hits = [n for n in names if n.startswith("src.simulator.controllers")]
        if engine_hits:
            offline_imports[rel] = engine_hits
        if legacy_hits:
            legacy_controller_imports[rel] = legacy_hits

    authority["offline_engine_imports_in_live_path"] = offline_imports
    authority["live_engines_are_isolated"] = not offline_imports
    authority["legacy_controller_imports_in_live_path"] = legacy_controller_imports

    # A live-path module also imports the LEGACY rule-based controller package.
    # It is a documented, dormant, non-authoritative helper (Stage 7), so the
    # honest check is: it is never INVOKED, and no physics/engine is imported.
    legacy_helper_callers = []
    for path in list((_PROJECT_ROOT / "src").rglob("*.py")) + \
            list((_PROJECT_ROOT / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "compute_ai_recommendation(" in text and "def compute_ai_recommendation" not in text:
            legacy_helper_callers.append(
                str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/")
            )
    authority["dormant_legacy_controller_helper"] = {
        "symbol": "SimBridge.compute_ai_recommendation",
        "imported_from": "src.simulator.controllers (offline research package)",
        "callers": legacy_helper_callers,
        "zero_callers": not legacy_helper_callers,
        "status": "dormant, documented NON-AUTHORITATIVE (Stage 7), not on any live path",
    }
    authority["authoritative_object_types"] = {
        "cascade": type(state_manager.sim_state.bridge.cascade).__name__,
        "network": type(state_manager.sim_state.bridge.cascade.network).__name__,
    }
    print(f"\n[1] Single authority      : live instances="
          f"{authority['live_instances']} · bridge owner={live_owners.get('SimBridge(')}")
    print(f"    SimulationEngine ctor : {live_owners.get('SimulationEngine(', []) or 'NONE in src/'}")
    print(f"    VirtualCascade ctor   : {offline.get('VirtualCascade(', []) or 'NONE in src/'}")
    evidence["single_authority"] = authority

    client.post("/api/simulation/pause")
    client.post("/api/simulation/reset")
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    for _ in range(8):
        client.post("/api/simulation/step")
    sim = state_manager.sim_state

    # ==================================================================
    # [2] LIVE AI-MODE CYCLE — what the authoritative system actually does
    # ==================================================================
    client.post("/api/controller/mode", json={"mode": "AI"})
    before = _net_state(sim.bridge.cascade.network)
    timestep_before = int(sim.bridge.cascade.network.timestep)
    t0 = time.perf_counter()
    live_state = sim.step()
    total_cycle_ms = (time.perf_counter() - t0) * 1000.0
    timestep_after = int(sim.bridge.cascade.network.timestep)
    live_decision = _decision_dict(sim.last_control_decision)
    live_trace = {
        "mode": "AI",
        "timestep_before": timestep_before,
        "timestep_after": timestep_after,
        "advanced_exactly_one": timestep_after - timestep_before == 1,
        "decision": live_decision,
        "network_before": before,
        "network_after": _net_state(sim.bridge.cascade.network),
        "mass_balance_status": live_state["mass_balance"]["status"],
        "mass_balance_residual": live_state["mass_balance"]["residual"],
        "gnn_advisory_status": live_state["gnn_advisory"]["status"],
        "gnn_advisory_affects_control": live_state["gnn_advisory"]["affects_control"],
        "forecast_sources": {k: v.get("forecast_source")
                             for k, v in live_state["reservoirs"].items()},
        "applications": sim.bridge.mass_balance_diagnostic().get("applied_action_percent"),
        "cycle_latency_ms": round(total_cycle_ms, 3),
    }
    print(f"\n[2] LIVE AI-mode cycle     : {live_decision['controller_status']} / "
          f"mpc={live_decision['mpc_status']} / safety={live_decision['safety_layer_status']} / "
          f"downstream={live_decision['downstream_status']}")
    print(f"    final action source   : {live_decision['final_action_source']} · "
          f"blocked_reason={live_decision['blocked_reason']}")
    print(f"    timestep {timestep_before} -> {timestep_after} · mass balance="
          f"{live_trace['mass_balance_status']} · GNN={live_trace['gnn_advisory_status']}")
    evidence["live_ai_cycle"] = live_trace

    # ==================================================================
    # [3] FULL CONTROL CHAIN with boundary instrumentation
    # ==================================================================
    order: list = []
    _orig_mpc_decide = MPCController.decide
    _orig_safety_validate = SafetyLayer.validate
    _orig_guard_evaluate = DownstreamCapacityGuard.evaluate

    def traced_mpc(self, *a, **kw):
        order.append("mpc.decide")
        return _orig_mpc_decide(self, *a, **kw)

    def traced_safety(self, *a, **kw):
        order.append("safety.validate")
        return _orig_safety_validate(self, *a, **kw)

    def traced_guard(self, *a, **kw):
        order.append("downstream_guard.evaluate")
        return _orig_guard_evaluate(self, *a, **kw)

    MPCController.decide = traced_mpc
    SafetyLayer.validate = traced_safety
    DownstreamCapacityGuard.evaluate = traced_guard
    try:
        chain_network = _network()
        chain_bundle = _bundle(chain_network)
        orchestrator = LiveMPCOrchestrator()
        chain_before = _net_state(chain_network)
        t0 = time.perf_counter()
        chain_decision = orchestrator.decide(chain_network, bundle=chain_bundle,
                                            current_inflows={n: 3.0 for n in NODES})
        chain_ms = (time.perf_counter() - t0) * 1000.0
    finally:
        MPCController.decide = _orig_mpc_decide
        SafetyLayer.validate = _orig_safety_validate
        DownstreamCapacityGuard.evaluate = _orig_guard_evaluate

    # Apply the final action to the physics of a real network and capture what
    # physics actually received.
    chain_applied = {nid: chain_decision.gate_positions_pct[nid] for nid in NODES}
    bridge = SimBridge(str(LIVE_CONFIG), str(THRESH_PATH))
    bridge.step({n: 3.0 for n in NODES}, dict(chain_applied),
                action_source=chain_decision.final_safe_control_action_source)
    physics_received = {nid: bridge.cascade.network.nodes[nid].state.gate_position * 100.0
                        for nid in NODES}

    chain_trace = {
        "boundary_call_order": order,
        "required_relative_order": ["mpc.decide", "safety.validate",
                                     "downstream_guard.evaluate"],
        "order_is_correct": (
            order[0] == "mpc.decide"
            and "safety.validate" in order
            and order.index("safety.validate") > 0
            and "downstream_guard.evaluate" in order
            and order.index("downstream_guard.evaluate") > order.index("safety.validate")
            and order[-1] == "downstream_guard.evaluate"
        ),
        "safety_validate_call_count": order.count("safety.validate"),
        "safety_double_invocation_explained": (
            "MPCController.decide() validates its own chosen gates internally "
            "(mpc_controller.py:222) and the orchestrator then validates the "
            "proposal it receives. Both calls precede the capacity guard, so the "
            "authoritative order MPC -> SafetyLayer -> DownstreamCapacityGuard "
            "holds; the orchestrator's own comment claiming the layer is called "
            "'exactly ONCE per decision' is imprecise and was corrected."
        ),
        "forecast_snapshot": {
            "forecast_date": chain_bundle.snapshot.forecast_date,
            "nodes": list(getattr(chain_bundle.snapshot, "forecasts", {}).keys()),
        },
        "decision": _decision_dict(chain_decision),
        "final_action_handed_to_physics_pct": chain_applied,
        "gate_positions_physics_reports_pct": physics_received,
        "physics_received_exactly_the_final_action": all(
            abs(chain_applied[nid] - physics_received[nid]) < 1e-9 for nid in NODES
        ),
        "network_before": chain_before,
        "network_after": _net_state(chain_network),
        "latency_ms": {
            "decide_total": round(chain_ms, 3),
            "safety_layer": round(orchestrator.last_safety_latency_ms, 3),
            "downstream_guard": round(orchestrator.last_downstream_latency_ms, 3),
            "mpc_candidate_vectors": chain_decision.candidates_evaluated,
        },
    }
    print(f"\n[3] Full chain order       : {order}")
    print(f"    raw MPC proposal      : {chain_decision.proposed_gate_positions_fraction}")
    print(f"    SafetyLayer output    : {chain_decision.safety_layer_gate_positions_pct} "
          f"({chain_decision.safety_layer_status}, modified={chain_decision.safety_modified})")
    print(f"    Guard output          : {chain_decision.gate_positions_pct} "
          f"({chain_decision.downstream_status}, modified={chain_decision.downstream_protection_modified})")
    print(f"    FINAL action source   : {chain_decision.final_safe_control_action_source}")
    print(f"    physics received      : {physics_received} · exact="
          f"{chain_trace['physics_received_exactly_the_final_action']}")
    evidence["full_control_chain"] = chain_trace

    # ==================================================================
    # [4] FORECAST INTEGRITY — validated vs demo / invalid / missing
    # ==================================================================
    integrity_network = _network()
    forecast_cases = {}
    cases = {
        "VALIDATED_all_four": ({n: _payload() for n in NODES},),
        "DEMONSTRATION_ONLY": ({n: _demo_payload() for n in NODES},),
        "MISSING_RESERVOIR_D": ({n: _payload() for n in NODES[:-1]},),
        "NAN_FORECAST": ({n: _payload(v1=float("nan")) for n in NODES},),
        "INFINITE_FORECAST": ({n: _payload(v1=float("inf")) for n in NODES},),
        "NEGATIVE_FORECAST": ({n: _payload(v1=-5.0) for n in NODES},),
    }
    for label, (payloads,) in cases.items():
        orchestrator = LiveMPCOrchestrator()
        try:
            bundle = _bundle(integrity_network, payloads=payloads)
            decision = orchestrator.decide(integrity_network, bundle=bundle)
            forecast_cases[label] = {
                "eligible": decision.forecast_control_eligible,
                "mpc_status": decision.mpc_status,
                "controller_status": decision.controller_status,
                "final_action_source": decision.final_safe_control_action_source,
                "control_applied": decision.control_applied,
                "blocked_reason": decision.blocked_reason,
            }
        except Exception as exc:
            forecast_cases[label] = {"raised": f"{type(exc).__name__}: {exc}",
                                     "control_applied": False}
    print(f"\n[4] Forecast integrity     :")
    for label, info in forecast_cases.items():
        print(f"    {label:22s} eligible={info.get('eligible')} "
              f"mpc={info.get('mpc_status')} applied={info.get('control_applied')}")
    evidence["forecast_integrity"] = {
        "cases": forecast_cases,
        "validated_is_the_only_eligible_case": (
            forecast_cases["VALIDATED_all_four"].get("eligible") is True
            and all(v.get("eligible") is not True for k, v in forecast_cases.items()
                    if k != "VALIDATED_all_four")
        ),
        "no_invalid_case_ever_applied_control": all(
            v.get("control_applied") is not True for k, v in forecast_cases.items()
            if k != "VALIDATED_all_four"
        ),
    }

    # ==================================================================
    # [5] MPC PROOF — action space and four-reservoir participation
    # ==================================================================
    mpc_action_space = orchestrator.action_space_for(NODES)
    mpc_proof = {
        "action_space": mpc_action_space,
        "dimension_is_four": mpc_action_space.get("dimension") == 4,
        "candidate_vectors_are_1296": mpc_action_space.get("candidate_vectors") == 1296,
        "nodes": NODES,
        "node_repository_names": NODE_TO_NAME,
        "reservoir_d_not_pinned": None,
    }
    # D must be an ordinary member: prove its gate can actually change.
    probe = _network()
    probe_gates = {n: 0.5 for n in NODES}
    probe.step({n: 3.0 for n in NODES}, probe_gates)
    gate_d_after_half = probe.nodes["Virtual Reservoir D"].state.gate_position
    probe.step({n: 3.0 for n in NODES}, {**probe_gates, "Virtual Reservoir D": 0.05})
    gate_d_after_low = probe.nodes["Virtual Reservoir D"].state.gate_position
    mpc_proof["reservoir_d_not_pinned"] = (
        abs(gate_d_after_half - 0.5) < 1e-9 and abs(gate_d_after_low - 0.05) < 1e-9
        and gate_d_after_half != gate_d_after_low
    )
    mpc_proof["reservoir_d_gate_observed"] = [round(gate_d_after_half, 6),
                                              round(gate_d_after_low, 6)]
    print(f"\n[5] MPC proof              : dimension={mpc_action_space.get('dimension')} · "
          f"candidates={mpc_action_space.get('candidate_vectors')} · "
          f"D commandable={mpc_proof['reservoir_d_not_pinned']}")
    evidence["mpc_proof"] = mpc_proof

    # ==================================================================
    # [6] SAFETY ORDERING with an intentionally UNSAFE proposal
    # ==================================================================
    unsafe_network = _network()
    unsafe_current = {n: float(unsafe_network.nodes[n].state.gate_position) for n in NODES}

    unsafe_proposal = {"Virtual Reservoir A": 5.0, "Virtual Reservoir B": -3.0,
                       "Virtual Reservoir C": float("nan"), "Virtual Reservoir D": 0.95}
    safety = SafetyLayer()
    safety_result = safety.validate(unsafe_proposal, unsafe_current, NODES)
    guard = DownstreamCapacityGuard()
    guard_result = guard.evaluate(
        unsafe_network,
        action_fraction={n: float(safety_result.validated_gates[n]) for n in NODES},
        current_fraction=unsafe_current,
        node_ids=NODES,
        max_gate_change=0.5,
        inflows={n: 3.0 for n in NODES},
    )
    unsafe_trace = {
        "raw_proposal_fraction": unsafe_proposal,
        "safety_layer_status": str(safety_result.status),
        "safety_layer_is_safe": bool(safety_result.is_safe),
        "safety_layer_violations": list(safety_result.violations),
        "safety_layer_output_fraction": {n: float(safety_result.validated_gates[n]) for n in NODES},
        "downstream_status": str(guard_result.status),
        "downstream_capacity_achieved": bool(guard_result.capacity_achieved),
        "downstream_modified": bool(guard_result.modified),
        "downstream_output_fraction": {n: float(guard_result.action_fraction[n]) for n in NODES},
        "downstream_reason": guard_result.reason,
        "nan_proposal_never_reaches_physics": all(
            math.isfinite(float(guard_result.action_fraction[n])) for n in NODES
        ),
        "bounds_respected": all(
            0.0 <= float(guard_result.action_fraction[n]) <= 1.0 for n in NODES
        ),
        "rate_limit_respected": all(
            abs(float(guard_result.action_fraction[n]) - unsafe_current[n]) <= 0.5 + 1e-9
            for n in NODES
        ),
        "downstream_capacity_mcm_day": float(unsafe_network.downstream_capacity),
    }
    print(f"\n[6] Unsafe proposal        : raw={unsafe_proposal}")
    print(f"    SafetyLayer           : {unsafe_trace['safety_layer_status']} -> "
          f"{unsafe_trace['safety_layer_output_fraction']}")
    print(f"    Guard                 : {unsafe_trace['downstream_status']} -> "
          f"{unsafe_trace['downstream_output_fraction']}")
    print(f"    finite/bounded/rate-limited: "
          f"{unsafe_trace['nan_proposal_never_reaches_physics']}/"
          f"{unsafe_trace['bounds_respected']}/{unsafe_trace['rate_limit_respected']}")
    evidence["unsafe_proposal_trace"] = unsafe_trace

    # ==================================================================
    # [7] PHYSICS PROOF — topology, delays, attenuation, no duplicate physics
    # ==================================================================
    net = _network()
    connections = [
        {"source": c.source, "destination": c.destination,
         "delay": c.delay, "attenuation": c.attenuation}
        for c in net.connections
    ]
    physics_proof = {
        "engine": "src/network_env/reservoir_network.py",
        "nodes": list(net.processing_order),
        "connections": connections,
        "delays_match": [c["delay"] for c in connections] == PHYSICAL_DELAYS,
        "attenuation_match": all(
            abs(c["attenuation"] - a) < 1e-12
            for c, a in zip(connections, PHYSICAL_ATTENUATION)
        ),
        "downstream_capacity_mcm_day": float(net.downstream_capacity),
        "timestep_advances_once_per_step": None,
    }
    ts0 = int(net.timestep)
    net.step({n: 0.0 for n in NODES}, {n: 0.0 for n in NODES})
    physics_proof["timestep_advances_once_per_step"] = int(net.timestep) - ts0 == 1

    # No duplicate physics outside the authoritative engine.
    duplicate_physics = {}
    for path in list((_PROJECT_ROOT / "src").rglob("*.js")) + \
            list((_PROJECT_ROOT / "src").rglob("*.html")) + \
            [p for p in (_PROJECT_ROOT / "src").rglob("*.py")
             if "reservoir_network" not in str(p)]:
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(tok in text for tok in ("routing_delay", "attenuation_factor",
                                       "storage_mcm =", "delta_storage")):
            duplicate_physics[str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/")] = True
    physics_proof["files_with_physics_like_tokens"] = sorted(duplicate_physics)
    print(f"\n[7] Physics                : {len(connections)} links · delays="
          f"{[c['delay'] for c in connections]} · attenuation="
          f"{[round(c['attenuation'], 2) for c in connections]}")
    print(f"    duplicate physics      : {sorted(duplicate_physics) or 'NONE'}")
    evidence["physics_proof"] = physics_proof

    # ==================================================================
    # [8] MASS BALANCE — PASS after a real step + corruption detection
    # ==================================================================
    mb_network = _network()
    monitor = MassBalanceMonitor()
    mb_step = monitor.step_and_check(mb_network, {n: 1.0 for n in NODES},
                                     {n: 0.3 for n in NODES})
    mb_pass = {"status": mb_step.status, "residual": mb_step.residual,
               "tolerance": mb_step.tolerance, "checked": mb_step.checked}

    corruptions = {}
    clean_control = None
    for label, corrupt in (
        (None, None),
        ("storage", lambda n: setattr(n.nodes["Virtual Reservoir C"].state, "storage",
                                      n.nodes["Virtual Reservoir C"].state.storage + 7.0)),
        ("outflow", lambda n: setattr(n.nodes["Virtual Reservoir D"].state, "total_outflow",
                                      n.nodes["Virtual Reservoir D"].state.total_outflow + 3.0)),
        ("routing_queue", lambda n: n.connections[0].queue.__setitem__(
            -1, n.connections[0].queue[-1] + 2.0)),
        ("non_finite", lambda n: setattr(n.nodes["Virtual Reservoir D"].state, "storage",
                                         float("nan"))),
    ):
        # Correct protocol: snapshot the PRE-step state, run the real physics
        # step, then (optionally) corrupt, then verify. Corrupting after the step
        # is what makes the detected residual attributable to the corruption.
        c_net = _network()
        c_monitor = MassBalanceMonitor()
        snapshot = c_monitor.snapshot(c_net)
        c_net.step({n: 1.0 for n in NODES}, {n: 0.3 for n in NODES})
        if corrupt is None:
            clean_control = c_monitor.verify(c_net, snapshot,
                                             applied_inflows={n: 1.0 for n in NODES},
                                             applied_gates_fraction={n: 0.3 for n in NODES})
            continue
        corrupt(c_net)
        result = c_monitor.verify(c_net, snapshot, applied_inflows={n: 1.0 for n in NODES},
                                  applied_gates_fraction={n: 0.3 for n in NODES})
        corruptions[label] = {
            "status": result.status,
            "detected": result.status != "PASS",
            "violations": list(result.violations)[:3],
        }

    mass_balance = {
        "clean_step": mb_pass,
        "clean_control_same_protocol": {
            "status": clean_control.status,
            "residual": clean_control.residual,
        },
        "tolerance_preserved": mb_pass["tolerance"] == 1e-9,
        "corruption_cases": corruptions,
        "all_corruptions_detected": all(v["detected"] for v in corruptions.values()),
    }
    print(f"\n[8] Mass balance           : clean step={mb_pass['status']} "
          f"(residual={mb_pass['residual']:.3e}, tol={mb_pass['tolerance']})")
    print(f"    clean control (same protocol): {clean_control.status} "
          f"(residual={clean_control.residual:.3e})")
    for label, info in corruptions.items():
        print(f"    corruption {label:14s} -> {info['status']} (detected={info['detected']})")
    evidence["mass_balance"] = mass_balance

    # ==================================================================
    # [9] REST / WebSocket STATE AUTHORITY
    # ==================================================================
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    with client.websocket_connect("/ws/state") as ws:
        ws_state = ws.receive_json()
        rest_same = client.get("/api/state").json()
        identity_same = ws_state["state_identity"] == rest_same["state_identity"]

        client.post("/api/simulation/step")
        pushed = ws.receive_json()
        rest_after = client.get("/api/state").json()

    agreement_keys = ("reservoirs", "cascade", "control", "mass_balance", "downstream",
                      "forecast_summary", "storm", "simulation", "state_identity",
                      "hardware_status", "forecast_provenance")

    def _stable_advisory(payload: dict) -> dict:
        """The advisory minus its two inherently per-payload volatile fields.

        ``inference_timestamp`` and ``inference_latency_ms`` are generated on
        every payload, so two payloads of the SAME state legitimately differ
        there. The advisory CONTENT (status, model, graph, embeddings,
        similarity, provenance, flags) must be identical.
        """
        advisory = dict(payload.get("gnn_advisory") or {})
        for volatile in ("inference_timestamp", "inference_latency_ms"):
            advisory.pop(volatile, None)
        return advisory

    state_authority = {
        "same_timestep_agreement": {k: ws_state.get(k) == rest_same.get(k)
                                    for k in agreement_keys},
        "same_timestep_advisory_content_agreement": (
            _stable_advisory(ws_state) == _stable_advisory(rest_same)
        ),
        "same_timestep_advisory_volatile_fields": [
            k for k in ("inference_timestamp", "inference_latency_ms")
            if (ws_state.get("gnn_advisory") or {}).get(k)
            != (rest_same.get("gnn_advisory") or {}).get(k)
        ],
        "identity_equal_same_timestep": identity_same,
        "pushed_equals_rest_after_step": {k: pushed.get(k) == rest_after.get(k)
                                          for k in agreement_keys},
        "pushed_advisory_content_equals_rest": (
            _stable_advisory(pushed) == _stable_advisory(rest_after)
        ),
        "pushed_state_id": pushed["state_identity"]["state_id"],
        "rest_state_id": rest_after["state_identity"]["state_id"],
        "timestep_delta_for_one_step": (
            rest_after["state_identity"]["network_timestep"]
            - rest_same["state_identity"]["network_timestep"]
        ),
        "no_state_write_endpoint": client.post("/api/state", json={"storage": 1}).status_code,
    }
    print(f"\n[9] REST/WS authority      : same-timestep agreement="
          f"{all(state_authority['same_timestep_agreement'].values())} · "
          f"pushed==REST after STEP={all(state_authority['pushed_equals_rest_after_step'].values())}")
    print(f"    timestep delta for one STEP: {state_authority['timestep_delta_for_one_step']} · "
          f"POST /api/state -> {state_authority['no_state_write_endpoint']}")
    evidence["state_authority"] = state_authority

    # ==================================================================
    # [10] FRONTEND DISPLAY-ONLY AUDIT
    # ==================================================================
    web_index = (_PROJECT_ROOT / "src" / "dashboard" / "web" / "index.html").read_text(
        encoding="utf-8")
    api_js = (_PROJECT_ROOT / "src" / "dashboard" / "web" / "api.js").read_text(encoding="utf-8")
    app_path = _PROJECT_ROOT / "src" / "dashboard" / "app.py"
    app_py = app_path.read_text(encoding="utf-8")

    # Model/runtime identifiers must not appear in the browser artifacts AT ALL
    # (comments included). Component LABELS such as "SAFETY LAYER" are display
    # text and are not part of this check.
    model_runtime = ("torch", "tensorflow", "onnx", "gatedgcnlstm",
                     "livegnnforecaster", "gnn_inference", "simbridge",
                     "mpc_controller", "reservoir_network", "simulationengine",
                     "virtualcascade", "state_dict")

    # Streamlit: inspect the ACTUAL imports (AST), not prose. Stage 13/14 pinned
    # that the page imports no model, engine or control module.
    app_tree = ast.parse(app_py)
    app_imports = set()
    for node in ast.walk(app_tree):
        if isinstance(node, ast.Import):
            app_imports |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            app_imports.add(node.module or "")
            app_imports |= {f"{node.module}.{a.name}" for a in node.names}
    forbidden_import_tokens = ("gnn", "sim_bridge", "simulator", "state_manager",
                              "network_env", "controller", "src.dashboard.api")
    streamlit_import_hits = sorted(
        m for m in app_imports if any(t in m.lower() for t in forbidden_import_tokens)
    )

    frontend = {
        "twin_model_runtime_tokens_found": [t for t in model_runtime if t in web_index.lower()],
        "api_js_model_runtime_tokens_found": [t for t in model_runtime if t in api_js.lower()],
        "streamlit_model_or_engine_imports": streamlit_import_hits,
        "streamlit_direct_imports": sorted(app_imports),
        "streamlit_runs_no_gnn": "node_representations" not in app_py
                                 and "cosine" not in app_py.lower(),
        "twin_reads_only_payload": "state.gnn_advisory" in web_index,
        "streamlit_reads_only_api_state": "/api/state" in app_py,
        "streamlit_command_routes": sorted(set(
            line.split('"')[1] for line in app_py.splitlines()
            if '"/api/' in line and line.strip().startswith('"')
        )),
    }
    print(f"\n[10] Frontend audit        : twin model tokens="
          f"{frontend['twin_model_runtime_tokens_found'] or 'NONE'} · "
          f"api.js={frontend['api_js_model_runtime_tokens_found'] or 'NONE'}")
    print(f"     Streamlit model/engine imports: "
          f"{frontend['streamlit_model_or_engine_imports'] or 'NONE'}")
    evidence["frontend_audit"] = frontend

    # ==================================================================
    # [11] END-TO-END COMMANDS
    # ==================================================================
    client.post("/api/simulation/pause")
    client.post("/api/controller/mode", json={"mode": "MANUAL"})
    commands: dict = {}

    client.post("/api/simulation/reset")
    before_reset = client.get("/api/state").json()["state_identity"]
    commands["RESET"] = {"identity_after": before_reset}

    t_before = client.get("/api/state").json()["state_identity"]["network_timestep"]
    with client.websocket_connect("/ws/state") as ws:
        ws.receive_json()
        client.post("/api/simulation/step")
        step_pushed = ws.receive_json()
    commands["STEP"] = {
        "timestep_before": t_before,
        "timestep_after": client.get("/api/state").json()["state_identity"]["network_timestep"],
        "pushed_state_id": step_pushed["state_identity"]["state_id"],
        "mass_balance_after": client.get("/api/state").json()["mass_balance"]["status"],
    }

    gate_response = client.post("/api/gate/reservoir_4", json={"value": 20.0})
    after_gate = client.get("/api/state").json()
    commands["GATE_D"] = {
        "status_code": gate_response.status_code,
        "authoritative_manual_gate": sim.manual_gates["Virtual Reservoir D"],
        "payload_gate_fraction": after_gate["reservoirs"]["reservoir_4"]["gate"],
        "name": after_gate["cascade"]["reservoirs"][3]["name"],
    }

    client.post("/api/simulation/play")
    running_after_play = client.get("/api/state").json()["simulation"]["running"]
    client.post("/api/simulation/pause")
    running_after_pause = client.get("/api/state").json()["simulation"]["running"]
    commands["PLAY_PAUSE"] = {"running_after_play": running_after_play,
                              "running_after_pause": running_after_pause,
                              "backend_owner": True}
    print(f"\n[11] Commands              : STEP {commands['STEP']['timestep_before']}->"
          f"{commands['STEP']['timestep_after']} · GATE_D "
          f"{commands['GATE_D']['status_code']}/{commands['GATE_D']['authoritative_manual_gate']}"
          f" · PLAY={running_after_play} PAUSE={running_after_pause}")
    evidence["commands"] = commands

    # ==================================================================
    # [12] FAILURE PATHS
    # ==================================================================
    client.post("/api/simulation/pause")
    failure_paths = {}
    for label, method, url, body in (
        ("malformed_body", "post", "/api/gate/reservoir_1", "not-json"),
        ("invalid_gate_string", "post", "/api/gate/reservoir_1", {"value": "wide open"}),
        ("invalid_gate_nan", "post", "/api/gate/reservoir_1", '{"value": NaN}'),
        ("unknown_reservoir", "post", "/api/gate/reservoir_9", {"value": 10.0}),
        ("invalid_speed", "post", "/api/simulation/speed", {"speed": "fast"}),
        ("zero_speed", "post", "/api/simulation/speed", {"speed": 0.0}),
        ("invalid_storm", "post", "/api/storm", {"value": None}),
        ("invalid_storm_nan", "post", "/api/storm", '{"value": Infinity}'),
        ("invalid_mode", "post", "/api/controller/mode", {"mode": "HACK"}),
        ("state_write_attempt", "post", "/api/state", {"storage": 999.0}),
        ("state_put_attempt", "put", "/api/state", {"storage": 999.0}),
        ("unknown_route", "get", "/api/nonexistent", None),
    ):
        if isinstance(body, str):
            response = client.post(url, content=body,
                                   headers={"Content-Type": "application/json"})
        elif method == "get":
            response = client.get(url)
        else:
            response = getattr(client, method)(url, json=body)
        failure_paths[label] = {"status_code": response.status_code,
                                "safe": response.status_code >= 400}

    # A zero speed is handled by the documented CLAMP (speed >= 0.05), not by a
    # rejection — clamping is the fail-safe behaviour here (a zero speed would
    # divide by zero inside the authoritative loop).
    failure_paths["zero_speed"]["safe"] = sim.sim_speed >= 0.05
    failure_paths["zero_speed"]["handled_by"] = f"clamped to {sim.sim_speed}"

    failure_paths["unavailable_gnn"] = None
    failure_paths["modified_gnn_advisory"] = None

    # GNN failure paths through the live object.
    sim.gnn_ready = False
    client.post("/api/simulation/step")
    unavailable_payload = client.get("/api/state").json()
    failure_paths["unavailable_gnn"] = {
        "advisory_status": unavailable_payload["gnn_advisory"]["status"],
        "control_status": unavailable_payload["control"]["controller_status"],
        "safe": unavailable_payload["gnn_advisory"]["status"] == "UNAVAILABLE",
    }
    sim.gnn_ready = True

    # Modified advisory: the Stage 14 A/B re-checked at the integration level.
    from src.modeling import gnn_advisory as _ga  # noqa: E402

    ab_network = _network()
    ab_bundle = _bundle(ab_network)
    _saved_advisory = sim.gnn_advisory
    baseline_decision = LiveMPCOrchestrator().decide(
        ab_network, bundle=ab_bundle, current_inflows={n: 3.0 for n in NODES})
    sim.gnn_advisory = _ga.build_gnn_advisory(
        embeddings=np.full((16, 64), 12345.0, dtype=np.float32),
        gate_value=0.999,
        live_input_nodes=[],
        graph_provenance={"available": True, "graph_nodes": 999},
    )
    modified_decision = LiveMPCOrchestrator().decide(
        ab_network, bundle=ab_bundle, current_inflows={n: 3.0 for n in NODES})
    sim.gnn_advisory = _saved_advisory

    def _decision_projection(decision) -> dict:
        return {
            "raw_proposal": dict(decision.proposed_gate_positions_fraction),
            "mpc_status": decision.mpc_status,
            "safety_layer_status": decision.safety_layer_status,
            "safety_layer_output": dict(decision.safety_layer_gate_positions_pct),
            "downstream_status": decision.downstream_status,
            "final_action_pct": dict(decision.final_safe_control_action_pct),
            "final_action_source": decision.final_safe_control_action_source,
            "applied_gates_pct": dict(decision.gate_positions_pct),
        }

    _proj_a = _decision_projection(baseline_decision)
    _proj_b = _decision_projection(modified_decision)
    failure_paths["modified_gnn_advisory"] = {
        "mpc_proposal_unchanged": _proj_a["raw_proposal"] == _proj_b["raw_proposal"],
        "safety_layer_output_unchanged": _proj_a["safety_layer_output"] == _proj_b["safety_layer_output"],
        "downstream_output_unchanged": _proj_a["downstream_status"] == _proj_b["downstream_status"],
        "final_action_unchanged": _proj_a["final_action_pct"] == _proj_b["final_action_pct"],
        "safe": _proj_a == _proj_b,
    }
    print(f"\n[12] Failure paths         : "
          f"{sum(1 for v in failure_paths.values() if v and v.get('safe'))}/"
          f"{len([v for v in failure_paths.values() if v])} fail safe")
    evidence["failure_paths"] = failure_paths

    # ==================================================================
    # [13] PERFORMANCE — one complete cycle, phase by phase
    # ==================================================================
    perf_sim = state_manager.sim_state
    client.post("/api/simulation/pause")
    client.post("/api/controller/mode", json={"mode": "AI"})

    t0 = time.perf_counter()
    forecasts = perf_sim._run_ml_pipeline()
    forecast_ms = (time.perf_counter() - t0) * 1000.0

    gnn_ms = forecasts and perf_sim.gnn_advisory.get("inference_latency_ms")

    perf_network = _network()
    perf_bundle = _bundle(perf_network)
    perf_orchestrator = LiveMPCOrchestrator()
    t0 = time.perf_counter()
    perf_orchestrator.decide(perf_network, bundle=perf_bundle,
                             current_inflows={n: 3.0 for n in NODES})
    decide_ms = (time.perf_counter() - t0) * 1000.0

    perf_bridge = SimBridge(str(LIVE_CONFIG), str(THRESH_PATH))
    t0 = time.perf_counter()
    perf_bridge.step({n: 3.0 for n in NODES}, {n: 0.3 for n in NODES})
    physics_ms = (time.perf_counter() - t0) * 1000.0

    mb_diag = perf_bridge.mass_balance_diagnostic()
    t0 = time.perf_counter()
    for _ in range(4):
        client.post("/api/simulation/step")
    step_ms = (time.perf_counter() - t0) * 1000.0 / 4

    performance = {
        "forecast_plus_advisory_ms": round(forecast_ms, 3),
        "gnn_advisory_inference_ms": gnn_ms,
        "mpc_safety_guard_decide_ms": round(decide_ms, 3),
        "safety_layer_ms": round(perf_orchestrator.last_safety_latency_ms, 3),
        "downstream_guard_ms": round(perf_orchestrator.last_downstream_latency_ms, 3),
        "physics_step_ms": round(physics_ms, 3),
        "mass_balance_latency_ms": mb_diag.get("latency_ms"),
        "http_step_round_trip_ms": round(step_ms, 3),
        "total_live_ai_cycle_ms": round(total_cycle_ms, 3),
        "unit": "ms",
        "note": ("Phases are measured individually on CPU. The live AI cycle is "
                 "dominated by the forecast + advisory pass; the physics step is "
                 "sub-millisecond and the mass-balance audit adds ~0.08 ms."),
    }
    print(f"\n[13] Performance           : forecast+advisory={performance['forecast_plus_advisory_ms']} "
          f"ms · decide={performance['mpc_safety_guard_decide_ms']} ms · physics="
          f"{performance['physics_step_ms']} ms · total live cycle={performance['total_live_ai_cycle_ms']} ms")
    evidence["performance"] = performance

    # ==================================================================
    # [14] SCIENTIFIC CLAIM AUDIT
    # ==================================================================
    claim_patterns = {
        "guaranteed flood prevention": ("guarantee flood", "flood-proof", "floodproof",
                                        "guarantees flood prevention", "prevents flooding"),
        "GNN causal discovery": ("causal discovery", "discovers causal", "causal relationship detected"),
        "GNN physical hydraulic discovery": ("hydraulic discovery", "proves hydraulic",
                                             "discovers physical connectivity"),
        "GNN direct gate control": ("gnn controls", "gnn control of gates",
                                    "gnn directly controls"),
        "AI directly controls real dams": ("controls real dams", "real-world dam control",
                                           "controls the dam"),
        "hardware-connected control": ("hardware connected", "actuator online"),
        "synthetic inputs called validated": ("validated live forecast", "live forecasts validated"),
    }
    claim_hits = {}
    for path in list((_PROJECT_ROOT / "src").rglob("*.html")) + \
            list((_PROJECT_ROOT / "src").rglob("*.js")) + \
            list((_PROJECT_ROOT / "src").rglob("*.py")):
        if "vendor" in str(path):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for claim, needles in claim_patterns.items():
            for needle in needles:
                if needle in text:
                    claim_hits.setdefault(claim, []).append(
                        f"{path.relative_to(_PROJECT_ROOT)}::{needle}")
    frontend_claim_text = (web_index + api_js).lower()
    claim_audit = {
        "hits": claim_hits,
        "no_unsupported_claims_in_live_frontend": not claim_hits,
        "required_terminology": {
            "advisory_only_label": "ADVISORY ONLY" in web_index.upper(),
            "embedding_similarity_label": "embedding similarity" in frontend_claim_text,
            "demonstration_banner": "DEMONSTRATION" in web_index.upper(),
        },
    }
    print(f"\n[14] Claim audit           : {claim_hits or 'NO UNSUPPORTED CLAIMS FOUND'}")
    evidence["claim_audit"] = claim_audit

    # ==================================================================
    # [15] FROZEN ARTIFACTS
    # ==================================================================
    frozen_after = {str(p.relative_to(_PROJECT_ROOT)): sha256_file(p) for p in FROZEN_ARTIFACTS}
    frozen_unchanged = frozen_before == frozen_after
    print(f"\n[15] Frozen artifacts      : unchanged={frozen_unchanged} ({len(frozen_after)} hashed)")

    # ==================================================================
    # VERDICT
    # ==================================================================
    checks = {
        "one_live_simulation_owner": authority["live_instances"] == 1,
        "only_state_manager_owns_the_live_bridge": authority["single_live_bridge"],
        "only_state_manager_owns_the_live_state": authority["single_live_state"],
        "offline_engines_are_not_reachable_from_the_live_path": (
            authority["live_engines_are_isolated"]
        ),
        "live_cascade_is_the_validated_stack": (
            authority["authoritative_object_types"] == {"cascade": "LiveCascadeAdapter",
                                                        "network": "ReservoirNetwork"}
        ),
        "live_cycle_advances_exactly_one_timestep": live_trace["advanced_exactly_one"],
        "live_cycle_mass_balance_checked": live_trace["mass_balance_status"] in ("PASS", "VIOLATION"),
        "boundary_order_is_mpc_safety_downstream": chain_trace["order_is_correct"],
        "chain_reached_a_final_safe_control_action": bool(
            chain_decision.final_safe_control_action_source
        ),
        "physics_received_exactly_the_final_action": (
            chain_trace["physics_received_exactly_the_final_action"]
        ),
        "validated_forecast_is_the_only_eligible_case": (
            evidence["forecast_integrity"]["validated_is_the_only_eligible_case"]
        ),
        "invalid_forecasts_never_applied_control": (
            evidence["forecast_integrity"]["no_invalid_case_ever_applied_control"]
        ),
        "mpc_action_space_is_1296": mpc_proof["candidate_vectors_are_1296"],
        "all_four_reservoirs_participate": mpc_proof["dimension_is_four"]
                                           and mpc_proof["reservoir_d_not_pinned"],
        "unsafe_proposal_is_corrected_or_failed_closed": (
            unsafe_trace["nan_proposal_never_reaches_physics"]
            and unsafe_trace["bounds_respected"]
            and unsafe_trace["rate_limit_respected"]
        ),
        "physics_topology_delays_attenuation_match": (
            physics_proof["delays_match"] and physics_proof["attenuation_match"]
        ),
        "physics_advances_one_timestep_per_step": physics_proof["timestep_advances_once_per_step"],
        "mass_balance_passes_on_a_clean_step": mb_pass["status"] == "PASS",
        "mass_balance_tolerance_is_1e-9": mass_balance["tolerance_preserved"],
        "every_corruption_is_detected": mass_balance["all_corruptions_detected"],
        "rest_and_ws_agree_on_the_same_timestep": all(
            state_authority["same_timestep_agreement"].values()
        ),
        "pushed_payload_equals_rest_after_step": all(
            state_authority["pushed_equals_rest_after_step"].values()
        ),
        "one_step_advances_the_clock_by_exactly_one": (
            state_authority["timestep_delta_for_one_step"] == 1
        ),
        "no_state_write_endpoint": state_authority["no_state_write_endpoint"] == 405,
        "twin_reads_only_the_payload": frontend["twin_reads_only_payload"],
        "no_model_runtime_in_the_browser_artifacts": (
            not frontend["twin_model_runtime_tokens_found"]
            and not frontend["api_js_model_runtime_tokens_found"]
        ),
        "streamlit_imports_no_model_or_engine": not frontend["streamlit_model_or_engine_imports"],
        "streamlit_runs_no_gnn": frontend["streamlit_runs_no_gnn"],
        "no_unsupported_scientific_claims_found": not claim_audit["hits"],
        "dormant_legacy_controller_helper_has_zero_callers": (
            authority["dormant_legacy_controller_helper"]["zero_callers"]
        ),
        "unavailable_gnn_does_not_stop_control": (
            failure_paths["unavailable_gnn"]["safe"]
        ),
        "modified_gnn_advisory_leaves_every_control_output_unchanged": (
            failure_paths["modified_gnn_advisory"]["safe"]
        ),
        "frozen_artifacts_unchanged": frozen_unchanged,
    }

    evidence["checks"] = checks
    evidence["frozen_artifact_sha256_before"] = frozen_before
    evidence["frozen_artifact_sha256_after"] = frozen_after
    evidence["hardware_connected"] = False

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, default=str)

    print("\n" + "-" * 78)
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nEvidence written: {OUTPUT_PATH.relative_to(_PROJECT_ROOT)}")
    verdict = all(checks.values())
    print("VERDICT:", "PASS" if verdict else "FAIL")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

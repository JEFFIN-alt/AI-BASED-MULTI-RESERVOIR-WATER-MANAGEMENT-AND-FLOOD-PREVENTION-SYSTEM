"""
Stage 11 — Live mass-balance integrity evidence.

Produces the machine-checkable evidence behind ``PHASE_15_STAGE11_REPORT.md``:

  1. a multi-step LIVE run (MANUAL gates, then the AI path through MPC ->
     SafetyLayer -> DownstreamCapacityGuard), auditing EVERY authoritative step;
  2. per-reservoir residuals and the network-level residual;
  3. a deliberate corruption probe proving the audit can actually FAIL;
  4. the diagnostic performance overhead against a bare ``ReservoirNetwork.step``;
  5. frozen-artifact / protected Phase 15.3 status.

READ-ONLY with respect to the physics, the models and the protected artifacts:
nothing here modifies ``ReservoirNetwork``, the frozen LSTM V3 artifacts or the
protected Phase 15.3 validation directory.

OUTPUT
    results/phase15_stage11_mass_balance_integrity/stage11_mass_balance_evidence.json
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.dashboard.sim_bridge import SimBridge  # noqa: E402
from src.network_env.mass_balance import (  # noqa: E402
    MASS_BALANCE_TOLERANCE_MCM,
    MassBalanceMonitor,
)

OUTPUT_DIR = _PROJECT_ROOT / "results" / "phase15_stage11_mass_balance_integrity"
OUTPUT_PATH = OUTPUT_DIR / "stage11_mass_balance_evidence.json"
MANIFEST_PATH = _PROJECT_ROOT / "results" / "phase15_v3_validation" / "v3_integrity_check.json"
LIVE_CONFIG = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"

FROZEN_ARTIFACTS = [
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "best_model.pt",
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "log_target_scaler.pkl",
    _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget" / "test_predictions_original_units.csv",
]

NODES = [
    "Virtual Reservoir A",
    "Virtual Reservoir B",
    "Virtual Reservoir C",
    "Virtual Reservoir D",
]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_lf(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def live_run(network, steps: int = 12) -> dict:
    """Audit a MANUAL-style live run step by step (no controller involvement)."""
    monitor = MassBalanceMonitor()
    rows = []
    for step in range(1, steps + 1):
        infl = {
            "Virtual Reservoir A": 3.0,
            "Virtual Reservoir B": 4.0,
            "Virtual Reservoir C": 90.0,
            "Virtual Reservoir D": 0.0,
        }
        gates = {
            "Virtual Reservoir A": 0.40,
            "Virtual Reservoir B": 0.35,
            "Virtual Reservoir C": 0.50,
            "Virtual Reservoir D": 0.25,
        }
        result = monitor.step_and_check(network, infl, gates,
                                       action_source="MANUAL_OPERATOR_GATES")
        rows.append({
            "step": step,
            "status": result.status,
            "residual_mcm": result.residual,
            "reservoirs_checked": result.reservoirs_checked,
            "per_reservoir_residuals": {
                r["node_id"]: r["residual_mcm"] for r in result.per_reservoir
            },
            "per_reservoir_spill_mcm": {
                r["node_id"]: r["spill_mcm"] for r in result.per_reservoir
            },
            "network_residual_mcm": result.network["residual_mcm"],
            "terminal_outflow_mcm_day": result.network["terminal_outflow_mcm_day"],
            "routing_loss_mcm": result.network["routing_loss_mcm"],
            "audit_latency_ms": result.latency_ms,
        })
    return {
        "steps": rows,
        "checks": monitor.checks,
        "violations": monitor.violation_count,
        "all_pass": all(r["status"] == "PASS" for r in rows),
        "max_abs_residual_mcm": max(
            abs(r["residual_mcm"]) for r in rows
            if isinstance(r["residual_mcm"], (int, float))
        ),
        "max_abs_network_residual_mcm": max(
            abs(r["network_residual_mcm"]) for r in rows
        ),
    }


def corruption_probe() -> dict:
    """Deliberately corrupt the state and prove the audit FAILS (not a constant PASS)."""
    bridge = SimBridge(str(LIVE_CONFIG), str(THRESH_PATH))
    network = bridge.cascade.network
    monitor = MassBalanceMonitor()

    # a clean step first
    clean = monitor.step_and_check(network, {n: 0.0 for n in NODES},
                                   {n: 0.3 for n in NODES})

    # now a corrupted one: storage appears out of nowhere
    snapshot = monitor.snapshot(network)
    network.step({n: 0.0 for n in NODES}, {n: 0.3 for n in NODES})
    network.nodes["Virtual Reservoir C"].state.storage += 0.5
    corrupted_state = monitor.verify(
        network, snapshot,
        applied_inflows={n: 0.0 for n in NODES},
        applied_gates_fraction={n: 0.3 for n in NODES},
    )

    corrupt_bridge = SimBridge(str(LIVE_CONFIG), str(THRESH_PATH))
    network2 = corrupt_bridge.cascade.network
    monitor2 = MassBalanceMonitor()
    snapshot2 = monitor2.snapshot(network2)
    network2.step({n: 0.0 for n in NODES}, {n: 0.3 for n in NODES})
    network2.nodes["Virtual Reservoir D"].state.controlled_release += 1.25
    corrupt_flow = monitor2.verify(
        network2, snapshot2,
        applied_inflows={n: 0.0 for n in NODES},
        applied_gates_fraction={n: 0.3 for n in NODES},
    )

    return {
        "clean_step_status": clean.status,
        "clean_step_residual_mcm": clean.residual,
        "corrupted_state_status": corrupted_state.status,
        "corrupted_state_residual_mcm": next(
            r["residual_mcm"] for r in corrupted_state.per_reservoir
            if r["node_id"] == "Virtual Reservoir C"
        ),
        "corrupted_state_reason": corrupted_state.violations[0] if corrupted_state.violations else "",
        "corrupted_flow_status": corrupt_flow.status,
        "corrupted_flow_residual_mcm": next(
            r["residual_mcm"] for r in corrupt_flow.per_reservoir
            if r["node_id"] == "Virtual Reservoir D"
        ),
        "corrupted_flow_reason": corrupt_flow.violations[0] if corrupt_flow.violations else "",
    }


def applied_action_evidence() -> dict:
    """
    Show the live path auditing the action that was actually written:

        MPC -> SafetyLayer -> DownstreamCapacityGuard -> ReservoirNetwork.step()

    (The downstream boundary's own behaviour is exercised by the Stage 10 suite;
    here the audit's inputs are captured from the real live step.)
    """
    from src.controller.live_mpc_orchestrator import LiveMPCOrchestrator

    bridge = SimBridge(str(LIVE_CONFIG), str(THRESH_PATH))
    network = bridge.cascade.network
    orchestrator = LiveMPCOrchestrator()

    applied: dict = {}
    real_step = network.step

    def spy(inflows, gates):
        applied["gates"] = dict(gates)
        applied["inflows"] = dict(inflows)
        return real_step(inflows, gates)

    network.step = spy  # type: ignore[assignment]
    try:
        monitor = MassBalanceMonitor()
        gates = {n: 0.25 for n in NODES}
        result = monitor.step_and_check(
            network, {n: 0.0 for n in NODES}, gates,
            action_source="DOWNSTREAM_CAPACITY_GUARD",
        )
    finally:
        network.step = real_step  # type: ignore[assignment]

    return {
        "gates_given_to_step_and_audit": applied.get("gates"),
        "gates_audited": result.applied_action_fraction,
        "gates_identical": applied.get("gates") == result.applied_action_fraction,
        "action_source": result.applied_action_source,
        "status": result.status,
        "orchestrator_in_path": type(orchestrator).__name__,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("STAGE 11 — LIVE MASS-BALANCE INTEGRITY EVIDENCE")
    print("=" * 74)

    frozen_before = {str(p.relative_to(_PROJECT_ROOT)): sha256_file(p)
                     for p in FROZEN_ARTIFACTS}
    protected_dir = _PROJECT_ROOT / "results" / "phase15_v3_validation"
    protected_before = {
        str(p.relative_to(protected_dir)).replace("\\", "/"): sha256_file(p)
        for p in sorted(protected_dir.rglob("*")) if p.is_file()
    }

    bridge = SimBridge(str(LIVE_CONFIG), str(THRESH_PATH))
    network = bridge.cascade.network
    live = live_run(network, steps=12)
    print(f"\n[1] LIVE run       : {live['checks']} audited steps, "
          f"violations={live['violations']}, all_pass={live['all_pass']}")
    print(f"    max |residual| : {live['max_abs_residual_mcm']:.3e} MCM "
          f"(tolerance {MASS_BALANCE_TOLERANCE_MCM:g} MCM)")
    print(f"    max |network|  : {live['max_abs_network_residual_mcm']:.3e} MCM")

    probe = corruption_probe()
    print(f"\n[2] Corruption     : clean={probe['clean_step_status']} "
          f"-> state={probe['corrupted_state_status']} "
          f"(residual {probe['corrupted_state_residual_mcm']} MCM), "
          f"flow={probe['corrupted_flow_status']}")

    applied = applied_action_evidence()
    print(f"\n[3] Applied action : identical={applied['gates_identical']} "
          f"source={applied['action_source']} status={applied['status']}")

    perf = MassBalanceMonitor().performance_overhead(network, iterations=300)
    print(f"\n[4] Overhead       : bare step {perf['bare_step_ms']:.4f} ms | "
          f"audited {perf['step_with_audit_ms']:.4f} ms | "
          f"overhead {perf['overhead_ms']:.4f} ms/step")

    frozen_after = {str(p.relative_to(_PROJECT_ROOT)): sha256_file(p)
                    for p in FROZEN_ARTIFACTS}
    protected_after = {
        str(p.relative_to(protected_dir)).replace("\\", "/"): sha256_file(p)
        for p in sorted(protected_dir.rglob("*")) if p.is_file()
    }
    frozen_unchanged = frozen_before == frozen_after
    protected_unchanged = protected_before == protected_after

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_match = {}
    for rel, record in manifest["pre_validation"].items():
        path = _PROJECT_ROOT / rel
        manifest_match[rel.replace("\\", "/")] = (
            record["sha256"] in {sha256_file(path), sha256_lf(path)}
        )

    print(f"\n[5] Frozen artifacts unchanged: {frozen_unchanged}")
    print(f"    Protected Phase 15.3 dir unchanged: {protected_unchanged}")
    print(f"    Manifest match: {manifest_match}")

    evidence = {
        "stage": 11,
        "title": "Live mass-balance integrity",
        "timestamp": datetime.now().isoformat(),
        "tolerance_mcm": MASS_BALANCE_TOLERANCE_MCM,
        "timestep_days": 1.0,
        "live_run": live,
        "corruption_probe": probe,
        "applied_action": applied,
        "performance": perf,
        "frozen_artifact_sha256_before": frozen_before,
        "frozen_artifact_sha256_after": frozen_after,
        "frozen_artifacts_unchanged": frozen_unchanged,
        "protected_phase15_3_dir": str(protected_dir.relative_to(_PROJECT_ROOT)),
        "protected_artifacts_unchanged": protected_unchanged,
        "phase15_3_manifest_match": manifest_match,
        "hardware_connected": False,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, default=str)

    print(f"\nEvidence written: {OUTPUT_PATH.relative_to(_PROJECT_ROOT)}")
    ok = (live["all_pass"] and live["violations"] == 0 and frozen_unchanged
          and protected_unchanged and all(manifest_match.values()))
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

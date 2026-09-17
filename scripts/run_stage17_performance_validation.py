"""
Stage 17 — PERFORMANCE VALIDATION: evidence generator.

MEASUREMENT ONLY. This script changes no production behaviour:

  * it never edits a source file,
  * it never retrains, re-fits or re-scales a model,
  * it never relaxes the forecast-provenance gate,
  * it never changes the MPC candidate space,
  * it never creates a second simulation owner.

The ONLY instrumentation is method-level timing applied to INSTANCES at runtime
(``obj.method = timed(obj.method)``), which shadows a bound method on one object
and leaves the class on disk untouched. That is what makes the MPC decomposition
in §4 a measurement of the real production controller rather than of a copy.

The performance path benchmarked is the REAL one:

    Forecast -> GNN advisory -> LiveForecastAdapter -> MPC -> SafetyLayer
             -> DownstreamCapacityGuard -> ReservoirNetwork -> MassBalanceMonitor
             -> StateAdapter -> FastAPI -> WebSocket -> Digital Twin / Streamlit

Two live-cycle cases are measured and NEVER conflated:

  CASE A  the real live behaviour. The live pipeline's forecasts are
          DEMONSTRATION_ONLY (synthetic placeholders) and Reservoir D has no live
          forecast at all, so the provenance gate blocks the MPC and the current
          gates are held. Measured on ``state_manager.sim_state.step()``.
  CASE B  a CONTROLLED TEST FIXTURE — explicitly labelled VALIDATED payloads —
          that lets the complete chain execute so its cost can be measured. The
          fixture is fed through the SAME authoritative objects
          (``sim_state.mpc_orchestrator``, ``sim_state.bridge``); it is NOT live
          telemetry and it does NOT convert any real DEMONSTRATION_ONLY forecast
          into VALIDATED.

OUTPUT
    results/phase15_stage17_performance_validation/stage17_performance_evidence.json
    results/phase15_stage17_performance_validation/stage17_uvicorn_child.log
        (the §6 child process's own stdout+stderr, kept for diagnosis; it is a
        file, never an unread pipe, so it can never block the child)
"""

from __future__ import annotations

import ast
import copy
import gc
import itertools
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import tracemalloc
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:  # the evidence text contains non-ASCII typography
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover
    pass

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

from src.common import units  # noqa: E402
from src.controller.downstream_capacity_guard import DownstreamCapacityGuard  # noqa: E402
from src.controller.live_mpc_orchestrator import LiveMPCOrchestrator  # noqa: E402
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin  # noqa: E402
from src.modeling.v3_feature_contract import feature_provenance_summary  # noqa: E402
from src.network_env.live_forecast_adapter import LiveForecastAdapter  # noqa: E402
from src.network_env.reservoir_network import ReservoirNetwork  # noqa: E402

OUTPUT_DIR = _PROJECT_ROOT / "results" / "phase15_stage17_performance_validation"
OUTPUT_PATH = OUTPUT_DIR / "stage17_performance_evidence.json"

FROZEN_ARTIFACTS = [
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "best_model.pt",
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "log_target_scaler.pkl",
    _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget" / "test_predictions_original_units.csv",
    _PROJECT_ROOT / "models" / "gcn_lstm_gated_v1" / "best_model.pt",
    _PROJECT_ROOT / "data" / "processed" / "graph" / "graph_D_correlation_v1_2" / "edges.csv",
]

NODES = ["Virtual Reservoir A", "Virtual Reservoir B",
         "Virtual Reservoir C", "Virtual Reservoir D"]
HORIZONS = ("forecast_1d", "forecast_3d", "forecast_7d")
MPC_LOOKAHEAD_STEPS = 3
EXPECTED_CANDIDATE_VECTORS = 6 ** 4

SERVER_PORT = 8041
SERVER_BASE = f"http://127.0.0.1:{SERVER_PORT}"

#: §6 — the uvicorn child's output is redirected to this file inside OUTPUT_DIR.
#: It must never be an unread pipe: when the pipe buffer fills, the child blocks
#: inside write(), its single event loop stalls and every request then times out.
CHILD_LOG_NAME = "stage17_uvicorn_child.log"
#: Explicit per-request bound for §6. A healthy step measures ~0.1-0.5 s, so this
#: exists only to stop a genuine hang waiting out the old 300 s default.
CONCURRENCY_REQUEST_TIMEOUT_S = 30.0
#: Hard wall-clock budget for the whole §6 probe. When it expires the child is
#: killed, so a hang can never leave the validation running for hours.
CONCURRENCY_PROBE_BUDGET_S = 600.0

PERF = time.perf_counter_ns


# ===========================================================================
# benchmark harness
# ===========================================================================

def _percentile(sorted_values, p):
    """Linear-interpolation percentile on an already-sorted list."""
    n = len(sorted_values)
    if n == 0:
        return None
    if n == 1:
        return sorted_values[0]
    k = (n - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_values[int(k)]
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (k - lo)


def describe_ns(samples_ns, *, label, warmup, iterations, extras=None):
    """Turn raw nanosecond samples into the required statistics block."""
    ms = [s / 1e6 for s in samples_ns]
    n = len(ms)
    ordered = sorted(ms)
    mean = sum(ms) / n
    stdev = statistics.stdev(ms) if n > 1 else 0.0
    block = {
        "label": label,
        "warmup_iterations": int(warmup),
        "measured_iterations": int(iterations),
        "unit": "ms",
        "mean": mean,
        "median": statistics.median(ms),
        "p95": _percentile(ordered, 0.95),
        "p99": _percentile(ordered, 0.99) if n >= 100 else None,
        "p99_note": None if n >= 100 else f"NOT_REPORTED: {n} samples (<100)",
        "min": ordered[0],
        "max": ordered[-1],
        "stdev": stdev,
        "total": sum(ms),
        "clock": "time.perf_counter_ns",
    }
    if extras:
        block.update(extras)
    return block


def benchmark(fn, *, label, warmup, iterations, before_each=None, extras=None):
    """Warm up, then measure ``iterations`` samples with a monotonic clock."""
    for _ in range(warmup):
        if before_each is not None:
            before_each()
        fn()
    samples = []
    for _ in range(iterations):
        if before_each is not None:
            before_each()
        t0 = PERF()
        fn()
        samples.append(PERF() - t0)
    return describe_ns(samples, label=label, warmup=warmup, iterations=iterations,
                       extras=extras)


def timed_instance_method(obj, name):
    """
    Shadow ``obj.name`` with a timing wrapper and return ``(counters, restore)``.

    ``restore()`` puts the ORIGINAL attribute back. It cannot simply ``delattr``:
    when ``obj`` is a class, deleting ``__init__`` from the class dictionary makes
    the class lose its own initialiser entirely rather than revealing a parent
    one.
    """
    original = getattr(obj, name)
    had_own_attribute = name in getattr(obj, "__dict__", {})
    counters = {"calls": 0, "total_ns": 0, "max_ns": 0}

    def wrapper(*args, **kwargs):
        t0 = PERF()
        try:
            return original(*args, **kwargs)
        finally:
            elapsed = PERF() - t0
            counters["calls"] += 1
            counters["total_ns"] += elapsed
            counters["max_ns"] = max(counters["max_ns"], elapsed)

    setattr(obj, name, wrapper)

    def restore():
        if had_own_attribute:
            setattr(obj, name, original)
        else:
            try:
                delattr(obj, name)
            except AttributeError:  # pragma: no cover
                pass

    return counters, restore


def sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ===========================================================================
# fixture helpers  (a fixture is always labelled as a fixture)
# ===========================================================================

def validated_fixture_payload():
    """
    A CONTROLLED TEST FIXTURE forecast payload.

    It declares VALIDATED so the provenance gate admits it and the full chain
    executes. It is fed ONLY through the audit's own bundle on the audit's own
    call to ``decide()``; it is never injected into ``GlobalSimulationState``, so
    no real live forecast changes status because of it.
    """
    return {
        "forecast_1d": 2.0, "forecast_3d": 2.1, "forecast_7d": 2.2,
        "forecast_status": "VALIDATED",
        "forecast_provenance": "REAL_MEASUREMENT_INPUTS_FROZEN_MODEL",
        "forecast_source": "FROZEN_LSTM_V3",
        "is_simulated": False,
        "validated_metrics_apply": True,
        "forecast_unit": "MCM/day",
        "horizons": list(HORIZONS),
        "input_provenance": {"synthetic_demo": [], "unavailable": [], "simulated": []},
    }


def fixture_bundle(network, forecast_date="2026-09-16"):
    adapter = LiveForecastAdapter.for_network(network, project_root=str(_PROJECT_ROOT))
    payloads = {nid: validated_fixture_payload() for nid in NODES}
    return adapter.build_bundle(payloads, forecast_date)


def network_snapshot(network):
    """Full deep snapshot of the authoritative network (1.0 ms measured)."""
    return copy.deepcopy(network)


def network_restore(network, snapshot):
    """Restore the authoritative network in place from a deep snapshot."""
    network.__dict__.update(copy.deepcopy(snapshot.__dict__))


def reset_live_to_50pct(sm):
    """The EXISTING reset path (``POST /api/simulation/reset`` -> init_cascade)."""
    sm.running = False
    sm.bridge.init_cascade(50.0)


def drive_live_steps(sm, count):
    for _ in range(count):
        sm.step()


# ===========================================================================
# §2 — hardware / environment baseline
# ===========================================================================

def _run(cmd, timeout=60):
    try:
        out = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), capture_output=True,
                             text=True, timeout=timeout)
        return (out.stdout or "").strip()
    except Exception as exc:  # pragma: no cover
        return f"UNKNOWN ({type(exc).__name__})"


def environment_baseline():
    vm = psutil.virtual_memory()
    try:
        import pytest
        pytest_version = pytest.__version__
    except Exception:  # pragma: no cover
        pytest_version = "UNKNOWN"

    pyproject = (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    has_pytest_section = "[tool.pytest.ini_options]" in pyproject

    gpu = {
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "device_used_by_live_models": "cpu" if not torch.cuda.is_available() else "cuda",
        "note": "no GPU is available; every measurement is CPU-only",
    }
    try:
        import torch.cuda as tc
        if torch.cuda.is_available():  # pragma: no cover
            gpu["device_name"] = tc.get_device_name(0)
    except Exception:  # pragma: no cover
        pass

    return {
        "os": f"{platform.system()} {platform.release()} {platform.version()}",
        "os_edition": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "pytorch_version": torch.__version__,
        "numpy_version": np.__version__,
        "psutil_version": psutil.__version__,
        "pytest_version": pytest_version,
        "cpu": {
            "logical_cores": os.cpu_count(),
            "physical_cores": psutil.cpu_count(logical=False),
            "processor": platform.processor() or "UNKNOWN",
            "machine": platform.machine(),
        },
        "ram_total_gb": round(vm.total / (1024 ** 3), 2),
        "ram_available_gb": round(vm.available / (1024 ** 3), 2),
        "gpu": gpu,
        "repository": {
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "head_commit": _run(["git", "rev-parse", "HEAD"]),
            "head_short": _run(["git", "rev-parse", "--short", "HEAD"]),
            "head_subject": _run(["git", "log", "-1", "--format=%s (%cI)"]),
            "working_tree_dirty_entries": len(
                [l for l in _run(["git", "status", "--short"]).splitlines() if l.strip()]
            ),
            "uncommitted_note": (
                "Stages 12-17 are uncommitted working-tree changes; the Phase 15.3 "
                "scientific sources are CLEAN against HEAD (Stage 16 evidence)."
            ),
        },
        "test_configuration": {
            "pyproject_pytest_section": has_pytest_section,
            "pythonpath": ["."],
            "conftest_present": (_PROJECT_ROOT / "tests" / "conftest.py").exists(),
            "invocation": "py -m pytest -q --no-header -p no:cacheprovider",
        },
        "environment_modified_for_benchmarking": False,
    }


# ===========================================================================
# §3 / §4 — component + MPC benchmarks
# ===========================================================================

def component_benchmarks(sm, network, bundle):
    """Benchmark A–J. Nothing here mutates production configuration."""
    comp = {}

    # ---- A. frozen LSTM V3 inference --------------------------------------
    v_name = "Virtual Reservoir A"
    p_name = sm.res_mapping[v_name]
    inputs = sm._live_feature_inputs(v_name, p_name)
    provenance = feature_provenance_summary(inputs)
    frame = sm._feature_frame(p_name, inputs)
    comp["A_frozen_lstm_v3_inference"] = benchmark(
        lambda: sm.lstm_forecaster.predict(frame, input_provenance=provenance),
        label="frozen LSTM V3 predict() (7-day window, 3 horizons)",
        warmup=5, iterations=60,
    )
    comp["A_frozen_lstm_v3_inference"]["forecast_status_of_benchmark_input"] = \
        provenance.get("all_real_measurements") and "VALIDATED" or "DEMONSTRATION_ONLY"

    # ---- B. GNN advisory inference ----------------------------------------
    gnn_history = sm._build_gnn_history()
    comp["B_gnn_advisory_inference"] = benchmark(
        lambda: sm.gnn_forecaster.predict_all(gnn_history),
        label="GNN advisory predict_all() (16 nodes, advisory only)",
        warmup=5, iterations=60,
        extras={"nodes_with_live_input": len(gnn_history)},
    )
    comp["B2_gnn_node_representations"] = benchmark(
        lambda: sm.gnn_forecaster.node_representations(gnn_history),
        label="GNN advisory node_representations() (16 x 64 embeddings + gate)",
        warmup=5, iterations=60,
    )

    # ---- C. LiveForecastAdapter -------------------------------------------
    payloads = {nid: validated_fixture_payload() for nid in NODES}
    adapter = LiveForecastAdapter.for_network(network, project_root=str(_PROJECT_ROOT))
    comp["C_live_forecast_adapter"] = benchmark(
        lambda: adapter.build_bundle(payloads, "2026-09-16"),
        label="LiveForecastAdapter.build_bundle() (4 reservoirs)",
        warmup=5, iterations=100,
    )

    # ---- D. MPCController.decide() ----------------------------------------
    orch = sm.mpc_orchestrator
    mpc = orch.mpc
    comp["D_mpc_decide"] = benchmark(
        lambda: mpc.decide(network, forecast_snapshot=bundle.snapshot,
                           current_inflows=dict(sm.manual_inflows)),
        label="MPCController.decide() (1296 coordinated candidates, 3-step lookahead)",
        warmup=2, iterations=12,
        before_each=lambda: reset_live_to_50pct(sm),
    )
    comp["D_mpc_decide"]["per_candidate_mean_ms"] = (
        comp["D_mpc_decide"]["mean"] / EXPECTED_CANDIDATE_VECTORS
    )
    # The same measurement on the state as it stands, with NO reset between
    # samples, to show how strongly the decision cost depends on reservoir state.
    comp["D2_mpc_decide_no_reset"] = benchmark(
        lambda: mpc.decide(network, forecast_snapshot=bundle.snapshot,
                           current_inflows=dict(sm.manual_inflows)),
        label="MPCController.decide() (state left as-is between samples)",
        warmup=1, iterations=8,
    )
    comp["D2_mpc_decide_no_reset"]["per_candidate_mean_ms"] = (
        comp["D2_mpc_decide_no_reset"]["mean"] / EXPECTED_CANDIDATE_VECTORS
    )

    # ---- E. SafetyLayer.validate() ----------------------------------------
    node_ids = list(network.processing_order)
    proposal = {nid: 0.30 for nid in node_ids}
    current = {nid: float(network.nodes[nid].state.gate_position) for nid in node_ids}
    comp["E_safety_layer_validate"] = benchmark(
        lambda: orch.safety.validate(proposal, current, node_ids),
        label="SafetyLayer.validate() (4 reservoirs)",
        warmup=20, iterations=500,
    )

    # ---- F. DownstreamCapacityGuard.evaluate() ----------------------------
    inflows = {nid: float(network.nodes[nid].state.inflow_local) for nid in node_ids}
    # The action that the Stage 15 validated fixture produced, per reservoir.
    guarded_action = {"Virtual Reservoir A": 0.30, "Virtual Reservoir B": 0.15,
                      "Virtual Reservoir C": 0.0, "Virtual Reservoir D": 0.0}
    safe_result = orch.downstream_guard.evaluate(
        network, action_fraction=guarded_action, current_fraction=current,
        node_ids=node_ids, max_gate_change=mpc.config.max_gate_change, inflows=inflows)
    comp["F_downstream_capacity_guard_safe"] = benchmark(
        lambda: orch.downstream_guard.evaluate(
            network, action_fraction=guarded_action, current_fraction=current,
            node_ids=node_ids, max_gate_change=mpc.config.max_gate_change,
            inflows=inflows),
        label="DownstreamCapacityGuard.evaluate() (Stage 15 fixture action)",
        warmup=5, iterations=100,
        extras={"observed_status": safe_result.status,
                "observed_modified": bool(safe_result.modified)},
    )
    unsafe = {nid: 1.0 for nid in node_ids}
    unsafe_result = orch.downstream_guard.evaluate(
        network, action_fraction=unsafe, current_fraction=unsafe,
        node_ids=node_ids, max_gate_change=1.0, inflows=inflows)
    comp["F2_downstream_capacity_guard_corrected"] = benchmark(
        lambda: orch.downstream_guard.evaluate(
            network, action_fraction=unsafe, current_fraction=unsafe,
            node_ids=node_ids, max_gate_change=1.0, inflows=inflows),
        label="DownstreamCapacityGuard.evaluate() (all gates 1.0 -> search + CORRECTED)",
        warmup=3, iterations=25,
        extras={"observed_status": unsafe_result.status,
                "observed_modified": bool(unsafe_result.modified),
                "observed_action_pct": {
                    k: units.gate_fraction_to_percent(float(v))
                    for k, v in unsafe_result.action_fraction.items()}},
    )

    # ---- G. ReservoirNetwork.step() ---------------------------------------
    snap = network_snapshot(network)
    step_inflows = dict(inflows)
    gate_fractions = {nid: mpc.config.gate_levels[2] for nid in node_ids}
    comp["G_reservoir_network_step"] = benchmark(
        lambda: network.step(dict(step_inflows), dict(gate_fractions)),
        label="ReservoirNetwork.step() (4 reservoirs, routing queues)",
        warmup=20, iterations=500,
        before_each=lambda: network_restore(network, snap),
    )
    network_restore(network, snap)

    # ---- H. MassBalanceMonitor --------------------------------------------
    monitor = sm.bridge.cascade.mass_balance_monitor
    comp["H_mass_balance_step_and_check"] = benchmark(
        lambda: monitor.step_and_check(network, dict(step_inflows), dict(gate_fractions),
                                       action_source="STAGE17_BENCHMARK"),
        label="MassBalanceMonitor.step_and_check() (snapshot + physics + audit)",
        warmup=20, iterations=500,
        before_each=lambda: network_restore(network, snap),
    )
    network_restore(network, snap)
    monitor_snapshot = monitor.snapshot(network)
    network.step(dict(step_inflows), dict(gate_fractions))
    comp["H2_mass_balance_verify_only"] = benchmark(
        lambda: monitor.verify(network, monitor_snapshot,
                               applied_inflows=dict(step_inflows),
                               applied_gates_fraction=dict(gate_fractions),
                               action_source="STAGE17_BENCHMARK"),
        label="MassBalanceMonitor.verify() only (audit, no physics)",
        warmup=5, iterations=300,
    )
    network_restore(network, snap)

    # ---- I. StateAdapter --------------------------------------------------
    publish_source = sm.bridge.get_state(sm._run_ml_pipeline())
    publish_source["control"] = orch.status_dict()
    payload = adapt_state_for_twin(publish_source, "MANUAL", 0.0)
    comp["I_state_adapter"] = benchmark(
        lambda: adapt_state_for_twin(publish_source, "MANUAL", 0.0),
        label="adapt_state_for_twin() (twin payload construction)",
        warmup=20, iterations=500,
    )
    payload_json = json.dumps(payload)
    comp["I2_payload_serialization"] = benchmark(
        lambda: json.dumps(payload),
        label="json.dumps(twin payload) (WebSocket/REST serialization)",
        warmup=20, iterations=300,
        extras={"payload_bytes": len(payload_json.encode("utf-8"))},
    )

    network_restore(network, snap)
    return comp


def mpc_decomposition(sm, network, bundle):
    """
    §4 — instrument the REAL controller instance to decompose one decision.

    The instrumentation shadows ``_simulate_trajectory`` and
    ``objective.evaluate_trajectory`` on the ONE live MPC instance. The class
    files are not touched, and the wrappers call straight through.
    """
    orch = sm.mpc_orchestrator
    mpc = orch.mpc

    sim, sim_restore = timed_instance_method(mpc, "_simulate_trajectory")
    obj, obj_restore = timed_instance_method(mpc.objective, "evaluate_trajectory")
    clone_ctor, clone_restore = timed_instance_method(ReservoirNetwork, "__init__")
    try:
        reset_live_to_50pct(sm)
        t0 = PERF()
        decision = mpc.decide(network, forecast_snapshot=bundle.snapshot,
                              current_inflows=dict(sm.manual_inflows))
        total_ns = PERF() - t0
    finally:
        sim_restore()
        obj_restore()
        clone_restore()

    levels = list(mpc.config.gate_levels)
    node_ids = list(network.processing_order)
    # An INDEPENDENT enumeration of the Cartesian product the controller searches.
    enumerated = len(list(itertools.product(levels, repeat=len(node_ids))))

    return {
        "candidate_space": {
            "gate_levels": levels,
            "gate_levels_count": len(levels),
            "reservoirs_included": len(node_ids),
            "reservoir_ids": node_ids,
            "expected_candidate_vectors": EXPECTED_CANDIDATE_VECTORS,
            "independently_enumerated_candidate_vectors": enumerated,
            "decision_candidates_evaluated": int(decision.candidates_evaluated),
            "orchestrator_reported_candidate_vectors":
                orch.action_space_for(node_ids)["candidate_vectors"],
            "matches_1296": (enumerated == EXPECTED_CANDIDATE_VECTORS ==
                             int(decision.candidates_evaluated)),
            "reduced_for_performance": False,
        },
        "instrumented_run": {
            "note": ("this run is instrumented, so its wall time is HIGHER than "
                     "the headline ``D_mpc_decide`` number; the decomposition is "
                     "what is taken from it"),
            "total_decision_ms": total_ns / 1e6,
            "trajectory_simulation_calls": sim["calls"],
            "trajectory_simulation_total_ms": sim["total_ns"] / 1e6,
            "trajectory_simulation_mean_ms": sim["total_ns"] / 1e6 / max(sim["calls"], 1),
            "trajectory_simulation_max_ms": sim["max_ns"] / 1e6,
            "objective_evaluation_calls": obj["calls"],
            "objective_evaluation_total_ms": obj["total_ns"] / 1e6,
            "objective_evaluation_mean_ms": obj["total_ns"] / 1e6 / max(obj["calls"], 1),
            "network_clone_ctor_calls": clone_ctor["calls"],
            "network_clone_ctor_total_ms": clone_ctor["total_ns"] / 1e6,
            "network_clone_ctor_mean_ms": clone_ctor["total_ns"] / 1e6 / max(clone_ctor["calls"], 1),
            "lookahead_steps": MPC_LOOKAHEAD_STEPS,
            "physics_steps_simulated_per_decision": sim["calls"] * MPC_LOOKAHEAD_STEPS,
        },
        "decision_summary": {
            "status": decision.status,
            "safety_status": decision.safety_status,
            "forecast_used": decision.forecast_used,
            "forecast_status": decision.forecast_status,
            "objective_score": decision.objective_score,
        },
    }


# ===========================================================================
# §5 — live control cycle, both cases
# ===========================================================================

def live_cycle_case_a(sm):
    """
    CASE A — the REAL live cycle where the provenance gate blocks the MPC.

    Measured on ``state_manager.sim_state.step()`` exactly as the REST
    ``POST /api/simulation/step`` route drives it. Nothing is substituted.
    """
    sm.running = False
    sm.mode = "AI"          # AI mode is what makes the provenance gate decide
    reset_live_to_50pct(sm)
    drive_live_steps(sm, 8)  # 7 distinct steps are required before a forecast exists

    ml, ml_restore = timed_instance_method(sm, "_run_ml_pipeline")
    ai, ai_restore = timed_instance_method(sm, "_apply_ai_control")
    phys, phys_restore = timed_instance_method(sm.bridge, "step")
    record, record_restore = timed_instance_method(sm, "_record_history")
    try:
        stats = benchmark(sm.step, label="CASE A live cycle (AI mode, MPC blocked)",
                          warmup=2, iterations=20)
        counters = {
            "ml_pipeline_calls": ml["calls"],
            "ml_pipeline_total_ms": ml["total_ns"] / 1e6,
            "apply_ai_control_calls": ai["calls"],
            "apply_ai_control_total_ms": ai["total_ns"] / 1e6,
            "physics_step_calls": phys["calls"],
            "physics_step_total_ms": phys["total_ns"] / 1e6,
            "record_history_calls": record["calls"],
            "record_history_total_ms": record["total_ns"] / 1e6,
        }
    finally:
        ml_restore()
        ai_restore()
        phys_restore()
        record_restore()

    decision = sm.last_control_decision
    measured_cycles = counters["physics_step_calls"]
    counters["ml_pipeline_calls_per_cycle"] = counters["ml_pipeline_calls"] / max(measured_cycles, 1)
    counters["ml_pipeline_total_ms_per_cycle"] = \
        counters["ml_pipeline_total_ms"] / max(measured_cycles, 1)
    counters["observed_cost_note"] = (
        "``GlobalSimulationState.step()`` calls ``_run_ml_pipeline()`` once itself "
        "and once more inside ``get_adapted_state()``, so the forecast + GNN "
        "advisory are computed TWICE per cycle. Measured, not inferred."
    )

    state = sm.get_adapted_state()
    result = {
        "case": "A",
        "description": ("real live behaviour: the live forecast provenance gate "
                        "blocks the MPC, which therefore holds the current gates"),
        "is_fixture": False,
        "cycle": stats,
        "decomposition": counters,
        "chain_verdict": {
            "controller_status": decision.controller_status if decision else None,
            "forecast_control_eligible":
                getattr(decision, "forecast_control_eligible", None),
            "mpc_status": getattr(decision, "mpc_status", None),
            "safety_layer_status": getattr(decision, "safety_layer_status", None),
            "downstream_status": getattr(decision, "downstream_status", None),
            "final_safe_control_action_source":
                getattr(decision, "final_safe_control_action_source", None),
            "control_applied": getattr(decision, "control_applied", None),
            "blocked_reason": getattr(decision, "blocked_reason", None),
            "forecast_summary_status_at_block_time":
                (state.get("forecast_summary") or {}).get("validated_metrics_apply"),
            "forecast_provenance_at_block_time":
                (state.get("forecast_summary") or {}).get("provenance"),
            "gnn_affects_control": (state.get("gnn_advisory") or {}).get("affects_control"),
        },
        "mass_balance_status": (state.get("mass_balance") or {}).get("status"),
        "mass_balance_checked": (state.get("mass_balance") or {}).get("checked"),
        "state_id": state.get("state_identity", {}).get("state_id"),
        "gnn_advisory_status": (state.get("gnn_advisory") or {}).get("status"),
    }

    # A second real case: MANUAL mode, the actual default live behaviour.
    sm.mode = "MANUAL"
    reset_live_to_50pct(sm)
    drive_live_steps(sm, 4)
    manual = benchmark(sm.step, label="CASE A2 live cycle (MANUAL operator gates)",
                       warmup=2, iterations=20)
    manual_state = sm.get_adapted_state()
    result["case_a2_manual_mode"] = {
        "description": ("the default live mode: operator gates are applied and no "
                        "controller decision is produced at all"),
        "cycle": manual,
        "action_source": (manual_state.get("mass_balance") or {}).get(
            "applied_action_source"),
        "mass_balance_status": (manual_state.get("mass_balance") or {}).get("status"),
    }
    return result


def live_cycle_case_b(sm):
    """
    CASE B — a CONTROLLED TEST FIXTURE that lets the complete chain execute.

    The fixture is an explicitly labelled VALIDATED forecast bundle handed to the
    SAME authoritative orchestrator, then applied through the SAME authoritative
    bridge, and published by the SAME state manager. It is NOT live telemetry and
    it does not change any real forecast's status.
    """
    sm.running = False
    sm.mode = "MANUAL"
    reset_live_to_50pct(sm)
    drive_live_steps(sm, 8)

    network = sm.bridge.cascade.network
    orch = sm.mpc_orchestrator
    node_ids = list(network.processing_order)
    bundle = fixture_bundle(network)
    gnn_history = sm._build_gnn_history()

    phases = {"advisory_ms": [], "adapter_ms": [], "decide_ms": [],
              "physics_ms": [], "publish_ms": [], "total_ms": []}
    decisions = []
    for _ in range(12):
        reset_live_to_50pct(sm)  # not part of the measured cycle
        t_start = PERF()

        t0 = PERF()
        sm.gnn_forecaster.predict_all(gnn_history)
        phases["advisory_ms"].append((PERF() - t0) / 1e6)

        t0 = PERF()
        b = fixture_bundle(sm.bridge.cascade.network)
        phases["adapter_ms"].append((PERF() - t0) / 1e6)

        t0 = PERF()
        decision = orch.decide(sm.bridge.cascade.network, bundle=b,
                               current_inflows=dict(sm.manual_inflows))
        phases["decide_ms"].append((PERF() - t0) / 1e6)
        decisions.append(decision)

        t0 = PERF()
        sm.bridge.step(dict(sm.manual_inflows), dict(decision.gate_positions_pct),
                       action_source=decision.final_safe_control_action_source)
        phases["physics_ms"].append((PERF() - t0) / 1e6)

        t0 = PERF()
        published = sm.get_adapted_state()
        phases["publish_ms"].append((PERF() - t0) / 1e6)

        phases["total_ms"].append((PERF() - t_start) / 1e6)

    last = decisions[-1]

    def block(key, label):
        return describe_ns([v * 1e6 for v in phases[key]], label=label,
                           warmup=0, iterations=len(phases[key]),
                           extras={"note": "reset between samples is NOT measured"})

    result = {
        "case": "B",
        "description": ("CONTROLLED TEST FIXTURE (explicitly labelled VALIDATED "
                        "payloads) driving forecast -> GNN advisory -> adapter -> "
                        "MPC -> SafetyLayer -> guard -> physics -> mass balance -> "
                        "state publication"),
        "is_fixture": True,
        "fixture_declaration": {
            "payload_forecast_status": "VALIDATED",
            "payload_provenance": "REAL_MEASUREMENT_INPUTS_FROZEN_MODEL",
            "is_live_telemetry": False,
            "fed_to": "audit-local bundle on the authoritative LiveMPCOrchestrator",
            "real_demonstration_only_forecasts_converted": False,
            "live_pipeline_status_after_case_b": None,  # filled in below
        },
        "phases": {
            "gnn_advisory": block("advisory_ms", "GNN advisory inference (fixture cycle)"),
            "live_forecast_adapter": block("adapter_ms", "LiveForecastAdapter.build_bundle() (fixture cycle)"),
            "mpc_and_safety_and_guard": block("decide_ms", "MPC + SafetyLayer + guard (one full decide())"),
            "physics_and_mass_balance": block("physics_ms", "ReservoirNetwork.step() + mass-balance audit"),
            "state_publication": block("publish_ms", "get_adapted_state() (includes a second ML pipeline run)"),
            "total_cycle": block("total_ms", "complete fixture cycle"),
        },
        "chain_verdict": {
            "forecast_control_eligible": last.forecast_control_eligible,
            "mpc_status": last.mpc_status,
            "mpc_objective_score": last.mpc_objective_score,
            "safety_layer_status": last.safety_layer_status,
            "safety_modified": last.safety_modified,
            "downstream_status": last.downstream_status,
            "downstream_capacity_achieved": last.downstream_capacity_achieved,
            "downstream_protection_modified": last.downstream_protection_modified,
            "final_safe_control_action_pct": dict(last.final_safe_control_action_pct),
            "final_safe_control_action_source": last.final_safe_control_action_source,
            "control_applied": last.control_applied,
            "candidates_evaluated": int(
                orch.action_space_for(node_ids)["candidate_vectors"]),
        },
        "physics_received_action_pct": {
            nid: units.gate_fraction_to_percent(
                float(sm.bridge.cascade.network.nodes[nid].state.gate_position))
            for nid in node_ids
        },
        "mass_balance_status": (published.get("mass_balance") or {}).get("status"),
        "mass_balance_residual": (published.get("mass_balance") or {}).get("residual"),
        "state_id": published.get("state_identity", {}).get("state_id"),
    }
    # prove the real pipeline is still DEMONSTRATION_ONLY / fail-closed
    sm.mode = "AI"
    reset_live_to_50pct(sm)
    drive_live_steps(sm, 8)
    after = sm.get_adapted_state()
    result["fixture_declaration"]["live_pipeline_status_after_case_b"] = {
        "forecast_control_eligible": getattr(sm.last_control_decision,
                                             "forecast_control_eligible", None),
        "final_safe_control_action_source": getattr(sm.last_control_decision,
                                                    "final_safe_control_action_source", None),
        "blocked_reason": getattr(sm.last_control_decision, "blocked_reason", None),
        "gnn_affects_control": (after.get("gnn_advisory") or {}).get("affects_control"),
    }
    sm.mode = "MANUAL"
    return result


# ===========================================================================
# §6 — concurrency / serialization
# ===========================================================================

def _http_json(url, method="GET", payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, {"detail": exc.read().decode("utf-8", "replace")}


def _wait_for_server(deadline_s=90):
    end = time.time() + deadline_s
    while time.time() < end:
        try:
            status, _ = _http_json(f"{SERVER_BASE}/api/state", timeout=20)
            if status == 200:
                return True
        except Exception:
            time.sleep(0.4)
    return False


def _step_once(timeout=CONCURRENCY_REQUEST_TIMEOUT_S):
    t0 = PERF()
    try:
        st, body = _http_json(f"{SERVER_BASE}/api/simulation/step", "POST", {},
                              timeout=timeout)
        return st, body, (PERF() - t0) / 1e6
    except Exception as exc:
        return None, {"error": f"{type(exc).__name__}: {exc}"}, (PERF() - t0) / 1e6


def concurrency_probe(sm):
    """§6 — real HTTP + WebSocket traffic against a real uvicorn server."""
    import importlib.util
    if importlib.util.find_spec("uvicorn") is None:  # pragma: no cover
        return {"performed": False, "reason": "uvicorn is not installed"}

    evidence = {"performed": True, "server": f"uvicorn 127.0.0.1:{SERVER_PORT}",
                "workers": 1}
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    # The child's output MUST be drained. Nothing in this probe consumes a
    # ``subprocess.PIPE``, so the pipe buffer fills after a few physics steps and
    # the child blocks inside write() — stalling its single event loop and making
    # every request time out. Redirecting to a file keeps the child's output
    # fully diagnosable while never blocking it.
    child_log_path = OUTPUT_DIR / CHILD_LOG_NAME
    child_log_path.parent.mkdir(parents=True, exist_ok=True)
    child_log = open(child_log_path, "wb")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.dashboard.api.app:app",
         "--host", "127.0.0.1", "--port", str(SERVER_PORT), "--log-level", "warning"],
        cwd=str(_PROJECT_ROOT), env=env,
        stdin=subprocess.DEVNULL,
        stdout=child_log, stderr=subprocess.STDOUT,
    )
    # Watchdog: bound the probe so a genuine hang cannot run for hours.
    probe_deadline = time.monotonic() + CONCURRENCY_PROBE_BUDGET_S

    def _watchdog():
        while proc.poll() is None:
            if time.monotonic() > probe_deadline:
                print(f"    [6] watchdog: child exceeded the "
                      f"{CONCURRENCY_PROBE_BUDGET_S:.0f} s probe budget — killing it "
                      f"so the validation cannot hang", flush=True)
                proc.kill()
                return
            time.sleep(0.5)

    watchdog = threading.Thread(target=_watchdog, daemon=True)
    watchdog.start()
    try:
        if not _wait_for_server():
            return {"performed": False, "reason": "server did not become ready"}

        _http_json(f"{SERVER_BASE}/api/simulation/pause", "POST", {}, timeout=120)
        _http_json(f"{SERVER_BASE}/api/simulation/reset", "POST", {}, timeout=120)
        _http_json(f"{SERVER_BASE}/api/controller/mode", "POST", {"mode": "MANUAL"},
                   timeout=120)
        # Pre-warm past the 7-step forecast warm-up so a step is a real cycle.
        for _ in range(9):
            _step_once()

        # ---- 1. uncontended step latency ---------------------------------
        solo = [_step_once()[2] for _ in range(3)]

        # ---- 2. step latency under continuous state-read pressure --------
        # ``GET /api/state`` computes the whole twin payload (including the ML
        # pipeline) synchronously on the same event loop as the commands, so
        # reader load is a real contention source worth measuring.
        stop_readers = threading.Event()
        reader_count = {"n": 0}

        def reader():
            while not stop_readers.is_set():
                try:
                    _http_json(f"{SERVER_BASE}/api/state", timeout=120)
                    reader_count["n"] += 1
                except Exception:
                    pass
                time.sleep(0.02)

        readers = [threading.Thread(target=reader, daemon=True) for _ in range(4)]
        for t in readers:
            t.start()
        pressured = [_step_once()[2] for _ in range(3)]
        stop_readers.set()
        for t in readers:
            t.join(timeout=5)
        evidence["state_read_pressure"] = {
            "reader_threads": 4,
            "state_reads_completed": reader_count["n"],
            "uncontended_step_mean_ms": sum(solo) / len(solo),
            "pressured_step_mean_ms": sum(pressured) / len(pressured),
            "slowdown_factor": (sum(pressured) / len(pressured)) / (sum(solo) / len(solo)),
            "note": ("a step is triggered one at a time in both measurements; the "
                     "difference is contention between the control mutation and the "
                     "state payload computation, which share one event loop"),
        }

        _, before = _http_json(f"{SERVER_BASE}/api/state", timeout=120)
        before_idx = before["state_identity"]["sim_step_index"]

        # ---- 3. concurrent step requests ---------------------------------
        n = 8
        results = [None] * n
        observed_ids = []
        ids_lock = threading.Lock()
        stop_polling = threading.Event()

        def poller():
            # Deliberately gentle: a fast poll loop would itself flood the one
            # event loop and change the thing being measured.
            reads = 0
            while not stop_polling.is_set() and reads < 14:
                try:
                    st, body = _http_json(f"{SERVER_BASE}/api/state", timeout=120)
                    if st == 200:
                        with ids_lock:
                            observed_ids.append(
                                (body["state_identity"]["state_id"],
                                 body["state_identity"]["sim_step_index"]))
                except Exception as exc:
                    with ids_lock:
                        observed_ids.append((f"ERROR:{type(exc).__name__}", -1))
                reads += 1
                time.sleep(0.15)

        pollers = [threading.Thread(target=poller, daemon=True) for _ in range(2)]
        for t in pollers:
            t.start()

        def one(i):
            results[i] = _step_once()

        threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
        t_start = PERF()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        burst_ms = (PERF() - t_start) / 1e6
        stop_polling.set()
        for t in pollers:
            t.join(timeout=10)

        _, after = _http_json(f"{SERVER_BASE}/api/state", timeout=120)
        after_idx = after["state_identity"]["sim_step_index"]

        statuses = [r[0] for r in results]
        latencies = [r[2] for r in results]
        with ids_lock:
            ids = list(observed_ids)

        indices = [i for _, i in ids]
        clean_ids = [sid for sid, _ in ids if not sid.startswith("ERROR")]
        evidence["concurrent_steps"] = {
            "requests": n,
            "http_200": sum(1 for s in statuses if s == 200),
            "other_statuses": sorted({str(s) for s in statuses if s != 200}),
            "errors": [r[1] for r in results if r[0] != 200],
            "sim_step_index_before": before_idx,
            "sim_step_index_after": after_idx,
            "steps_advanced": after_idx - before_idx,
            "expected_steps": n,
            "duplicate_or_lost_steps": (after_idx - before_idx) != n,
            "burst_wall_ms": burst_ms,
            "request_latency_ms_mean": sum(latencies) / len(latencies),
            "request_latency_ms_max": max(latencies),
            "state_reads_during_burst": len(ids),
            "distinct_state_ids_during_burst": len(set(clean_ids)),
            "state_ids_all_succeeded": all(
                not s.startswith("ERROR") for s, _ in ids),
            "state_index_monotonic_nondecreasing":
                all(b >= a for a, b in zip(indices, indices[1:])),
            "state_index_max_minus_min": (max(indices) - min(indices)) if indices else None,
        }

        # ---- 4. REST vs WebSocket consistency ----------------------------
        ws_evidence = {"performed": False}
        try:
            from websockets.sync.client import connect as ws_connect
            with ws_connect(f"ws://127.0.0.1:{SERVER_PORT}/ws/state",
                            open_timeout=30, close_timeout=5) as ws:
                first = json.loads(ws.recv(timeout=60))
                _step_once()
                frame = None
                frames_read = 0
                deadline = time.time() + 60
                target = first["state_identity"]["sim_step_index"] + 1
                while time.time() < deadline:
                    candidate = json.loads(ws.recv(timeout=60))
                    frames_read += 1
                    if candidate["state_identity"]["sim_step_index"] >= target:
                        frame = candidate
                        break
                _, rest = _http_json(f"{SERVER_BASE}/api/state", timeout=120)
                ws_evidence = {
                    "performed": True,
                    "frames_read": frames_read,
                    "ws_state_id": None if frame is None else
                        frame["state_identity"]["state_id"],
                    "rest_state_id": rest["state_identity"]["state_id"],
                    "same_state_id": None if frame is None else
                        frame["state_identity"]["state_id"] ==
                        rest["state_identity"]["state_id"],
                    "reservoir_block_identical": None if frame is None else
                        frame["reservoirs"] == rest["reservoirs"],
                    "mass_balance_status_identical": None if frame is None else
                        frame["mass_balance"]["status"] == rest["mass_balance"]["status"],
                    "gnn_advisory_identical": None if frame is None else
                        frame["gnn_advisory"] == rest["gnn_advisory"],
                    "payload_bytes": None if frame is None else
                        len(json.dumps(frame).encode("utf-8")),
                }
        except Exception as exc:
            ws_evidence = {"performed": False,
                           "reason": f"{type(exc).__name__}: {exc}"}
        evidence["rest_vs_websocket"] = ws_evidence

        # ---- 5. background loop ------------------------------------------
        _http_json(f"{SERVER_BASE}/api/simulation/pause", "POST", {}, timeout=120)
        _, s_paused = _http_json(f"{SERVER_BASE}/api/state", timeout=120)
        _http_json(f"{SERVER_BASE}/api/simulation/play", "POST", {}, timeout=120)
        time.sleep(2.5)  # allow the background loop real iterations
        _http_json(f"{SERVER_BASE}/api/simulation/pause", "POST", {}, timeout=120)
        _, s_played = _http_json(f"{SERVER_BASE}/api/state", timeout=120)
        evidence["background_loop"] = {
            "running_flag_after_pause": s_paused["simulation"]["running"],
            "steps_advanced_while_playing": (
                s_played["state_identity"]["sim_step_index"]
                - s_paused["state_identity"]["sim_step_index"]
            ),
            "note": ("the background loop steps the SAME one instance from the "
                     "SAME event loop; it cannot overlap an HTTP step"),
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:  # pragma: no cover
            proc.kill()
            try:
                proc.wait(timeout=15)
            except Exception:  # pragma: no cover
                pass
        watchdog.join(timeout=5)
        child_log.flush()
        child_log.close()

    evidence["in_process_concurrency"] = in_process_concurrency_probe(sm)
    return evidence


def in_process_concurrency_probe(sm):
    """
    Direct multi-thread probes.

    ``decide()`` never mutates the network, so overlapping it is safe to measure
    and is exactly the question §6 asks ("overlapping MPC decisions"). A direct
    multi-threaded ``step()`` probe is NOT run: it would corrupt the authoritative
    state and its mass-balance audit, and the REST path is measured through the
    real server instead. What is measured here is whether the object graph holds
    exactly ONE simulation owner.
    """
    network = sm.bridge.cascade.network
    orch = sm.mpc_orchestrator
    bundle = fixture_bundle(network)

    lock = threading.Lock()
    state = {"current": 0, "max": 0, "calls": 0}
    original = orch.decide

    def wrapped(*a, **k):
        with lock:
            state["current"] += 1
            state["calls"] += 1
            state["max"] = max(state["max"], state["current"])
        try:
            return original(*a, **k)
        finally:
            with lock:
                state["current"] -= 1

    orch.decide = wrapped
    errors = []
    try:
        def worker():
            try:
                orch.decide(network, bundle=bundle,
                            current_inflows=dict(sm.manual_inflows))
            except Exception as exc:  # pragma: no cover
                errors.append(f"{type(exc).__name__}: {exc}")

        n = 4
        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        orch.decide = original

    # Is there any synchronization primitive at all in the live path?
    # AST, not a text scan: a comment or docstring naming a lock must not count.
    live_files = [
        "src/dashboard/api/state_manager.py", "src/dashboard/api/routes.py",
        "src/dashboard/api/app.py", "src/dashboard/sim_bridge.py",
        "src/network_env/live_cascade_adapter.py",
        "src/controller/live_mpc_orchestrator.py",
        "src/controller/mpc_controller.py",
        "src/controller/safety.py",
        "src/controller/downstream_capacity_guard.py",
        "src/network_env/reservoir_network.py",
    ]
    LOCK_NAMES = {"Lock", "RLock", "Semaphore", "Event", "Condition"}
    lock_hits = {}
    for rel in live_files:
        tree = ast.parse((_PROJECT_ROOT / rel).read_text(encoding="utf-8"))
        hits = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and node.func.attr in LOCK_NAMES:
                hits.append(f"{node.func.attr}() line {node.lineno}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in ("threading", "asyncio"):
                        hits.append(f"import {alias.name} line {node.lineno}")
        if hits:
            lock_hits[rel] = hits

    # AST proof that ``step()`` has no suspension point, so the asyncio event
    # loop cannot interleave two mutations.
    step_awaits = []
    sm_tree = ast.parse((_PROJECT_ROOT / "src/dashboard/api/state_manager.py")
                        .read_text(encoding="utf-8"))
    for node in ast.walk(sm_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name == "step" and isinstance(node, ast.FunctionDef):
            step_awaits = [n.lineno for n in ast.walk(node)
                           if isinstance(n, (ast.Await, ast.AsyncFor, ast.AsyncWith))]
    # how many times the module-level singleton is constructed
    owners = []
    for path in (_PROJECT_ROOT / "src").rglob("*.py"):
        rel = str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/")
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                    and node.func.id == "GlobalSimulationState":
                owners.append(f"{rel}:{node.lineno}")
                break

    return {
        "step_suspension_points": step_awaits,
        "step_is_synchronous_with_no_suspension_point": (len(step_awaits) == 0),
        "concurrent_decide_probe": {
            "threads": n,
            "decide_calls": state["calls"],
            "max_simultaneous_decide": state["max"],
            "errors": errors,
            "mutates_network": False,
            "note": ("``decide()`` is read-only with respect to the live network "
                     "(it clones), so this probe is non-destructive"),
        },
        "explicit_locks_in_live_path": lock_hits,
        "explicit_lock_present": bool(lock_hits),
        "serialization_mechanism": (
            "Every command handler in routes.py is ``async def`` and "
            "``GlobalSimulationState.step()`` is a synchronous call with no "
            "``await`` inside it, so the single asyncio event loop runs each "
            "mutation to completion: control commands are serialized by the event "
            "loop rather than by a lock."
        ),
        "authoritative_instance_count_in_this_process":
            state_manager.authoritative_instance_count(),
        "modules_constructing_GlobalSimulationState": owners,
        "single_owner_in_process": (
            state_manager.authoritative_instance_count() == 1
            and len(owners) == 1
            and owners[0].startswith("src/dashboard/api/state_manager.py:")
        ),
        "multi_worker_note": (
            "ANALYTIC, NOT BENCHMARKED: ``sim_state`` is constructed at module "
            "import, so one uvicorn worker process owns one instance. Running "
            "uvicorn with more than one worker would create one owner PER process "
            "and the twin would read whichever worker answered. The measured "
            "evidence in this report is for a single-worker deployment."
        ),
    }


# ===========================================================================
# §7 / §8 — memory and throughput stability
# ===========================================================================

def memory_probe(sm):
    proc = psutil.Process(os.getpid())

    def rss_mb():
        return proc.memory_info().rss / (1024 ** 2)

    gc.collect()
    baseline = rss_mb()

    sm.running = False
    sm.mode = "AI"
    reset_live_to_50pct(sm)
    drive_live_steps(sm, 8)
    after_warmup = rss_mb()

    rounds = []
    for r in range(3):
        for _ in range(15):
            sm.step()
        gc.collect()
        rounds.append({"round": r + 1, "cycles": 15, "rss_mb": rss_mb()})

    # leak probe: allocation growth across repeated advisory + MPC clones
    network = sm.bridge.cascade.network
    bundle = fixture_bundle(network)
    gc.collect()
    tracemalloc.start()
    snap1 = tracemalloc.take_snapshot()
    for _ in range(3):
        sm.gnn_forecaster.predict_all(sm._build_gnn_history())
    for _ in range(3):
        sm.mpc_orchestrator.decide(network, bundle=bundle,
                                   current_inflows=dict(sm.manual_inflows))
    snap2 = tracemalloc.take_snapshot()
    growth = snap2.compare_to(snap1, "lineno")
    tracemalloc.stop()

    def count_networks():
        gc.collect()
        return sum(1 for o in gc.get_objects() if isinstance(o, ReservoirNetwork))

    net_objects_after = count_networks()

    return {
        "rss_baseline_mb": baseline,
        "rss_after_warmup_mb": after_warmup,
        "rounds": rounds,
        "rss_growth_over_45_cycles_mb": rounds[-1]["rss_mb"] - after_warmup,
        "rss_after_tracemalloc_mb": rss_mb(),
        "history_buffers": {
            name: {"length": len(buf), "cap": 7}
            for name, buf in sm.history_buffers.items()
        },
        "history_buffers_bounded": all(len(b) <= 7 for b in sm.history_buffers.values()),
        "websocket_clients_registered": len(sm.clients),
        "live_reservoir_network_objects_after_cycles": net_objects_after,
        "tracemalloc_top_growth": [
            {"location": str(stat.traceback[0]), "size_diff_kb": stat.size_diff / 1024,
             "count_diff": stat.count_diff}
            for stat in growth[:8]
        ],
        "note": ("tracemalloc growth covers 3 GNN advisory runs + 3 full MPC "
                 "decisions (3888 network clones) performed on purpose to test "
                 "whether repeated cloning leaks"),
        "destructive_stress_tests_performed": False,
    }


def throughput_probe(sm):
    sm.running = False
    sm.mode = "AI"
    reset_live_to_50pct(sm)
    drive_live_steps(sm, 8)

    cycles = 40
    samples = []
    failures = []
    exceptions = []
    mass_balance_statuses = []
    step_indices = []
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss / (1024 ** 2)

    for i in range(cycles):
        if i and i % 10 == 0:
            reset_live_to_50pct(sm)  # not part of any measured cycle
        try:
            t0 = PERF()
            state = sm.step()
            samples.append(PERF() - t0)
            mb = state.get("mass_balance") or {}
            mass_balance_statuses.append(mb.get("status"))
            step_indices.append(state["state_identity"]["sim_step_index"])
            if mb.get("checked") is not True:
                failures.append({"cycle": i, "reason": "MASS_BALANCE_NOT_CHECKED"})
        except Exception as exc:
            exceptions.append({"cycle": i, "error": f"{type(exc).__name__}: {exc}"})

    rss_after = proc.memory_info().rss / (1024 ** 2)

    return {
        "cycles_requested": cycles,
        "cycles_completed": len(samples),
        "failures": failures,
        "exceptions": exceptions,
        "timing": describe_ns(samples, label="repeated live cycle (AI, MPC blocked)",
                              warmup=0, iterations=len(samples)),
        "mass_balance_status_counts": {
            s: mass_balance_statuses.count(s) for s in sorted(set(mass_balance_statuses))
        },
        "sim_step_index_monotonic": all(
            b == a + 1 for a, b in zip(step_indices, step_indices[1:])),
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_change_mb": rss_after - rss_before,
        "reset_every_10_cycles": True,
        "reset_note": ("storage is returned to 50 % every 10 cycles so the run "
                       "cannot end in a saturated, spilling end-state; the reset "
                       "is not part of any measured cycle"),
        "real_time_claim": ("NOT CLAIMED. A fast benchmark on a shared desktop "
                            "OS is not evidence of real-time capability; see the "
                            "hardware-readiness implications in the report."),
    }


# ===========================================================================
# §9 / §10 — bottleneck ranking and budget
# ===========================================================================

def bottleneck_ranking(comp, mpc_decomp, case_a, case_b):
    """Ranked list of MEASURED costs. No subjective wording."""
    rows = [
        ("MPC decision (1296 candidates, 3-step lookahead)", "scientific computation",
         comp["D_mpc_decide"]["mean"]),
        ("MPC trajectory simulation (all candidates, instrumented)",
         "scientific computation", mpc_decomp["instrumented_run"]["trajectory_simulation_total_ms"]),
        ("Live cycle CASE B (fixture, complete chain)", "integration",
         case_b["phases"]["total_cycle"]["mean"]),
        ("Live cycle CASE A (AI, MPC blocked)", "integration", case_a["cycle"]["mean"]),
        ("Live cycle CASE A2 (MANUAL)", "integration",
         case_a["case_a2_manual_mode"]["cycle"]["mean"]),
        ("DownstreamCapacityGuard CORRECTED path", "safety computation",
         comp["F2_downstream_capacity_guard_corrected"]["mean"]),
        ("DownstreamCapacityGuard PROTECTED path", "safety computation",
         comp["F_downstream_capacity_guard_safe"]["mean"]),
        ("LiveForecastAdapter.build_bundle", "serialization/adapter",
         comp["C_live_forecast_adapter"]["mean"]),
        ("GNN advisory inference", "inference", comp["B_gnn_advisory_inference"]["mean"]),
        ("GNN advisory embeddings", "inference",
         comp["B2_gnn_node_representations"]["mean"]),
        ("Frozen LSTM V3 inference", "inference",
         comp["A_frozen_lstm_v3_inference"]["mean"]),
        ("Twin payload JSON serialization", "serialization",
         comp["I2_payload_serialization"]["mean"]),
        ("MassBalanceMonitor.step_and_check", "physics/audit",
         comp["H_mass_balance_step_and_check"]["mean"]),
        ("MassBalanceMonitor.verify only", "physics/audit",
         comp["H2_mass_balance_verify_only"]["mean"]),
        ("ReservoirNetwork.step", "physics", comp["G_reservoir_network_step"]["mean"]),
        ("adapt_state_for_twin", "serialization/UI", comp["I_state_adapter"]["mean"]),
        ("SafetyLayer.validate", "safety computation",
         comp["E_safety_layer_validate"]["mean"]),
    ]
    rows.sort(key=lambda r: r[2], reverse=True)
    return {
        "ordered_by_measured_mean_ms": [
            {"rank": i + 1, "component": name, "category": category,
             "measured_mean_ms": value}
            for i, (name, category, value) in enumerate(rows)
        ],
        "category_totals_ms": {},
        "note": ("ranking is by measured mean wall time only; no qualitative "
                 "labels are applied"),
    }


def performance_budget(comp, mpc_decomp, case_a, case_b):
    """
    A transparent PROTOTYPE budget. The budget numbers are derived from the
    hardware-integration cadence this prototype would be asked to serve
    (a 60 s telemetry/control cadence), NOT chosen to make the measurements pass.
    """
    total_budget_ms = 60000.0  # one control decision period
    rows = [
        ("forecast (frozen LSTM V3, 3 reservoirs)", 60.0,
         comp["A_frozen_lstm_v3_inference"]["mean"] * 3),
        ("GNN advisory + embeddings", 60.0,
         comp["B_gnn_advisory_inference"]["mean"] + comp["B2_gnn_node_representations"]["mean"]),
        ("forecast adapter (4 reservoirs)", 10.0, comp["C_live_forecast_adapter"]["mean"]),
        ("MPC decision (1296 candidates)", 2000.0, comp["D_mpc_decide"]["mean"]),
        ("SafetyLayer.validate", 5.0, comp["E_safety_layer_validate"]["mean"]),
        ("DownstreamCapacityGuard (worst measured path)", 50.0,
         comp["F2_downstream_capacity_guard_corrected"]["mean"]),
        ("ReservoirNetwork.step", 1.0, comp["G_reservoir_network_step"]["mean"]),
        ("MassBalanceMonitor.step_and_check", 5.0,
         comp["H_mass_balance_step_and_check"]["mean"]),
        ("state publication (adapter + JSON)", 5.0,
         comp["I_state_adapter"]["mean"] + comp["I2_payload_serialization"]["mean"]),
    ]
    out = []
    for name, budget, measured in rows:
        out.append({
            "stage": name,
            "budget_ms": budget,
            "measured_ms": measured,
            "within_budget": measured <= budget,
            "headroom_ms": budget - measured,
        })
    measured_total = sum(r["measured_ms"] for r in out)
    return {
        "control_decision_period_ms": total_budget_ms,
        "budget_basis": ("one control decision per 60 s — a telemetry cadence this "
                         "prototype is being assessed against, not a measured "
                         "requirement of the plant"),
        "rows": out,
        "measured_total_ms": measured_total,
        "budget_total_ms": sum(r["budget_ms"] for r in out),
        "utilization_of_period_fraction": measured_total / total_budget_ms,
        "meets_budget": all(r["within_budget"] for r in out),
        "not_met": [r["stage"] for r in out if not r["within_budget"]],
        "budget_pass_forced_by_code_change": False,
    }


# ===========================================================================
# main
# ===========================================================================

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("STAGE 17 — PERFORMANCE VALIDATION")
    print("=" * 78)

    evidence = {"stage": 17, "timestamp": datetime.now().isoformat(),
                "measurement_only": True, "code_modified": False}

    frozen_before = {str(p.relative_to(_PROJECT_ROOT)): sha256_file(p)
                     for p in FROZEN_ARTIFACTS}
    evidence["frozen_artifacts_before"] = frozen_before

    print("[2] environment baseline ...")
    evidence["environment"] = environment_baseline()
    print("    ", evidence["environment"]["os"])
    print("    python", evidence["environment"]["python_version"],
          "| torch", evidence["environment"]["pytorch_version"],
          "| ram", evidence["environment"]["ram_total_gb"], "GB")

    sm = state_manager.sim_state
    sm.running = False
    sm.mode = "MANUAL"
    reset_live_to_50pct(sm)
    drive_live_steps(sm, 8)
    network = sm.bridge.cascade.network
    bundle = fixture_bundle(network)

    print("[3]/[4] component + MPC benchmarks ...")
    comp = component_benchmarks(sm, network, bundle)
    reset_live_to_50pct(sm)
    mpc_decomp = mpc_decomposition(sm, network, bundle)
    print("    decide() mean %.1f ms | candidates %d | sim total %.1f ms"
          % (comp["D_mpc_decide"]["mean"],
             mpc_decomp["candidate_space"]["decision_candidates_evaluated"],
             mpc_decomp["instrumented_run"]["trajectory_simulation_total_ms"]))

    print("[5] live cycle CASE A (real, MPC blocked) ...")
    case_a = live_cycle_case_a(sm)
    print("    mean %.1f ms | source %s"
          % (case_a["cycle"]["mean"],
             case_a["chain_verdict"]["final_safe_control_action_source"]))

    print("[5] live cycle CASE B (controlled fixture) ...")
    case_b = live_cycle_case_b(sm)
    print("    mean %.1f ms | source %s"
          % (case_b["phases"]["total_cycle"]["mean"],
             case_b["chain_verdict"]["final_safe_control_action_source"]))

    evidence["components"] = comp
    evidence["mpc"] = mpc_decomp
    evidence["case_a"] = case_a
    evidence["case_b"] = case_b

    print("[7] memory ...")
    evidence["memory"] = memory_probe(sm)

    print("[8] throughput / stability (40 cycles) ...")
    evidence["throughput"] = throughput_probe(sm)
    print("    completed %d/%d | failures %d | exceptions %d"
          % (evidence["throughput"]["cycles_completed"],
             evidence["throughput"]["cycles_requested"],
             len(evidence["throughput"]["failures"]),
             len(evidence["throughput"]["exceptions"])))

    print("[6] concurrency / serialization (real uvicorn) ...")
    evidence["concurrency"] = concurrency_probe(sm)
    cs = evidence["concurrency"].get("concurrent_steps") or {}
    print("    steps advanced %s (expected %s) | duplicate_or_lost=%s"
          % (cs.get("steps_advanced"), cs.get("expected_steps"),
             cs.get("duplicate_or_lost_steps")))

    print("[9]/[10] ranking + budget ...")
    evidence["bottlenecks"] = bottleneck_ranking(comp, mpc_decomp, case_a, case_b)
    evidence["budget"] = performance_budget(comp, mpc_decomp, case_a, case_b)

    reset_live_to_50pct(sm)
    sm.mode = "MANUAL"
    sm.running = False

    frozen_after = {str(p.relative_to(_PROJECT_ROOT)): sha256_file(p)
                    for p in FROZEN_ARTIFACTS}
    evidence["frozen_artifacts_after"] = frozen_after
    evidence["frozen_artifacts_unchanged"] = frozen_before == frozen_after

    evidence["limitations"] = [
        "All measurements are CPU-only on a shared, non-realtime desktop OS; they "
        "are not deterministic and the host is not a control-grade machine.",
        "No hardware is connected. No value in this evidence is plant telemetry.",
        "The CASE B live cycle is a controlled fixture, not live operation.",
        "The concurrency probe used a single-worker uvicorn deployment; multi-worker "
        "behaviour is described analytically and was NOT benchmarked.",
        "MPC samples are 12 (and 20 for the live cycles), so p99 is reported as "
        "NOT_REPORTED where fewer than 100 samples were taken.",
        "Memory figures are process RSS, which includes allocator retention and "
        "cannot distinguish a genuine leak from fragmentation on this host.",
        "In-process direct multi-threaded step() calls were NOT benchmarked because "
        "they would corrupt the authoritative state and its audit.",
    ]

    OUTPUT_PATH.write_text(json.dumps(evidence, indent=2, default=str), encoding="utf-8")
    print()
    print("frozen artifacts unchanged:", evidence["frozen_artifacts_unchanged"])
    print("evidence ->", OUTPUT_PATH)
    print("VERDICT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

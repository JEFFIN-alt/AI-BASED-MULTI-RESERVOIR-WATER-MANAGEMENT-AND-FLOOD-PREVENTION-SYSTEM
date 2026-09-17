"""
Stage 12 — Authoritative Digital Twin state: evidence.

Produces the machine-checkable evidence behind
``results/phase15_stage12_authoritative_twin_state/PHASE_15_STAGE12_REPORT.md``:

  1. the ONE authoritative live simulation owner;
  2. the state flow (owner -> physics -> mass balance -> adapter -> transports);
  3. REST / WebSocket payload agreement + the authoritative state identity;
  4. the command round-trip (command -> backend -> broadcast -> state);
  5. the frontend audit (what the browser no longer computes/injects);
  6. protected/frozen artifact status;
  7. performance of the presentation adapter.

READ-ONLY with respect to the physics, the models and the protected artifacts.

OUTPUT
    results/phase15_stage12_authoritative_twin_state/stage12_state_authority_evidence.json
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from src.dashboard.api import routes, state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin  # noqa: E402

OUTPUT_DIR = _PROJECT_ROOT / "results" / "phase15_stage12_authoritative_twin_state"
OUTPUT_PATH = OUTPUT_DIR / "stage12_state_authority_evidence.json"

WEB_INDEX = _PROJECT_ROOT / "src" / "dashboard" / "web" / "index.html"
MIRROR = _PROJECT_ROOT / "src" / "dashboard" / "twin_component" / "reservoir_twin.html"
API_JS = _PROJECT_ROOT / "src" / "dashboard" / "web" / "api.js"
MANIFEST_PATH = _PROJECT_ROOT / "results" / "phase15_v3_validation" / "v3_integrity_check.json"
PROTECTED_DIR = _PROJECT_ROOT / "results" / "phase15_v3_validation"
FROZEN_ARTIFACTS = [
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "best_model.pt",
    _PROJECT_ROOT / "models" / "lstm_pytorch_v3_logtarget" / "log_target_scaler.pkl",
    _PROJECT_ROOT / "results" / "lstm_pytorch_v3_logtarget" / "test_predictions_original_units.csv",
]
FROZEN_PHYSICS = [
    _PROJECT_ROOT / "src" / "network_env" / "reservoir_network.py",
    _PROJECT_ROOT / "src" / "controller" / "mpc_controller.py",
    _PROJECT_ROOT / "src" / "controller" / "safety.py",
    _PROJECT_ROOT / "src" / "controller" / "downstream_capacity_guard.py",
    _PROJECT_ROOT / "src" / "controller" / "live_mpc_orchestrator.py",
]

#: Patterns the browser must NOT contain in executable code (Stage 12 removals).
FRONTEND_FORBIDDEN = (
    "const INITIAL_STATE = {",
    "150 m³/s",
    "ds < 120",
    "clamp01(ds / 150)",
    "d.inflow - d.release",
    "d4.inflow - d.release",
    "d4.inflow - d4.release",
    "0.14 * Math.sin",
    "window.__twin = ",
    "streamlit:render",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def state_owners() -> list:
    """Every module that constructs the ONE live simulation instance."""
    hits = []
    for path in (_PROJECT_ROOT / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^sim_state = GlobalSimulationState\(\)", text, re.M):
            hits.append(str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/"))
    return hits


def frontend_audit() -> dict:
    index = WEB_INDEX.read_text(encoding="utf-8")
    mirror = MIRROR.read_text(encoding="utf-8")
    js = API_JS.read_text(encoding="utf-8")
    return {
        "index_html_forbidden_patterns_present": {
            p: (p in index) for p in FRONTEND_FORBIDDEN
        },
        "mirror_html_forbidden_patterns_present": {
            p: (p in mirror) for p in FRONTEND_FORBIDDEN
            if p not in ("streamlit:render", "window.__twin = ")
        },
        "index_html_consumes_backend_blocks": {
            key: (key in index) for key in (
                "state.downstream", "state.forecast_summary", "state.state_identity",
                "state.simulation", "state.mass_balance", "state.control",
                "src.trend", "d4.trend",
            )
        },
        "console_state_push_handle": ("window.__twin = " in index) or ("window.__twin = " in mirror),
        "read_only_ready_flag": ("window.__twinReady = true" in index),
        "api_js_is_commands_only": {
            "endpoints": sorted({
                m for m in re.findall(r"post\(`(/[a-z/{}$._\w-]+)`", js)
            }),
            "pushes_state": any(w in js for w in ("reservoirs", "mass_balance", "storage_mcm")),
            "websocket_feed": "/ws/state" in js,
        },
        "mirror_accepts_only_authoritative_payloads": (
            "hasAuthoritativeMarker" in mirror
            and "if (!hasAuthoritativeMarker) return;" in mirror
        ),
        "mirror_has_no_write_back": not any(
            w in mirror for w in ("fetch(", "XMLHttpRequest", "sendBeacon")
        ),
    }


#: Top-level blocks that both transports must deliver identically.
AGREEMENT_KEYS = (
    "reservoirs", "cascade", "control", "mass_balance", "forecast_provenance",
    "state_identity", "downstream", "forecast_summary", "storm", "simulation",
    "hardware_status",
)


def field_agreement(a: dict, b: dict) -> dict:
    """Field-by-field equality of two payloads of the SAME authoritative state.

    ``simulation_time`` is a wall-clock stamp generated per payload and is
    deliberately excluded; the identity is ``state_identity``.
    """
    return {key: (a.get(key) == b.get(key)) for key in AGREEMENT_KEYS}


def live_flow_probe() -> dict:
    client = TestClient(app)
    client.post("/api/simulation/pause")
    client.post("/api/simulation/reset")
    for i in (1, 2, 3, 4):
        client.post(f"/api/gate/reservoir_{i}", json={"value": 0.0})
    client.post("/api/controller/mode", json={"mode": "MANUAL"})

    before = client.get("/api/state").json()
    command_log = []

    # --- command -> backend -> broadcast -> state (STEP) ---
    client.post("/api/simulation/step")
    after_step = client.get("/api/state").json()
    command_log.append({
        "command": "POST /api/simulation/step",
        "backend_network_timestep_before": before["state_identity"]["network_timestep"],
        "backend_network_timestep_after": after_step["state_identity"]["network_timestep"],
        "state_id_after": after_step["state_identity"]["state_id"],
    })

    # --- REST and WebSocket deliver the SAME authoritative state ---
    # Both transports are read for one and the same timestep. Comparing across a
    # step would compare two different authoritative states (reservoirs,
    # mass-balance counters and the state identity all advance), which is a
    # property of the probe, not of the system.
    with client.websocket_connect("/ws/state") as ws:
        ws_state = ws.receive_json()                 # initial authoritative push
        rest_state = client.get("/api/state").json()  # same timestep, no step
        agreement = field_agreement(rest_state, ws_state)

        # --- a real WebSocket client sees the state the command produced ---
        client.post("/api/simulation/step")
        pushed = ws.receive_json()
        rest_after_broadcast = client.get("/api/state").json()
        pushed_agreement = field_agreement(pushed, rest_after_broadcast)

        agreement["pushed_timestep_is_backend_timestep"] = (
            pushed["state_identity"]["network_timestep"]
            == state_manager.sim_state.bridge.cascade.network.timestep
        )
        rest_state = rest_after_broadcast

    # --- PLAY / PAUSE round-trip through the backend ---
    client.post("/api/simulation/play")
    play_state = client.get("/api/state").json()["simulation"]
    client.post("/api/simulation/pause")
    pause_state = client.get("/api/state").json()["simulation"]

    client.post("/api/simulation/reset")
    reset_state = client.get("/api/state").json()

    return {
        "single_owner_instance_count": state_manager.authoritative_instance_count(),
        "owner_is_the_routes_owner": routes.sim_state is state_manager.sim_state,
        "owner_module_declaration_sites": state_owners(),
        "commands": command_log,
        "rest_websocket_agreement": agreement,
        "pushed_payload_equals_rest_after_step": pushed_agreement,
        "play_state": play_state,
        "pause_state": pause_state,
        "reset_state_identity": reset_state["state_identity"],
        "reset_mass_balance": reset_state["mass_balance"]["status"],
        "mass_balance_after_step": after_step["mass_balance"]["status"],
        "mass_balance_reservoirs_checked": after_step["mass_balance"]["reservoirs_checked"],
        "idukki": {
            "repository_name": rest_state["reservoirs"]["reservoir_4"]["repository_name"],
            "node_id": rest_state["reservoirs"]["reservoir_4"]["node_id"],
            "is_terminal": rest_state["reservoirs"]["reservoir_4"]["is_terminal"],
        },
        "downstream_block": rest_state["downstream"],
        "state_identity": rest_state["state_identity"],
    }


def adapter_performance() -> dict:
    sim_state = state_manager.sim_state.bridge.get_state({})
    sim_state["control"] = state_manager.sim_state.mpc_orchestrator.status_dict()
    sim_state["mass_balance"] = state_manager.sim_state.bridge.mass_balance_diagnostic()
    sim_state["state_identity"] = {"sim_step_index": 1, "network_timestep": 1,
                                   "state_id": "step1-t1", "source": "s"}
    sim_state["simulation"] = {"running": False, "speed": 1.0, "source": "s"}
    iterations = 300
    adapt_state_for_twin(sim_state, "MANUAL", 0.0)
    t0 = time.perf_counter()
    for _ in range(iterations):
        adapt_state_for_twin(sim_state, "MANUAL", 0.0)
    return {
        "iterations": iterations,
        "adapter_ms_per_call": (time.perf_counter() - t0) * 1000.0 / iterations,
        "unit": "ms",
        "note": "presentation transform only; MPC decide() is ~660 ms",
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("STAGE 12 — AUTHORITATIVE DIGITAL TWIN STATE — EVIDENCE")
    print("=" * 74)

    frozen_before = {str(p.relative_to(_PROJECT_ROOT)): sha256_file(p) for p in FROZEN_ARTIFACTS}
    protected_before = {
        str(p.relative_to(PROTECTED_DIR)).replace("\\", "/"): sha256_file(p)
        for p in sorted(PROTECTED_DIR.rglob("*")) if p.is_file()
    }

    owners = state_owners()
    print(f"\n[1] Authoritative owner : {owners} "
          f"(instances={state_manager.authoritative_instance_count()})")

    audit = frontend_audit()
    left = [k for k, v in audit["index_html_forbidden_patterns_present"].items() if v]
    print(f"[2] Frontend audit      : forbidden patterns present in index.html: {left or 'NONE'}")
    print(f"    console push handle : {audit['console_state_push_handle']}")
    print(f"    consumes backend    : {audit['index_html_consumes_backend_blocks']}")

    flow = live_flow_probe()
    print(f"\n[3] Command round-trip  : {flow['commands']}")
    print(f"    REST/WS agreement   : {flow['rest_websocket_agreement']}")
    print(f"    pushed == REST      : {flow['pushed_payload_equals_rest_after_step']}")
    print(f"    PLAY/PAUSE state    : {flow['play_state']} -> {flow['pause_state']}")
    print(f"    mass balance        : {flow['mass_balance_after_step']} "
          f"({flow['mass_balance_reservoirs_checked']} reservoirs)")
    print(f"    state identity      : {flow['state_identity']}")
    print(f"    downstream block    : {flow['downstream_block']}")

    perf = adapter_performance()
    print(f"\n[4] Adapter performance : {perf['adapter_ms_per_call']:.4f} ms/call")

    frozen_after = {str(p.relative_to(_PROJECT_ROOT)): sha256_file(p) for p in FROZEN_ARTIFACTS}
    protected_after = {
        str(p.relative_to(PROTECTED_DIR)).replace("\\", "/"): sha256_file(p)
        for p in sorted(PROTECTED_DIR.rglob("*")) if p.is_file()
    }
    frozen_unchanged = frozen_before == frozen_after
    protected_unchanged = protected_before == protected_after

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest_match = {}
    for rel, record in manifest["pre_validation"].items():
        path = _PROJECT_ROOT / rel
        raw = sha256_file(path)
        lf = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        manifest_match[rel.replace("\\", "/")] = record["sha256"] in {raw, lf}

    physics_intact = {}
    for path in FROZEN_PHYSICS:
        text = path.read_text(encoding="utf-8")
        physics_intact[str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/")] = not any(
            token in text for token in ("state_identity", "twin_component",
                                        "forecast_summary", "state_adapter")
        )

    print(f"\n[5] Frozen artifacts unchanged   : {frozen_unchanged}")
    print(f"    Protected Phase 15.3 unchanged: {protected_unchanged}")
    print(f"    Manifest match               : {manifest_match}")
    print(f"    Frozen physics intact        : {physics_intact}")

    evidence = {
        "stage": 12,
        "title": "Authoritative Digital Twin state",
        "timestamp": datetime.now().isoformat(),
        "architecture": [
            "authoritative backend simulation (GlobalSimulationState)",
            "ReservoirNetwork.step() audited by MassBalanceMonitor",
            "StateAdapter (adapt_state_for_twin)",
            "FastAPI /api/state + /ws/state",
            "Three.js Digital Twin (DISPLAY ONLY)",
            "REST commands -> backend -> broadcast -> display",
        ],
        "authoritative_owner": {
            "module": "src/dashboard/api/state_manager.py",
            "object": "sim_state (GlobalSimulationState)",
            "instance_count": state_manager.authoritative_instance_count(),
            "declaration_sites": owners,
        },
        "frontend_audit": audit,
        "live_flow_probe": flow,
        "performance": perf,
        "frozen_artifact_sha256_before": frozen_before,
        "frozen_artifact_sha256_after": frozen_after,
        "frozen_artifacts_unchanged": frozen_unchanged,
        "protected_phase15_3_unchanged": protected_unchanged,
        "phase15_3_manifest_match": manifest_match,
        "frozen_physics_untouched": physics_intact,
        "hardware_connected": False,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, default=str)

    print(f"\nEvidence written: {OUTPUT_PATH.relative_to(_PROJECT_ROOT)}")
    ok = (not left and not audit["console_state_push_handle"] and frozen_unchanged
          and protected_unchanged and all(manifest_match.values())
          and all(physics_intact.values())
          and all(flow["rest_websocket_agreement"].values())
          and all(flow["pushed_payload_equals_rest_after_step"].values()))
    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

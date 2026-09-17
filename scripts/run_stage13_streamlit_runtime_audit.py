"""
Stage 13 — Streamlit read-only / command-proxy RUNTIME audit.

This is the Phase 12 behavioural test. It is NOT a mock: it

  1. starts the REAL authoritative FastAPI backend (uvicorn) on a real port;
  2. runs the REAL Streamlit script (``src/dashboard/app.py``) through
     Streamlit's own ``AppTest`` runner, pointed at that backend;
  3. clicks the real command-proxy widgets and measures the REAL authoritative
     state before and after, over HTTP and through the authoritative singleton.

What it proves (per the Stage 13 requirement list):

  * Streamlit loads and obtains state from the authoritative backend.
  * All four reservoirs are displayed, with the backend's own names.
  * The displayed state identity is the backend's state identity.
  * STEP through Streamlit advances the authoritative backend by exactly one
    timestep, and the page then displays the NEW backend state.
  * A gate command through Streamlit reaches the authoritative simulation
    (Reservoir D / Idukki included) without advancing physics.
  * Mass-balance / control / safety / downstream blocks shown are the backend's.
  * Running the script creates ZERO extra simulation instances.

READ-ONLY with respect to physics, models and frozen artifacts.

OUTPUT
    results/phase15_stage13_streamlit_command_proxy/
        stage13_streamlit_runtime_evidence.json
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "dashboard"))

#: The authoritative state flash messages contain "→"; a Windows console is
#: cp1252 by default, so make the audit's own output encodable rather than
#: degrading the evidence text.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
    pass

#: host/port of the audit's own authoritative backend. The Streamlit script
#: reads these through the documented environment overrides, so the page is
#: genuinely pointed at THIS backend over real HTTP.
HOST = "127.0.0.1"
PORT = int(os.environ.get("STAGE13_AUDIT_PORT", "8031"))
BASE = f"http://{HOST}:{PORT}"
os.environ["AQUAFLOW_TWIN_HOST"] = HOST
os.environ["AQUAFLOW_TWIN_PORT"] = str(PORT)

APP_PATH = _PROJECT_ROOT / "src" / "dashboard" / "app.py"
OUTPUT_DIR = _PROJECT_ROOT / "results" / "phase15_stage13_streamlit_command_proxy"
OUTPUT_PATH = OUTPUT_DIR / "stage13_streamlit_runtime_evidence.json"


# ---------------------------------------------------------------------------
# HTTP helpers (the audit's own operator, talking to the authoritative backend)
# ---------------------------------------------------------------------------

def request(method: str, path: str, payload=None, timeout: float = 30.0):
    """``(status_code, body)`` — body is decoded JSON, or a diagnostic dict."""
    url = f"{BASE}{path}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return resp.status, (json.loads(raw) if raw.strip() else {})
    except urllib.error.HTTPError as exc:
        return exc.code, {"detail": exc.reason}
    except (urllib.error.URLError, OSError) as exc:
        return 0, {"detail": str(exc)}


def get_state() -> dict:
    return request("GET", "/api/state")[1]


def post(path: str, payload=None):
    return request("POST", path, payload)


# ---------------------------------------------------------------------------
# Backend lifecycle
# ---------------------------------------------------------------------------

def start_backend():
    import uvicorn

    from src.dashboard.api.app import app as fastapi_app

    server = uvicorn.Server(
        uvicorn.Config(fastapi_app, host=HOST, port=PORT, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True, name="stage13-uvicorn")
    thread.start()
    return server, thread


def wait_until_ready(timeout: float = 90.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, body = request("GET", "/api/state", timeout=3.0)
        if status == 200 and isinstance(body, dict) and "reservoirs" in body:
            return True
        time.sleep(0.4)
    return False


# ---------------------------------------------------------------------------
# AppTest helpers
# ---------------------------------------------------------------------------

def find_widget(widget_list, key: str):
    for widget in widget_list:
        if getattr(widget, "key", None) == key:
            return widget
    available = [getattr(w, "key", None) for w in widget_list]
    raise AssertionError(f"widget {key!r} not found; present: {available}")


def metrics_of(at) -> dict:
    return {m.label: m.value for m in at.metric}


def exceptions_of(at) -> list:
    out = []
    for exc in at.exception:
        out.append(str(getattr(exc, "value", None) or getattr(exc, "message", "") or exc))
    return out


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("STAGE 13 — STREAMLIT READ-ONLY / COMMAND-PROXY — RUNTIME AUDIT")
    print("=" * 74)

    server, _thread = start_backend()
    if not wait_until_ready():
        print("FAILED: the authoritative backend never became ready.")
        return 1
    print(f"\n[1] Authoritative backend  : {BASE} (real uvicorn process, real HTTP)")

    # Deterministic starting point: paused, MANUAL, reset physics clock.
    post("/api/simulation/pause")
    post("/api/simulation/reset")
    post("/api/controller/mode", {"mode": "MANUAL"})

    from streamlit.testing.v1 import AppTest

    from src.dashboard.api import state_manager

    instances_before = state_manager.authoritative_instance_count()
    backend_before = get_state()

    # ------------------------------------------------------------------
    # Run the REAL Streamlit script
    # ------------------------------------------------------------------
    at = AppTest.from_file(str(APP_PATH), default_timeout=300)
    at.run()

    instances_after = state_manager.authoritative_instance_count()
    exceptions = exceptions_of(at)
    metrics = metrics_of(at)

    print(f"[2] Streamlit script       : ran ({len(at.metric)} metrics rendered)")
    print(f"    exceptions            : {exceptions or 'NONE'}")
    print(f"    authoritative instances: before={instances_before} after={instances_after}")

    names = [
        entry.get("name")
        for entry in backend_before.get("cascade", {}).get("reservoirs", [])
    ]
    reservoir_cards = {n: (f"{n} · storage" in metrics) for n in names}
    print(f"[3] Four-reservoir cards   : {reservoir_cards}")

    identity_shown = metrics.get("State #")
    connected = [s.value for s in at.success]
    print(f"    state identity shown  : {identity_shown!r} "
          f"(backend {backend_before['state_identity']['state_id']!r})")
    print(f"    backend connection    : {connected}")

    # ------------------------------------------------------------------
    # STEP through Streamlit
    # ------------------------------------------------------------------
    before_step = get_state()["state_identity"]
    find_widget(at.button, "cmd_step").click()
    at.run()
    after_step = get_state()["state_identity"]
    metrics_after_step = metrics_of(at)

    step_evidence = {
        "backend_network_timestep_before": before_step["network_timestep"],
        "backend_network_timestep_after": after_step["network_timestep"],
        "advanced_exactly_one": (
            after_step["network_timestep"] - before_step["network_timestep"] == 1
        ),
        "displayed_state_id": metrics_after_step.get("State #"),
        "backend_state_id": after_step["state_id"],
        "display_matches_backend": metrics_after_step.get("State #") == after_step["state_id"],
        "flash": [s.value for s in at.success],
    }
    print(f"\n[4] STEP through Streamlit : {step_evidence}")

    # ------------------------------------------------------------------
    # Mass balance visible and equal to the backend's verdict
    # ------------------------------------------------------------------
    mb_backend = get_state()["mass_balance"]
    mb_evidence = {
        "backend_status": mb_backend.get("status"),
        "displayed_status": metrics_after_step.get("Mass balance"),
        "display_matches_backend": (
            metrics_after_step.get("Mass balance") == mb_backend.get("status")
        ),
        "backend_residual": mb_backend.get("residual"),
        "backend_reservoirs_checked": mb_backend.get("reservoirs_checked"),
        "backend_reservoirs_expected": mb_backend.get("reservoirs_expected"),
    }
    print(f"[5] Mass balance           : {mb_evidence}")

    # ------------------------------------------------------------------
    # Gate command through Streamlit, targeting Idukki (Reservoir D)
    # ------------------------------------------------------------------
    timestep_before_gate = get_state()["state_identity"]["network_timestep"]
    gate_before = get_state()["reservoirs"]["reservoir_4"]["gate"]

    find_widget(at.slider, "cmd_gate_reservoir_4").set_value(20.0)
    at.run()
    find_widget(at.button, "cmd_gates_apply").click()
    at.run()

    gate_state = get_state()
    gate_after = gate_state["reservoirs"]["reservoir_4"]["gate"]
    authoritative_gate = state_manager.sim_state.manual_gates["Virtual Reservoir D"]
    gate_evidence = {
        "target": "reservoir_4 / Virtual Reservoir D (Idukki)",
        "displayed_gate_before": gate_before,
        "displayed_gate_after": gate_after,
        "authoritative_manual_gate_after": authoritative_gate,
        "reached_authoritative_singleton": abs(authoritative_gate - 20.0) < 1e-9,
        "physics_clock_unchanged_by_gate_command": (
            gate_state["state_identity"]["network_timestep"] == timestep_before_gate
        ),
        "four_gate_sliders_present": sorted(
            w.key for w in at.slider if str(getattr(w, "key", "")).startswith("cmd_gate_")
        ),
        "flash": [s.value for s in at.success] + [e.value for e in at.error],
    }
    print(f"\n[6] Gate command (Idukki)  : {gate_evidence}")

    # ------------------------------------------------------------------
    # Provenance blocks the page shows must be the backend's own verdicts
    # ------------------------------------------------------------------
    state_now = get_state()
    provenance = {
        "control_controller_type": state_now["control"].get("controller_type"),
        "control_controller_status": state_now["control"].get("controller_status"),
        "safety_layer_status": state_now["control"].get("safety_layer_status"),
        "downstream_status": state_now["control"].get("downstream_status"),
        "downstream_block_status": state_now["downstream"].get("status"),
        "downstream_capacity_m3_s": state_now["downstream"].get("capacity_m3_s"),
        "forecast_provenance_live_validated": (
            state_now["forecast_provenance"].get("live_forecasts_are_validated")
        ),
        "forecast_summary_status": state_now["forecast_summary"].get("status"),
        "hardware_connected": state_now["mass_balance"].get("hardware_connected"),
        "displayed_safety_layer": metrics_of(at).get("Safety layer"),
        "displayed_downstream_guard": metrics_of(at).get("Downstream guard"),
    }
    provenance["display_matches_backend"] = (
        provenance["displayed_safety_layer"] == provenance["safety_layer_status"]
        and provenance["displayed_downstream_guard"] == provenance["downstream_status"]
    )
    print(f"\n[7] Provenance blocks      : {provenance}")

    # ------------------------------------------------------------------
    # No simulation object was created by the page
    # ------------------------------------------------------------------
    owner_evidence = {
        "authoritative_instance_count_after_running_streamlit": (
            state_manager.authoritative_instance_count()
        ),
        "expected": 1,
        "streamlit_cannot_create_one": (
            state_manager.authoritative_instance_count() == 1
        ),
    }
    print(f"\n[8] Simulation owners      : {owner_evidence}")

    # ------------------------------------------------------------------
    # Performance of the read + command proxy (no physics work added)
    # ------------------------------------------------------------------
    reads = 20
    t0 = time.perf_counter()
    for _ in range(reads):
        get_state()
    read_ms = (time.perf_counter() - t0) * 1000.0 / reads

    commands = 10
    t0 = time.perf_counter()
    for _ in range(commands):
        post("/api/simulation/step")
    command_ms = (time.perf_counter() - t0) * 1000.0 / commands

    t0 = time.perf_counter()
    at.run()
    rerender_ms = (time.perf_counter() - t0) * 1000.0

    performance = {
        "authoritative_state_read_ms": read_ms,
        "proxied_command_round_trip_ms": command_ms,
        "full_streamlit_script_rerun_ms": rerender_ms,
        "unit": "ms",
        "note": (
            "the page adds no physics work: a proxied command costs one HTTP "
            "round trip, and a state read costs one GET. A control cycle is "
            "~660 ms (MPC decide()), so the proxy is overhead-free relative to it."
        ),
    }
    print(f"\n[9] Performance           : {performance}")

    checks = {
        "streamlit_script_ran_without_exception": not exceptions,
        "streamlit_connected_to_backend": bool(connected),
        "state_identity_from_backend": identity_shown == backend_before["state_identity"]["state_id"],
        "all_four_reservoirs_displayed": all(reservoir_cards.values()) and len(reservoir_cards) == 4,
        "step_advanced_backend_exactly_one": step_evidence["advanced_exactly_one"],
        "display_followed_backend_after_step": step_evidence["display_matches_backend"],
        "mass_balance_displayed_from_backend": mb_evidence["display_matches_backend"],
        "gate_command_reached_authoritative_singleton": gate_evidence["reached_authoritative_singleton"],
        "gate_command_did_not_advance_physics": gate_evidence["physics_clock_unchanged_by_gate_command"],
        "all_four_gate_sliders_present": len(gate_evidence["four_gate_sliders_present"]) == 4,
        "no_extra_simulation_instance_created": owner_evidence["streamlit_cannot_create_one"],
        "provenance_display_matches_backend": provenance["display_matches_backend"],
    }

    evidence = {
        "stage": 13,
        "title": "Streamlit read-only / command-proxy runtime evidence",
        "timestamp": datetime.now().isoformat(),
        "method": (
            "real uvicorn authoritative backend + real Streamlit script executed "
            "by streamlit.testing.v1.AppTest + real widget clicks + real HTTP "
            "measurements of the authoritative state"
        ),
        "backend_url": BASE,
        "streamlit_app": "src/dashboard/app.py",
        "streamlit_version": __import__("streamlit").__version__,
        "checks": checks,
        "step": step_evidence,
        "mass_balance": mb_evidence,
        "gate_command": gate_evidence,
        "provenance": provenance,
        "owners": owner_evidence,
        "performance": performance,
        "hardware_connected": False,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(evidence, fh, indent=2, default=str)

    print("\n" + "-" * 74)
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nEvidence written: {OUTPUT_PATH.relative_to(_PROJECT_ROOT)}")
    verdict = all(checks.values())
    print("VERDICT:", "PASS" if verdict else "FAIL")

    server.should_exit = True
    time.sleep(0.5)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

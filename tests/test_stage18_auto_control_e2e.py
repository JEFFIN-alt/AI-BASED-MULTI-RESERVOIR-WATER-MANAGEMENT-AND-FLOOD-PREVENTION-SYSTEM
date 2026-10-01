"""
STAGE 18 — AUTO / AI CONTROL: end-to-end verification of the live control path.

What this suite proves, against the REAL authoritative backend (no mocks):

1.  The operator can arm AUTO from the API and the backend reports a truthful
    AUTO state for every phase of the run (AUTO_DISABLED -> AUTO_READY ->
    AUTO_CONTROL_APPLIED / AUTO_BLOCKED).
2.  With the ``VALIDATED_REPLAY`` forecast source the Stage 7 provenance gate is
    genuinely satisfied — every controlled reservoir declares ``VALIDATED`` —
    so the MPC, the SafetyLayer and the DownstreamCapacityGuard all actually
    run, and the action they produce moves the REAL gates of the REAL
    ``ReservoirNetwork``.
3.  The AUTOMATIC action applied to the physics is the controller's
    ``FINAL_SAFE_CONTROL_ACTION``: it is re-measured here from the network, not
    taken from the payload.
4.  Every event in the backend event log corresponds to something the backend
    really did, and the gate-change events match the measured gate deltas.
5.  The WebSocket feed carries the same AUTO state, forecast-source selection
    and event log as REST, under the same authoritative state id.
6.  The honest default is NOT weakened: with the ``SIMULATION`` forecast source
    the provenance gate still refuses control (``BLOCKED``), AUTO reports
    ``AUTO_BLOCKED`` and the gates are HELD bit-for-bit.
7.  MANUAL keeps full operator authority, and the AUTO -> MANUAL handover moves
    no physics.
8.  The twin page renders these backend blocks verbatim and emits no event and
    no control state of its own (static + ``node --check`` verification).
"""

import contextlib
import io
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.common import units  # noqa: E402
from src.dashboard.api import state_manager  # noqa: E402
from src.dashboard.api.app import app  # noqa: E402

client = TestClient(app)

WEB_DIR = _PROJECT_ROOT / "src" / "dashboard" / "web"
TWIN_PATH = WEB_DIR / "index.html"

NODES = [
    "Virtual Reservoir A",
    "Virtual Reservoir B",
    "Virtual Reservoir C",
    "Virtual Reservoir D",
]

SIMULATION = "SIMULATION"
VALIDATED_REPLAY = "VALIDATED_REPLAY"

#: Gate tolerance — the same 1e-9 the backend uses for its own action check.
GATE_TOL = 1e-9


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _quiet(fn, *args, **kwargs):
    """Run a command with the physics engine's stdout suppressed."""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _network():
    """The ONE authoritative reservoir network."""
    return state_manager.sim_state.bridge.cascade.network


def _gate_pct() -> dict:
    """Re-measure the REAL gate positions (percent) from the physics itself."""
    network = _network()
    return {
        str(nid): units.gate_fraction_to_percent(network.nodes[nid].state.gate_position)
        for nid in network.processing_order
    }


def _state() -> dict:
    """The adapted authoritative payload the twin/API serves."""
    response = client.get("/api/state")
    assert response.status_code == 200
    return response.json()


def _arm(mode: str, source: str | None = None):
    body = {"mode": mode} if source is None else {"mode": mode, "source": source}
    return _quiet(client.post, "/api/controller/mode", json=body)


def _step():
    response = _quiet(client.post, "/api/simulation/step")
    assert response.status_code == 200, response.text
    return response


def _pause():
    response = _quiet(client.post, "/api/simulation/pause")
    assert response.status_code == 200, response.text


def _reset():
    response = _quiet(client.post, "/api/simulation/reset")
    assert response.status_code == 200, response.text


def _moved_nodes(before: dict, after: dict) -> dict:
    """Nodes whose REAL gate position actually changed, with their deltas."""
    return {
        nid: (before[nid], after[nid])
        for nid in NODES
        if abs(after[nid] - before[nid]) > GATE_TOL
    }


def _kinds(events) -> list:
    return [str(e.get("kind")) for e in events or []]


def _count(events, kind: str) -> int:
    """How many events of ``kind`` are in a backend event log."""
    return _kinds(events).count(kind)


def _assert_transitions_match_physics(transitions, moved):
    """
    The reported gate transitions must be exactly the measured gate deltas.

    A transition the physics did not make, or a real movement that was not
    reported, both fail here.
    """
    reported = {str(t["node"]): t for t in transitions}
    assert set(reported) == set(moved), (reported, moved)
    for nid, (previous, new) in moved.items():
        assert reported[nid]["previous_pct"] == pytest.approx(previous, abs=GATE_TOL)
        assert reported[nid]["new_pct"] == pytest.approx(new, abs=GATE_TOL)
        if "delta_pct" in reported[nid]:
            assert reported[nid]["delta_pct"] == pytest.approx(
                new - previous, abs=GATE_TOL
            )


def _assert_event_log_is_self_consistent(events):
    """seq is strictly increasing and unique; every entry is a real record."""
    seqs = [e["seq"] for e in events]
    assert seqs == sorted(seqs)
    assert len(seqs) == len(set(seqs))
    for event in events:
        assert str(event["kind"]).strip()
        assert str(event["text"]).strip()
        assert str(event["timestamp"]).strip()


def _assert_actions_are_physically_true(actions, applied, before: dict, after: dict):
    """
    Every per-reservoir action in the AUTO block must be anchored in physics.

    ``previous_pct`` is the gate measured BEFORE the step, ``final_pct`` is the
    gate measured AFTER it, and the applied action equals the authoritative
    ``control.final_safe_control_action_pct`` for all four reservoirs.
    """
    by_node = {str(a["node"]): a for a in actions}
    assert set(by_node) == set(NODES)
    assert set(applied) == set(NODES)
    for nid in NODES:
        assert by_node[nid]["previous_pct"] == pytest.approx(before[nid], abs=GATE_TOL)
        assert by_node[nid]["final_pct"] == pytest.approx(after[nid], abs=GATE_TOL)
        assert by_node[nid]["final_pct"] == pytest.approx(applied[nid], abs=GATE_TOL)


def _module_script(html: str) -> str:
    marker = '<script type="module">'
    start = html.index(marker)
    end = html.index("</script>", start)
    return html[start + len(marker):end]


def _node_check(js: str, name: str = "twin_page.mjs"):
    node = shutil.which("node")
    if node is None:  # pragma: no cover - node is present in this environment
        pytest.skip("node is not available to parse-check the twin page")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / name
        path.write_text(js, encoding="utf-8")
        return subprocess.run([node, "--check", str(path)], capture_output=True, text=True)


@pytest.fixture(autouse=True)
def _restore_authoritative_state():
    """
    This module drives the ONE authoritative singleton. Restore the honest
    default afterwards (paused, MANUAL, SIMULATION source, fresh run) so no
    other suite inherits an armed AUTO controller or a replay forecast source.
    """
    yield
    _pause()
    _arm("MANUAL", SIMULATION)
    _reset()


@pytest.fixture
def manual_reset():
    """A fresh run, paused, in MANUAL, on the honest SIMULATION source."""
    _pause()
    _arm("MANUAL", SIMULATION)
    _reset()
    return state_manager.sim_state


# ===========================================================================
# A. RESET -> VALIDATED_REPLAY -> AUTO -> STEP -> REAL GATE MOVEMENT
# ===========================================================================

def test_auto_control_end_to_end_moves_real_gates_on_validated_forecasts(manual_reset):
    """
    The Stage 18 demo path, end to end, on the real backend.

    RESET -> arm AUTO with the VALIDATED_REPLAY source -> one authoritative
    step. The provenance gate must pass, the MPC / SafetyLayer /
    DownstreamCapacityGuard must all really run, and the resulting
    FINAL_SAFE_CONTROL_ACTION must be what the REAL network gates hold.
    """
    # ── 1. RESET baseline: a new run, MANUAL, controller idle, log cleared.
    state = _state()
    auto = state["auto_control"]
    assert auto["state"] == "AUTO_DISABLED"
    assert auto["control_mode"] == "MANUAL"
    assert auto["auto_enabled"] is False
    assert state["control"]["controller_status"] == "UNAVAILABLE"
    assert state["control"]["forecast_control_eligible"] is False
    assert state["forecast_source_selection"]["selected"] == SIMULATION
    assert state["forecast_source_selection"]["control_forecast_validated"] is False
    # The log is a NEW log: the RESET is event 1 of this run, not a leftover.
    assert _kinds(state["event_log"]) == ["reset"]
    assert state["event_log"][0]["seq"] == 1
    assert [a["final_pct"] for a in auto["actions"]] == [None] * len(NODES)

    # ── 2. Arm AUTO on the VALIDATED_REPLAY source (one command, as the UI does).
    response = _arm("AI", VALIDATED_REPLAY)
    assert response.status_code == 200
    assert response.json()["mode"] == "AI"
    assert response.json()["forecast_source"] == VALIDATED_REPLAY

    state = _state()
    auto = state["auto_control"]
    assert auto["control_mode"] == "AI"
    assert auto["auto_enabled"] is True
    assert auto["forecast_source"] == VALIDATED_REPLAY
    # Armed but not yet decided — reported honestly, not as a success.
    assert auto["state"] == "AUTO_READY"
    assert state["forecast_source_selection"]["selected"] == VALIDATED_REPLAY
    assert state["forecast_source_selection"]["control_forecast_validated"] is True
    assert _kinds(state["event_log"]) == ["reset", "forecast_source", "auto_enabled"]

    # ── 3. One authoritative step. Gates are measured from the physics itself.
    before = _gate_pct()
    _step()
    after = _gate_pct()
    state = _state()
    control = state["control"]
    auto = state["auto_control"]
    events = state["event_log"]
    moved = _moved_nodes(before, after)

    # 3a. The provenance gate genuinely passed — the replay forecasts are the
    #     frozen V3 model's predictions on its held-out REAL test split.
    assert control["forecast_control_eligible"] is True
    assert auto["forecast_control_eligible"] is True
    provenance_nodes = control["forecast_provenance"]["nodes"]
    assert set(provenance_nodes) == set(NODES)
    for nid in NODES:
        node = provenance_nodes[nid]
        assert node["declared_status"] == "VALIDATED", nid
        assert node["validated_metrics_apply"] is True, nid
        assert node["is_simulated"] is False, nid
        assert node["eligible"] is True, nid
    # The HONEST statement about the *live* path is unchanged: this is a
    # historical replay, so the live-forecast claim stays false.
    assert state["forecast_provenance"]["live_forecasts_are_validated"] is False

    # 3b. The MPC really optimised (1,296 coordinated candidate vectors).
    assert control["controller_status"] == "ACTIVE"
    assert control["mpc_status"] in ("OPTIMAL", "CORRECTED", "SAFE")
    assert control["candidates_evaluated"] > 0
    assert auto["controller_status"] == "ACTIVE"
    assert auto["mpc_status"] == control["mpc_status"]

    # 3c. The SafetyLayer validated it and the downstream guard protected the
    #     channel — reported separately, as the backend reports them.
    assert control["safety_layer_status"] in ("SAFE", "CORRECTED")
    assert control["downstream_status"] in ("PROTECTED", "CORRECTED")
    assert auto["safety_layer_status"] == control["safety_layer_status"]
    assert auto["downstream_status"] == control["downstream_status"]
    assert control["final_safe_control_action_source"] == "DOWNSTREAM_CAPACITY_GUARD"
    assert (auto["final_safe_control_action_source"]
            == control["final_safe_control_action_source"])

    # 3d. …and the gate positions the physics now holds ARE that safe action.
    applied = control["final_safe_control_action_pct"]
    assert set(applied) == set(NODES)
    for nid in NODES:
        assert after[nid] == pytest.approx(applied[nid], abs=GATE_TOL), nid

    # 3e. The AUTO block anchors every per-reservoir action in that physics.
    assert auto["control_applied"] is True
    _assert_actions_are_physically_true(auto["actions"], applied, before, after)

    # 3f. The gates really moved, and the reported deltas are the measured ones.
    assert moved, (before, after)
    assert auto["state"] == "AUTO_CONTROL_APPLIED"
    assert auto["reason"] == (
        "Control action applied to the authoritative reservoir network."
    )
    _assert_transitions_match_physics(auto["gate_transitions"], moved)

    # 3g. The backend's own mass-balance / applied-action audit agrees.
    audit = state["mass_balance"]
    assert audit["status"] == "PASS"
    assert audit["controller_action_checked"] is True
    assert audit["matches_final_safe_control_action"] is True

    # 3h. Every event corresponds to something the backend really did.
    kinds = _kinds(events)
    assert kinds[:6] == [
        "reset", "forecast_source", "auto_enabled",
        "mpc_decision", "safety_validation", "downstream_protection",
    ]
    assert kinds[6:] == ["gate_change"] * len(moved)
    _assert_event_log_is_self_consistent(events)
    by_kind = {e["kind"]: e for e in events}
    assert by_kind["mpc_decision"]["controller_status"] == "ACTIVE"
    assert (by_kind["safety_validation"]["safety_layer_status"]
            == control["safety_layer_status"])
    assert (by_kind["downstream_protection"]["downstream_status"]
            == control["downstream_status"])
    gate_events = [e for e in events if e["kind"] == "gate_change"]
    assert {e["reservoir"] for e in gate_events} == set(moved)
    for event in gate_events:
        previous, new = moved[event["reservoir"]]
        assert event["previous_pct"] == pytest.approx(previous, abs=GATE_TOL)
        assert event["new_pct"] == pytest.approx(new, abs=GATE_TOL)


def test_every_auto_step_applies_its_own_action_and_chains_from_the_last_one(manual_reset):
    """
    CONTINUITY — each AUTO step decides from the state the PREVIOUS step left.

    Three consecutive steps: the gates measured after step N must be the
    ``previous_pct`` the AUTO block reports on step N+1, so the reported
    "previous" is real history, never a value reconstructed for display.
    """
    _arm("AI", VALIDATED_REPLAY)

    previous_after = _gate_pct()
    for index in range(3):
        before = _gate_pct()
        for nid in NODES:
            assert before[nid] == pytest.approx(previous_after[nid], abs=GATE_TOL)

        _step()
        after = _gate_pct()
        state = _state()
        auto = state["auto_control"]
        control = state["control"]
        moved = _moved_nodes(before, after)

        assert control["controller_status"] == "ACTIVE", index
        assert auto["control_applied"] is True, index
        expected_state = "AUTO_CONTROL_APPLIED" if moved else "AUTO_HOLD"
        assert auto["state"] == expected_state, index
        if moved:
            assert auto["gate_transitions"], index
        else:
            assert auto["gate_transitions"] == [], index
        _assert_transitions_match_physics(auto["gate_transitions"], moved)
        _assert_actions_are_physically_true(
            auto["actions"], control["final_safe_control_action_pct"], before, after
        )
        assert state["mass_balance"]["matches_final_safe_control_action"] is True

        previous_after = after

    # Three real optimisations happened; the log records one triple of
    # controller events per step (plus gate changes and any risk transitions).
    kinds = _kinds(_state()["event_log"])
    assert kinds.count("mpc_decision") == 3
    assert kinds.count("safety_validation") == 3
    assert kinds.count("downstream_protection") == 3
    assert kinds.count("gate_change") >= 1
    assert "control_blocked" not in kinds
    assert "control_error" not in kinds


# ===========================================================================
# B. WEBSOCKET PARITY
# ===========================================================================

def test_websocket_carries_the_same_auto_state_and_event_log_as_rest(manual_reset):
    """
    The twin's ONLY state source must carry the Stage 18 blocks.

    The WebSocket payload is compared against REST for the same authoritative
    state id, so a client that reads nothing but the socket still sees the
    backend's AUTO state, forecast-source selection and event log — unchanged.
    """
    _arm("AI", VALIDATED_REPLAY)
    _step()

    rest = _state()
    with client.websocket_connect("/ws/state") as websocket:
        socket_state = websocket.receive_json()

    assert (socket_state["state_identity"]["state_id"]
            == rest["state_identity"]["state_id"])
    assert socket_state["controller_mode"] == rest["controller_mode"] == "AI"

    assert socket_state["auto_control"] == rest["auto_control"]
    assert (socket_state["forecast_source_selection"]
            == rest["forecast_source_selection"])
    assert ([(e["seq"], e["kind"], e["text"]) for e in socket_state["event_log"]]
            == [(e["seq"], e["kind"], e["text"]) for e in rest["event_log"]])

    assert socket_state["auto_control"]["auto_enabled"] is True
    assert socket_state["auto_control"]["state"] == "AUTO_CONTROL_APPLIED"
    assert socket_state["auto_control"]["forecast_source"] == VALIDATED_REPLAY
    assert (socket_state["forecast_source_selection"]["selected"]
            == VALIDATED_REPLAY)
    assert _kinds(socket_state["event_log"])[:3] == [
        "reset", "forecast_source", "auto_enabled",
    ]


# ===========================================================================
# C. THE HONEST DEFAULT IS NOT WEAKENED: SIMULATION STILL FAILS CLOSED
# ===========================================================================

def test_simulation_source_still_fails_closed_and_holds_the_gates(manual_reset):
    """
    The demo must NOT have been bought by relaxing the provenance gate.

    With the SIMULATION forecast source the control forecast is
    DEMONSTRATION_ONLY (and Reservoir D has no record at all), so the Stage 7
    gate refuses control: AUTO reports AUTO_BLOCKED and the gates are held
    exactly as they were — no fabricated action and no fabricated event.
    """
    for _ in range(8):
        _step()                      # warm the simulation forecast window
    _arm("AI", SIMULATION)

    state = _state()
    assert state["forecast_source_selection"]["selected"] == SIMULATION
    assert state["forecast_source_selection"]["control_forecast_validated"] is False
    assert state["auto_control"]["state"] == "AUTO_READY"

    events_before = state["event_log"]
    before = _gate_pct()
    _step()
    after = _gate_pct()
    state = _state()
    control = state["control"]
    auto = state["auto_control"]

    # ── The gate refused control, and said exactly why.
    assert control["forecast_control_eligible"] is False
    assert control["controller_status"] == "BLOCKED"
    assert control["mpc_status"] == "NOT_INVOKED"
    assert control["candidates_evaluated"] == 0
    assert control["blocked_reason"] == "FORECAST_NOT_ELIGIBLE_FOR_CONTROL"
    assert control["final_safe_control_action_source"] == "HELD_CURRENT_GATES"
    assert control["safety_layer_status"] == "NOT_APPLIED_MPC_BLOCKED"
    assert control["downstream_status"] == "NOT_APPLIED_MPC_BLOCKED"
    assert auto["state"] == "AUTO_BLOCKED"
    assert auto["reason"] == "FORECAST_NOT_ELIGIBLE_FOR_CONTROL"
    assert auto["control_applied"] is False

    # …and the forecast provenance is what blocks it: nothing is VALIDATED.
    nodes = control["forecast_provenance"]["nodes"]
    assert all((nodes.get(nid) or {}).get("declared_status") != "VALIDATED"
               for nid in NODES)
    assert state["forecast_provenance"]["live_forecasts_are_validated"] is False

    # ── Gates HELD bit-for-bit: the "action" is the current gate, not a new one.
    assert _moved_nodes(before, after) == {}
    assert auto["gate_transitions"] == []
    by_node = {a["node"]: a for a in auto["actions"]}
    for nid in NODES:
        assert (control["final_safe_control_action_pct"][nid]
                == pytest.approx(after[nid], abs=GATE_TOL)), nid
        assert by_node[nid]["previous_pct"] == pytest.approx(before[nid], abs=GATE_TOL)
        assert by_node[nid]["final_pct"] == pytest.approx(after[nid], abs=GATE_TOL)
        # The MPC never ran, so there is no proposal and no SafetyLayer output.
        assert by_node[nid]["mpc_proposal_pct"] is None, nid
        assert by_node[nid]["safety_pct"] is None, nid

    # ── The audit agrees that no controller action was applied this step.
    audit = state["mass_balance"]
    assert audit["status"] == "PASS"
    assert audit["controller_action_checked"] is False
    assert audit["matches_final_safe_control_action"] is None

    # ── Exactly one real event was added: the refusal (risk changes may also be
    #    reported — they are real classifications of this step).
    appended = state["event_log"][len(events_before):]
    assert appended, "the blocked step produced no event"
    assert appended[-1]["kind"] == "control_blocked"
    assert {e["kind"] for e in appended} <= {"control_blocked", "risk_change"}
    assert appended[-1]["controller_status"] == "BLOCKED"
    assert appended[-1]["reason"] == "FORECAST_NOT_ELIGIBLE_FOR_CONTROL"
    kinds = _kinds(state["event_log"])
    assert "mpc_decision" not in kinds
    assert "safety_validation" not in kinds
    assert "downstream_protection" not in kinds
    assert "gate_change" not in kinds
    _assert_event_log_is_self_consistent(state["event_log"])


# ===========================================================================
# D. MANUAL AUTHORITY AND THE HANDOVER
# ===========================================================================

def test_manual_mode_keeps_operator_authority_and_handover_moves_no_physics(
        manual_reset):
    """
    MANUAL means the OPERATOR owns the gates; AUTO -> MANUAL changes the mode
    only. Neither direction may move physics outside a step.
    """
    state = _state()
    auto = state["auto_control"]
    assert auto["state"] == "AUTO_DISABLED"
    assert auto["control_mode"] == "MANUAL"
    assert auto["auto_enabled"] is False
    assert state["control"]["controller_status"] == "UNAVAILABLE"
    assert [a["final_pct"] for a in auto["actions"]] == [None] * len(NODES)
    assert state["mass_balance"]["controller_action_checked"] is False

    # ── The operator's gate value reaches the physics, unreinterpreted.
    assert client.post("/api/gate/reservoir_1", json={"value": 42.0}).status_code == 200
    assert client.post("/api/gate/reservoir_2", json={"value": 77.0}).status_code == 200
    _step()
    after = _gate_pct()
    assert after["Virtual Reservoir A"] == pytest.approx(42.0, abs=GATE_TOL)
    assert after["Virtual Reservoir B"] == pytest.approx(77.0, abs=GATE_TOL)

    state = _state()
    kinds = _kinds(state["event_log"])
    # MANUAL emits no controller events at all: the operator acted, not AUTO.
    assert "mpc_decision" not in kinds
    assert "safety_validation" not in kinds
    assert "gate_change" not in kinds
    assert state["auto_control"]["state"] == "AUTO_DISABLED"
    assert state["mass_balance"]["controller_action_checked"] is False
    assert state["mass_balance"]["status"] == "PASS"

    # ── AUTO takes over the next step: its action is what the gates hold.
    _arm("AI", VALIDATED_REPLAY)
    before = _gate_pct()
    _step()
    after = _gate_pct()
    state = _state()
    assert state["auto_control"]["state"] == "AUTO_CONTROL_APPLIED"
    assert _moved_nodes(before, after)
    for nid in NODES:
        assert (after[nid]
                == pytest.approx(state["control"]["final_safe_control_action_pct"][nid],
                                 abs=GATE_TOL)), nid
    decisions = _count(state["event_log"], "mpc_decision")
    assert decisions == 1

    # ── …and handing control back changes only the mode.
    before_handover = _gate_pct()
    assert _arm("MANUAL").status_code == 200
    state = _state()
    assert state["auto_control"]["state"] == "AUTO_DISABLED"
    assert state["auto_control"]["auto_enabled"] is False
    assert state["controller_mode"] == "MANUAL"
    after_handover = _gate_pct()
    for nid in NODES:
        assert after_handover[nid] == pytest.approx(before_handover[nid], abs=GATE_TOL)
    assert _count(state["event_log"], "auto_disabled") == 1
    assert _count(state["event_log"], "mpc_decision") == decisions

    # ── Operator authority is back: a manual gate command reaches the physics.
    assert client.post("/api/gate/reservoir_3", json={"value": 90.0}).status_code == 200
    _step()
    assert _gate_pct()["Virtual Reservoir C"] == pytest.approx(90.0, abs=GATE_TOL)
    state = _state()
    assert state["auto_control"]["state"] == "AUTO_DISABLED"
    assert _count(state["event_log"], "mpc_decision") == decisions


def test_an_unknown_forecast_source_is_rejected_and_mutates_nothing(manual_reset):
    """
    The command boundary cannot inject a forecast source the backend does not
    have, and a rejected command leaves the authoritative state untouched.
    """
    _arm("AI", VALIDATED_REPLAY)
    before = _state()

    response = client.post("/api/controller/mode",
                           json={"mode": "AI", "source": "BOGUS_SOURCE"})
    assert response.status_code == 422

    after = _state()
    assert after["forecast_source_selection"]["selected"] == VALIDATED_REPLAY
    assert after["controller_mode"] == before["controller_mode"] == "AI"
    assert state_manager.sim_state.forecast_source == VALIDATED_REPLAY
    assert state_manager.sim_state.mode == "AI"
    assert _kinds(after["event_log"]) == _kinds(before["event_log"])
    assert _kinds(after["event_log"]) == ["reset", "forecast_source", "auto_enabled"]


# ===========================================================================
# E. THE TWIN PAGE RENDERS THE BACKEND BLOCKS — AND AUTHORS NOTHING
# ===========================================================================

def _strip_js_comments(js: str) -> str:
    """Remove // and /* */ comments (string-aware) so assertions test CODE."""
    out = []
    i, n = 0, len(js)
    mode = None
    while i < n:
        char = js[i]
        next_char = js[i + 1] if i + 1 < n else ""
        if mode is None:
            if char == "/" and next_char == "/":
                mode = "//"; i += 2; continue
            if char == "/" and next_char == "*":
                mode = "/*"; i += 2; continue
            if char in ('"', "'", "`"):
                mode = char; out.append(char); i += 1; continue
            out.append(char); i += 1; continue
        if mode == "//":
            if char == "\n":
                mode = None; out.append(char)
            i += 1; continue
        if mode == "/*":
            if char == "*" and next_char == "/":
                mode = None; i += 2; continue
            i += 1; continue
        if char == "\\":
            out.append(char); out.append(next_char); i += 2; continue
        if char == mode:
            mode = None
        out.append(char); i += 1
    return "".join(out)


def test_twin_page_consumes_the_backend_auto_state_forecast_source_and_event_log():
    """
    The page must RENDER the three Stage 18 backend blocks.

    It stores them verbatim, mirrors the backend mode onto its controls, and
    builds the activity feed from the backend's own event array — so what the
    operator sees is what the backend reported.
    """
    html = TWIN_PATH.read_text(encoding="utf-8")
    code = _module_script(html)

    # ── The three blocks are stored VERBATIM from the authoritative payload.
    assert "this.data.autoControl = state.auto_control || null;" in code
    assert ("this.data.forecastSourceSelection = state.forecast_source_selection "
            "|| null;") in code
    assert ("this.data.eventLog = Array.isArray(state.event_log) "
            "? state.event_log : null;") in code

    # ── The sync runs on every authoritative state update, after the rest.
    assert "updateState(state) {" in code
    assert "this._syncAutoControls();" in code
    assert "_syncAutoControls() {" in code

    # ── Control mode / forecast source come from the BACKEND, never from the
    #    click that was sent.
    assert "const autoEnabled = !!(ac && ac.auto_enabled === true) || mode === 'AI';" in code
    assert "#ctl-mode [data-mode]" in code
    assert "#ctl-source [data-source]" in code
    assert "btn.dataset.source === selected" in code

    # ── Gate sliders are locked exactly while the backend is deciding gates.
    assert "const lockGates = hasState && autoEnabled;" in code
    assert "s.disabled = lockGates;" in code
    assert "GATES · AUTO (LOCKED)" in code
    assert "GATES · OPERATOR" in code

    # ── The AUTO CONTROL panel prints the backend's own fields (and hides the
    #    rows the backend did not send rather than defaulting them).
    assert "AUTO CONTROL" in html
    assert 'data-ref="auto-state"' in html
    assert 'data-ref="auto-action"' in html
    assert 'data-ref="auto-reason"' in html
    assert 'data-ref="auto-feed"' in html
    assert "row.style.display = 'none';" in code

    # ── The activity feed is built ONLY from the backend event log.
    assert "const events = this.data.eventLog;" in code
    assert "const list = Array.isArray(events) ? events : [];" in code
    assert "ev.kind" in code
    assert "NO BACKEND EVENTS YET" in code

    # ── Presentation hooks for the new panel exist in the page's CSS.
    assert ".auto-feed{" in html
    assert ".ai-lock{" in html


def test_twin_page_cannot_author_backend_state_or_emit_its_own_events():
    """
    Display-only, enforced on the code (comments stripped).

    The page cannot record a backend event, cannot append to the backend log,
    and cannot reach the REST boundary except through the bounded ``api.js``
    client.
    """
    code = _strip_js_comments(_module_script(TWIN_PATH.read_text(encoding="utf-8")))

    assert "log_event(" not in code
    assert "eventLog.push" not in code
    assert "event_log.push" not in code
    assert "events.push(" not in code
    assert "'/controller/mode'" not in code
    assert '"/controller/mode"' not in code
    assert "'/gate/" not in code
    assert '"/gate/' not in code

    # …and the commands it does send go through the api.js client only.
    assert "api.setMode(" in code
    assert "api.setGate(" in code


def test_twin_page_javascript_parses():
    """Stage 18 additions must not break the page's module script."""
    html = TWIN_PATH.read_text(encoding="utf-8")
    proc = _node_check(_module_script(html), "twin_page_stage18.mjs")
    assert proc.returncode == 0, proc.stderr

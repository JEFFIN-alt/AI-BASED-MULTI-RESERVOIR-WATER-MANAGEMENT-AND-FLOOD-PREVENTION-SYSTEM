import datetime

from src.common import units


def _control_block(control: dict | None) -> dict:
    """
    Normalise the Stage 7/8 controller provenance for the twin payload.

    NEVER invents a favourable status: an absent control block is reported as
    UNKNOWN/UNAVAILABLE rather than being presented as an active MPC, and
    ``safety_modified`` is reported as ``None`` (unknown) rather than ``False``.
    """
    if not control:
        return {
            "controller_type": "UNKNOWN",
            "controller_status": "UNKNOWN",
            "forecast_control_eligible": False,
            "control_applied": False,
            "safety_layer_status": "UNKNOWN",
            # Stage 8 — unknown, NOT "False"/"unchanged". We do not know whether
            # a safety layer touched a control action, so we say so.
            "safety_modified": None,
            "proposed_gate_positions_fraction": {},
            # Stage 10 — no downstream-capacity guarantee may be implied when no
            # control provenance exists at all.
            "downstream_status": "UNKNOWN",
            "downstream_capacity_achieved": None,
            "downstream_protection_modified": None,
            "final_safe_control_action_source": "UNKNOWN",
            "forecast_provenance": {},
            "blocked_reason": "NO_CONTROL_PROVENANCE",
            "reasons": [],
        }
    out = dict(control)
    out.setdefault("controller_type", "UNKNOWN")
    out.setdefault("controller_status", "UNKNOWN")
    out.setdefault("forecast_control_eligible", False)
    out.setdefault("control_applied", False)
    out.setdefault("safety_layer_status", "UNKNOWN")
    out.setdefault("safety_modified", None)
    out.setdefault("proposed_gate_positions_fraction", {})
    out.setdefault("downstream_status", "UNKNOWN")
    out.setdefault("downstream_capacity_achieved", None)
    out.setdefault("downstream_protection_modified", None)
    out.setdefault("final_safe_control_action_source", "UNKNOWN")
    out.setdefault("forecast_provenance", {})
    out.setdefault("blocked_reason", "")
    out.setdefault("reasons", [])
    return out


def _mass_balance_block(mass_balance: dict | None) -> dict:
    """
    STAGE 11 — normalise the live mass-balance audit for the Digital Twin.

    NEVER invents a favourable result: an absent audit is reported as
    ``NOT_CHECKED`` with ``checked`` False, never as PASS. The status string, the
    residual, the tolerance and the number of reservoirs audited are all passed
    through from the authoritative backend — the UI computes none of them.
    """
    if not mass_balance:
        return {
            "status": "NOT_CHECKED",
            "checked": False,
            "residual": None,
            "tolerance": None,
            "unit": "MCM",
            "flow_unit": "MCM/day",
            "timestep_days": None,
            "timestamp": None,
            "step_index": None,
            "reason": "NO_MASS_BALANCE_PROVENANCE",
            "reservoirs_checked": 0,
            "reservoirs_expected": 0,
            "per_reservoir": [],
            "applied_action_source": "UNKNOWN",
            "state_valid": None,
            "ever_violated": None,
            "violation_count": None,
            "violations": [],
            "controller_action_checked": False,
            "matches_final_safe_control_action": None,
            "action_check_note": "NO_MASS_BALANCE_PROVENANCE",
            "fail_safe": "UNKNOWN",
            "fail_safe_gap": "",
            "equation": None,
            "physics": None,
            "hardware_connected": False,
            "latency_ms": None,
        }
    return {
        "status": str(mass_balance.get("status") or "NOT_CHECKED"),
        "checked": mass_balance.get("checked") is True,
        "residual": mass_balance.get("residual"),
        "tolerance": mass_balance.get("tolerance"),
        "unit": mass_balance.get("unit", "MCM"),
        "flow_unit": mass_balance.get("flow_unit", "MCM/day"),
        "timestep_days": mass_balance.get("timestep_days"),
        "timestamp": mass_balance.get("timestamp"),
        "step_index": mass_balance.get("step_index"),
        "timestep": mass_balance.get("timestep"),
        "reason": mass_balance.get("reason", ""),
        "reservoirs_checked": int(mass_balance.get("reservoirs_checked") or 0),
        "reservoirs_expected": int(mass_balance.get("reservoirs_expected") or 0),
        "per_reservoir": [dict(r) for r in (mass_balance.get("per_reservoir") or [])],
        "applied_action_source": mass_balance.get("applied_action_source") or "UNKNOWN",
        "applied_action_fraction": dict(mass_balance.get("applied_action_fraction") or {}),
        "applied_action_percent": dict(mass_balance.get("applied_action_percent") or {}),
        "state_valid": mass_balance.get("state_valid"),
        "ever_violated": mass_balance.get("ever_violated"),
        "violation_count": mass_balance.get("violation_count"),
        "violations": list(mass_balance.get("violations") or []),
        "non_finite": list(mass_balance.get("non_finite") or []),
        "action_fully_applied": mass_balance.get("action_fully_applied"),
        "controller_action_checked": mass_balance.get("controller_action_checked") is True,
        "matches_final_safe_control_action": mass_balance.get(
            "matches_final_safe_control_action"
        ),
        "checked_action_source": mass_balance.get("checked_action_source"),
        "action_check_note": mass_balance.get("action_check_note", ""),
        "fail_safe": mass_balance.get("fail_safe", "UNKNOWN"),
        "fail_safe_gap": mass_balance.get("fail_safe_gap", ""),
        "equation": mass_balance.get("equation"),
        "physics": mass_balance.get("physics"),
        "hardware_connected": bool(
            (mass_balance.get("telemetry") or {}).get("hardware_connected") is True
        ),
        "latency_ms": mass_balance.get("latency_ms"),
    }


# ── STAGE 12 — DISPLAY CLASSIFICATIONS THAT BELONG TO THE BACKEND ────────────
# Each of these used to be computed IN JAVASCRIPT from raw payload numbers. The
# browser must not derive physical/risk conclusions, so the ONE backend that owns
# the authoritative state classifies it once and the client renders the string.
# The thresholds are the SAME ones the frontend used, now applied to the
# AUTHORITATIVE quantities (previously the UI compared against a hardcoded
# "150 m³/s" safe limit that had no backend source).

#: |net flux| below this is reported as STEADY (m³/s). Was ±1 in the JS.
TREND_STEADY_THRESHOLD_M3_S = 1.0
#: Downstream utilisation at which the flow is reported as a WARNING / CRITICAL.
DOWNSTREAM_WARNING_RATIO = 0.8
DOWNSTREAM_CRITICAL_RATIO = 1.0

TREND_RISING = "RISING"
TREND_FALLING = "FALLING"
TREND_STEADY = "STEADY"

DOWNSTREAM_STATUS_NORMAL = "NORMAL"
DOWNSTREAM_STATUS_WARNING = "WARNING"
DOWNSTREAM_STATUS_CRITICAL = "CRITICAL"
DOWNSTREAM_STATUS_UNKNOWN = "UNKNOWN"

STORM_LEVELS = ((0.25, "LIGHT"), (0.55, "NORMAL"), (0.85, "HEAVY"))
STORM_LEVEL_SEVERE = "SEVERE"


def classify_trend(net_flux_m3_s: float | None) -> str | None:
    """RISING / FALLING / STEADY from the backend-computed net flux (display only)."""
    if net_flux_m3_s is None:
        return None
    if net_flux_m3_s > TREND_STEADY_THRESHOLD_M3_S:
        return TREND_RISING
    if net_flux_m3_s < -TREND_STEADY_THRESHOLD_M3_S:
        return TREND_FALLING
    return TREND_STEADY


def classify_downstream_status(flow_m3_s: float | None,
                               capacity_m3_s: float | None) -> str:
    """
    Classify the AUTHORITATIVE terminal outflow against the AUTHORITATIVE
    downstream capacity. 80 % of the capacity is a warning, above it is critical.
    Nothing is classified when either quantity is unknown.
    """
    if flow_m3_s is None or capacity_m3_s is None or capacity_m3_s <= 0:
        return DOWNSTREAM_STATUS_UNKNOWN
    ratio = flow_m3_s / capacity_m3_s
    if ratio > DOWNSTREAM_CRITICAL_RATIO:
        return DOWNSTREAM_STATUS_CRITICAL
    if ratio >= DOWNSTREAM_WARNING_RATIO:
        return DOWNSTREAM_STATUS_WARNING
    return DOWNSTREAM_STATUS_NORMAL


def classify_storm_level(intensity: float | None) -> str | None:
    """LIGHT / NORMAL / HEAVY / SEVERE from the authoritative storm intensity."""
    if intensity is None:
        return None
    for threshold, label in STORM_LEVELS:
        if intensity < threshold:
            return label
    return STORM_LEVEL_SEVERE


def _state_identity_block(identity: dict | None) -> dict:
    """
    STAGE 12 — what identifies a state update.

    The identity of a state is the AUTHORITATIVE simulation progress, not a
    browser clock:

      * ``sim_step_index``   — steps taken by the ONE ``GlobalSimulationState``
      * ``network_timestep`` — ``ReservoirNetwork.timestep``, incremented ONLY by
        an authoritative ``ReservoirNetwork.step()``
      * ``state_id``         — label derived from those two numbers

    A RESET is visible because ``network_timestep`` returns to 0 while
    ``sim_step_index`` keeps counting. Nothing new is counted here.
    """
    if not identity:
        return {
            "sim_step_index": None,
            "network_timestep": None,
            "state_id": None,
            "reason": "NO_STATE_IDENTITY_PROVENANCE",
        }
    return {
        "sim_step_index": identity.get("sim_step_index"),
        "network_timestep": identity.get("network_timestep"),
        "state_id": identity.get("state_id"),
        "source": identity.get("source"),
    }


def _downstream_block(sim_state: dict) -> dict:
    """
    STAGE 12 — the downstream flow, its AUTHORITATIVE limit and the severity,
    all computed by the backend. Replaces the hardcoded ``150 m³/s`` safe limit
    and the ``ds < 120 ? … : ds < 150 ? …`` thresholds that used to live in
    JavaScript.
    """
    flow_m3_s = mcm_per_day_to_m3_per_s(sim_state.get("downstream_flow"))
    capacity_m3_s = mcm_per_day_to_m3_per_s(sim_state.get("downstream_capacity"))
    utilisation = None
    if flow_m3_s is not None and capacity_m3_s not in (None, 0):
        utilisation = flow_m3_s / capacity_m3_s
    return {
        "flow_m3_s": flow_m3_s,
        "capacity_m3_s": capacity_m3_s,
        "utilisation": utilisation,
        "status": classify_downstream_status(flow_m3_s, capacity_m3_s),
        "unit": "m3/s",
        "source": "ReservoirNetwork.terminal_outflow vs network.downstream_capacity",
    }


def _storm_block(intensity: float | None) -> dict:
    """STAGE 12 — the storm intensity plus its backend-computed level label."""
    intensity = None if intensity is None else float(intensity)
    return {
        "intensity": intensity,
        "level": classify_storm_level(intensity),
        "source": "authoritative storm_intensity",
    }


def _simulation_block(simulation: dict | None) -> dict:
    """
    STAGE 12 — the authoritative run state of the ONE live simulation.

    The frontend printed a hardcoded ``RUNNING``. It now shows what the backend
    reports, and ``None`` (rendered as ``--``) when the backend reports nothing.
    """
    if not simulation:
        return {
            "running": None,
            "speed": None,
            "reason": "NO_SIMULATION_PROVENANCE",
        }
    return {
        "running": simulation.get("running"),
        "speed": simulation.get("speed"),
        "source": simulation.get("source"),
    }


def _forecast_summary_block(sim_state: dict) -> dict:
    """
    STAGE 12 — one backend-computed forecast series for the UI to draw.

    The weather panel used to FABRICATE a 7-bar forecast shape in JavaScript with
    a sine function of the render-loop clock. The bars are now a purely visual
    interpolation between these authoritative horizon values (declared in the
    frontend), and the aggregation happens here rather than in the browser.
    """
    reservoirs = sim_state.get("reservoirs", {}) or {}
    horizons = ("forecast_1d", "forecast_3d", "forecast_7d")
    totals = {h: 0.0 for h in horizons}
    available = {h: 0 for h in horizons}
    missing: list = []
    provenance: set = set()
    validated = True
    counted = 0

    for name, res in reservoirs.items():
        if not isinstance(res, dict):
            continue
        row_values = [res.get(h) for h in horizons]
        if all(v is None for v in row_values):
            missing.append(str(res.get("repository_name") or name))
            validated = False
            continue
        counted += 1
        prov = res.get("forecast_provenance")
        if prov:
            provenance.add(str(prov))
        if res.get("validated_metrics_apply") is not True:
            validated = False
        for h, v in zip(horizons, row_values):
            if v is None:
                continue
            totals[h] += float(v)
            available[h] += 1

    total_reservoirs = len([r for r in reservoirs.values() if isinstance(r, dict)])
    if counted == 0:
        status = "UNAVAILABLE"
    elif counted >= total_reservoirs and total_reservoirs > 0:
        status = "COMPLETE"
    else:
        status = "PARTIAL"

    return {
        "unit": "m3/s",
        "horizons": {
            "1d": mcm_per_day_to_m3_per_s(totals["forecast_1d"]) if available["forecast_1d"] else None,
            "3d": mcm_per_day_to_m3_per_s(totals["forecast_3d"]) if available["forecast_3d"] else None,
            "7d": mcm_per_day_to_m3_per_s(totals["forecast_7d"]) if available["forecast_7d"] else None,
        },
        "horizons_available": dict(available),
        "reservoirs_with_forecast": counted,
        "reservoirs_total": total_reservoirs,
        "missing_forecast": missing,
        "status": status,
        "provenance": sorted(provenance),
        "validated_metrics_apply": bool(validated and counted > 0),
        "source": "authoritative per-reservoir forecasts (summed by the backend)",
    }


def mcm_per_day_to_m3_per_s(mcm: float) -> float:
    """
    Converts MCM/day (Million Cubic Meters per day) to m³/s (Cubic Meters per second).

    This is the presentation-layer flow boundary: the project computes in
    MCM/day, while the Digital Twin payload renders m³/s. The conversion itself
    is defined once in ``src/common/units.py``.

    ``None`` is passed through unchanged, because the twin schema uses null to
    mean "forecast unavailable".
    """
    if mcm is None:
        return None
    return units.mcm_per_day_to_cubic_metres_per_second(mcm)


#: ── STAGE 9 — THE COMPLETE FOUR-RESERVOIR TWIN INVENTORY ────────────────────
#: Live reservoir name -> twin key, for ALL FOUR reservoirs of the validated
#: cascade. Before Stage 9 this stopped at ``reservoir_3``, so Reservoir D
#: (Idukki) was dropped from every WebSocket/twin payload even though it is a
#: first-class member of the state, the MPC, the SafetyLayer and the network.
#:
#: The names below are only a FALLBACK for the reservoir → key mapping. The
#: displayed reservoir NAME always comes from the authoritative payload
#: (``SimBridge.get_state()`` reads it from the validated config), never from
#: this table and never from JavaScript.
TWIN_RESERVOIR_KEYS = {
    "Virtual Reservoir A": "reservoir_1",
    "Virtual Reservoir B": "reservoir_2",
    "Virtual Reservoir C": "reservoir_3",
    "Virtual Reservoir D": "reservoir_4",
}

#: Order in which the four twin keys are emitted / validated.
TWIN_KEY_ORDER = ("reservoir_1", "reservoir_2", "reservoir_3", "reservoir_4")


def _cascade_block(sim_state: dict, twin_reservoirs: dict) -> dict:
    """
    Describe the authoritative cascade inventory for the Digital Twin.

    This is what makes "all four reservoirs are first-class" checkable rather
    than implied: the twin receives the ordered list of reservoirs, their node
    ids, their real (repository) names, their position in the cascade and which
    one is terminal — all read from the authoritative state, so JavaScript never
    has to hardcode a reservoir count or a reservoir name.
    """
    entries = []
    for key in TWIN_KEY_ORDER:
        res = twin_reservoirs.get(key) or {}
        entries.append({
            "key": key,
            "node_id": res.get("node_id"),
            "name": res.get("repository_name"),
            "position": res.get("cascade_position"),
            "terminal": res.get("is_terminal"),
        })
    terminal = next((e for e in entries if e["terminal"]), None)
    return {
        "order": [e["key"] for e in entries],
        "reservoirs": entries,
        "count": len(entries),
        "terminal": terminal,
        "downstream_capacity_m3_s": mcm_per_day_to_m3_per_s(
            sim_state.get("downstream_capacity")
        ),
    }


def adapt_state_for_twin(sim_state, current_mode="MANUAL", storm_intensity=0.0):
    """
    Converts the output of SimBridge.get_state() into the exact JSON schema
    expected by the GLM 3D Reservoir Digital Twin UI.
    """

    mapping = TWIN_RESERVOIR_KEYS

    twin_state = {
        "reservoirs": {},
        "storm_intensity": float(storm_intensity),
        "downstream_flow": mcm_per_day_to_m3_per_s(sim_state.get("downstream_flow", 0.0)),
        "controller_mode": current_mode,
        "simulation_time": datetime.datetime.now().isoformat(),
        "hardware_status": {
            "esp32": "NOT_CONNECTED",
            "water_level_sensor": "NOT_CONNECTED",
            "flow_sensor": "NOT_CONNECTED",
            "gate_actuator": "NOT_CONNECTED"
        },
        "metadata": {
            "flow_unit": "m3/s",
            "storage_unit": "ratio",
            "level_unit": "proxy_ratio"
        },
        # ── STAGE 7 — AUTHORITATIVE CONTROLLER PROVENANCE (top level) ─────
        # States exactly which controller produced the gate positions, whether
        # it was allowed to, and what the SafetyLayer did to the proposal.
        # A rule-based controller is never labelled as MPC, and the reported
        # gate positions are the SAFETY-VALIDATED ones (Stage 8), never the raw
        # MPC proposal.
        "control": _control_block(sim_state.get("control")),
        # ── STAGE 11 — LIVE MASS-BALANCE INTEGRITY ──────────────────────
        # The audit of the last authoritative simulation step, straight from the
        # backend physics. An absent audit is NOT_CHECKED / checked=False — the
        # payload can never imply "conserved" before a check has actually run.
        "mass_balance": _mass_balance_block(sim_state.get("mass_balance")),
        # ── STAGE 12 — AUTHORITATIVE STATE IDENTITY + DISPLAY VERDICTS ───
        # `state_identity` tells the client WHICH authoritative state this is.
        # `downstream` / `storm` / `forecast_summary` carry the classifications
        # the browser used to derive for itself (safe limit, severity, storm
        # level, forecast shape). The client renders them; it computes none.
        "state_identity": _state_identity_block(sim_state.get("state_identity")),
        "simulation": _simulation_block(sim_state.get("simulation")),
        "downstream": _downstream_block(sim_state),
        "storm": _storm_block(sim_state.get("storm_intensity")),
        "forecast_summary": _forecast_summary_block(sim_state),
        # ── STAGE 5 — FORECAST PROVENANCE (top level) ────────────────────
        # States plainly what the forecast numbers ARE. A demonstration
        # forecast (simulation / synthetic inputs) must never be presented as
        # the validated real-data evaluation, and the validated V3 metrics do
        # not apply to it.
        "forecast_provenance": {
            "model": "LSTM_V3_LOGTARGET",
            "model_status": "FROZEN_UNMODIFIED",
            "forecast_unit": "MCM/day",
            "horizons": ["1d", "3d", "7d"],
            "validated_evaluation": "historical held-out data -> frozen LSTM V3",
            "live_path": "authoritative simulation state -> simulation/demo feature inputs -> frozen LSTM V3",
            "live_forecasts_are_validated": False,
            "provenance_note": (
                "Live forecasts use simulation-derived inputs and explicitly "
                "labelled synthetic placeholders for features the live "
                "simulation cannot produce (water_level in metres, rainfall in mm). "
                "Validated V3 test metrics do NOT apply to them."
            ),
        },
    }

    for res_name, res_data in sim_state.get("reservoirs", {}).items():
        twin_key = mapping.get(res_name)
        if not twin_key:
            continue

        # STAGE 12 — the net flux and its trend are computed HERE, not in the
        # browser. The twin renders the backend's verdict.
        inflow_m3_s = mcm_per_day_to_m3_per_s(
            res_data.get("inflow", 0.0) + res_data.get("routed_inflow", 0.0)
        )
        release_m3_s = mcm_per_day_to_m3_per_s(res_data.get("outflow", 0.0))
        net_flux_m3_s = None
        if inflow_m3_s is not None and release_m3_s is not None:
            net_flux_m3_s = inflow_m3_s - release_m3_s

        twin_state["reservoirs"][twin_key] = {
            # The twin renderer consumes RATIOS in [0, 1], derived from the
            # EXTERNAL percentage representation via the single boundary.
            "water_level": units.storage_percent_to_fraction(res_data.get("storage_pct", 0.0)),
            "storage": units.storage_percent_to_fraction(res_data.get("storage_pct", 0.0)),
            "inflow": inflow_m3_s,
            "release": release_m3_s,
            "gate": units.gate_percent_to_fraction(res_data.get("gate_position_pct", 0.0)),
            "risk": res_data.get("risk_status", "NORMAL").lower(),
            # ── STAGE 12 — backend-computed display classifications ──────
            "net_flux_m3_s": net_flux_m3_s,
            "trend": classify_trend(net_flux_m3_s),
            "forecast_1d": mcm_per_day_to_m3_per_s(res_data.get("forecast_1d")),
            "forecast_3d": mcm_per_day_to_m3_per_s(res_data.get("forecast_3d")),
            "forecast_7d": mcm_per_day_to_m3_per_s(res_data.get("forecast_7d")),

            # ── STAGE 9 — RESERVOIR IDENTITY ─────────────────────────────
            # Emitted for ALL FOUR reservoirs, read from the authoritative
            # state, so the UI never has to guess which reservoir a card is.
            "node_id": res_data.get("node_id", res_name),
            "repository_name": res_data.get("repository_name"),
            "cascade_position": res_data.get("cascade_position"),
            "is_terminal": res_data.get("is_terminal"),

            # ── STAGE 5 FORECAST PROVENANCE ──────────────────────────────
            # Passed through verbatim from the forecast pipeline so the UI can
            # never present a demonstration forecast as a validated one.
            # NOTE: the `water_level` field ABOVE is a *display ratio* for the
            # 3D renderer (see "level_unit": "proxy_ratio"). It is NOT a water
            # level in metres and must never be used as a model feature.
            "forecast_status": res_data.get("forecast_status"),
            "forecast_provenance": res_data.get("forecast_provenance"),
            "forecast_source": res_data.get("forecast_source"),
            "forecast_is_simulated": res_data.get("is_simulated"),
            "forecast_validated_metrics_apply": res_data.get("validated_metrics_apply"),
            "forecast_unavailable_features": res_data.get("forecast_unavailable_features"),
        }
        
    # Provide defaults if missing — for ALL FOUR reservoirs (Stage 9).
    for i in range(1, len(TWIN_KEY_ORDER) + 1):
        key = f"reservoir_{i}"
        if key not in twin_state["reservoirs"]:
            twin_state["reservoirs"][key] = {
                "water_level": 0.0,
                "storage": 0.0,
                "inflow": 0.0,
                "release": 0.0,
                "gate": 0.0,
                "risk": "normal",
                "node_id": None,
                "repository_name": None,
                "cascade_position": i - 1,
                "is_terminal": i == len(TWIN_KEY_ORDER),
                # Stage 12 — same schema as a populated entry (no data ⇒ None).
                "net_flux_m3_s": None,
                "trend": None,
            }

    # ── STAGE 9 — the authoritative four-reservoir cascade inventory ──────
    twin_state["cascade"] = _cascade_block(sim_state, twin_state["reservoirs"])

    return twin_state

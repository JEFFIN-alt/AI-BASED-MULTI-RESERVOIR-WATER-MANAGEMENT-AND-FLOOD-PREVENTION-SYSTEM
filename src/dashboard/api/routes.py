"""
Digital Twin REST API routes.

STAGE 4 — SINGLE AUTHORITATIVE SIMULATION
=========================================
Every route in this module operates on ONE authoritative simulation instance:
``src.dashboard.api.state_manager.sim_state`` (imported below). There is no
second simulation, and the API exposes **no endpoint that accepts simulation
STATE** — clients may only issue bounded COMMANDS.

Frontend injection surface
--------------------------
Before Stage 4 the command models were bare ``float`` / ``str`` fields, so a
browser could push arbitrary values (``NaN``, ``Inf``, ``1e308``, zero speed,
arbitrary controller modes) straight into the authoritative simulation. The
models below now:

  * reject non-finite / non-numeric values (422),
  * clamp finite values into their documented domain, matching the clamping
    semantics of ``src/common/units.py``,
  * constrain ``mode`` to the two supported controller modes.

A request can therefore influence the authoritative simulation ONLY through
these bounded commands. It can never set storage, release, spill, routing or
any other physical state directly.
"""

import math
import asyncio
import re
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, field_validator

from src.dashboard.api.state_manager import sim_state
from src.dashboard.api.gmail_auth import gmail_oauth

router = APIRouter()


# ---------------------------------------------------------------------------
# Command models — bounded, finite, validated
# ---------------------------------------------------------------------------

def _finite_number(v, label: str) -> float:
    """Reject bools / non-numbers / NaN / Inf; return a plain float."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{label} must be a real number")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{label} must be finite")
    return f


class GateCommand(BaseModel):
    """Gate command in the EXTERNAL percent representation (0-100)."""
    value: float

    @field_validator("value", mode="before")
    @classmethod
    def _validate(cls, v):
        f = _finite_number(v, "gate value")
        # Clamp finite values into the documented external gate domain,
        # mirroring src/common/units.gate_percent_to_fraction().
        return max(0.0, min(100.0, f))


class StormCommand(BaseModel):
    """Storm intensity multiplier, documented domain [0.0, 1.0]."""
    value: float

    @field_validator("value", mode="before")
    @classmethod
    def _validate(cls, v):
        f = _finite_number(v, "storm value")
        return max(0.0, min(1.0, f))


class ModeCommand(BaseModel):
    """Controller mode and the forecast source offered to it.

    The two supported modes remain ``MANUAL`` and ``AI``.  ``source`` is an
    optional bounded forecast-source selector: ``SIMULATION`` is the unchanged
    live source, and ``VALIDATED_REPLAY`` is the explicitly selected frozen-V3
    historical-replay source.  The selector never relaxes the provenance gate.
    """

    mode: Literal["MANUAL", "AI"]
    source: Literal["SIMULATION", "VALIDATED_REPLAY"] | None = None


class SpeedCommand(BaseModel):
    """
    Simulation playback speed.

    MUST be strictly positive: the simulation loop computes
    ``asyncio.sleep(1.0 / sim_speed)``, so a zero value would raise
    ZeroDivisionError inside the authoritative loop.
    """
    speed: float

    @field_validator("speed", mode="before")
    @classmethod
    def _validate(cls, v):
        f = _finite_number(v, "speed")
        return max(0.05, min(50.0, f))


class NotificationPreferences(BaseModel):
    recipient: str | None = None
    high_enabled: bool | None = None
    critical_enabled: bool | None = None
    telegram_enabled: bool | None = None
    discord_enabled: bool | None = None

    @field_validator("recipient")
    @classmethod
    def valid_recipient(cls, value):
        if value is None:
            return value
        value = value.strip()
        if value and (len(value) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value)):
            raise ValueError("Enter a valid recipient email address")
        return value


@router.get("/notifications/settings")
async def get_notification_settings():
    return sim_state.notification_manager.settings_payload()


@router.put("/notifications/settings")
async def save_notification_settings(settings: NotificationPreferences):
    return sim_state.notification_manager.update_preferences(**settings.model_dump(exclude_unset=True))


@router.post("/notifications/test")
async def send_test_notification():
    sim_state._notification_loop = asyncio.get_running_loop()
    try:
        return sim_state.notification_manager.send_test()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None


@router.post("/notifications/telegram/test")
async def send_test_telegram_notification():
    return sim_state.notification_manager.telegram.send_test()


@router.post("/notifications/discord/test")
async def send_test_discord_notification():
    sim_state._notification_loop = asyncio.get_running_loop()
    try:
        return sim_state.notification_manager.discord.send_test(sim_state.get_adapted_state())
    except RuntimeError:
        raise HTTPException(status_code=409, detail="Discord test cooldown is active") from None


@router.post("/notifications/incidents/{incident_id}/acknowledge")
async def acknowledge_notification_incident(incident_id: str):
    sim_state._notification_loop = asyncio.get_running_loop()
    try:
        return sim_state.notification_manager.acknowledge(incident_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Incident not found") from None
    except ValueError:
        raise HTTPException(status_code=409, detail="Incident is already resolved") from None


@router.post("/notifications/gmail/connect")
async def connect_gmail():
    authorization_url = gmail_oauth.authorization_url()
    if not authorization_url:
        raise HTTPException(status_code=503, detail="Google OAuth is not configured. Administrator OAuth credentials are required.")
    return {"authorization_url": authorization_url}


@router.get("/notifications/gmail/callback")
async def gmail_oauth_callback_get(code: str | None = None, state: str | None = None, error: str | None = None):
    if error or not code:
        return RedirectResponse("/?settings=notifications&gmail_oauth=error")
    try:
        await asyncio.to_thread(gmail_oauth.complete, code, state)
    except Exception:
        return RedirectResponse("/?settings=notifications&gmail_oauth=error")
    return RedirectResponse("/?settings=notifications&gmail_oauth=connected")


@router.post("/notifications/gmail/callback")
async def gmail_oauth_callback_post(request: Request):
    # Accept Google's form_post response mode for deployments that explicitly use it.
    from urllib.parse import parse_qs
    form = parse_qs((await request.body()).decode("utf-8", errors="replace"))
    return await gmail_oauth_callback_get((form.get("code") or [None])[0],
                                          (form.get("state") or [None])[0],
                                          (form.get("error") or [None])[0])

@router.post("/notifications/gmail/disconnect")
async def disconnect_gmail():
    gmail_oauth.disconnect()
    return sim_state.notification_manager.settings_payload()


@router.get("/state")
async def get_state():
    """Read the ONE authoritative Digital Twin state."""
    return sim_state.get_adapted_state()


#: Live reservoir keys -> authoritative network node ids.
#:
#: STAGE 9 — ALL FOUR reservoirs are commandable. Before Stage 9 this mapping
#: stopped at ``reservoir_3``, so Reservoir D (Idukki) had no API command at all
#: and its hardcoded manual baseline was literally unchangeable ("permanently
#: pinned"). D is now an ordinary, first-class commandable reservoir.
RESERVOIR_ID_TO_NODE = {
    "reservoir_1": "Virtual Reservoir A",
    "reservoir_2": "Virtual Reservoir B",
    "reservoir_3": "Virtual Reservoir C",
    "reservoir_4": "Virtual Reservoir D",
}


@router.post("/gate/{reservoir_id}")
async def set_gate(reservoir_id: str, cmd: GateCommand):
    mapping = RESERVOIR_ID_TO_NODE
    v_res = mapping.get(reservoir_id)
    if v_res:
        sim_state.manual_gates[v_res] = cmd.value
        if not sim_state.running:
            await sim_state.broadcast_state()
        return {"status": "success", "reservoir": reservoir_id, "gate": cmd.value}
    raise HTTPException(status_code=400, detail="Invalid reservoir_id")

@router.post("/simulation/play")
async def play_simulation():
    sim_state.running = True
    sim_state.log_event("simulation_play", "Simulation PLAY")
    # STAGE 12 — a command must come back to the client as the resulting
    # AUTHORITATIVE state, so the twin never shows a stale run state.
    await sim_state.broadcast_state()
    return {"status": "success", "running": True}

@router.post("/simulation/pause")
async def pause_simulation():
    sim_state.running = False
    sim_state.log_event("simulation_pause", "Simulation PAUSE")
    # STAGE 12 — PAUSE previously stopped the backend WITHOUT broadcasting, so a
    # connected twin kept displaying "RUNNING" until some other command happened
    # to push a state. The pause now reports the resulting state like every other
    # command does.
    await sim_state.broadcast_state()
    return {"status": "success", "running": False}

@router.post("/simulation/step")
async def step_simulation():
    sim_state.running = False
    sim_state._notification_loop = __import__("asyncio").get_running_loop()
    try:
        sim_state.step()
    except Exception as exc:
        sim_state.record_simulation_error(exc)
        await sim_state.broadcast_state()
        raise HTTPException(status_code=409, detail=f"Simulation paused: {exc}") from None
    await sim_state.broadcast_state()
    return {"status": "success", "stepped": True}

@router.post("/simulation/classroom-demo")
async def classroom_demo():
    sim_state.load_classroom_demo()
    sim_state.close_resolved_notification_after_reset()
    await sim_state.broadcast_state()
    return {"status": "success", "preset": "classroom", "running": False}


@router.post("/simulation/reset")
async def reset_simulation():
    sim_state.classroom_demo = False
    sim_state.bridge.init_cascade(50.0)
    # STAGE 18 — a reset begins a NEW run, so the previous run's controller
    # decision, gate transitions, risk memory and event log are cleared. This is
    # what stops AUTO from replaying a decision that belonged to the discarded
    # run. No loop is created or stopped here: exactly ONE authoritative loop
    # exists (GlobalSimulationState.simulation_loop), so there is no second
    # automatic loop to leak.
    sim_state.reset_auto_control_state()
    # Close a prior incident against the new cascade state without warming the
    # display forecast cache or invoking any controller logic.
    sim_state.close_resolved_notification_after_reset()
    if not sim_state.running:
        await sim_state.broadcast_state()
    return {"status": "success", "reset": True}

@router.post("/simulation/speed")
async def set_speed(cmd: SpeedCommand):
    sim_state.sim_speed = cmd.speed
    # STAGE 12 — the payload reports the authoritative speed, so the command
    # result is broadcast too.
    await sim_state.broadcast_state()
    return {"status": "success", "speed": cmd.speed}

@router.post("/storm")
async def set_storm(cmd: StormCommand):
    previous = sim_state.storm_intensity
    sim_state.storm_intensity = cmd.value
    # STAGE 18 — a real storm change is a real event.
    if cmd.value != previous:
        sim_state.log_event(
            "storm_change",
            f"Storm intensity {previous:.2f} -> {cmd.value:.2f}"
            + (" (increased)" if cmd.value > previous else " (decreased)"),
            previous_storm=previous, storm=cmd.value,
        )
    if not sim_state.running:
        await sim_state.broadcast_state()
    return {"status": "success", "storm": cmd.value}

@router.post("/controller/mode")
async def set_mode(cmd: ModeCommand):
    # ── STAGE 18 — MANUAL | AUTO ──────────────────────────────────────────────
    # Switching to MANUAL stops the backend from deciding gates WITHOUT touching
    # the authoritative reservoir state, so the current gates and storages are
    # preserved and the operator's manual controls behave exactly as before.
    # Switching back to AUTO resets nothing: the controller simply re-evaluates
    # the CURRENT authoritative state and forecast on the next step.
    #
    # The mode change itself is recorded as a real backend event.
    if cmd.source is not None:
        sim_state.set_forecast_source(cmd.source)
    sim_state.set_mode(cmd.mode)
    if not sim_state.running:
        await sim_state.broadcast_state()
    return {
        "status": "success",
        "mode": cmd.mode,
        "forecast_source": sim_state.forecast_source,
    }

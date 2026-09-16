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
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from src.dashboard.api.state_manager import sim_state

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
    """Controller mode — only the two implemented modes are accepted."""
    mode: Literal["MANUAL", "AI"]


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
    # STAGE 12 — a command must come back to the client as the resulting
    # AUTHORITATIVE state, so the twin never shows a stale run state.
    await sim_state.broadcast_state()
    return {"status": "success", "running": True}

@router.post("/simulation/pause")
async def pause_simulation():
    sim_state.running = False
    # STAGE 12 — PAUSE previously stopped the backend WITHOUT broadcasting, so a
    # connected twin kept displaying "RUNNING" until some other command happened
    # to push a state. The pause now reports the resulting state like every other
    # command does.
    await sim_state.broadcast_state()
    return {"status": "success", "running": False}

@router.post("/simulation/step")
async def step_simulation():
    sim_state.running = False
    sim_state.step()
    await sim_state.broadcast_state()
    return {"status": "success", "stepped": True}

@router.post("/simulation/reset")
async def reset_simulation():
    sim_state.bridge.init_cascade(50.0)
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
    sim_state.storm_intensity = cmd.value
    if not sim_state.running:
        await sim_state.broadcast_state()
    return {"status": "success", "storm": cmd.value}

@router.post("/controller/mode")
async def set_mode(cmd: ModeCommand):
    sim_state.mode = cmd.mode
    if not sim_state.running:
        await sim_state.broadcast_state()
    return {"status": "success", "mode": cmd.mode}

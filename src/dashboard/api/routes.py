from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.dashboard.api.state_manager import sim_state

router = APIRouter()

class GateCommand(BaseModel):
    value: float

class StormCommand(BaseModel):
    value: float

class ModeCommand(BaseModel):
    mode: str

class SpeedCommand(BaseModel):
    speed: float

@router.get("/state")
async def get_state():
    return sim_state.get_adapted_state()

@router.post("/gate/{reservoir_id}")
async def set_gate(reservoir_id: str, cmd: GateCommand):
    mapping = {
        "reservoir_1": "Virtual Reservoir A",
        "reservoir_2": "Virtual Reservoir B",
        "reservoir_3": "Virtual Reservoir C"
    }
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
    return {"status": "success", "running": True}

@router.post("/simulation/pause")
async def pause_simulation():
    sim_state.running = False
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

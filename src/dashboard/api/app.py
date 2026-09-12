import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from src.dashboard.api.state_manager import sim_state
from src.dashboard.api.routes import router

_THIS_DIR = Path(__file__).resolve().parent
_WEB_DIR = _THIS_DIR.parent / "web"
_WEB_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Digital Twin API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.on_event("startup")
async def startup_event():
    sim_state.loop_task = asyncio.create_task(sim_state.simulation_loop())

@app.on_event("shutdown")
async def shutdown_event():
    if sim_state.loop_task:
        sim_state.loop_task.cancel()

@app.websocket("/ws/state")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    sim_state.clients.add(websocket)
    # Send immediate initial state
    await websocket.send_text(json.dumps(sim_state.get_adapted_state()))
    try:
        while True:
            # Keep connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in sim_state.clients:
            sim_state.clients.remove(websocket)

# Mount static files
app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")

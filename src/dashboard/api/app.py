import asyncio
import json
import math
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from src.dashboard.api.state_manager import sim_state
from src.dashboard.api.routes import router

_THIS_DIR = Path(__file__).resolve().parent
_WEB_DIR = _THIS_DIR.parent / "web"
_WEB_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Digital Twin API")


# ---------------------------------------------------------------------------
# STAGE 4 — validation errors must never themselves fail
# ---------------------------------------------------------------------------
# A hostile (or merely buggy) client can put ``NaN`` / ``Infinity`` on the wire.
# Python's ``json.loads`` parses those into non-finite floats, our command
# validators correctly reject them, and FastAPI then echoes the offending value
# back inside the 422 payload. Starlette's JSONResponse renders with
# ``allow_nan=False``, so encoding that echo raises and the request turns into a
# 500 — i.e. the client could still crash the handler by "injecting" a value.
#
# The handler below renders non-finite floats as strings, so the API always
# answers with a clean 422 and never a 500.
# ---------------------------------------------------------------------------

def _json_safe_float(value: float):
    """Render non-finite floats as their string form so JSON can encode them."""
    return value if math.isfinite(value) else repr(value)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    try:
        detail = jsonable_encoder(exc.errors(), custom_encoder={float: _json_safe_float})
    except Exception:  # pragma: no cover - defensive: never fail to fail safely
        detail = [{"msg": "invalid request payload"}]
    return JSONResponse(status_code=422, content={"detail": detail})


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

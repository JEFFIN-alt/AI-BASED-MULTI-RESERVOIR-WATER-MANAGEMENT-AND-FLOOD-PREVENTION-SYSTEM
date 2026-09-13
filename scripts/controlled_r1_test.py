"""
Controlled R1 gate experiment against the LIVE backend (http://127.0.0.1:8000).

Proves (or disproves) the full chain:
    UI gate command -> FastAPI /api/gate -> SimBridge.manual_gates
    -> VirtualCascade.step() -> release/storage/level physics
    -> RiskEngine.assess_risk() -> /api/state (same payload WebSocket sends)

Usage:
    python scripts/controlled_r1_test.py [--steps N]
"""
import json
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:8000/api"


def post(endpoint, payload=None):
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(
        f"{BASE}{endpoint}", data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def get_state():
    with urllib.request.urlopen(f"{BASE}/state", timeout=30) as r:
        return json.loads(r.read().decode())


def r1(state):
    return state["reservoirs"]["reservoir_1"]


def row(label, s):
    d = r1(s)
    print(f"{label:<14} gate={d['gate']*100:6.1f}%  inflow={d['inflow']:8.2f}  "
          f"release={d['release']:8.2f}  storage={d['storage']*100:6.2f}%  "
          f"level={d['water_level']*100:6.2f}%  risk={d['risk'].upper()}")


def main():
    n_steps = 8
    if "--steps" in sys.argv:
        n_steps = int(sys.argv[sys.argv.index("--steps") + 1])

    print("=== CONTROLLED R1 EXPERIMENT ===")
    print("[1] Reset simulation (storage -> 50%)")
    post("/simulation/reset")
    post("/simulation/pause")
    print("[2] Storm -> 0, mode -> MANUAL")
    post("/storm", {"value": 0.0})
    post("/controller/mode", {"mode": "MANUAL"})
    print("[3] R1 gate -> 0 (fill to high level)")
    post("/gate/reservoir_1", {"value": 0.0})
    post("/gate/reservoir_2", {"value": 0.0})
    post("/gate/reservoir_3", {"value": 0.0})

    # Fill R1 to high level with gate closed
    for i in range(6):
        post("/simulation/step")
        s = get_state()
        row(f"fill step {i+1}", s)
        if r1(s)["water_level"] >= 0.95:
            print("    -> R1 at high level, stop filling")
            break

    print(f"\n[4] R1 gate -> 100%, then {n_steps} STEP operations")
    resp = post("/gate/reservoir_1", {"value": 100.0})
    print(f"    gate command response: {resp}")
    for i in range(n_steps):
        post("/simulation/step")
        s = get_state()
        row(f"step {i+1}", s)

    print("\n[5] Control: R1 gate -> 0, 2 more steps (release should drop)")
    post("/gate/reservoir_1", {"value": 0.0})
    for i in range(2):
        post("/simulation/step")
        row(f"ctrl step {i+1}", get_state())


if __name__ == "__main__":
    main()

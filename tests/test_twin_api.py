import pytest
from fastapi.testclient import TestClient
from src.dashboard.api.app import app
from src.dashboard.api.state_manager import sim_state

client = TestClient(app)

def test_get_state():
    response = client.get("/api/state")
    assert response.status_code == 200
    data = response.json()
    assert "reservoirs" in data
    assert "reservoir_1" in data["reservoirs"]
    assert "reservoir_2" in data["reservoirs"]
    assert "reservoir_3" in data["reservoirs"]
    assert "hardware_status" in data

def test_gate_command():
    # Test valid reservoir
    response = client.post("/api/gate/reservoir_1", json={"value": 75.0})
    assert response.status_code == 200
    assert response.json()["gate"] == 75.0
    
    # Verify state manager was updated
    assert sim_state.manual_gates["Virtual Reservoir A"] == 75.0

    # Test invalid reservoir
    response = client.post("/api/gate/invalid_res", json={"value": 50.0})
    assert response.status_code == 400

def test_simulation_controls():
    response = client.post("/api/simulation/play")
    assert response.status_code == 200
    assert sim_state.running == True

    response = client.post("/api/simulation/pause")
    assert response.status_code == 200
    assert sim_state.running == False

    response = client.post("/api/simulation/step")
    assert response.status_code == 200
    assert response.json()["stepped"] == True

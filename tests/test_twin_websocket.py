import pytest
from fastapi.testclient import TestClient
from src.dashboard.api.app import app

client = TestClient(app)

def test_websocket_connection():
    with client.websocket_connect("/ws/state") as websocket:
        # Upon connection, we should receive the initial state immediately
        data = websocket.receive_json()
        assert "reservoirs" in data
        assert "reservoir_1" in data["reservoirs"]
        
        # Verify hardware is NOT_CONNECTED
        hw = data["hardware_status"]
        assert hw["esp32"] == "NOT_CONNECTED"
        assert hw["gate_actuator"] == "NOT_CONNECTED"

        # Verify unit metadata
        assert "metadata" in data
        assert data["metadata"]["flow_unit"] == "m3/s"

import asyncio
import json
from pathlib import Path
from src.dashboard.sim_bridge import SimBridge
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent

CONFIG_PATH = _PROJECT_ROOT / "configs" / "simulation" / "four_reservoir_demo.json"
THRESH_PATH = _PROJECT_ROOT / "data" / "processed" / "historical_inflow_thresholds.json"

class GlobalSimulationState:
    def __init__(self):
        self.bridge = SimBridge(str(CONFIG_PATH), str(THRESH_PATH))
        self.storm_intensity = 0.0
        self.mode = "MANUAL"
        self.running = False
        self.sim_speed = 1.0
        self.manual_inflows = {
            "Virtual Reservoir A": 10.0,
            "Virtual Reservoir B": 20.0,
            "Virtual Reservoir C": 100.0,
            "Virtual Reservoir D": 0.0
        }
        self.manual_gates = {
            "Virtual Reservoir A": 40.0,
            "Virtual Reservoir B": 35.0,
            "Virtual Reservoir C": 50.0,
            "Virtual Reservoir D": 100.0
        }
        self.clients = set()
        self.loop_task = None
        self._update_inflows()
        
    def _update_inflows(self):
        # Update inflows based on storm intensity
        for res in ["Virtual Reservoir A", "Virtual Reservoir B", "Virtual Reservoir C"]:
            default_inflow = 10.0 if "A" in res else (20.0 if "B" in res else 100.0)
            self.manual_inflows[res] = default_inflow * (1.0 + (self.storm_intensity * 3.0))

    def step(self):
        self._update_inflows()
        forecasts = {}
        for res in ["Virtual Reservoir A", "Virtual Reservoir B", "Virtual Reservoir C"]:
            f_base = self.manual_inflows[res]
            forecasts[res] = {"forecast_1d": f_base, "forecast_3d": f_base, "forecast_7d": f_base}
            
        ai_recommendations = self.bridge.compute_ai_recommendation(forecasts)
        
        gate_commands = self.manual_gates.copy()
        if self.mode == "AI":
            for res in ["Virtual Reservoir A", "Virtual Reservoir B", "Virtual Reservoir C"]:
                gate_commands[res] = ai_recommendations.get(res, 0.0)
                
        self.bridge.step(self.manual_inflows, gate_commands)
        return self.get_adapted_state()

    def get_adapted_state(self):
        forecasts = {}
        for res in ["Virtual Reservoir A", "Virtual Reservoir B", "Virtual Reservoir C"]:
            f_base = self.manual_inflows[res]
            forecasts[res] = {"forecast_1d": f_base, "forecast_3d": f_base, "forecast_7d": f_base}
        
        current_state = self.bridge.get_state(forecasts)
        current_state["storm_intensity"] = self.storm_intensity
        
        # Inject live manual gate state into current_state for immediate visual feedback
        # if not in AI mode, since the bridge state will only reflect AFTER the next step
        if self.mode == "MANUAL":
            for res, gate_val in self.manual_gates.items():
                if res in current_state["reservoirs"]:
                    current_state["reservoirs"][res]["gate_position_pct"] = gate_val
                    
        return adapt_state_for_twin(current_state, self.mode, self.storm_intensity)

    async def broadcast_state(self):
        if not self.clients:
            return
        state = self.get_adapted_state()
        state_json = json.dumps(state)
        # Handle disconnections during iteration safely
        for client in list(self.clients):
            try:
                await client.send_text(state_json)
            except Exception:
                self.clients.remove(client)
                
    async def simulation_loop(self):
        while True:
            if self.running:
                self.step()
                await self.broadcast_state()
                # Assuming 1.0 speed is 1 step per second
                await asyncio.sleep(1.0 / self.sim_speed)
            else:
                await asyncio.sleep(0.5)

sim_state = GlobalSimulationState()

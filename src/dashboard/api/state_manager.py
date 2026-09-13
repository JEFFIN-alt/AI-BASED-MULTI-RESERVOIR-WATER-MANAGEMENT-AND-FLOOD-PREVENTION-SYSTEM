import asyncio
import json
import numpy as np
import pandas as pd
from pathlib import Path
from src.dashboard.sim_bridge import SimBridge
from src.dashboard.twin_component.state_adapter import adapt_state_for_twin
from src.modeling.inference import LiveForecaster
from src.modeling.gnn_inference import LiveGNNForecaster
from src.network_env.gnn_forecast_adapter import GNNForecastAdapter

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
        # Baseline inflows (MCM/day). MUST stay below each reservoir's
        # max_release_capacity_mcm_day (A:5, B:10, C:150, D:200) so that a
        # fully-open gate can actually drain the reservoir — including routed
        # inflow from upstream (A max release 5 → B local 4 + routed 5 = 9 < 10;
        # C local 90 + routed 10 = 100 < 150). At storm=0 the multipliers below
        # keep every total inflow under its release capacity.
        self.manual_inflows = {
            "Virtual Reservoir A": 3.0,
            "Virtual Reservoir B": 4.0,
            "Virtual Reservoir C": 90.0,
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
        
        # ── ML Integration ──
        # Map virtual reservoirs to physical ones for models
        with open(CONFIG_PATH) as f:
            config = json.load(f)
        self.res_mapping = {
            node_id: res_cfg["repository_derived_source"] 
            for node_id, res_cfg in config["reservoirs"].items()
        }
        
        try:
            self.lstm_forecaster = LiveForecaster(str(_PROJECT_ROOT))
            self.gnn_forecaster = LiveGNNForecaster(str(_PROJECT_ROOT))
            self.gnn_adapter = GNNForecastAdapter(self.res_mapping)
            self.ml_ready = True
        except Exception as e:
            print(f"[StateManager] ML initialization failed: {e}")
            self.ml_ready = False
            
        # Keep 7 days of simulated history for inference
        self.history_buffers = {v: [] for k, v in self.res_mapping.items()}
        for _ in range(7):
            self._record_history()

    def _record_history(self):
        """Maintain a 7-day rolling window of state for live inference."""
        for v_name, p_name in self.res_mapping.items():
            inflow = self.manual_inflows.get(v_name, 0.0)
            res_obj = self.bridge.cascade.reservoirs.get(v_name)
            wl = res_obj.get_simulated_water_level_proxy() if res_obj else 50.0
            storage = res_obj.state.storage_mcm if res_obj else 50.0
            outflow = res_obj.state.release_mcm_day if res_obj else inflow
            
            # Feature order: [inflow, water_level, live_storage, rainfall, total_outflow]
            row = [inflow, wl, storage, 0.0, outflow]
            self.history_buffers[p_name].append(row)
            if len(self.history_buffers[p_name]) > 7:
                self.history_buffers[p_name].pop(0)

    def _update_inflows(self):
        # Storm multiplier scales baseline inflow up to 4x (1 + 3*storm).
        # Baselines are chosen so that at storm=0 every inflow is below the
        # reservoir's max release capacity — full gate can then drain it.
        for res in ["Virtual Reservoir A", "Virtual Reservoir B", "Virtual Reservoir C"]:
            default_inflow = 3.0 if "A" in res else (4.0 if "B" in res else 90.0)
            self.manual_inflows[res] = default_inflow * (1.0 + (self.storm_intensity * 3.0))

    def _run_ml_pipeline(self) -> dict:
        """Run LSTM + GNN and return control forecasts."""
        lstm_forecasts = {}
        gnn_history = {}
        
        if not self.ml_ready:
            # Fallback mock
            for res in ["Virtual Reservoir A", "Virtual Reservoir B", "Virtual Reservoir C"]:
                f_base = self.manual_inflows[res]
                lstm_forecasts[res] = {"forecast_1d": f_base, "forecast_3d": f_base, "forecast_7d": f_base}
            return lstm_forecasts
            
        for v_name, p_name in self.res_mapping.items():
            if v_name == "Virtual Reservoir D": 
                continue
                
            arr = np.array(self.history_buffers[p_name], dtype=np.float32)
            gnn_history[p_name] = arr
            
            df = pd.DataFrame(arr, columns=["inflow", "water_level", "live_storage", "rainfall", "total_outflow"])
            df["date"] = pd.date_range(end=pd.Timestamp.now(), periods=7).strftime("%Y-%m-%d")
            
            try:
                fc = self.lstm_forecaster.predict(df)
                lstm_forecasts[v_name] = fc
            except Exception as e:
                # If scaling or something fails for the mock data, fallback
                f_base = self.manual_inflows[v_name]
                lstm_forecasts[v_name] = {"forecast_1d": f_base, "forecast_3d": f_base, "forecast_7d": f_base}

        # Run GNN
        try:
            gnn_result = self.gnn_forecaster.predict_all(gnn_history)
            self.gnn_adapter.update_from_inference(gnn_result)
        except Exception as e:
            print(f"[StateManager] GNN Inference failed: {e}")
            
        # Get production control policy (lstm_primary)
        ctrl_forecasts = self.gnn_adapter.get_control_forecasts(lstm_forecasts, policy="lstm_primary")
        return ctrl_forecasts

    def step(self):
        self._update_inflows()
        self._record_history()
        
        ctrl_forecasts = self._run_ml_pipeline()
        ai_recommendations = self.bridge.compute_ai_recommendation(ctrl_forecasts)
        
        gate_commands = self.manual_gates.copy()
        if self.mode == "AI":
            for res in ["Virtual Reservoir A", "Virtual Reservoir B", "Virtual Reservoir C"]:
                gate_commands[res] = ai_recommendations.get(res, 0.0)
                
        self.bridge.step(self.manual_inflows, gate_commands)
        return self.get_adapted_state()

    def get_adapted_state(self):
        ctrl_forecasts = self._run_ml_pipeline()
        current_state = self.bridge.get_state(ctrl_forecasts)
        current_state["storm_intensity"] = self.storm_intensity
        
        # Inject live manual gate state into current_state for immediate visual feedback
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
                await asyncio.sleep(1.0 / self.sim_speed)
            else:
                await asyncio.sleep(0.5)

sim_state = GlobalSimulationState()

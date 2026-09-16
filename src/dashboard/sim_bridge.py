import sys
from pathlib import Path
import json

# Setup paths
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.network_env.live_cascade_adapter import LiveCascadeAdapter
from src.simulator.controllers import ForecastAwareController
from src.management.risk_engine import assess_risk
from src.common import units

#: Canonical validated physics configuration. This is the SINGLE authority for
#: live Digital Twin topology, routing delays and attenuation.
TOPOLOGY_PATH = _PROJECT_ROOT / "src" / "network_env" / "topology_config.yaml"


class SimBridge:
    """
    Live Digital Twin bridge.

    PHYSICS AUTHORITY (Stage 3)
    ---------------------------
    The live cascade is backed by the validated ``ReservoirNetwork`` through
    ``LiveCascadeAdapter``. The legacy Phase 14.4 ``VirtualCascade`` is NOT
    used on the live path; it remains in ``src/simulator/environment.py``
    untouched for the research/backtest simulator.

    Live facade contract (unchanged for all callers):
      * gate commands are EXTERNAL PERCENT (0-100)
      * storages/flows are MCM and MCM/day
      * state is exposed per reservoir via ``self.cascade.reservoirs``
    """

    def __init__(self, config_path: str, thresholds_path: str):
        self.config_path = config_path
        self.thresholds_path = thresholds_path
        
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
            
        with open(self.thresholds_path, 'r') as f:
            self.historical_thresholds = json.load(f)
            
        self.ai_controller = ForecastAwareController(self.config)
        self.cascade = None
        self.init_cascade(50.0) # default 50% start
        
    def init_cascade(self, initial_storage_pct: float):
        """Initializes or resets the live cascade.

        ``initial_storage_pct`` is the EXTERNAL storage percentage (0-100).
        The simulator stores MCM internally, so the percentage is converted to a
        capacity fraction once, at the boundary.

        Physics is delegated to the authoritative ``LiveCascadeAdapter``
        (validated ``ReservoirNetwork`` + ``topology_config.yaml``).
        """
        # Clone config so we don't pollute the base template
        run_config = json.loads(json.dumps(self.config))
        for res_name in run_config["reservoirs"]:
            cap = run_config["reservoirs"][res_name]["capacity_mcm"]["value"]
            fraction = units.storage_percent_to_fraction(initial_storage_pct)
            run_config["reservoirs"][res_name]["initial_storage_mcm"]["value"] = cap * fraction

        self.cascade = LiveCascadeAdapter(run_config, topology_path=str(TOPOLOGY_PATH))
        
    def step(self, inflows: dict, gate_commands: dict, action_source: str = None):
        """Advances simulation by one day.

        STAGE 11 — the authoritative step is audited for mass balance inside the
        cascade adapter (the only place ``ReservoirNetwork.step()`` is called on
        the live path). ``action_source`` records where the applied action came
        from for that diagnostic; it never influences the physics.
        """
        self.cascade.step(inflows, gate_commands, action_source=action_source)

    def mass_balance_diagnostic(self) -> dict:
        """STAGE 11 — the last audited authoritative step (never a fabricated PASS)."""
        return self.cascade.mass_balance_diagnostic()
        
    def compute_ai_recommendation(self, forecasts: dict) -> dict:
        """
        Rule-based gate advisor (``ForecastAwareController``).

        STAGE 7 — NON-AUTHORITATIVE. This is NOT the live controller any more.
        The authoritative live AI path is
        ``GlobalSimulationState -> LiveMPCOrchestrator -> MPCController`` behind
        the forecast-provenance gate. This method is retained only as an
        offline/analytics helper and must never be presented as MPC output.

        Runs the ForecastAwareController to generate AI gate recommendations.
        forecasts: dict mapping virtual reservoir name to dict of {"forecast_1d": x, ...}
        """
        recommendations = {}
        # We need downstream statuses for the controller cascade coordination
        # Since we just want recommendations, we'll assume everything is NORMAL
        # unless we want to do a full top-down pass. For simplicity, assume NORMAL.
        ds_status = "NORMAL"
        
        # Process bottom-up to match engine behavior
        reversed_order = reversed(self.config["topology"]["cascade_order"])
        statuses = {"Terminal": "NORMAL"}
        
        for i, res_name in enumerate(reversed_order):
            res_obj = self.cascade.reservoirs[res_name]
            
            # Determine downstream status
            if i == 0:
                ds = "NORMAL" if self.cascade.current_downstream_flow <= self.config["topology"]["downstream_capacity"]["value"] else "HIGH RISK"
            else:
                ds = statuses[self.config["topology"]["cascade_order"][-(i)]]
                
            f_data = forecasts.get(res_name, {})
            gate, reason = self.ai_controller.compute_gate(
                res_name, res_obj, f_data, self.historical_thresholds, ds
            )
            recommendations[res_name] = gate
            
            # Predict status for upstream
            if gate == 100.0: statuses[res_name] = "HIGH RISK"
            elif gate >= 50.0: statuses[res_name] = "ALERT"
            else: statuses[res_name] = "NORMAL"
            
        return recommendations
        
    def get_state(self, forecasts: dict) -> dict:
        """
        Returns a complete state snapshot of the cascade for the 3D frontend.
        """
        state = {
            "reservoirs": {},
            "downstream_flow": self.cascade.current_downstream_flow,
            "downstream_capacity": self.cascade.downstream_capacity,
            # ── STAGE 11 — LIVE MASS-BALANCE INTEGRITY ────────────────────
            # The audit of the most recent authoritative step, produced by the
            # physics itself. It is the backend's record; the UI displays it and
            # never computes it.
            "mass_balance": self.cascade.mass_balance_diagnostic(),
        }
        
        for res_name in self.cascade.cascade_order:
            res_obj = self.cascade.reservoirs[res_name]
            
            # Get physical state
            storage_pct = res_obj.get_simulated_water_level_proxy()
            
            # Compute risk
            real_name = self.config["reservoirs"][res_name]["repository_derived_source"]
            hist_thresh = self.historical_thresholds.get(real_name)            
            metadata = {
                "blueLevel": 75.0,
                "orangeLevel": 85.0,
                "redLevel": 95.0,
                "historical_95th_inflow": hist_thresh
            }
            
            f_data = forecasts.get(res_name, {})
            f1, f3, f7 = f_data.get("forecast_1d"), f_data.get("forecast_3d"), f_data.get("forecast_7d")
            
            if f1 is None:
                risk = assess_risk({"waterLevel": storage_pct}, metadata, None, None, None)
            else:
                risk = assess_risk({"waterLevel": storage_pct}, metadata, f1, f3, f7)
            
            state["reservoirs"][res_name] = {
                "storage_mcm": res_obj.state.storage_mcm,
                "capacity_mcm": res_obj.capacity_mcm,
                "storage_pct": storage_pct,
                "inflow": res_obj.state.inflow_mcm_day,
                "routed_inflow": res_obj.state.upstream_routed_inflow,
                "outflow": res_obj.state.release_mcm_day,
                "gate_position_pct": res_obj.state.gate_position_pct,
                "risk_status": risk["overall_status"],
                "risk_reason": risk["reason"],
                "overflow": res_obj.state.overflow_events > 0,
                "forecast_1d": f1,
                "forecast_3d": f3,
                "forecast_7d": f7,
                # ── STAGE 9 — reservoir identity (authoritative, not UI-hardcoded) ──
                # The Digital Twin must name each reservoir from the SAME source
                # the physics uses, so the payload can never present Reservoir A
                # data under Reservoir D's name (which is what the hardcoded
                # three-name list in the UI used to do).
                "node_id": res_name,
                "repository_name": real_name,
                "cascade_position": self.cascade.cascade_order.index(res_name),
                "is_terminal": res_name == self.cascade.cascade_order[-1],
                # ── Stage 5 forecast provenance (pass-through, never invented) ──
                "forecast_status": f_data.get("forecast_status"),
                "forecast_provenance": f_data.get("forecast_provenance"),
                "forecast_source": f_data.get("forecast_source"),
                "is_simulated": f_data.get("is_simulated"),
                "validated_metrics_apply": f_data.get("validated_metrics_apply"),
                "forecast_unavailable_features": f_data.get("unavailable_features"),
            }
            
        return state
